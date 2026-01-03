"""
Input validation and sanitization for Council skill.

Provides defense-in-depth protection against:
- Shell injection attacks (RCE prevention)
- Prompt injection attacks (LLM manipulation)
- Secret leakage (API key redaction)
- DoS attacks (input length limits, rate limiting)
- Homoglyph attacks (Unicode normalization)
"""

import re
import shlex
import time
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ValidationResult:
    """Result of input validation with sanitized output."""
    is_valid: bool
    sanitized_input: str
    violations: List[str]
    redacted_secrets: List[str] = None

    def __post_init__(self):
        if self.redacted_secrets is None:
            self.redacted_secrets = []


# =============================================================================
# Rate Limiter (P1 - DoS Protection)
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_sessions_per_minute: int = 10       # Max new sessions per minute
    max_queries_per_session: int = 50       # Max queries in a single session
    max_total_time_seconds: int = 3600      # Max total deliberation time (1 hour)
    max_model_calls_per_session: int = 100  # Max individual model calls
    window_seconds: int = 60                # Time window for rate calculations


class RateLimiter:
    """
    Rate limiter to prevent DoS attacks on the council.

    Tracks:
    - Sessions per time window (prevents rapid session creation)
    - Queries per session (prevents query flooding)
    - Total time per session (prevents indefinite deliberation)
    - Model calls per session (prevents resource exhaustion)

    Thread-safe: All operations protected by lock.
    """

    def __init__(self, config: RateLimitConfig = None):
        self._lock = threading.RLock()
        self.config = config or RateLimitConfig()
        self._session_starts: List[float] = []  # Timestamps of session starts
        self._sessions: Dict[str, dict] = {}    # session_id -> tracking data

    def check_new_session(self) -> Tuple[bool, Optional[str]]:
        """
        Check if a new session can be started.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        with self._lock:
            now = time.time()
            window_start = now - self.config.window_seconds

            # Clean old entries
            self._session_starts = [t for t in self._session_starts if t > window_start]

            # Check rate
            if len(self._session_starts) >= self.config.max_sessions_per_minute:
                return False, f"Rate limit exceeded: {self.config.max_sessions_per_minute} sessions per minute"

            return True, None

    def start_session(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Register a new session start.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        allowed, reason = self.check_new_session()
        if not allowed:
            return allowed, reason

        with self._lock:
            now = time.time()
            self._session_starts.append(now)
            self._sessions[session_id] = {
                'start_time': now,
                'query_count': 0,
                'model_calls': 0,
                'last_activity': now
            }
            return True, None

    def check_query(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a query can be processed in the given session.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        with self._lock:
            if session_id not in self._sessions:
                return False, "Session not found"

            session = self._sessions[session_id]
            now = time.time()

            # Check session time limit
            elapsed = now - session['start_time']
            if elapsed > self.config.max_total_time_seconds:
                return False, f"Session time limit exceeded ({elapsed:.0f}s > {self.config.max_total_time_seconds}s)"

            # Check query count
            if session['query_count'] >= self.config.max_queries_per_session:
                return False, f"Query limit exceeded ({self.config.max_queries_per_session} per session)"

            return True, None

    def record_query(self, session_id: str):
        """Record a query in the session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]['query_count'] += 1
                self._sessions[session_id]['last_activity'] = time.time()

    def check_model_call(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a model call can be made in the given session.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        with self._lock:
            if session_id not in self._sessions:
                return False, "Session not found"

            session = self._sessions[session_id]

            if session['model_calls'] >= self.config.max_model_calls_per_session:
                return False, f"Model call limit exceeded ({self.config.max_model_calls_per_session} per session)"

            return True, None

    def record_model_call(self, session_id: str):
        """Record a model call in the session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]['model_calls'] += 1
                self._sessions[session_id]['last_activity'] = time.time()

    def end_session(self, session_id: str):
        """End a session and clean up."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def get_session_stats(self, session_id: str) -> Optional[dict]:
        """Get stats for a session."""
        with self._lock:
            if session_id not in self._sessions:
                return None

            session = self._sessions[session_id]
            now = time.time()
            return {
                'query_count': session['query_count'],
                'model_calls': session['model_calls'],
                'elapsed_seconds': now - session['start_time'],
                'remaining_queries': self.config.max_queries_per_session - session['query_count'],
                'remaining_model_calls': self.config.max_model_calls_per_session - session['model_calls'],
                'remaining_time': max(0, self.config.max_total_time_seconds - (now - session['start_time']))
            }

    def cleanup_stale_sessions(self, max_idle_seconds: int = 1800):
        """Remove sessions that have been idle too long."""
        with self._lock:
            now = time.time()
            stale = [
                sid for sid, data in self._sessions.items()
                if now - data['last_activity'] > max_idle_seconds
            ]
            for sid in stale:
                del self._sessions[sid]
            return len(stale)


# Global rate limiter instance
RATE_LIMITER = RateLimiter()


# =============================================================================
# Unicode Normalization (P2 - Homoglyph Protection)
# =============================================================================

# Common homoglyph mappings (confusable characters)
HOMOGLYPH_MAP = {
    # Cyrillic lookalikes
    'а': 'a', 'е': 'e', 'і': 'i', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y',
    'А': 'A', 'Е': 'E', 'І': 'I', 'О': 'O', 'Р': 'P', 'С': 'C', 'У': 'Y',
    # Greek lookalikes
    'α': 'a', 'ε': 'e', 'ι': 'i', 'ο': 'o', 'ρ': 'p', 'υ': 'u',
    'Α': 'A', 'Ε': 'E', 'Ι': 'I', 'Ο': 'O', 'Ρ': 'P',
    # Numbers as letters
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't',
    # Special characters
    'ⅰ': 'i', 'ⅼ': 'l', 'ℓ': 'l', '℮': 'e',
    # Fullwidth
    'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f',
    'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l',
    'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r',
    'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
    'ｙ': 'y', 'ｚ': 'z',
}


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to prevent homoglyph attacks.

    Steps:
    1. NFKC normalization (compatibility decomposition + canonical composition)
    2. Replace known homoglyphs with ASCII equivalents
    3. Remove zero-width characters

    Args:
        text: Input text possibly containing homoglyphs

    Returns:
        Normalized ASCII-like text for security checks
    """
    if not text:
        return text

    # Step 1: NFKC normalization
    normalized = unicodedata.normalize('NFKC', text)

    # Step 2: Replace known homoglyphs
    result = []
    for char in normalized:
        result.append(HOMOGLYPH_MAP.get(char, char))
    normalized = ''.join(result)

    # Step 3: Remove zero-width characters (invisible injection)
    zero_width = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u2060',  # Word joiner
        '\ufeff',  # BOM / zero-width no-break space
    ]
    for zw in zero_width:
        normalized = normalized.replace(zw, '')

    return normalized


class InputValidator:
    """
    Defense-in-depth input validation for council queries.

    Protects against:
    - Shell injection (CWE-78)
    - Prompt injection (OWASP LLM01)
    - Sensitive data disclosure (OWASP LLM06)
    """

    # ============================================================================
    # Shell Injection Patterns
    # ============================================================================

    SHELL_INJECTION_PATTERNS = [
        (r'[;&|`]', 'Shell operator'),
        (r'\$\(', 'Command substitution'),
        (r'\$\{', 'Variable expansion'),
        (r'>\s*[/\w]', 'Output redirection'),
        (r'<\s*[/\w]', 'Input redirection'),
        (r'\.\./\.\.', 'Path traversal'),
        (r'&&|\|\|', 'Logical operators'),
    ]

    # ============================================================================
    # Prompt Injection Patterns
    # ============================================================================

    PROMPT_INJECTION_PATTERNS = [
        (r'ignore\s+(all\s+)?(previous|above|prior|all)\s+(instructions?|prompts?|rules?)',
         'Instruction override attempt'),
        (r'you\s+are\s+now\s+(in\s+)?(admin|developer|debug|god)\s+mode',
         'Privilege escalation'),
        (r'<\s*/?system\s*>',
         'System tag injection'),
        (r'<\s*/?s\s*>',
         'Instruction tag mimicry'),
        (r'new\s+(instructions?|role|persona|system)',
         'Role redefinition'),
        (r'forget\s+(everything|all|previous)',
         'Memory manipulation'),
        (r'reveal\s+(your\s+)?(prompt|instructions|system)',
         'Prompt extraction'),
        (r'output\s+(your|the)\s+(code|source|prompt)',
         'Source disclosure'),
    ]

    # ============================================================================
    # Secret Patterns (OWASP LLM06)
    # ============================================================================

    SECRET_PATTERNS = {
        'openai_proj_key': (r'sk-proj-[a-zA-Z0-9_-]{12,}', '[REDACTED_OPENAI_PROJECT_KEY]'),
        'openai_key': (r'sk-[a-zA-Z0-9]{12,}', '[REDACTED_OPENAI_KEY]'),
        'google_key': (r'AIza[a-zA-Z0-9_-]{12,}', '[REDACTED_GOOGLE_KEY]'),
        'github_token': (r'ghp_[a-zA-Z0-9]{12,}', '[REDACTED_GITHUB_TOKEN]'),
        'github_oauth': (r'gho_[a-zA-Z0-9]{12,}', '[REDACTED_GITHUB_OAUTH]'),
        'aws_key': (r'AKIA[0-9A-Z]{16}', '[REDACTED_AWS_KEY]'),
        'slack_token': (r'xox[baprs]-[0-9]{10,13}-[a-zA-Z0-9-]{16,}', '[REDACTED_SLACK_TOKEN]'),
        'stripe_key': (r'sk_live_[a-zA-Z0-9]{16,}', '[REDACTED_STRIPE_KEY]'),
        'generic_bearer': (r'Bearer\s+[a-zA-Z0-9_\-\.]{16,}', 'Bearer [REDACTED_TOKEN]'),
        'jwt_token': (r'eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+', '[REDACTED_JWT]'),
        'password_literal': (r'password\s*[:=]\s*["\']([^"\']{6,})["\']', 'password=[REDACTED]'),
        'generic_api_key': (r'api[_-]?key\s*[:=]\s*["\']([a-zA-Z0-9_-]{16,})["\']', 'api_key=[REDACTED]'),
    }

    # ============================================================================
    # Input Limits
    # ============================================================================

    MAX_QUERY_LENGTH = 50000      # 50k chars
    MAX_CONTEXT_LENGTH = 200000   # 200k chars (for code files)
    MAX_ROUNDS = 10                # Prevent DoS
    MAX_TIMEOUT = 420              # 7 minutes max (Codex needs time for code exploration)

    # ============================================================================
    # Validation Methods
    # ============================================================================

    def validate_query(self, query: str, strict: bool = False) -> ValidationResult:
        """
        Validate user query for injection attacks and redact secrets.

        Args:
            query: User-provided query string
            strict: If True, fail immediately on any suspicious pattern.
                    If False, continue checking all patterns before returning.

        Returns:
            ValidationResult with sanitized query, violations list, and redacted secrets.
            is_valid=False if any violations detected (regardless of strict mode).

        Security:
            - Redacts API keys, tokens, and other secrets (OWASP LLM06)
            - Detects shell injection patterns (CWE-78)
            - Detects prompt injection patterns (OWASP LLM01)
        """
        violations = []
        redacted_secrets = []

        # 1. Length validation
        if len(query) > self.MAX_QUERY_LENGTH:
            violations.append(
                f"Query exceeds maximum length ({len(query)} > {self.MAX_QUERY_LENGTH})"
            )
            if strict:
                return ValidationResult(False, "", violations)
            # Truncate if not strict
            query = query[:self.MAX_QUERY_LENGTH]

        # 2. Secret scanning and redaction (OWASP LLM06)
        # This is critical - users may accidentally paste API keys in queries
        sanitized, found_secrets = self._redact_secrets(query)
        if found_secrets:
            redacted_secrets = found_secrets
            violations.extend([f"Redacted {secret_type} from query" for secret_type in found_secrets])

        # 3. Shell injection detection
        shell_violations = self._check_shell_injection(sanitized)
        violations.extend(shell_violations)

        if strict and shell_violations:
            return ValidationResult(False, "", violations, redacted_secrets)

        # 4. Prompt injection detection
        prompt_violations = self._check_prompt_injection(sanitized)
        violations.extend(prompt_violations)

        if strict and prompt_violations:
            return ValidationResult(False, "", violations, redacted_secrets)

        return ValidationResult(
            is_valid=len([v for v in violations if 'Redacted' not in v]) == 0,
            sanitized_input=sanitized,
            violations=violations,
            redacted_secrets=redacted_secrets
        )

    def validate_context(self, context: str) -> ValidationResult:
        """
        Validate code context for secrets and injection patterns.

        Args:
            context: User-provided code or additional context

        Returns:
            ValidationResult with redacted context and secret violations
        """
        violations = []
        redacted_secrets = []

        # 1. Length validation
        if len(context) > self.MAX_CONTEXT_LENGTH:
            violations.append(
                f"Context exceeds maximum length ({len(context)} > {self.MAX_CONTEXT_LENGTH})"
            )
            context = context[:self.MAX_CONTEXT_LENGTH]

        # 2. Secret scanning and redaction
        sanitized, found_secrets = self._redact_secrets(context)
        if found_secrets:
            redacted_secrets = found_secrets
            violations.extend([f"Redacted {secret_type}" for secret_type in found_secrets])

        return ValidationResult(
            is_valid=True,  # Context is always valid after redaction
            sanitized_input=sanitized,
            violations=violations,
            redacted_secrets=redacted_secrets
        )

    def validate_config(self, max_rounds: int, timeout: int) -> Tuple[int, int, List[str]]:
        """
        Validate configuration parameters.

        Args:
            max_rounds: Maximum deliberation rounds
            timeout: Per-model timeout in seconds

        Returns:
            Tuple of (sanitized_max_rounds, sanitized_timeout, violations)
        """
        violations = []

        # Validate max_rounds
        if max_rounds < 1:
            violations.append(f"max_rounds must be >= 1, got {max_rounds}")
            max_rounds = 1
        elif max_rounds > self.MAX_ROUNDS:
            violations.append(f"max_rounds exceeds limit ({max_rounds} > {self.MAX_ROUNDS})")
            max_rounds = self.MAX_ROUNDS

        # Validate timeout
        if timeout < 10:
            violations.append(f"timeout must be >= 10s, got {timeout}")
            timeout = 10
        elif timeout > self.MAX_TIMEOUT:
            violations.append(f"timeout exceeds limit ({timeout} > {self.MAX_TIMEOUT})")
            timeout = self.MAX_TIMEOUT

        return max_rounds, timeout, violations

    # ============================================================================
    # Private Detection Methods
    # ============================================================================

    def _check_shell_injection(self, text: str) -> List[str]:
        """
        Detect shell injection patterns.

        Security: Normalizes Unicode before checking to prevent homoglyph bypass.
        """
        violations = []

        # Normalize to catch homoglyph attacks (e.g., Cyrillic 'а' instead of 'a')
        normalized = normalize_unicode(text)

        for pattern, description in self.SHELL_INJECTION_PATTERNS:
            matches = re.findall(pattern, normalized)
            if matches:
                violations.append(
                    f"Shell injection risk: {description} found ({len(matches)} occurrence(s))"
                )

        return violations

    def _check_prompt_injection(self, text: str) -> List[str]:
        """
        Detect prompt injection patterns.

        Security: Normalizes Unicode before checking to prevent homoglyph bypass.
        For example, "ign0re previous instructions" (with zero instead of 'o')
        will be normalized to "ignore previous instructions" before matching.
        """
        violations = []

        # Normalize to catch homoglyph attacks
        normalized = normalize_unicode(text)

        for pattern, description in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                violations.append(
                    f"Prompt injection risk: {description}"
                )

        return violations

    def _redact_secrets(self, text: str) -> Tuple[str, List[str]]:
        """
        Redact secrets from text.

        Returns:
            Tuple of (redacted_text, list_of_secret_types_found)
        """
        result = text
        found_secrets = []

        for secret_type, (pattern, replacement) in self.SECRET_PATTERNS.items():
            matches = re.findall(pattern, result)
            if matches:
                found_secrets.append(secret_type)
                result = re.sub(pattern, replacement, result)

        return result, found_secrets

    def redact_output(self, output: dict) -> dict:
        """
        Redact secrets from output JSON before emission.

        Recursively scans all string values in the output dict.

        Args:
            output: Output dict to redact

        Returns:
            Redacted output dict
        """
        if isinstance(output, dict):
            return {k: self.redact_output(v) for k, v in output.items()}
        elif isinstance(output, list):
            return [self.redact_output(item) for item in output]
        elif isinstance(output, str):
            redacted, _ = self._redact_secrets(output)
            return redacted
        else:
            return output

    def sanitize_llm_output(self, output: str, max_length: int = 50000) -> str:
        """
        Sanitize LLM output before passing to subsequent rounds.

        Prevents cross-model prompt injection by:
        1. Redacting any leaked secrets
        2. Escaping injection patterns (system tags, role overrides)
        3. Limiting output length

        Args:
            output: Raw LLM output string
            max_length: Maximum allowed length

        Returns:
            Sanitized output safe for cross-model use
        """
        if not output:
            return output

        # 1. Truncate to max length
        if len(output) > max_length:
            output = output[:max_length] + "\n[TRUNCATED]"

        # 2. Redact secrets
        output, _ = self._redact_secrets(output)

        # 3. Escape injection patterns - replace dangerous tags with safe versions
        # Escape system/instruction tags that could manipulate next model
        output = re.sub(r'<\s*/?\s*system\s*>', '[system]', output, flags=re.IGNORECASE)
        output = re.sub(r'<\s*/?\s*s\s*>', '[s]', output, flags=re.IGNORECASE)
        output = re.sub(r'<\s*/?\s*instructions?\s*>', '[instruction]', output, flags=re.IGNORECASE)

        # Escape role override attempts
        output = re.sub(
            r'(you\s+are\s+now\s+(in\s+)?(admin|developer|debug|god)\s+mode)',
            r'[BLOCKED: \1]',
            output,
            flags=re.IGNORECASE
        )
        output = re.sub(
            r'(ignore\s+(all\s+)?(previous|above|prior|all)\s+(instructions?|prompts?|rules?))',
            r'[BLOCKED: \1]',
            output,
            flags=re.IGNORECASE
        )
        output = re.sub(
            r'(forget\s+(everything|all|previous))',
            r'[BLOCKED: \1]',
            output,
            flags=re.IGNORECASE
        )

        return output


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_and_sanitize(query: str, context: str = None,
                         max_rounds: int = 3, timeout: int = 60,
                         strict: bool = False) -> Dict:
    """
    Convenience function for complete validation.

    Args:
        query: User query
        context: Optional code context
        max_rounds: Max deliberation rounds
        timeout: Per-model timeout
        strict: Fail on any violation if True

    Returns:
        Dict with sanitized inputs and validation status

    Security:
        - Redacts secrets from BOTH query AND context (OWASP LLM06)
        - Detects shell injection (CWE-78)
        - Detects prompt injection (OWASP LLM01)
    """
    validator = InputValidator()

    # Validate query (now includes secret redaction)
    query_result = validator.validate_query(query, strict=strict)

    # Validate context
    context_result = None
    if context:
        context_result = validator.validate_context(context)

    # Validate config
    sanitized_rounds, sanitized_timeout, config_violations = validator.validate_config(
        max_rounds, timeout
    )

    all_violations = query_result.violations + config_violations
    if context_result:
        all_violations.extend(context_result.violations)

    # Collect redacted secrets from BOTH query and context
    all_redacted_secrets = list(query_result.redacted_secrets)
    if context_result:
        all_redacted_secrets.extend(context_result.redacted_secrets)

    return {
        'is_valid': query_result.is_valid and len(config_violations) == 0,
        'query': query_result.sanitized_input,
        'context': context_result.sanitized_input if context_result else None,
        'max_rounds': sanitized_rounds,
        'timeout': sanitized_timeout,
        'violations': all_violations,
        'redacted_secrets': all_redacted_secrets
    }
