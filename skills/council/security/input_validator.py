"""
Input validation and sanitization for Council skill.

Provides defense-in-depth protection against:
- Shell injection attacks (RCE prevention)
- Prompt injection attacks (LLM manipulation)
- Secret leakage (API key redaction)
- DoS attacks (input length limits)
"""

import re
import shlex
from dataclasses import dataclass
from typing import List, Dict, Tuple


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
        Validate user query for injection attacks.

        Args:
            query: User-provided query string
            strict: If True, fail immediately on any suspicious pattern.
                    If False, continue checking all patterns before returning.

        Returns:
            ValidationResult with sanitized query and violations list.
            is_valid=False if any violations detected (regardless of strict mode).
        """
        violations = []

        # 1. Length validation
        if len(query) > self.MAX_QUERY_LENGTH:
            violations.append(
                f"Query exceeds maximum length ({len(query)} > {self.MAX_QUERY_LENGTH})"
            )
            if strict:
                return ValidationResult(False, "", violations)
            # Truncate if not strict
            query = query[:self.MAX_QUERY_LENGTH]

        # 2. Shell injection detection
        shell_violations = self._check_shell_injection(query)
        violations.extend(shell_violations)

        if strict and shell_violations:
            return ValidationResult(False, "", violations)

        # 3. Prompt injection detection
        prompt_violations = self._check_prompt_injection(query)
        violations.extend(prompt_violations)

        if strict and prompt_violations:
            return ValidationResult(False, "", violations)

        # 4. Sanitize (shell escape)
        # NOTE: We don't use shlex.quote() on the entire query since it's passed
        # as a CLI argument, not through shell. The subprocess uses args list, not shell.
        # Instead, we just detect and warn.
        sanitized = query

        return ValidationResult(
            is_valid=len(violations) == 0,
            sanitized_input=sanitized,
            violations=violations
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
        """Detect shell injection patterns."""
        violations = []

        for pattern, description in self.SHELL_INJECTION_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                violations.append(
                    f"Shell injection risk: {description} found ({len(matches)} occurrence(s))"
                )

        return violations

    def _check_prompt_injection(self, text: str) -> List[str]:
        """Detect prompt injection patterns."""
        violations = []

        for pattern, description in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
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
    """
    validator = InputValidator()

    # Validate query
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

    return {
        'is_valid': query_result.is_valid and len(config_violations) == 0,
        'query': query_result.sanitized_input,
        'context': context_result.sanitized_input if context_result else None,
        'max_rounds': sanitized_rounds,
        'timeout': sanitized_timeout,
        'violations': all_violations,
        'redacted_secrets': context_result.redacted_secrets if context_result else []
    }
