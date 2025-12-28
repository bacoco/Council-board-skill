# Security Guidelines

Based on OWASP LLM Top 10 (2025).

## LLM01: Prompt Injection

**Risk**: Malicious instructions via query or inter-model responses.

**Mitigations**:
1. XML sandwich prompts with clear DATA/INSTRUCTION separation
2. Input sanitization detecting attack patterns
3. Reminder blocks after user content

```python
INJECTION_PATTERNS = [
    r"ignore.*(?:previous|above).*instructions",
    r"you are now",
    r"new instruction:",
    r"```system",
]
```

## LLM06: Sensitive Information Disclosure

**Risk**: Secrets leaked to models or in outputs.

**Mitigations**:

Input Redaction:
```python
SECRET_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{48}', '[REDACTED_OPENAI_KEY]'),
    (r'AIza[a-zA-Z0-9_-]{35}', '[REDACTED_GOOGLE_KEY]'),
    (r'ghp_[a-zA-Z0-9]{36}', '[REDACTED_GITHUB_TOKEN]'),
    (r'(?i)password\s*[:=]\s*\S+', 'password=[REDACTED]'),
]
```

Egress Filtering:
- Scan outputs for patterns that shouldn't appear
- Filter private keys, paths, bearer tokens

## LLM07: Insecure Plugin Design

**Risk**: CLIs may execute code or access files.

**Mitigations**:
- `tools_allowed=false` by default
- Isolated workdir: `/tmp/council-sandbox-{session_id}/`
- Minimal environment variables
- `--no-tools`, `--text-only` flags when supported

## Anonymization (Anti-Bias)

Stage 2 responses anonymized to prevent:
- Brand bias (Claude favoring Claude)
- Targeted attacks between models

```python
def anonymize_responses(responses: dict) -> tuple[dict, dict]:
    labels = ['A', 'B', 'C', 'D', 'E']
    shuffled = list(responses.keys())
    random.shuffle(shuffled)
    anonymized = {label: normalize_style(responses[model]) 
                  for label, model in zip(labels, shuffled)}
    mapping = dict(zip(labels, shuffled))
    return anonymized, mapping
```

## Style Normalization

Responses normalized to hide model signatures:
- Consistent length (truncate/pad)
- Remove markdown formatting
- Neutral tone
- Strip model-specific phrases
