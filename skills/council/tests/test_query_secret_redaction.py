"""
Tests for secret redaction in query validation.

These tests verify that secrets (API keys, tokens, passwords) are
properly redacted from user queries, not just from context.
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.input_validator import InputValidator, validate_and_sanitize


class TestQuerySecretRedaction:
    """Test that secrets are redacted from queries."""

    def test_redacts_openai_key_from_query(self):
        """OpenAI API keys should be redacted from queries."""
        validator = InputValidator()

        query = "Why is my API call failing with key sk-proj-abc123xyz456789012345?"
        result = validator.validate_query(query)

        assert "sk-proj" not in result.sanitized_input
        assert "[REDACTED_OPENAI_PROJECT_KEY]" in result.sanitized_input
        assert "openai_proj_key" in result.redacted_secrets

    def test_redacts_github_token_from_query(self):
        """GitHub tokens should be redacted from queries."""
        validator = InputValidator()

        query = "My git push failed with token ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        result = validator.validate_query(query)

        assert "ghp_" not in result.sanitized_input
        assert "[REDACTED_GITHUB_TOKEN]" in result.sanitized_input
        assert "github_token" in result.redacted_secrets

    def test_redacts_aws_key_from_query(self):
        """AWS access keys should be redacted from queries."""
        validator = InputValidator()

        query = "Configure S3 with access key AKIAIOSFODNN7EXAMPLE"
        result = validator.validate_query(query)

        assert "AKIAIOSFODNN7EXAMPLE" not in result.sanitized_input
        assert "[REDACTED_AWS_KEY]" in result.sanitized_input
        assert "aws_key" in result.redacted_secrets

    def test_redacts_jwt_from_query(self):
        """JWT tokens should be redacted from queries."""
        validator = InputValidator()

        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        query = f"Decode this token: {jwt}"
        result = validator.validate_query(query)

        assert "eyJ" not in result.sanitized_input
        assert "[REDACTED_JWT]" in result.sanitized_input
        assert "jwt_token" in result.redacted_secrets

    def test_redacts_bearer_token_from_query(self):
        """Bearer tokens should be redacted from queries."""
        validator = InputValidator()

        query = "Authorization header: Bearer abc123def456ghi789jkl012mno345"
        result = validator.validate_query(query)

        assert "abc123def456" not in result.sanitized_input
        assert "Bearer [REDACTED_TOKEN]" in result.sanitized_input
        assert "generic_bearer" in result.redacted_secrets

    def test_redacts_password_from_query(self):
        """Passwords in common formats should be redacted."""
        validator = InputValidator()

        query = 'Database connection with password="mysecretpassword123"'
        result = validator.validate_query(query)

        assert "mysecretpassword123" not in result.sanitized_input
        assert "password=[REDACTED]" in result.sanitized_input
        assert "password_literal" in result.redacted_secrets

    def test_redacts_multiple_secrets_from_query(self):
        """Multiple secrets in one query should all be redacted."""
        validator = InputValidator()

        query = """
        My config uses:
        - OpenAI key: sk-proj-abc123xyz456789012345
        - GitHub token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Please help debug.
        """
        result = validator.validate_query(query)

        assert "sk-proj" not in result.sanitized_input
        assert "ghp_" not in result.sanitized_input
        assert len(result.redacted_secrets) >= 2
        assert "openai_proj_key" in result.redacted_secrets
        assert "github_token" in result.redacted_secrets

    def test_query_still_valid_after_redaction(self):
        """Query should remain valid after secret redaction."""
        validator = InputValidator()

        query = "Why does sk-proj-abc123xyz456789012345 not work?"
        result = validator.validate_query(query)

        # Query is still valid - secrets are informational, not blocking
        assert result.is_valid == True
        # Violation was logged but doesn't make query invalid
        assert any("Redacted" in v for v in result.violations)

    def test_strict_mode_doesnt_fail_on_secrets(self):
        """Strict mode should not fail on secrets (only on injection)."""
        validator = InputValidator()

        query = "Check my key sk-proj-abc123xyz456789012345"
        result = validator.validate_query(query, strict=True)

        # Should still be valid - secrets are redacted, not rejected
        assert result.is_valid == True
        assert "sk-proj" not in result.sanitized_input


class TestValidateAndSanitizeSecrets:
    """Test the convenience function handles secrets from both query and context."""

    def test_collects_secrets_from_query(self):
        """validate_and_sanitize should collect secrets from query."""
        result = validate_and_sanitize(
            query="My key is sk-proj-abc123xyz456789012345",
            context=None
        )

        assert "openai_proj_key" in result['redacted_secrets']
        assert "sk-proj" not in result['query']

    def test_collects_secrets_from_context(self):
        """validate_and_sanitize should collect secrets from context."""
        result = validate_and_sanitize(
            query="Check this config",
            context="API_KEY=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )

        assert "github_token" in result['redacted_secrets']
        assert "ghp_" not in result['context']

    def test_collects_secrets_from_both(self):
        """validate_and_sanitize should collect secrets from both query and context."""
        result = validate_and_sanitize(
            query="Debug my sk-proj-abc123xyz456789012345 usage",
            context="GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )

        assert "openai_proj_key" in result['redacted_secrets']
        assert "github_token" in result['redacted_secrets']
        assert len(result['redacted_secrets']) >= 2

    def test_result_is_valid_with_secrets(self):
        """Query with secrets should still be valid after redaction."""
        result = validate_and_sanitize(
            query="Test sk-proj-abc123xyz456789012345",
            context=None
        )

        assert result['is_valid'] == True
        assert "openai_proj_key" in result['redacted_secrets']


class TestSecretPatterns:
    """Test all secret patterns are properly detected."""

    def test_google_api_key(self):
        """Google API keys (AIza...) should be detected."""
        validator = InputValidator()
        result = validator.validate_query("Key: AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe")

        assert "google_key" in result.redacted_secrets
        assert "AIza" not in result.sanitized_input

    def test_stripe_key(self):
        """Stripe live keys should be detected."""
        validator = InputValidator()
        result = validator.validate_query("Stripe: sk_live_abc123def456ghi789jkl")

        assert "stripe_key" in result.redacted_secrets
        assert "sk_live_" not in result.sanitized_input

    def test_generic_api_key(self):
        """Generic api_key patterns should be detected."""
        validator = InputValidator()
        result = validator.validate_query('Config: api_key="abcdef1234567890abcdef"')

        assert "generic_api_key" in result.redacted_secrets
        assert "abcdef1234567890abcdef" not in result.sanitized_input


class TestNoFalsePositives:
    """Test that normal text is not incorrectly flagged as secrets."""

    def test_short_strings_not_flagged(self):
        """Short strings that look like prefixes should not be flagged."""
        validator = InputValidator()
        result = validator.validate_query("The sk- prefix is used for OpenAI keys")

        # Short "sk-" alone should not trigger (pattern requires 12+ chars)
        assert len(result.redacted_secrets) == 0

    def test_normal_code_not_flagged(self):
        """Normal code without secrets should pass cleanly."""
        validator = InputValidator()
        result = validator.validate_query("""
        def calculate_sum(a, b):
            return a + b

        result = calculate_sum(1, 2)
        print(result)
        """)

        assert len(result.redacted_secrets) == 0
        assert result.is_valid == True

    def test_base64_not_always_flagged(self):
        """Random base64 should not be flagged as JWT unless it has JWT structure."""
        validator = InputValidator()
        # This is base64 but NOT a JWT (doesn't have the eyJ...eyJ... structure)
        result = validator.validate_query("Encode: SGVsbG8gV29ybGQ=")

        assert "jwt_token" not in result.redacted_secrets


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
