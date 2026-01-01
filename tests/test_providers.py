"""
Tests for the SDK provider abstraction layer.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure council package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "council"))

from model_providers.base import ProviderProtocol, BaseProvider
from model_providers.cli_provider import CLIProvider, CLI_CONFIGS
from model_providers.factory import get_provider, get_available_providers, PROVIDER_PREFERENCES
from core.models import LLMResponse


class TestBaseProvider:
    """Tests for BaseProvider."""

    def test_create_response_includes_timing(self):
        """Verify _create_response calculates latency."""
        import time

        class TestProvider(BaseProvider):
            def is_available(self):
                return True

            async def query(self, prompt, timeout):
                pass

        provider = TestProvider("test")
        start = time.time() - 0.5  # Simulate 500ms delay

        response = provider._create_response("content", start, success=True)

        assert response.success is True
        assert response.content == "content"
        assert response.latency_ms >= 500  # At least 500ms
        assert response.model == "test"

    def test_create_error_response(self):
        """Verify _create_error_response sets error fields."""
        import time

        class TestProvider(BaseProvider):
            def is_available(self):
                return True

            async def query(self, prompt, timeout):
                pass

        provider = TestProvider("test")
        start = time.time()

        response = provider._create_error_response("connection failed", start)

        assert response.success is False
        assert response.error == "connection failed"
        assert response.content == ""


class TestCLIProvider:
    """Tests for CLIProvider."""

    def test_cli_configs_exist(self):
        """Verify CLI configs exist for all models."""
        assert "claude" in CLI_CONFIGS
        assert "gemini" in CLI_CONFIGS
        assert "codex" in CLI_CONFIGS

    def test_codex_uses_stdin(self):
        """Codex should use stdin for prompts."""
        assert CLI_CONFIGS["codex"].use_stdin is True

    def test_claude_uses_json_output(self):
        """Claude should use JSON output format."""
        assert "--output-format" in CLI_CONFIGS["claude"].args
        assert "json" in CLI_CONFIGS["claude"].args

    def test_unknown_model_raises_error(self):
        """Unknown models should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown CLI model"):
            CLIProvider("unknown_model")

    @patch("shutil.which")
    def test_is_available_with_cli(self, mock_which):
        """is_available should check CLI presence."""
        mock_which.return_value = "/usr/bin/claude"

        provider = CLIProvider("claude")
        # Clear cache before test
        provider.is_available.cache_clear()

        assert provider.is_available() is True
        mock_which.assert_called_with("claude")

    @patch("shutil.which")
    def test_is_available_without_cli(self, mock_which):
        """is_available should return False if CLI not found."""
        mock_which.return_value = None

        provider = CLIProvider("claude")
        provider.is_available.cache_clear()

        assert provider.is_available() is False


class TestProviderFactory:
    """Tests for the provider factory."""

    def test_provider_preferences_exist(self):
        """Verify preferences are defined for all models."""
        assert "claude" in PROVIDER_PREFERENCES
        assert "gemini" in PROVIDER_PREFERENCES
        assert "codex" in PROVIDER_PREFERENCES

    def test_codex_only_has_cli(self):
        """Codex should only have CLI provider (no SDK without API key)."""
        assert PROVIDER_PREFERENCES["codex"] == ["cli"]

    def test_gemini_only_has_cli(self):
        """Gemini should only have CLI provider (SDK requires Google Cloud)."""
        assert PROVIDER_PREFERENCES["gemini"] == ["cli"]

    def test_claude_prefers_sdk(self):
        """Claude should prefer SDK over CLI."""
        prefs = PROVIDER_PREFERENCES["claude"]
        assert prefs[0] == "sdk"
        assert "cli" in prefs

    @patch("model_providers.factory._get_claude_sdk_provider")
    @patch("shutil.which")
    def test_get_provider_returns_cli_fallback(self, mock_which, mock_sdk):
        """When SDK unavailable, should fall back to CLI."""
        mock_sdk.return_value = None  # No SDK
        mock_which.return_value = "/usr/bin/claude"  # CLI available

        provider = get_provider("claude")

        assert isinstance(provider, CLIProvider)
        assert provider.name == "claude"

    def test_unknown_model_raises_error(self):
        """Unknown models should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_provider("unknown_model")


class TestClaudeSDKProvider:
    """Tests for Claude SDK provider (mocked)."""

    def test_import_check(self):
        """Verify claude_sdk module can be imported (even if SDK not installed)."""
        try:
            from model_providers.claude_sdk import ClaudeSDKProvider, _check_claude_sdk
            # Module exists, test passes
            assert True
        except ImportError:
            pytest.skip("claude-agent-sdk not installed")

    @patch("model_providers.claude_sdk.shutil.which")
    def test_is_available_checks_cli(self, mock_which):
        """ClaudeSDKProvider.is_available should check CLI presence."""
        try:
            from model_providers.claude_sdk import ClaudeSDKProvider, _check_claude_sdk

            # Reset the cached check
            import model_providers.claude_sdk
            model_providers.claude_sdk._claude_sdk_available = None

            mock_which.return_value = "/usr/bin/claude"
            provider = ClaudeSDKProvider()

            # is_available checks both SDK availability and CLI presence
            result = provider.is_available()

            # If SDK not installed, it returns False before checking CLI
            # If SDK is installed, it should call shutil.which
            if _check_claude_sdk():
                mock_which.assert_called_with("claude")
        except ImportError:
            pytest.skip("claude-agent-sdk not installed")


class TestGeminiSDKProvider:
    """Tests for Gemini SDK provider (mocked)."""

    def test_import_check(self):
        """Verify gemini_sdk module can be imported (even if SDK not installed)."""
        try:
            from model_providers.gemini_sdk import GeminiSDKProvider, _check_gemini_sdk
            assert True
        except ImportError:
            pytest.skip("google-generativeai not installed")

    def test_supports_streaming(self):
        """GeminiSDKProvider should support streaming."""
        try:
            from model_providers.gemini_sdk import GeminiSDKProvider
            provider = GeminiSDKProvider()
            assert provider.supports_streaming() is True
        except ImportError:
            pytest.skip("google-generativeai not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
