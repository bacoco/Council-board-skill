"""
Local authentication token retrieval.

Reads auth tokens from local files stored by CLI tools.
NO API keys, NO cloud auth - just local tokens from existing CLI logins.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LocalAuth:
    """Authentication credentials from local storage."""
    token: str
    token_type: str  # 'api_key', 'oauth', 'refresh'
    expires_at: Optional[int] = None
    refresh_token: Optional[str] = None


def get_claude_auth() -> Optional[LocalAuth]:
    """
    Read Claude Code OAuth token from ~/.claude/.credentials.json

    Returns:
        LocalAuth with access token, or None if not found.
    """
    credentials_path = Path.home() / '.claude' / '.credentials.json'

    if not credentials_path.exists():
        return None

    try:
        with open(credentials_path, 'r') as f:
            data = json.load(f)

        oauth = data.get('claudeAiOauth', {})
        access_token = oauth.get('accessToken')

        if not access_token:
            return None

        return LocalAuth(
            token=access_token,
            token_type='oauth',
            expires_at=oauth.get('expiresAt'),
            refresh_token=oauth.get('refreshToken')
        )
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def get_gemini_auth() -> Optional[LocalAuth]:
    """
    Read Gemini auth from ~/.gemini/settings.json or environment.

    Gemini stores tokens in system keychain, but we can check:
    1. GEMINI_API_KEY env var
    2. ~/.gemini/settings.json for OAuth state

    Returns:
        LocalAuth with token, or None if not found.
    """
    # Check environment first
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        return LocalAuth(token=api_key, token_type='api_key')

    # Check settings file for OAuth
    settings_path = Path.home() / '.gemini' / 'settings.json'

    if not settings_path.exists():
        return None

    try:
        with open(settings_path, 'r') as f:
            data = json.load(f)

        # Check if using Google OAuth login
        auth_type = data.get('selectedType')
        if auth_type == 'LOGIN_WITH_GOOGLE':
            # OAuth tokens are in keychain, not file
            # We'll need to use the CLI for this
            return None

        # Check for stored API key reference
        if auth_type == 'USE_GEMINI':
            # Key is in keychain, not directly accessible
            return None

    except (json.JSONDecodeError, IOError):
        pass

    return None


def get_codex_auth() -> Optional[LocalAuth]:
    """
    Read Codex auth from ~/.codex/auth.json

    The auth.json file contains the API key and OAuth tokens
    stored after running `codex login`.

    Returns:
        LocalAuth with API key or token, or None if not found.
    """
    # Check CODEX_HOME env var, default to ~/.codex
    codex_home = os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))
    auth_path = Path(codex_home) / 'auth.json'

    if not auth_path.exists():
        return None

    try:
        with open(auth_path, 'r') as f:
            data = json.load(f)

        # Try different possible keys in auth.json
        # The exact structure depends on OpenAI's implementation
        api_key = data.get('api_key') or data.get('apiKey') or data.get('access_token')

        if api_key:
            return LocalAuth(
                token=api_key,
                token_type='api_key',
                refresh_token=data.get('refresh_token') or data.get('refreshToken')
            )

        # Check for OAuth tokens
        access_token = data.get('accessToken') or data.get('access_token')
        if access_token:
            return LocalAuth(
                token=access_token,
                token_type='oauth',
                expires_at=data.get('expiresAt') or data.get('expires_at'),
                refresh_token=data.get('refreshToken') or data.get('refresh_token')
            )

    except (json.JSONDecodeError, IOError):
        pass

    return None


def get_all_local_auth() -> Dict[str, Optional[LocalAuth]]:
    """
    Get all available local auth tokens.

    Returns:
        Dict mapping model name to LocalAuth (or None if not available).
    """
    return {
        'claude': get_claude_auth(),
        'gemini': get_gemini_auth(),
        'codex': get_codex_auth(),
    }


def check_local_auth_available(model: str) -> bool:
    """Check if local auth is available for a model."""
    auth_funcs = {
        'claude': get_claude_auth,
        'gemini': get_gemini_auth,
        'codex': get_codex_auth,
    }

    func = auth_funcs.get(model)
    if func:
        return func() is not None
    return False
