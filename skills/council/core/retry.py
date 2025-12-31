"""
Error classification for retry logic.

Classifies errors as transient (retriable) vs permanent (not retriable).
"""

# Error patterns for classification
TRANSIENT_ERRORS = {'timeout', 'rate_limit', 'connection', '503', '429', 'temporarily', 'overloaded'}
PERMANENT_ERRORS = {'auth', 'authentication', 'not found', 'invalid', 'permission', 'denied', '401', '403', '404'}


def is_retriable_error(error: str) -> bool:
    """
    Classify error as transient (retriable) vs permanent (not retriable).

    Transient errors (network issues, rate limits) may resolve with retry.
    Permanent errors (auth failures, invalid requests) will never resolve.

    Args:
        error: Error message string

    Returns:
        True if error is transient and retry may help, False for permanent errors
    """
    if not error:
        return True  # Unknown errors default to retriable

    error_lower = error.lower()

    # Check for permanent errors first (these should never retry)
    if any(p in error_lower for p in PERMANENT_ERRORS):
        return False

    # Check for known transient errors
    if any(t in error_lower for t in TRANSIENT_ERRORS):
        return True

    # Default: assume transient (safer to retry than miss recoverable errors)
    return True
