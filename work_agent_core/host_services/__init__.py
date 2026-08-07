"""Narrow, user-authorized bridges to macOS-only capabilities.

These providers are intentionally separate from the generic execution plane:
they can reach hardware or OS-managed data, but they do not accept arbitrary
shell commands and expose only typed operations.
"""

from .apple_pim import ApplePimService, ApplePimServiceError

__all__ = ["ApplePimService", "ApplePimServiceError"]
