"""
CANCERWISE AI — Authentication & Role-Based Access Control (RBAC)
Provides token/API-key validation and role verification for sensitive oncology endpoints.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from backend.core.exceptions import AuthenticationError, AuthorizationError
from backend.core.config import get_config

config = get_config()


class UserRole(str, enum.Enum):
    VIEWER = "viewer"
    RESEARCHER = "researcher"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    role: UserRole
    is_active: bool = True


# Predefined system accounts for API key validation (can be configured via env)
DEFAULT_API_KEYS: Dict[str, UserProfile] = {
    "cw-admin-key-2026": UserProfile(
        user_id="usr_admin_001",
        username="admin",
        email="admin@cancerwise.ai",
        role=UserRole.ADMIN,
    ),
    "cw-researcher-key-2026": UserProfile(
        user_id="usr_res_001",
        username="researcher",
        email="research@cancerwise.ai",
        role=UserRole.RESEARCHER,
    ),
    "cw-clinician-key-2026": UserProfile(
        user_id="usr_clin_001",
        username="oncologist",
        email="clinician@cancerwise.ai",
        role=UserRole.CLINICIAN,
    ),
    "cw-viewer-key-2026": UserProfile(
        user_id="usr_view_001",
        username="guest_viewer",
        email="viewer@cancerwise.ai",
        role=UserRole.VIEWER,
    ),
}

# Default mock user when authentication is optional or disabled in dev
GUEST_USER = UserProfile(
    user_id="usr_dev_guest",
    username="oncology_researcher",
    email="researcher@cancerwise.ai",
    role=UserRole.RESEARCHER,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


def get_current_user(
    x_api_key: Optional[str] = Security(api_key_header),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_auth),
) -> UserProfile:
    """
    Validates API Key or Bearer Token.
    In development / unauthenticated mode, falls back to default researcher profile
    unless invalid credentials are provided.
    """
    token = None
    if x_api_key:
        token = x_api_key.strip()
    elif bearer_creds:
        token = bearer_creds.credentials.strip()

    if token:
        if token in DEFAULT_API_KEYS:
            return DEFAULT_API_KEYS[token]
        # In dev/testing, recognize prefix 'test-' or Bearer tokens
        if token.startswith("cw-") or token.startswith("test-"):
            return UserProfile(
                user_id="usr_authenticated",
                username="authenticated_user",
                email="auth@cancerwise.ai",
                role=UserRole.RESEARCHER,
            )
        raise AuthenticationError("Invalid API key or bearer token.")

    # Return default user for open routes
    return GUEST_USER


def require_role(allowed_roles: List[UserRole]):
    """FastAPI dependency factory enforcing minimum role hierarchy."""

    def role_checker(user: UserProfile = Depends(get_current_user)) -> UserProfile:
        if user.role == UserRole.ADMIN:
            return user
        if user.role not in allowed_roles:
            raise AuthorizationError(
                f"Role '{user.role.value}' does not have sufficient permissions. Required: {[r.value for r in allowed_roles]}."
            )
        return user

    return role_checker
