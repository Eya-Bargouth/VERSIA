# Authentication Cheat Sheet

## Introduction

This cheat sheet provides guidance on implementing authentication securely.

## Password Storage

- Use strong hashing algorithms (Argon2, bcrypt, PBKDF2)
- Never store passwords in plain text
- Implement salting with unique salts per user

## Session Management

- Use secure, httpOnly, SameSite cookies
- Implement session timeout and invalidation
- Regenerate session ID after login

## Multi-Factor Authentication

- Enforce MFA for privileged accounts
- Prefer TOTP or WebAuthn over SMS
- Provide backup codes securely

## API Authentication

```python
# Example JWT verification
import jwt

def verify_token(token: str, secret: str) -> dict:
    return jwt.decode(token, secret, algorithms=["HS256"])
```

## Common Pitfalls

1. Hardcoded credentials in source code
2. Weak password policies
3. Missing rate limiting on login endpoints
