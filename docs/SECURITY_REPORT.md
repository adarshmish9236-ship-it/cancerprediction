# CANCERWISE AI — Security, Privacy & Vulnerability Report

---

> [!IMPORTANT]
> **Biomedical Cyber-Security Commitment:**
> Patient data privacy, model integrity, and operational security are non-negotiable requirements in clinical decision-support architectures. This report details the defenses implemented against the OWASP Top 10 API Security Risks and medical informatics threats.

---

## 1. Threat Modeling & Defense Matrix (OWASP API Top 10)

| OWASP Vulnerability | Risk Level | Defense Implemented in CANCERWISE AI | Verification Status |
| :--- | :---: | :--- | :---: |
| **API1: Broken Object Level Auth** | High | Dataset and model identifiers are validated against internal registry namespaces; unauthorized path traversal is rejected. | **Passed** |
| **API2: Broken Authentication** | High | Centralized API Key / Bearer Token middleware (`backend/middleware/auth.py`). Configurable via environment variables. | **Passed** |
| **API3: Broken Object Property Auth** | Medium | Strict Pydantic v2 schemas reject extraneous or unmodeled request attributes (`extra='forbid'`). | **Passed** |
| **API4: Unrestricted Resource Consumption** | High | Request body size limit enforced at 100 MB (`backend/middleware/request_limits.py`); asynchronous task queue prevents thread starvation. | **Passed** |
| **API5: Broken Function Level Auth** | Medium | Separate endpoints for operational inference vs administrative model deletion and background job cancellation. | **Passed** |
| **API6: Unrestricted Access to Business Flows** | Medium | Upload validation rejects non-biomedical file formats and enforces minimum tabular dimension constraints. | **Passed** |
| **API7: Server-Side Request Forgery (SSRF)** | Low | External connectors only fetch from hardcoded, whitelisted academic endpoints (UCI, NCI GDC, UFPR BreakHis). | **Passed** |
| **API8: Security Misconfiguration** | Medium | Explicit CORS configuration (`backend/main.py`); debug modes disabled in production; verbose stack traces suppressed. | **Passed** |
| **API9: Improper Inventory Management** | Low | Versioned API paths (`/api/v1`) with automated OpenAPI/Swagger schema documentation. | **Passed** |
| **API10: Unsafe Consumption of APIs** | Low | SSL certificate verification enabled on all outgoing connector HTTPS requests. | **Passed** |

---

## 2. Safe File Path & Ingestion Protections

To prevent directory traversal attacks (e.g. `../../etc/passwd` or `..\Windows\win.ini`):
- All incoming dataset upload filenames are passed through `pathlib.Path(filename).name` and regex sanitization (`re.sub(r'[^a-zA-Z0-9_\.-]', '_', name)`).
- Resolved target file paths are checked against platform roots:
  ```python
  resolved_path = (target_dir / safe_filename).resolve()
  if not resolved_path.is_relative_to(target_dir.resolve()):
      raise SecurityException("Path traversal attempt detected.")
  ```

---

## 3. Secret & Credential Management

- **Zero Hardcoded Secrets**: The entire repository is free of embedded secrets, passwords, or cloud credentials.
- **`.gitignore` Enforcement**: Comprehensive rules block all `.env`, `.env.local`, `*.pem`, `*.key`, and private authentication tokens from git tracking.
- **Configuration Templates**: Parameterized `.env.example` provides exhaustive documentation of all operational environment variables with safe development defaults.

---

## 4. Container & Runtime Isolation

- **Non-Root Execution**: The production Docker image creates a dedicated unprivileged user `cancerwise` (`UID=10001, GID=10001`) and executes all processes without root privileges.
- **Minimal Attack Surface**: Built upon `python:3.12-slim-bookworm`, stripping unnecessary build utilities (gcc, g++, make) in a multi-stage Docker build.
- **Read-Only Code Segments**: Container filesystem mounts separate executable code from persistent data and log volumes.
