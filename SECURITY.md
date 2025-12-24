# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within PDM-PILOT, please send an email to security@example.com. All security vulnerabilities will be promptly addressed.

Please include the following in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Security Measures

### Authentication & Authorization
- [ ] JWT-based authentication (planned)
- [ ] Role-based access control (RBAC)
- [x] Audit logging middleware
- [x] Security headers (CSP, HSTS, X-Frame-Options)

### Data Protection
- [x] Password hashing using bcrypt
- [x] Minimum password length: 12 characters
- [x] SQL injection prevention via parameterized queries
- [x] XSS prevention via input validation
- [x] Environment variables for secrets

### Infrastructure
- [x] Encrypted database backups (Fernet)
- [x] S3 server-side encryption for backups
- [ ] TLS/SSL enforcement (planned)

---

## Dependency Licenses

This project uses open-source dependencies. See below for license summary:

| License Type | Count |
|--------------|-------|
| MIT / MIT License | 30 |
| BSD / BSD-3-Clause | 17 |
| Apache 2.0 | 14 |
| LGPL | 2 |
| MPL 2.0 | 2 |
| ISC | 2 |
| PSF | 1 |
| Unlicense | 1 |

> **Note:** LGPL dependencies (psycopg2-binary, chardet) are used as-is without modification.

### License Compatibility

All dependencies are compatible with commercial use:
- ✅ MIT - Permissive, commercial-friendly
- ✅ BSD - Permissive, commercial-friendly
- ✅ Apache 2.0 - Permissive, commercial-friendly
- ⚠️ LGPL - Commercial use OK, but modifications must be shared
- ⚠️ MPL 2.0 - File-level copyleft, commercial use OK

### High-Risk Dependencies

| Package | License | Risk Level | Notes |
|---------|---------|------------|-------|
| psycopg2-binary | LGPL | LOW | Database driver, no modification |
| chardet | LGPLv2+ | LOW | Character detection, no modification |
| certifi | MPL 2.0 | LOW | CA certificates, no modification |

---

## Software Bill of Materials (SBOM)

A complete SBOM in CycloneDX format is available at:
- [software_bill_of_materials.xml](./software_bill_of_materials.xml)

Generated using CycloneDX Python v7.2.1 compliant with:
- CycloneDX v1.6 specification
- Package URL (purl) identifiers
- License SPDX identifiers

---

## Change Management

### Branch Protection (Recommended)

```yaml
# .github/branch-protection.yml
main:
  required_reviews: 1
  require_code_owner_reviews: true
  require_status_checks:
    - lint
    - test
    - security-scan
```

### Commit Signing

All commits should be signed with GPG:
```bash
git config --global commit.gpgsign true
```

---

## Vulnerability Disclosure Timeline

| Stage | Timeline |
|-------|----------|
| Report received | Day 0 |
| Initial response | Within 48 hours |
| Issue confirmed | Within 7 days |
| Patch developed | Within 30 days |
| Public disclosure | 90 days after report |

---

## Security Contacts

- **Security Team:** security@example.com
- **Primary Maintainer:** Andrew (andrew@example.com)

---

*Last Updated: 2025-12-24*
