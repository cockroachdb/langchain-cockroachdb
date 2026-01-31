# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The langchain-cockroachdb team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

To report a security issue, please use the GitHub Security Advisory ["Report a Vulnerability"](https://github.com/cockroachdb/langchain-cockroachdb/security/advisories/new) tab.

The team will send a response indicating the next steps in handling your report. After the initial reply to your report, the security team will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

### What to Include in Your Report

Please include the following information in your report:

- Type of issue (e.g., SQL injection, authentication bypass, information disclosure)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

## Security Best Practices

When using langchain-cockroachdb, follow these security best practices:

### 1. Connection Security

Always use secure connections in production:

```python
# Good: Use SSL/TLS
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://user:password@host:26257/db?sslmode=require"
)

# Bad: Insecure connection
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://user:password@host:26257/db?sslmode=disable"
)
```

### 2. Credentials Management

Never hardcode credentials:

```python
import os

# Good: Use environment variables
connection_string = os.getenv("COCKROACHDB_CONNECTION_STRING")
engine = CockroachDBEngine.from_connection_string(connection_string)

# Bad: Hardcoded credentials
engine = CockroachDBEngine.from_connection_string(
    "cockroachdb://root:password123@localhost:26257/db"
)
```

### 3. Input Validation

The package uses parameterized queries to prevent SQL injection, but always validate user inputs:

```python
# The package handles parameterization safely
results = vectorstore.similarity_search(
    user_query,  # Automatically escaped
    filter={"source": user_source}  # Filter values are parameterized
)
```

### 4. Metadata Filtering

Be cautious with user-provided filter expressions:

```python
# Validate user input before constructing filters
if not is_valid_filter(user_filter):
    raise ValueError("Invalid filter")

results = vectorstore.similarity_search(
    query, 
    filter=user_filter
)
```

### 5. Connection Pooling

Configure appropriate connection limits to prevent resource exhaustion:

```python
engine = CockroachDBEngine.from_connection_string(
    connection_string,
    pool_size=10,          # Reasonable pool size
    max_overflow=20,       # Limit overflow connections
    pool_pre_ping=True,    # Validate connections
)
```

### 6. Access Control

Use CockroachDB's built-in access control:

```sql
-- Create read-only user for application
CREATE USER app_readonly WITH PASSWORD 'secure_password';
GRANT SELECT ON TABLE vectors TO app_readonly;

-- Limit permissions
REVOKE ALL ON DATABASE defaultdb FROM app_readonly;
GRANT CONNECT ON DATABASE defaultdb TO app_readonly;
```

### 7. Logging and Monitoring

Avoid logging sensitive information:

```python
import logging

# Good: Log non-sensitive information
logging.info(f"Vector search completed, {len(results)} results")

# Bad: Logging sensitive data
logging.info(f"Query: {user_query}, Results: {results}")  # May contain PII
```

### 8. Rate Limiting

Implement rate limiting for public-facing applications:

```python
from functools import lru_cache
from time import time

@lru_cache(maxsize=1000)
def rate_limited_search(query: str, timestamp: int) -> list:
    # timestamp changes every second, providing rate limiting
    return vectorstore.similarity_search(query)

# Usage
results = rate_limited_search(query, int(time()))
```

### 9. Dependency Management

Keep dependencies up to date:

```bash
# Regularly update dependencies
uv pip list --outdated
uv pip install --upgrade langchain-cockroachdb
```

### 10. CockroachDB Security

Follow CockroachDB security best practices:

- Enable authentication
- Use TLS certificates
- Configure firewall rules
- Regular security updates
- Monitor audit logs

## Known Security Considerations

### Vector Injection

While not a direct vulnerability, be aware that embedding models can be susceptible to adversarial inputs. Consider:

- Input sanitization before embedding
- Rate limiting on vector operations
- Monitoring for unusual query patterns

### Metadata Storage

Metadata is stored as JSONB and should not contain:

- Passwords or secrets
- Personal Identifiable Information (PII) without proper consent
- Sensitive business data without encryption

### Transaction Retries

CockroachDB may retry transactions automatically. Ensure idempotency in your operations to prevent unintended side effects.

## Security Updates

Security updates will be released as patch versions (e.g., 0.1.1 → 0.1.2) and announced via:

- GitHub Security Advisories
- Release notes
- Project README

Subscribe to repository notifications to stay informed about security updates.

## Compliance

This package is designed to work with CockroachDB's security features:

- SERIALIZABLE isolation level (strong consistency)
- TLS encryption in transit
- Support for audit logging (via CockroachDB)
- Role-based access control (via CockroachDB)

Ensure your CockroachDB deployment meets your compliance requirements (GDPR, HIPAA, SOC 2, etc.).

## Contact

For security inquiries that don't constitute a vulnerability, please contact:
- GitHub Issues: https://github.com/cockroachdb/langchain-cockroachdb/issues
- GitHub Discussions: https://github.com/cockroachdb/langchain-cockroachdb/discussions

## Acknowledgments

We would like to thank all security researchers who responsibly disclose vulnerabilities to us. Contributors who report valid security issues will be acknowledged in our release notes (unless they prefer to remain anonymous).
