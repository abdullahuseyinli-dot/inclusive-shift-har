# Security and sensitive-data reporting

## Supported code

Security fixes target the current development branch and the latest published
tag, if one exists. Historical experiment tags and evidence records remain
immutable; a fix is released through a new commit and version.

## Reporting

Use GitHub private vulnerability reporting or a private security advisory for
vulnerabilities, credentials, raw participant data, or other sensitive material.
Do not place secrets, exploit details, personal data, location fields, or
unredacted logs in a public issue.

If private reporting is unavailable, open a public issue containing only a short
request for a private contact channel. Do not include the sensitive content.

## Research-data incidents

Treat accidental inclusion of any of the following as a sensitive-data incident:

- raw or restricted sensor recordings;
- participant identity or free-text records;
- precise location/GPS values;
- credentials, access tokens, or signed download links;
- unrestricted predictions or checkpoints that were not approved for release.

Preserve the affected evidence locally, revoke exposed credentials, and create a
sanitized incident record. Do not rewrite historical evidence or force-push as an
informal fix; coordinate an explicit remediation plan.

## Safety scope

InclusiveShift-HAR is research software. It is not intended for medical diagnosis,
rehabilitation decisions, emergency monitoring, motor control, or other
safety-critical use. Model-performance limitations should be reported as model,
data, or interface failures rather than participant deficits.
