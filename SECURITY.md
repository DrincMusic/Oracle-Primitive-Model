# Security Policy

Please report suspected credential exposure, unsafe deserialization, dependency vulnerabilities, or
integrity-verification bypasses privately to the repository owner through GitHub's private
vulnerability-reporting feature when available. Do not include secrets or exploit details in a public
issue.

The v1.1.4 snapshot is immutable. A security fix that would alter frozen bytes should be implemented
in a new software version, with the historical risk documented rather than silently rewriting the
study record.

The public export intentionally excludes checkpoints and large row-level evidence from normal Git
history. Their identities remain bound through SHA-256 manifests.
