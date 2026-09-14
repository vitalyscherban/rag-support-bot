# Authentication

How users and services authenticate against the Acme Cloud API.

## Password reset

Users reset their own password from the sign-in page via **Forgot password**.
The reset link is valid for one hour and can be used once.

Workspace admins can force a reset from **Settings > Members**, selecting the
user and choosing **Require password reset**. The user is signed out of all
sessions immediately.

If the reset email does not arrive, check that `auth.email_domain_allowlist`
does not exclude the user's domain. Messages to excluded domains are dropped
silently by design.

## API keys

Create an API key from **Settings > API keys**. Keys are shown once at creation
and cannot be retrieved afterwards; rotate rather than recover a lost key.

Keys carry the permissions of the role assigned at creation. A key created with
the Viewer role cannot write, even if the creating user is later promoted.

### Key rotation

Rotate with `POST /v1/keys/{key_id}/rotate`. The old key remains valid for a
24-hour grace period so deployments can roll forward without downtime. Pass
`"immediate": true` to revoke the old key instantly during an incident.

## Single sign-on

SSO is available on Enterprise plans and supports SAML 2.0 and OIDC. Configure
it under **Settings > Security > SSO**.

Once SSO is enforced, password sign-in is disabled for all members except
break-glass accounts explicitly listed in `auth.breakglass_users`. Keep at least
one break-glass account or a misconfigured identity provider will lock the
workspace out entirely.

### SCIM provisioning

SCIM 2.0 user provisioning is supported alongside SSO. Deprovisioning through
SCIM deactivates the user but retains their audit history. Groups map to Acme
roles through the `scim.group_role_map` setting.

## Session limits

Sessions expire after 12 hours of inactivity by default. Enterprise workspaces
can set `auth.session_ttl_minutes` between 15 and 10080.
