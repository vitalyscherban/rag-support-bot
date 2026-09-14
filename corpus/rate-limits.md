# Rate limits

How Acme Cloud throttles API traffic and what to do when you hit a limit.

## Default limits

Every workspace starts at 1,000 requests per minute per API key, measured in a
sliding 60-second window. Burst traffic up to 1,500 requests is tolerated for a
maximum of 10 seconds.

Limits are per key, not per workspace. Splitting traffic across multiple keys
multiplies your effective ceiling, which is the supported way to scale.

## Response headers

Every response carries `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and
`X-RateLimit-Reset`. The reset value is a Unix timestamp, not a duration.

When throttled, the API returns HTTP 429 with a `Retry-After` header in seconds.
Clients should honour `Retry-After` rather than applying a fixed backoff.

## Handling 429s

Retry with exponential backoff and full jitter. A fixed retry interval causes
synchronised retry storms across your fleet, which is the most common reason a
workspace stays throttled after traffic has already dropped.

The official SDKs implement this automatically. Set `max_retries` on the client
to tune it; the default is 3.

## Requesting an increase

Increases are granted per key and require a sustained usage pattern over at
least seven days. Open a ticket including the key ID and the peak rate you need.

Enterprise plans can set limits directly through
`PATCH /v1/keys/{key_id}/limits` without a ticket.
