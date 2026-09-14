# Frequently asked questions

Short answers to the questions support sees most. Fuller detail lives in the
topic guides; this page deliberately restates the essentials, which is exactly
the near-duplicate content a retriever must learn to collapse.

## How do I get a refund?

Refunds are available within 30 days of the original charge. Requests after the
window require a support ticket and manager approval.

## Can I get a partial refund?

Partial refunds are supported for annual plans only. The refunded amount is
prorated by unused days and returned to the original payment method.

## How long does a refund take?

Refunds post to the original payment method within 5-10 business days. ACH
refunds can take up to 15 business days.

## How do I reset my password?

Use **Forgot password** on the sign-in page. The reset link is valid for one
hour and can be used once.

## Why didn't my reset email arrive?

Check that `auth.email_domain_allowlist` does not exclude the user's domain.
Messages to excluded domains are dropped silently by design.

## What is the API rate limit?

Every workspace starts at 1,000 requests per minute per API key, measured in a
sliding 60-second window.

## What should I do when I get a 429?

Retry with exponential backoff and full jitter, honouring the `Retry-After`
header rather than a fixed interval.

## Can I rotate an API key without downtime?

Yes. The old key remains valid for a 24-hour grace period after rotation.
