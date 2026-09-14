# Billing

Everything about invoices, refunds, and payment methods for Acme Cloud.

## Payment methods

Acme Cloud accepts Visa, Mastercard, American Express, and ACH bank transfer
for annual plans. Cards are charged in USD regardless of billing address.

To add a card, open **Settings > Billing > Payment methods** and choose
**Add method**. The card is authorised for $1.00 to verify it; the hold is
released within three business days.

### Failed payments

When a charge fails we retry on days 1, 3, and 7. After the third failure the
workspace moves to a read-only state. Existing data is retained for 30 days.

Set `billing.dunning_email` in workspace settings to route failure notices to a
shared inbox rather than the workspace owner.

## Refunds

Refunds are available within 30 days of the original charge. Requests after the
window require a support ticket and manager approval.

### Partial refunds

Partial refunds are supported for annual plans only. The refunded amount is
prorated by unused days and returned to the original payment method. Partial
refunds on monthly plans are not supported; cancel instead and the plan runs to
the end of the current period.

To request one, call `POST /v1/charges/{charge_id}/refund` with an `amount_cents`
field. Omitting `amount_cents` refunds the full charge.

### Refund timing

Refunds post to the original payment method within 5-10 business days. ACH
refunds can take up to 15 business days. The refund appears as a separate line
item on the statement rather than reversing the original charge.

## Invoices

Invoices are generated on the first of each month and emailed to every user with
the Billing Admin role. Download past invoices from **Settings > Billing >
History**; invoices are retained for seven years.

Set a purchase order number with `billing.po_number` and it appears in the header
of every subsequent invoice.
