Instead you have:

customer_zip_code_prefix
geolocation_zip_code_prefix


But:

Many zip prefixes map to multiple lat/lng points

Cities/states are noisy

No guaranteed 1-to-1 match

You can do:
Option A — crude join (lossy)
customer_zip_code_prefix == geolocation_zip_code_prefix


❌ duplicates
❌ arbitrary coordinates
❌ introduces noise

Option B — spatial / approximate join (this is where Joiner shines)

You can:

Aggregate geolocation to a centroid per zip prefix

Join customers to geolocation using:

zip prefix

city

state

(optionally) distance threshold

This is conceptually identical to the flight + airport + weather example.

✔ Multi-column
✔ Imperfect keys

print(customers.columns)
print(orders.columns)
print(order_items.columns)
print(payments.columns)
print(reviews.columns)
print(order_payments.columns)
print(geolocation.columns)
print(products.columns)