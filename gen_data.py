"""Generate synthetic e-commerce orders for the RFM segmentation project.

Deterministic: seed=42. Produces ~12,000 orders from 2,500 customers
(Jan–Sep 2026) with realistic behaviour archetypes:
one-time buyers, occasional shoppers, regulars, loyals and a few whales.
"""
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

N_CUSTOMERS = 2500
START = np.datetime64("2026-01-01")
END = np.datetime64("2026-09-30")
N_DAYS = int((END - START).astype(int)) + 1  # 273 days

# (archetype, n_customers, orders range, category weights
#  [Electronics, Fashion, Home & Kitchen, Grocery, Beauty])
ARCHETYPES = [
    ("one_time",  1125, (1, 1),   [0.10, 0.35, 0.15, 0.20, 0.20]),
    ("occasional", 750, (2, 4),   [0.15, 0.30, 0.20, 0.20, 0.15]),
    ("regular",    375, (5, 12),  [0.20, 0.25, 0.25, 0.20, 0.10]),
    ("loyal",      175, (13, 25), [0.25, 0.20, 0.25, 0.20, 0.10]),
    ("whale",       75, (20, 40), [0.40, 0.15, 0.20, 0.15, 0.10]),
]

CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Grocery", "Beauty"]
# lognormal (mu, sigma) -> median order value per category (INR)
PRICE_PARAMS = {
    "Electronics":    (8.90, 0.50),   # median ~ ₹7,350
    "Fashion":        (7.00, 0.60),   # median ~ ₹1,100
    "Home & Kitchen": (7.80, 0.50),   # median ~ ₹2,440
    "Grocery":        (6.30, 0.40),   # median ~ ₹545
    "Beauty":         (6.70, 0.50),   # median ~ ₹810
}

# Business is growing: later dates are more likely (linear ramp).
day_weights = np.linspace(1.0, 2.2, N_DAYS)
day_weights /= day_weights.sum()


def sample_dates(n, first_day_lo=0):
    """Sample n order dates (day offsets) with growth weighting."""
    return rng.choice(N_DAYS, size=n, replace=True, p=day_weights)


rows = []
order_seq = 1
cust_seq = 1

for archetype, n_cust, (lo, hi), cat_w in ARCHETYPES:
    for _ in range(n_cust):
        customer_id = f"CUST{cust_seq:04d}"
        cust_seq += 1
        n_orders = int(rng.integers(lo, hi + 1))
        if archetype == "one_time":
            # one-time buyers skew recent (acquired during growth)
            offsets = sample_dates(n_orders)
        else:
            # repeat buyers: first order any time, later orders after it
            first = int(rng.choice(N_DAYS, p=day_weights))
            if n_orders == 1:
                offsets = np.array([first])
            else:
                # subsequent orders fall after the first order, biased toward
                # recent dates (active customers keep buying); sample with
                # replacement from the growth-weighted day distribution
                w = day_weights[first:].copy()
                w /= w.sum()
                later = rng.choice(np.arange(first, N_DAYS), size=n_orders - 1,
                                   replace=True, p=w)
                offsets = np.sort(np.concatenate([[first], later]))
        offsets = np.sort(offsets)
        for off in offsets:
            order_date = START + np.timedelta64(int(off), "D")
            category = rng.choice(CATEGORIES, p=cat_w)
            mu, sigma = PRICE_PARAMS[category]
            amount = float(rng.lognormal(mu, sigma))
            # whales occasionally place very large orders
            if archetype == "whale" and rng.random() < 0.10:
                amount *= rng.uniform(2.0, 4.0)
            amount = round(max(amount, 99.0), 2)  # min ticket ₹99
            rows.append({
                "order_id": f"ORD{order_seq:06d}",
                "customer_id": customer_id,
                "order_date": str(order_date),
                "amount_inr": amount,
                "category": category,
                "archetype": archetype,  # hidden label, dropped before save
            })
            order_seq += 1

df = pd.DataFrame(rows)
# Shuffle so archetypes are not grouped together
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df = df.drop(columns=["archetype"])
df.to_csv("data/orders.csv", index=False)
print(f"orders.csv: {len(df):,} orders, {df['customer_id'].nunique():,} customers")
print(df.head(3).to_string(index=False))
print("\nOrders per customer (describe):")
print(df.groupby("customer_id").size().describe().round(2).to_string())
