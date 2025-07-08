import numpy as np
import pandas as pd

np.random.seed(42)

# Parameters
n_components = 2500  # 2500 unique components, each with local & import = 5000 rows
component_names = [f"Comp_{i:05d}" for i in range(n_components)]
metal_types = ['Copper', 'Tin', 'Aluminum', 'Nickel', 'Silver']
form_factors = ['Foil', 'Sheet', 'Coil', 'Wire']
industry_usages = ['Automotive', 'Toys', 'Wearable', 'Consumer', 'Industrial']
local_countries = ['India']
import_countries = ['China', 'Germany', 'USA', 'Japan', 'South Korea']

rows = []
for i, name in enumerate(component_names):
    # Random attributes
    metal = np.random.choice(metal_types)
    form = np.random.choice(form_factors)
    usage = np.random.choice(industry_usages)
    weight = np.round(np.random.uniform(0.1, 10), 2)  # 0.1 to 10 kg


    # Local sourcing (make local sometimes more expensive)
    base_price_local = np.round(np.random.uniform(900, 1600), 2)
    freight_local = np.round(np.random.uniform(40, 120), 2)
    lead_time_local = np.random.randint(2, 15)
    customs_local = 0
    local_tax_local = 18
    exch_local = 1
    country_local = np.random.choice(local_countries)

    # Import sourcing (make import sometimes cheaper)
    base_price_import = np.round(base_price_local * np.random.uniform(0.6, 1.1), 2)
    freight_import = np.round(np.random.uniform(10, 60), 2)
    lead_time_import = np.random.randint(10, 40)
    customs_import = np.round(np.random.uniform(0, 8), 2)
    local_tax_import = 18
    exch_import = np.round(np.random.uniform(0.95, 1.15), 2)
    country_import = np.random.choice(import_countries)

    # Landed cost calculations
    total_local = (base_price_local + freight_local) * weight * (1 + local_tax_local/100)
    total_import = ((base_price_import + freight_import) * weight * exch_import) * (1 + customs_import/100) * (1 + local_tax_import/100)

    # Append local
    rows.append([
        i, name, metal, form, usage, weight, 'Local', country_local, lead_time_local,
        base_price_local, freight_local, customs_local, local_tax_local, exch_local, total_local, None
    ])
    # Append import
    rows.append([
        i, name, metal, form, usage, weight, 'Import', country_import, lead_time_import,
        base_price_import, freight_import, customs_import, local_tax_import, exch_import, total_import, None
    ])

# Create DataFrame
cols = [
    'component_id', 'component_name', 'metal_type', 'form_factor', 'industry_usage', 'unit_weight_kg',
    'source_type', 'source_country', 'lead_time_days', 'base_price_per_kg', 'freight_cost_per_kg',
    'customs_duty_percent', 'local_tax_percent', 'exchange_rate_multiplier', 'total_landed_cost_inr',
    'recommended_source'
]
df = pd.DataFrame(rows, columns=cols)

# Recommend source with lower landed cost per component
for cid in df['component_id'].unique():
    subset = df[df['component_id'] == cid]
    idx_min = subset['total_landed_cost_inr'].idxmin()
    df.loc[df['component_id'] == cid, 'recommended_source'] = df.loc[idx_min, 'source_type']

# Save to CSV
df.to_csv('synthetic_sourcing_data.csv', index=False)
print("Synthetic dataset saved as 'synthetic_sourcing_data.csv'")
