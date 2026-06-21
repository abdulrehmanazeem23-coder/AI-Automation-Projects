# test_data.py
# This file just checks that we can read our Excel data correctly.

import pandas as pd

print("🔄 Loading Bid History sheet...")

df_bids = pd.read_excel(
    "data/sample_data.xlsx", 
    sheet_name="PS1 \u2013 Bid History",   # This is the sheet name with a special dash
    header=1,      # Row 2 (index 1) is our header row
    skiprows=[0]   # Skip the title row at the very top
)

print("🔄 Loading Capability Library sheet...")

df_caps = pd.read_excel(
    "data/sample_data.xlsx", 
    sheet_name="PS1 \u2013 Capability Library",
    header=1,
    skiprows=[0]
)

# Drop any completely empty rows
df_bids = df_bids.dropna(how='all')
df_caps = df_caps.dropna(how='all')

print("\n✅ Bid History loaded successfully!")
print(f"   → Rows: {len(df_bids)}  |  Columns: {len(df_bids.columns)}")
print(f"   → Column names: {list(df_bids.columns)}")

print("\n✅ Capability Library loaded successfully!")
print(f"   → Rows: {len(df_caps)}  |  Columns: {len(df_caps.columns)}")
print(f"   → Column names: {list(df_caps.columns)}")

print("\n📋 First 2 rows of Bid History:")
print(df_bids.head(2).to_string())

print("\n📋 First 2 rows of Capability Library:")
print(df_caps.head(2).to_string())

print("\n🎉 Step 1 Complete! Your data is loading correctly.")