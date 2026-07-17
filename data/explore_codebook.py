import pandas as pd
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 200)

df = pd.read_excel("2018microdata_codebook.xlsx", header=1)
print(df.columns.tolist())
print(df.shape)

targets = ["PBA", "PBAPLUS", "SQFT", "SQFTC", "NFLOOR", "YRCONC", "WKHRS", "WKHRSC",
           "NWKER", "NWKERC", "HT1", "COOL", "WLCNS", "MFBTU", "ELBTU", "REGION", "CENDIV"]

varcol = df.columns[1]  # variable name column
for t in targets:
    rows = df[df[varcol].astype(str).str.strip() == t]
    if rows.empty:
        print(f"\n=== {t}: NOT FOUND (exact match) ===")
        continue
    print(f"\n=== {t} ===")
    for _, r in rows.iterrows():
        print(r.to_dict())
