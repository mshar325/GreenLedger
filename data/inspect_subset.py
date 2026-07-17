import pandas as pd

cols = ["PUBID","PBA","PBAPLUS","REGION","SQFT","SQFTC","NFLOOR","YRCONC","WKHRS","WKHRSC",
        "NWKER","NWKERC","HT1","COOL","WLCNS","RFCNS","GLSSPC","FLCEILHT","ATTIC","BASEMNT",
        "ELEVTR","MONUSE","OCCUPYP","MFBTU","ELBTU"]

df = pd.read_csv("cbecs2018_final_public.csv", usecols=cols)
print("Full shape:", df.shape)

small_biz_pba = {6: "Food sales", 15: "Food service", 23: "Strip shopping center",
                  25: "Retail other than mall", 26: "Service"}
sub = df[df["PBA"].isin(small_biz_pba.keys())].copy()
print("\nAfter PBA filter (food sales/service, strip mall, retail, service):", sub.shape)
print(sub["PBA"].map(small_biz_pba).value_counts())

print("\nSQFT distribution in this subset:")
print(sub["SQFT"].describe())

for cap in [10000, 25000, 50000]:
    n = (sub["SQFT"] <= cap).sum()
    print(f"  SQFT <= {cap}: {n} buildings")

print("\nMFBTU missing count in subset:", sub["MFBTU"].isna().sum(), "/", len(sub))
print("NWKER missing:", sub["NWKER"].isna().sum())
print("WKHRS missing:", sub["WKHRS"].isna().sum())

sub2 = sub[(sub["SQFT"] <= 25000) & sub["MFBTU"].notna()]
print("\nFinal candidate subset (PBA filter + SQFT<=25000 + MFBTU present):", sub2.shape)
print(sub2["PBA"].map(small_biz_pba).value_counts())
