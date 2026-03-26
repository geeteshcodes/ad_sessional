import pandas as pd
import numpy as np
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from pyECLAT import ECLAT

df_raw = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = []
for _, row in df_raw.iterrows():
    items = [str(i).strip() for i in row if pd.notna(i) and str(i).strip() != 'nan']
    if items:
        transactions.append(items)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print(f"Transactions: {len(transactions)}")
print(f"Unique items: {len(te.columns_)}")
print(f"Avg basket: {np.mean([len(t) for t in transactions]):.2f}")
print()

MIN_SUPPORT = 0.04
MIN_CONFIDENCE = 0.20

t0 = time.time()
freq_ap = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
rules_ap = association_rules(freq_ap, metric='confidence', min_threshold=MIN_CONFIDENCE)
ap_time = round(time.time() - t0, 4)

rules_ap = rules_ap.sort_values('lift', ascending=False).reset_index(drop=True)

print(f"[Apriori] Time: {ap_time}s | Itemsets: {len(freq_ap)} | Rules: {len(rules_ap)}")
print(rules_ap[['antecedents','consequents','support','confidence','lift']].head(10).to_string())
print()

t0 = time.time()
freq_fp = fpgrowth(df, min_support=MIN_SUPPORT, use_colnames=True)
rules_fp = association_rules(freq_fp, metric='confidence', min_threshold=MIN_CONFIDENCE)
fp_time = round(time.time() - t0, 4)

rules_fp = rules_fp.sort_values('lift', ascending=False).reset_index(drop=True)

print(f"[FP-Growth] Time: {fp_time}s | Itemsets: {len(freq_fp)} | Rules: {len(rules_fp)}")
print(rules_fp[['antecedents','consequents','support','confidence','lift']].head(10).to_string())
print()

t0 = time.time()

df_eclat = pd.DataFrame(transactions)

eclat = ECLAT(data=df_eclat, verbose=False)

_, supports = eclat.fit(
    min_support=MIN_SUPPORT,
    min_combination=1,
    max_combination=2
)

ec_time = round(time.time() - t0, 4)

eclat_df = pd.DataFrame({
    'itemset': list(supports.keys()),
    'support': list(supports.values())
}).sort_values('support', ascending=False).reset_index(drop=True)

print(f"[ECLAT] Time: {ec_time}s | Itemsets: {len(eclat_df)}")
print(eclat_df.head(10).to_string())
print()

print("="*55)
print(f"{'Metric':<30} {'Apriori':>10} {'FP-Growth':>12} {'ECLAT':>10}")
print("-"*55)
print(f"{'Execution Time (s)':<30} {ap_time:>10} {fp_time:>12} {ec_time:>10}")
print(f"{'Frequent Itemsets':<30} {len(freq_ap):>10} {len(freq_fp):>12} {len(eclat_df):>10}")
print(f"{'Rules Generated':<30} {len(rules_ap):>10} {len(rules_fp):>12} {'N/A':>10}")
print(f"{'Top Lift':<30} {rules_ap['lift'].max():>10.4f} {rules_fp['lift'].max():>12.4f} {'N/A':>10}")