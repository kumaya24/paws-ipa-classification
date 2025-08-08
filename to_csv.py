# python to_csv.py

import ast
import pandas as pd

input_file = "ipa_ko_dev.txt"   # your .txt file
output_file = "ipa_ko_dev.csv"

data = []

with open(input_file, "r", encoding="utf-16") as f:  # <-- changed to utf-16
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            tup = ast.literal_eval(line)
            data.append(tup)
        except Exception as e:
            print(f"Skipping line due to error: {line[:50]}...  {e}")

df = pd.DataFrame(data, columns=["ipa_s1", "ipa_s2", "label"])
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Converted {len(df)} rows to {output_file}")
