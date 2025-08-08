# python process_paws.py
import pandas as pd

def safe_remove_spaces(val):
    if pd.isna(val):
        return ""  # replace NaN with empty string
    return str(val).replace(" ", "")

def process_paws_to_tsv(char_file, ipa_file, output_char_tsv, output_ipa_tsv, sep_token="|"):
    df_char = pd.read_csv(char_file)
    df_ipa = pd.read_csv(ipa_file)

    assert len(df_char) == len(df_ipa), "Mismatch in number of rows!"

    char_lines = []
    ipa_lines = []

    for idx in range(len(df_char)):
        # Debug prints to check values before processing
        s1_char_raw = df_char.iloc[idx, 0]
        s2_char_raw = df_char.iloc[idx, 1]
        label = df_char.iloc[idx, 2]

        s1_ipa_raw = df_ipa.iloc[idx, 0]
        s2_ipa_raw = df_ipa.iloc[idx, 1]
        label_ipa = df_ipa.iloc[idx, 2]

        print(f"Row {idx} raw char s1: {s1_char_raw}, s2: {s2_char_raw}")
        print(f"Row {idx} raw ipa s1: {s1_ipa_raw}, s2: {s2_ipa_raw}")

        s1_char = safe_remove_spaces(s1_char_raw)
        s2_char = safe_remove_spaces(s2_char_raw)

        s1_ipa = safe_remove_spaces(s1_ipa_raw)
        s2_ipa = safe_remove_spaces(s2_ipa_raw)

        assert label == label_ipa, f"Label mismatch at row {idx}"

        char_combined = f"{s1_char}{sep_token}{s2_char}"
        ipa_combined = f"{s1_ipa}{sep_token}{s2_ipa}"

        char_lines.append(f"{char_combined}\t{label}")
        ipa_lines.append(f"{ipa_combined}\t{label}")

    with open(output_char_tsv, "w", encoding="utf-8") as f_char:
        f_char.write("\n".join(char_lines))

    with open(output_ipa_tsv, "w", encoding="utf-8") as f_ipa:
        f_ipa.write("\n".join(ipa_lines))

    print(f"Processed {len(char_lines)} lines.")
    print(f"Chinese char TSV saved to: {output_char_tsv}")
    print(f"IPA TSV saved to: {output_ipa_tsv}")

if __name__ == "__main__":
    process_paws_to_tsv(
        char_file="char_ko_dev.csv",
        ipa_file="ipa_ko_dev.csv",
        output_char_tsv="char_ko_dev.tsv",
        output_ipa_tsv="ipa_ko_dev.tsv",
        sep_token="|"
    ) 