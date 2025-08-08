# python .\transcribe_epitran.py > ko.txt

import pandas as pd
import epitran
import re
import jieba

# Load dataset
df = pd.read_csv(r"x-final\ko\dev_2k.tsv", sep="\t")  # Dev dataset
dataset = list(zip(df['sentence1'], df['sentence2'], df['label']))

epi = epitran.Epitran('kor-Hang', ligatures=False)

def remove_english(text):
    return re.sub(r'[A-Za-z]', '', text)

def clean_text(text):
    ipa_tokens = []
    text_no_eng = remove_english(text)
    # Cut text by jieba 
    for token in jieba.cut(text_no_eng):
        if re.search(r'[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]', token):
            ipa_tokens.append(epi.transliterate(token))
        else:
            ipa_tokens.append(token)
    return ' '.join(ipa_tokens)

# Transcribe into IPA
ipa_dataset = [(clean_text(s1), clean_text(s2), label) for (s1, s2, label) in dataset]

for s in ipa_dataset:
    print(s)
