# python clean_columns.py

import csv
import re

def remove_english(text):
    # Remove all English letters (uppercase and lowercase)
    return re.sub(r'[A-Za-z]', '', text)

with open("test_2k.tsv", "r", newline="", encoding="utf-8") as source:
    reader = csv.reader(source, delimiter="\t")  # specify tab delimiter since input is TSV
    
    with open("char_ko_test.csv", "w", newline="", encoding="utf-8") as result:
        writer = csv.writer(result)
        
        for r in reader:
            # Skip empty or short rows
            if len(r) < 4:
                continue
            
            # Remove spaces first
            col1 = r[1].replace(" ", "")
            col3 = r[2].replace(" ", "")
            col2 = r[3].replace(" ", "")
            
            # Remove English letters
            col1 = remove_english(col1)
            col3 = remove_english(col3)
            col2 = remove_english(col2)
            
            # Write columns in desired order
            writer.writerow([col1, col3, col2])
