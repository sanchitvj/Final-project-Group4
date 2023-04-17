import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("./data")

train_peptides = pd.read_csv("./data/train_peptides.csv")
train_proteins = pd.read_csv("./data/train_proteins.csv")
train_clinical_data = pd.read_csv("./data/train_clinical_data.csv")
supp_clinical_data = pd.read_csv("./data/supplemental_clinical_data.csv")

# print(train_peptides.head().to_string())
# print(train_proteins.head().to_string())
# print(train_clinical_data.head().to_string())
# print(supp_clinical_data.head().to_string())

print(train_peptides.isna().sum())
print(train_proteins.isna().sum())
print(train_clinical_data.isna().sum())
print(supp_clinical_data.isna().sum())

print(train_peptides.describe().to_string())
print(train_proteins.describe().to_string())
print(train_clinical_data.describe().to_string())
print(supp_clinical_data.describe().to_string())



