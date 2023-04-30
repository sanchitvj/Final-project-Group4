import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_features = pd.read_csv("./data/train_features.csv")
train_drugs = pd.read_csv("./data/train_drug.csv")
train_targets_scored = pd.read_csv("./data/train_targets_scored.csv")

print(train_features.shape)
print(train_drugs.shape)
print(train_targets_scored.shape)

sns.set_style("darkgrid")
train_features["treatments"] = ["Compound (drugs)" if i == "trt_cp" else "Control (no drugs)" for i in train_features["cp_type"]]
ax = sns.countplot(data=train_features, x=train_features.treatments, palette='colorblind', alpha=0.75)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext = (0, 5), textcoords='offset points')
ax.set(xlabel=None, ylabel=None)
plt.title("Treatment type")
plt.show()
train_features = train_features.drop(["treatments"], axis=1)

ax = sns.countplot(data=train_features, x=train_features.cp_time, palette='muted', alpha=0.75)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext = (0, 5), textcoords='offset points')
ax.set(xlabel=None, ylabel=None)
plt.title("Dosage time in hours")
plt.show()

train_features["dose_levels"] = ["High" if i == "D2" else "Low" for i in train_features["cp_dose"]]
ax = sns.countplot(data=train_features, x=train_features.dose_levels, palette='magma', alpha=0.75)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
ax.set(xlabel=None, ylabel=None)
plt.title("Dosage level")
plt.show()
train_features = train_features.drop(["dose_levels"], axis=1)

counts = train_targets_scored.sum(axis=1).value_counts().sort_index()
ax = sns.barplot(x=counts.index, y=counts.values)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.title("Number of activations per sample")
plt.show()

labels = []
counts = []
for col in train_targets_scored.columns:
    count = train_targets_scored[col].value_counts()[1]
    if (count * 100) / len(train_targets_scored) >= 1:
        counts.append(count)
        labels.append(col)

ax = sns.barplot(y=labels, x=counts)
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=10)
plt.title("Targets with occurrence over 1%")
plt.xlabel("count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(17, 15))
sns.heatmap(train_targets_scored.corr())
plt.tight_layout()
plt.show()

corr = train_targets_scored.corr().abs()
threshold = 0.4
correlated_cols = []
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > threshold:
            col_dict = {'predictor1': corr.columns[i], 'predictor2': corr.columns[j], 'correlation': corr.iloc[i, j]}
            correlated_cols.append(col_dict)

correlated_df = pd.DataFrame(correlated_cols)

print(correlated_df)


