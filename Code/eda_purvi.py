# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import log_loss

# from xgboost import XGBClassifier

# Loading Data
train_features = pd.read_csv("./data/train_features.csv")
train_drugs = pd.read_csv("./data/train_drug.csv")
train_target_scored = pd.read_csv("./data/train_targets_scored.csv")

print(train_features.shape)
print(train_drugs.shape)
print(train_target_scored.shape)


# train_features.describe()

# def plot_selected_features(features, data, title):
# plt.figure(figsize=(15, 5))
# for feature in features:
# sns.kdeplot(data[feature], lw=2)
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title(title)
# plt.legend(features)
# plt.show()
# selected_features = ['c-10', 'c-50', 'c-70', 'c-90']
# plot_selected_features(selected_features, train_features, 'Selected c- Features')

def plot_individual_histograms(features, data, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, feature in enumerate(features):
        axes[i].hist(data[feature], bins=30, alpha=0.6)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Histogram of {feature}')

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


selected_features = ['c-10', 'c-50', 'c-70', 'c-90']
plot_individual_histograms(selected_features, train_features, 'Histograms of Selected c- Features')

selected_genes = ['g-10', 'g-100', 'g-200', 'g-400']
plot_individual_histograms(selected_genes, train_features, 'Selected g- Features')

def plot_cell_viability_difference(feature, data, control_data):
    plt.figure(figsize=(15, 5))
    sns.kdeplot(data[feature], lw=2, label="Treated")
    sns.kdeplot(control_data[feature], lw=2, label="Control")
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Cell Viability Difference for {feature}')
    plt.legend()
    plt.show()


control_samples = train_features[train_features['cp_type'] == 'ctl_vehicle']
treated_samples = train_features[train_features['cp_type'] == 'trt_cp']


plot_cell_viability_difference('c-30', treated_samples, control_samples)


def plot_treatment_time_impact(feature, data):
    plt.figure(figsize=(15, 5))
    for time in data['cp_time'].unique():
        sns.kdeplot(data[data['cp_time'] == time][feature], lw=2, label=f'{time} hours')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Impact of Treatment Time on {feature}')
    plt.legend()
    # plt.show()


plot_treatment_time_impact('c-30', treated_samples)

def correlation_matrix(data, title):
    corr = data.corr()

    # Get the upper triangular of the correlation matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Get the top 10 highest and lowest correlated features
    top_10_highest = upper.stack().nlargest(10)
    top_10_lowest = upper.stack().nsmallest(10)

    # # Plot the heatmap
    # plt.figure(figsize=(15, 12))
    # sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, square=True,
    #             cbar_kws={"shrink": .8})
    # plt.title(title)
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)
    # # plt.show()


    # Print the top 10 highest and lowest correlated features
    print("Top 10 Highest Correlated Features:\n")
    print(top_10_highest)
    print("\nTop 10 Lowest Correlated Features:\n")
    print(top_10_lowest)
    plt.show()
# c_cols = [col for col in treated_samples.columns if 'c-' in col]
# correlation_matrix(treated_samples[c_cols], 'Correlation Between Cell Viability Features in Treated Samples')
# g_cols = [col for col in treated_samples.columns if 'g-' in col]
# correlation_matrix(treated_samples[g_cols], 'Correlation Between Gene Expression Features in Treated Samples')
# def correlation_matrix(data, title):
#     corr = data.corr()
#     mask = np.triu(np.ones_like(corr, dtype=bool))

#     plt.figure(figsize=(15, 12))
#     sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": .8})
#     plt.title(title)
#     plt.xticks(rotation=90)
#     plt.yticks(rotation=0)
#     #plt.show()

# c_cols = [col for col in treated_samples.columns if 'c-' in col]
# # correlation_matrix(treated_samples[c_cols], 'Correlation Between Cell Viability Features in Treated Samples')


# g_cols = [col for col in treated_samples.columns if 'g-' in col]
# # correlation_matrix(treated_samples[g_cols], 'Correlation Between Gene Expression Features in Treated Samples')

control_samples = train_features[train_features['cp_type'] == 'ctl_vehicle']
treatment_samples = train_features[train_features['cp_type'] == 'trt_cp']
# Get gene expression columns
# gene_columns = [col for col in train_features.columns if col.startswith('g-')]
# Perform t-test on each gene expression column
# t_test_results = []
# for gene in gene_columns:
# control_group = control_samples[gene]
# treatment_group = treatment_samples[gene]

# t_stat, p_value = ttest_ind(control_group, treatment_group, equal_var=False)

# t_test_results.append({
# 'gene': gene,
# 't_stat': t_stat,
# 'p_value': p_value
# })
# t_test_results_df = pd.DataFrame(t_test_results)
# from statsmodels.stats.multitest import multipletests
# t_test_results_df['fdr_bh'] = multipletests(t_test_results_df['p_value'], method='fdr_bh')[1]
# Set a significance threshold, such as 0.05
# significance_threshold = 0.05
# Filter differentially expressed genes
# differentially_expressed_genes = t_test_results_df[t_test_results_df['fdr_bh'] < significance_threshold]
# Show the differentially expressed genes
# print(differentially_expressed_genes)


# Select gene expression columns ('g-' columns)
gene_expression_columns = [col for col in train_features.columns if col.startswith('g-')]
gene_expression_data = train_features[gene_expression_columns]
# Standardize the gene expression data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(gene_expression_data)
# Apply PCA
# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(scaled_data)
# Create a DataFrame with the principal components
# pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
# Visualize the results
# plt.figure(figsize=(8, 6))
# plt.scatter(pca_df['PC1'], pca_df['PC2'], edgecolors='k')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA of Gene Expression Data')
# plt.show()


# Merge train_features and train_drugs
merged_data = train_features.merge(train_drugs, on="sig_id")

# Drop sig_id, cp_type, and cp_time from the merged_data
# merged_data = merged_data.drop(["sig_id", "cp_type", "cp_time"], axis=1)

# Merge the merged_data with train_target_scored
merged_data = merged_data.merge(train_target_scored, on="sig_id")

# Separate out the categorical columns
cat_cols = ['cp_type', 'cp_time', 'cp_dose']
cat_data = merged_data[cat_cols]

# Drop the categorical columns and the 'sig_id' column
num_data = merged_data.drop(cat_cols + ['sig_id'], axis=1)

# Standardize the numerical data using the RobustScaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(num_data.filter(regex=r'^g-', axis=1))

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

#Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Gene Expression Data')
plt.show()

# Merge the scaled numerical data with the categorical data
merged_data = np.concatenate((scaled_data, num_data.filter(regex=r'^c-'), pd.get_dummies(cat_data)), axis=1)

# Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(
# merged_data,
# train_target_scored.iloc[:, 1:],  # Assuming the target variable starts from the second column
# test_size=0.2,
# random_state=42
# )
X_train, y_train, X_val, y_val = iterative_train_test_split(
    merged_data,
    train_target_scored.iloc[:, 1:].values,
    test_size=0.2
)

# Scale the data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Reduce the dimensionality of the data using PCA
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)

# Train and evaluate various models
models = {
    "Logistic Regression": MultiOutputClassifier(LogisticRegression()),
    #"SVM": MultiOutputClassifier(SVC(probability=True)),
    "Random Forest": RandomForestClassifier()
    # "XGBoost": XGBClassifier()
}

# for name, model in models.items():
#     print(f"Training {name}...")
#     model.fit(X_train, y_train)

#     y_pred = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else model.predict(X_val)

#     if isinstance(y_pred[0], list):
#         y_pred_concat = np.concatenate([y[:, np.newaxis] for y in y_pred], axis=1)
#     elif len(y_pred.shape) == 1:
#         y_pred_concat = y_pred[:, np.newaxis]
#     else:
#         y_pred_concat = y_pred

#     loss = log_loss(y_val, y_pred_concat)
#     print(f"{name} log loss: {loss}")

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(X_train_scaled.shape)
print(y_train.shape)


def Extract(lst):
    return [item[:, 1] for item in lst]


def pred_transform(preds):
    out = Extract(preds)
    arr = np.array(out)
    arr1 = np.transpose(arr)
    return arr1


# Update the logistic regression model with a higher max_iter value
# models['Logistic Regression'] = LogisticRegression(max_iter=100)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)  # remove [:1000,]

    y_pred = model.predict_proba(X_val_scaled)  # if hasattr(model, 'predict_proba') else model.predict(X_val_scaled)
    # y_pred = np.array(y_pred)  
    # if y_pred.ndim == 3:  # If y_pred has 3 dimensions
    #     y_pred_concat = y_pred[:, :, 1]  # Select only the positive class probabilities
    # else:
    #     y_pred_concat = y_pred
    # y_pred_concat = y_pred_concat.T

    y_pred_concat = pred_transform(y_pred)
    print(y_val.shape)
    print(y_pred_concat.shape)

    # unique_labels = np.unique(y_val)
    loss = log_loss(y_val, y_pred_concat)  # labels=unique_labels)
    print(f"{name} log loss: {loss}")
    # if isinstance(y_pred, list):
    # y_pred_concat = np.concatenate([y[:, np.newaxis] for y in y_pred], axis=1)
    # y_pred_concat = np.concatenate(y_pred, axis=1)

    # # elif isinstance(y_pred, np.ndarray) and len(y_pred.shape) == 1:
    # #     y_pred_concat = y_pred[:, np.newaxis]
    # # else:
    # #     y_pred_concat = y_pred[:, 1][:, np.newaxis] 
    # print(y_val.shape)
    # print(y_pred_concat.shape)
    # loss = log_loss(y_val, y_pred_concat)
    # print(f"{name} log loss: {loss}")
