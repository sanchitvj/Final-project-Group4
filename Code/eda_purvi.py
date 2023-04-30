#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Loading Data
train_features = pd.read_csv("./data/train_features.csv")
train_drugs = pd.read_csv("./data/train_drug.csv")
train_target_scored = pd.read_csv("./data/train_targets_scored.csv")

print(train_features.shape)
print(train_drugs.shape)
print(train_target_scored.shape)

#train_features.describe()

#def plot_selected_features(features, data, title):
    #plt.figure(figsize=(15, 5))
    #for feature in features:
        #sns.kdeplot(data[feature], lw=2)
    #plt.xlabel('Value')
    #plt.ylabel('Density')
    #plt.title(title)
    #plt.legend(features)
    #plt.show()
#selected_features = ['c-10', 'c-50', 'c-70', 'c-90']
#plot_selected_features(selected_features, train_features, 'Selected c- Features')

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
    plt.show()

plot_treatment_time_impact('c-30', treated_samples)


def correlation_matrix(data, title):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": .8})
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

c_cols = [col for col in treated_samples.columns if 'c-' in col]
correlation_matrix(treated_samples[c_cols], 'Correlation Between Cell Viability Features in Treated Samples')


g_cols = [col for col in treated_samples.columns if 'g-' in col]
correlation_matrix(treated_samples[g_cols], 'Correlation Between Gene Expression Features in Treated Samples')

control_samples = train_features[train_features['cp_type'] == 'ctl_vehicle']
treatment_samples = train_features[train_features['cp_type'] == 'trt_cp']
# Get gene expression columns
#gene_columns = [col for col in train_features.columns if col.startswith('g-')]
# Perform t-test on each gene expression column
#t_test_results = []
#for gene in gene_columns:
    #control_group = control_samples[gene]
    #treatment_group = treatment_samples[gene]
    
    #t_stat, p_value = ttest_ind(control_group, treatment_group, equal_var=False)
    
    #t_test_results.append({
        #'gene': gene,
        #'t_stat': t_stat,
        #'p_value': p_value
    #})
#t_test_results_df = pd.DataFrame(t_test_results)
#from statsmodels.stats.multitest import multipletests
#t_test_results_df['fdr_bh'] = multipletests(t_test_results_df['p_value'], method='fdr_bh')[1]
# Set a significance threshold, such as 0.05
#significance_threshold = 0.05
# Filter differentially expressed genes
#differentially_expressed_genes = t_test_results_df[t_test_results_df['fdr_bh'] < significance_threshold]
# Show the differentially expressed genes
#print(differentially_expressed_genes)


# Select gene expression columns ('g-' columns)
gene_expression_columns = [col for col in train_features.columns if col.startswith('g-')]
gene_expression_data = train_features[gene_expression_columns]
# Standardize the gene expression data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(gene_expression_data)
# Apply PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_data)
# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], edgecolors='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Gene Expression Data')
plt.show()


