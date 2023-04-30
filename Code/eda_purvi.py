#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading Data
train_features = pd.read_csv("./data/train_features.csv")
train_drugs = pd.read_csv("./data/train_drug.csv")
train_target_scored = pd.read_csv("./data/train_targets_scored.csv")

train_features.describe()

def plotf(col1, col2, col3, col4):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    sns.histplot(data=train_features, x=col1, kde=True, ax=axes[0], color='blue')
    axes[0].set_title(col1)
    
    sns.histplot(data=train_features, x=col2, kde=True, ax=axes[1], color='red')
    axes[1].set_title(col2)
    
    sns.histplot(data=train_features, x=col3, kde=True, ax=axes[2], color='green')
    axes[2].set_title(col3)
    
    sns.histplot(data=train_features, x=col4, kde=True, ax=axes[3], color='purple')
    axes[3].set_title(col4)
    
    plt.tight_layout()
    plt.show()


plotf('c-10', 'c-50', 'c-70', 'c-90')

def plotd(col):
    sns.histplot(data=train_features, x=col, hue='cp_type', kde=True)
    plt.title(col)
    plt.show()

plotd("c-30")

def plott(col):
    sns.histplot(data=train_features, x=col, hue='cp_time', kde=True)
    plt.title(col)
    plt.show()

plott('c-30')



#Datasets for treated and control experiments
treated= train_features[train_features['cp_type']=='trt_cp']
control= train_features[train_features['cp_type']=='ctl_vehicle']


#Treatment time datasets
cp24= train_features[train_features['cp_time']== 24]
cp48= train_features[train_features['cp_time']== 48]
cp72= train_features[train_features['cp_time']== 72]


#Treated drugs without control
treated_list = treated['sig_id'].to_list()
drugs_tr= train_target_scored[train_target_scored['sig_id'].isin(treated_list)]


#adt= All Drugs Treated
adt= train_drugs[train_drugs['sig_id'].isin(treated_list)]

#Select the columns c-
c_cols = [col for col in train_features.columns if 'c-' in col]
#Filter the columns c-
cells=treated[c_cols]

#Select the columns g-
g_cols = [col for col in train_features.columns if 'g-' in col]
#Filter the columns g-
genes=treated[g_cols]



def plotd(f1):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15,5))
    #1 rows 2 cols
    #first row, first col
    ax1 = plt.subplot2grid((1,2),(0,0))
    plt.hist(control[f1], bins=4, color='mediumpurple',alpha=0.5)
    plt.title(f'control: {f1}',weight='bold', fontsize=18)
    #first row sec col
    ax1 = plt.subplot2grid((1,2),(0,1))
    plt.hist(treated[f1], bins=4, color='darkcyan',alpha=0.5)
    plt.title(f'Treated with drugs: {f1}',weight='bold', fontsize=18)


def plott(f1):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(15,5))
    #1 rows 2 cols
    #first row, first col
    ax1 = plt.subplot2grid((1,3),(0,0))
    plt.hist(cp24[f1], bins=3, color='deepskyblue',alpha=0.5)
    plt.title(f'Treatment duration 24h: {f1}',weight='bold', fontsize=14)
    #first row sec col
    ax1 = plt.subplot2grid((1,3),(0,1))
    plt.hist(cp48[f1], bins=3, color='lightgreen',alpha=0.5)
    plt.title(f'Treatment duration 48h: {f1}',weight='bold', fontsize=14)
    #first row 3rd column
    ax1 = plt.subplot2grid((1,3),(0,2))
    plt.hist(cp72[f1], bins=3, color='gold',alpha=0.5)
    plt.title(f'Treatment duration 72h: {f1}',weight='bold', fontsize=14)
    plt.show()


plt.figure(figsize=(15, 6))
sns.heatmap(cells.corr(), cmap='coolwarm', alpha=0.9)
plt.title('Correlation: Cell viability', fontsize=15, weight='bold')
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()


def plotf(f1, f2, f3, f4):
    plt.style.use('seaborn')
    sns.set_style('whitegrid')

    fig= plt.figure(figsize=(15,10))
    #2 rows 2 cols
    #first row, first col
    ax1 = plt.subplot2grid((2,2),(0,0))
    sns.distplot(a[f1], color='crimson')
    plt.title(f1,weight='bold', fontsize=18)
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    #first row sec col
    ax1 = plt.subplot2grid((2,2), (0, 1))
    sns.distplot(a[f2], color='gainsboro')
    plt.title(f2,weight='bold', fontsize=18)
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    #Second row first column
    ax1 = plt.subplot2grid((2,2), (1, 0))
    sns.distplot(a[f3], color='deepskyblue')
    plt.title(f3,weight='bold', fontsize=18)
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')
    #second row second column
    ax1 = plt.subplot2grid((2,2), (1, 1))
    sns.distplot(a[f4], color='black')
    plt.title(f4,weight='bold', fontsize=18)
    plt.yticks(weight='bold')
    plt.xticks(weight='bold')

    return plt.show()

def corrs(data, col1='Gene 1', col2='Gene 2',rows=5,thresh=0.8, pos=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53]):
        #Correlation between genes
        corre= data.corr()
         #Unstack the dataframe
        s = corre.unstack()
        so = s.sort_values(kind="quicksort", ascending=False)
        #Create new dataframe
        so2= pd.DataFrame(so).reset_index()
        so2= so2.rename(columns={0: 'correlation', 'level_0':col1, 'level_1': col2})
        #Filter out the coef 1 correlation between the same drugs
        so2= so2[so2['correlation'] != 1]
        #Drop pair duplicates
        so2= so2.reset_index()
        pos = pos
        so3= so2.drop(so2.index[pos])
        so3= so3.drop('index', axis=1)
        #Show the first 10 high correlations
        cm = sns.light_palette("Red", as_cmap=True)
        s = so3.head(rows).style.background_gradient(cmap=cm)
        print(f"{len(so2[so2['correlation']>thresh])/2} {col1} pairs have +{thresh} correlation.")
        return s
