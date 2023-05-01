import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

train_features = pd.read_csv("./data/train_features.csv")
train_drugs = pd.read_csv("./data/train_drug.csv")
train_targets_scored = pd.read_csv("./data/train_targets_scored.csv")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]

# GENES
n_comp = 50

data = pd.DataFrame(train_features[GENES])
data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[GENES]))

train2 = pd.DataFrame(data2, columns=[f'pca_G-{i}' for i in range(n_comp)])
train_features = pd.concat((train_features, train2), axis=1)

n_comp = 15

data = pd.DataFrame(train_features[CELLS])
data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))

train2 = pd.DataFrame(data2, columns=[f'pca_C-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train2), axis=1)

# train_features_new = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4),
#                                   columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])
# train_features = pd.concat((train_features_c, train_features_new), axis=1)
print(train_features.shape)

var_thresh = VarianceThreshold(threshold=0.5)
data = train_features
data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

train_features_transformed = data_transformed[: train_features.shape[0]]

train_features = pd.DataFrame(train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']].values.reshape(-1, 4),
                              columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose'])

train_features = pd.concat([train_features, pd.DataFrame(data_transformed)], axis=1)

print(train_features.shape)

train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]
train = train.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

folds = train.copy()

mskf = MultilabelStratifiedKFold(n_splits=5)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
# print(folds)


class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        dct = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        # print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time', 'cp_dose'])
    return data


feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = "mps" if torch.has_mps else "cpu"  # https://stackoverflow.com/questions/72535034/module-torch-has-no-attribute-has-mps
# NotImplementedError: The operator 'aten::_weight_norm_interface' is not currently implemented
# for the MPS device. If you want this op to be added in priority during the prototype phase
# of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a
# temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to
# use the CPU as a fallback for this op. WARNING: this will be slower than running natively
# on MPS.
# print(DEVICE)
EPOCHS = 20  # 10 : CV log_loss:  0.015007138448626117
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

num_features = len(feature_cols)
num_targets = len(target_cols)
hidden_size = 1024
model = Model(num_features=num_features, num_targets=num_targets, hidden_size=hidden_size)
print(model)
# for name, param in model.named_parameters():
#     print(name, param.shape)


def run_training(fold, seed):
    seed_everything(seed)

    train = process_data(folds)

    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index

    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)

    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values

    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    loss_fn = nn.BCEWithLogitsLoss()

    early_stopping_steps = EARLY_STOPPING_STEPS
    early_step = 0

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    tr_loss, vl_loss = [], []
    for epoch in range(EPOCHS):

        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
        tr_loss.append(train_loss), vl_loss.append(valid_loss)
        if valid_loss < best_loss:

            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"NN_FOLD{fold}_.pth")

        elif EARLY_STOP:

            early_step += 1
            if early_step >= early_stopping_steps:
                break

    return oof, tr_loss, vl_loss


def run_k_fold(NFOLDS, seed):
    oof = np.zeros((len(train), len(target_cols)))
    fold_loss = []
    for fold in range(NFOLDS):
        oof_, tr_loss, vl_loss = run_training(fold, seed)

        oof += oof_
        fold_loss.append(vl_loss)
        # plt.plot(vl_loss)
        # plt.xlabel("Epochs")
        # plt.title(f"Validation loss for fold {fold}")
        # plt.show()

    return oof, fold_loss


SEED = [6202]
# oof = np.zeros((len(train), len(target_cols)))

# for seed in SEED:
oof_, fold_loss = run_k_fold(NFOLDS, SEED[0])

plt.plot(range(EPOCHS), fold_loss[0], label="Fold 1")
plt.plot(range(EPOCHS), fold_loss[1], label="Fold 2")
plt.plot(range(EPOCHS), fold_loss[2], label="Fold 3")
plt.plot(range(EPOCHS), fold_loss[3], label="Fold 4")
plt.plot(range(EPOCHS), fold_loss[4], label="Fold 5")
plt.title("Validation loss for 5 folds")
plt.legend()
plt.xlabel("Epochs")
plt.show()

train[target_cols] = oof_
valid_results = train_targets_scored.drop(columns=target_cols).merge(train[['sig_id'] + target_cols], on='sig_id',
                                                                     how='left').fillna(0)

y_true = train_targets_scored[target_cols].values
y_pred = valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]

print("CV log_loss: ", score)
