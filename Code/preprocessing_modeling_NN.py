import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import log_loss

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = "mps" if torch.has_mps else "cpu"  # https://stackoverflow.com/questions/72535034/module-torch-has-no-attribute-has-mps
# NotImplementedError: The operator 'aten::_weight_norm_interface' is not currently implemented
# for the MPS device. If you want this op to be added in priority during the prototype phase
# of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a
# temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to
# use the CPU as a fallback for this op. WARNING: this will be slower than running natively
# on MPS.
# print(DEVICE) # Mac GPU and neural engine not utilizing -> CPU only
EPOCHS = 5  # 10 : CV log_loss:  0.015007138448626117
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5

train_features = pd.read_csv("./data/train_features.csv")
train_drugs = pd.read_csv("./data/train_drug.csv")
train_targets_scored = pd.read_csv("./data/train_targets_scored.csv")


def seed_everything(seed=6202):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.cuda.manual_seed(seed)  NO CUDA in mac
    # torch.backends.cudnn.deterministic = True


seed_everything(seed=6202)

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

def process_data(data):
    data = pd.get_dummies(data, columns=['cp_time', 'cp_dose'])
    return data


feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]


# Reference: https://www.kaggle.com/code/yasufuminakama/moa-pytorch-nn-starter?scriptVersionId=42246440&cellId=16
class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        data_dict = {
            'x': torch.tensor(self.features[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return data_dict


def train_model(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x, y = batch['x'].to(device), batch['y'].to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss


def validate(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch['x'].to(device), batch['y'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            predictions.append(outputs.sigmoid())
            # Use below if you have CUDA or MPS(Apple Silicon chip) and comment above.
            # predictions.append(outputs.sigmoid().detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(predictions)

    return avg_loss, predictions


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(input_dim, hidden_dim))

        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim))

        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(0.5)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_dim, output_dim))

        # Causing some issue at fold 2/3, giving Nan, not working
        # self.batch_norm4 = nn.BatchNorm1d(hidden_dim)
        # self.dropout4 = nn.Dropout(0.5)
        # self.dense4 = nn.PReLU(hidden_dim, output_dim)

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
        # x = self.batch_norm4(x)
        # x = self.dropout4(x)
        # x = self.dense4(x)

        return x


num_features = len(feature_cols)
num_targets = len(target_cols)
hidden_size = 1024
model = Model(input_dim=num_features, output_dim=num_targets, hidden_dim=hidden_size)
print(model)


# for name, param in model.named_parameters():
#     print(name, param.shape)


# def run_training(fold, seed):
#     seed_everything(seed)
#
#     train = process_data(folds)
#     # trn_idx = train[train['kfold'] != fold].index
#     val_idx = train[train['kfold'] == fold].index
#
#     train_df = train[train['kfold'] != fold].reset_index(drop=True)
#     valid_df = train[train['kfold'] == fold].reset_index(drop=True)
#
#     x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
#     x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values
#
#     train_dataset = MoADataset(x_train, y_train)
#     valid_dataset = MoADataset(x_valid, y_valid)
#     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#     model = Model(input_dim=num_features, output_dim=num_targets, hidden_dim=hidden_size)
#     model.to(DEVICE)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
#                                               max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))
#
#     # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
#     loss_fn = nn.BCEWithLogitsLoss()
#
#     early_stopping_steps = EARLY_STOPPING_STEPS
#     early_step = 0
#
#     oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
#     best_loss = np.inf
#     tr_loss, vl_loss = [], []
#     for epoch in range(EPOCHS):
#
#         train_loss = train_model(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
#         print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")
#         valid_loss, valid_preds = validate(model, loss_fn, validloader, DEVICE)
#         print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")
#         tr_loss.append(train_loss), vl_loss.append(valid_loss)
#         if valid_loss < best_loss:
#
#             best_loss = valid_loss
#             oof[val_idx] = valid_preds
#             torch.save(model.state_dict(), f"NN_FOLD{fold}_.pth")
#
#     return oof, tr_loss, vl_loss
def run_training(fold, seed):
    """
    Train the neural network model for one fold of the dataset.

    Args:
        fold (int): The fold to train the model on.
        seed (int): The random seed to use for the training process.

    Returns:
        tuple: A tuple containing the out-of-fold predictions, training loss, and validation loss.
    """

    # Set the random seed for reproducibility
    seed_everything(seed)

    # Load and preprocess the data for this fold
    train = process_data(folds)
    val_idx = train[train['kfold'] == fold].index
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)
    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values

    # Create PyTorch datasets and dataloaders for training and validation
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the neural network model and move it to the device
    model = Model(input_dim=num_features, output_dim=num_targets, hidden_dim=hidden_size)
    model.to(DEVICE)

    # Initialize the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(trainloader))

    # Use binary cross-entropy loss with logits
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    loss_fn = nn.BCEWithLogitsLoss()

    # Initialize arrays to store losses
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf
    tr_loss, vl_loss = [], []

    # Train the model for the specified number of epochs
    for epoch in range(EPOCHS):
        # Train the model for one epoch
        train_loss = train_model(model, optimizer, scheduler, loss_fn, trainloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")

        # Compute the validation loss and predictions for this epoch
        valid_loss, valid_preds = validate(model, loss_fn, validloader, DEVICE)
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")

        # Store the training and validation losses for this epoch
        tr_loss.append(train_loss), vl_loss.append(valid_loss)

        # Check if the validation loss improved and save the model if it did
        if valid_loss < best_loss:
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"NN_FOLD{fold}_.pth")

    # Return the out-of-fold predictions, training losses, and validation losses
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


# https://www.kaggle.com/competitions/lish-moa/discussion/193099
# Seed averaging needs memory and better compute. Time taking as well, not trying for now.
SEED = [6202]
# oof -> out of folds
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
valid_results = train_targets_scored.drop(columns=target_cols)
valid_results = valid_results.merge(train[['sig_id'] + target_cols], on='sig_id',
                                    how='left').fillna(0)
y_true, y_pred = train_targets_scored[target_cols].values, valid_results[target_cols].values

score = 0
for i in range(len(target_cols)):
    score_ = log_loss(y_true[:, i], y_pred[:, i])
    score += score_ / target.shape[1]

print("5 fold Cross Validation Logit Loss: ", score)
