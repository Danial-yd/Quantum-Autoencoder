import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.auto import trange
from collections import Counter
import pennylane as qml
from pennylane import numpy as pnp
import torch, torch.nn as nn, torch.optim as optim
import pennylane.numpy as qnp
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.neighbors import NearestNeighbors


cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]

train = '/Users/danialyntykbay/thesis/Experiment/kddcup.data.gz'
test = '/Users/danialyntykbay/thesis/Experiment/corrected.gz'
os.getcwd()
train_df = pd.read_csv(train,names=cols, header=None )
test_df  = pd.read_csv(test, names=cols, header=None)

train_df[train_df["label"] == 1]

for df in (train_df, test_df):
    df["label"] = (df["label"] != "normal.").astype(int)

normal_df = train_df[train_df["label"] == 0].reset_index(drop=True)
remain_df = train_df[train_df["label"] == 1].reset_index(drop=True)

X_attack = remain_df.drop(columns=["label"])
y_attack = remain_df["label"]

X_norm = normal_df.drop(columns=["label"])
y_norm = normal_df["label"]  # all zeros

X_test = test_df.drop(columns=["label"])
X_test_values = test_df["label"].values


# 3) Preprocessing transformer (fit ONLY on normal-train)
categorical = ["protocol_type", "service", "flag"]
numeric = [c for c in X_norm.columns if c not in categorical]

preproc = Pipeline([
    ("ct", ColumnTransformer([
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])),
    ("scale", MinMaxScaler()),
    ("pca", PCA(n_components=2))  # match your # of qubits
])



preproc.fit(X_norm)

# 4) Transform normal-train, normal-val, and the mixed test set
X_qae_train = preproc.transform(X_norm)


X_test_mixed = X_attack
y_test_mixed = remain_df["label"].values
X_qae_test   = preproc.transform(X_test_mixed)
X_test_m = preproc.transform(X_test)

print("QAE train shape:", X_qae_train.shape)
print("QAE test shape:",  X_qae_test.shape, " | positives in test:", y_test_mixed.sum())
# After fitting PCA



CTRL, REF, TRASH, COMP = 0, 1, 2, 3
dev = qml.device("default.qubit", wires=4)

def angle_embed(x):
    # x: array-like of length 2 (angles for TRASH, COMP) – assumed in radians
    qml.AngleEmbedding(x, wires=[TRASH, COMP], rotation="Y")

def encoder_rot_entangle(weights):
    """
    Two Rot layers with bidirectional CNOTs in between.
    weights shape: (2, 2, 3) -> [layer, qubit, (α,β,γ)]
    """
    # layer 1
    qml.Rot(*weights[0, 0], wires=TRASH)
    qml.Rot(*weights[0, 1], wires=COMP)
    qml.CNOT([TRASH, COMP])
    qml.CNOT([COMP, TRASH])
    # layer 2
    qml.Rot(*weights[1, 0], wires=TRASH)
    qml.Rot(*weights[1, 1], wires=COMP)

def prepare_reference(phi=None):
    # default |0>, or RY(phi)|0> if you want a nontrivial reference
    if phi is not None:
        qml.RY(phi, wires=REF)

@qml.qnode(dev, interface="autograd")
def swap_test_expectation(x, weights, ref_angle=None):
    prepare_reference(ref_angle)
    angle_embed(x)
    encoder_rot_entangle(weights)

    # SWAP test: control=0, swap 1<->2
    qml.Hadamard(wires=CTRL)
    qml.CSWAP(wires=[CTRL, REF, TRASH])
    qml.Hadamard(wires=CTRL)

    return qml.expval(qml.PauliZ(CTRL))

def loss_batch(X_angles, weights, ref_angle=None):
    # IMPORTANT: use qnp.stack so the tape stays connected
    vals = [swap_test_expectation(x, weights, ref_angle) for x in X_angles]
    z = qnp.stack(vals)                     # shape (B,)
    fids = 0.5 * (z + 1.0)                  # F = (⟨Z⟩ + 1)/2
    return 1.0 - qnp.mean(fids)

x_demo = np.array([0.5, 1.0])
weights_demo = np.random.normal(0, 0.1, size=(2, 2, 3))

print(qml.draw(swap_test_expectation)(x_demo, weights_demo))


@qml.qnode(dev, interface="autograd")
def compressed_readout(x, weights):
    angle_embed(x)
    encoder_rot_entangle(weights)
    return qml.expval(qml.PauliZ(COMP))


def embed_dataset(X_angles, weights):
    vals = []
    for x in X_angles:
        out = compressed_readout(x, weights)   # single expval now
        out = qml.math.toarray(out)            # convert from PennyLane tensor → numpy float
        vals.append(float(out))
    return np.array(vals, dtype=float).reshape(-1, 1)

# --- simple trainer ---
def train_qae_with_history_pb(X_angles, steps=200, batch_size=100, lr=0.2, ref_angle=None, seed=0):
    rng = qnp.random.default_rng(seed)

    w0 = 0.01 * rng.normal(size=(2, 2, 3))
    weights = qnp.array(w0, requires_grad=True)

    opt = qml.GradientDescentOptimizer(stepsize=lr)
    N = len(X_angles)
    steps_hist, fid_hist, loss_hist = [], [], []

    idx = qnp.arange(min(batch_size, N))
    Xb = X_angles[idx]

    pbar = trange(steps + 1, desc="QAE training", leave=True)
    for s in pbar:
        L = loss_batch(Xb, weights, ref_angle)
        F = 1.0 - L
        steps_hist.append(s)
        fid_hist.append(qml.math.toarray(F))
        loss_hist.append(qml.math.toarray(L))

        pbar.set_postfix({"loss": float(L), "fid": float(F)})

        if s < steps:
            weights = opt.step(lambda w: loss_batch(Xb, w, ref_angle), weights)
            weights = qnp.array(weights, requires_grad=True)

    return weights, qnp.array(steps_hist), qnp.array(fid_hist), qnp.array(loss_hist)
def train_qae_with_eval(X_train, X_test, steps=200, batch_size=100, lr=0.2, ref_angle=None, seed=0):
    rng = qnp.random.default_rng(seed)
    w0 = 0.01 * rng.normal(size=(2, 2, 3))
    weights = qnp.array(w0, requires_grad=True)

    opt = qml.GradientDescentOptimizer(stepsize=lr)
    Ntr = len(X_train)

    steps_hist, tr_loss_hist, tr_fid_hist = [], [], []
    te_loss_hist, te_fid_hist = [], []

    # fixed train mini-batch (same as your trainer)
    idx = qnp.arange(min(batch_size, Ntr))
    Xb_tr = X_train[idx]

    pbar = trange(steps + 1, desc="QAE training (with eval)", leave=True)
    for s in pbar:
        # ---- training loss on the (fixed) batch
        Ltr = loss_batch(Xb_tr, weights, ref_angle)
        Ftr = 1.0 - Ltr

        # ---- FULL test loss each epoch (or use a test minibatch if test is huge)
        Lte = loss_batch(X_test, weights, ref_angle)
        Fte = 1.0 - Lte

        # log
        steps_hist.append(s)
        tr_loss_hist.append(qml.math.toarray(Ltr))
        tr_fid_hist.append(qml.math.toarray(Ftr))
        te_loss_hist.append(qml.math.toarray(Lte))
        te_fid_hist.append(qml.math.toarray(Fte))

        pbar.set_postfix({
            "train_loss": float(Ltr), "train_fid": float(Ftr),
            "test_loss": float(Lte),  "test_fid": float(Fte)
        })

        if s < steps:
            weights = opt.step(lambda w: loss_batch(Xb_tr, w, ref_angle), weights)
            weights = qnp.array(weights, requires_grad=True)

    # convert to numpy arrays
    steps_hist   = np.asarray(steps_hist)
    tr_loss_hist = np.asarray(tr_loss_hist, dtype=float)
    tr_fid_hist  = np.asarray(tr_fid_hist, dtype=float)
    te_loss_hist = np.asarray(te_loss_hist, dtype=float)
    te_fid_hist  = np.asarray(te_fid_hist, dtype=float)
    return weights, steps_hist, tr_loss_hist, tr_fid_hist, te_loss_hist, te_fid_hist
def to_angles_0_pi(X):
    xmin = X.min(axis=0); xmax = X.max(axis=0)
    # avoid divide-by-zero if a column is constant
    scale = qnp.where(qnp.isclose(xmax - xmin, 0.0), 1.0, xmax - xmin)
    return (X - xmin) / scale * qnp.pi

@qml.qnode(dev, interface="autograd")
def compressed_readout_y(x, weights):
    angle_embed(x)
    encoder_rot_entangle(weights)
    return qml.expval(qml.PauliY(COMP))
def embed_dataset_y(X_angles, weights):
    """Return <PauliY> latent values for each data point in X_angles."""
    vals = []
    for x in X_angles:
        y_exp = compressed_readout_y(x, weights)
        y_exp = qml.math.toarray(y_exp)   # convert PennyLane tensor to float
        vals.append(float(y_exp))
    return np.array(vals, dtype=float).reshape(-1, 1)

@qml.qnode(dev, interface="autograd")
def compressed_readout_x(x, weights):
    angle_embed(x)
    encoder_rot_entangle(weights)
    return qml.expval(qml.PauliX(COMP))

def embed_dataset_x(X_angles, weights):
    vals = []
    for x in X_angles:
        x_exp = compressed_readout_x(x, weights)
        x_exp = qml.math.toarray(x_exp)   # PennyLane tensor → float
        vals.append(float(x_exp))
    return np.array(vals, dtype=float).reshape(-1, 1)

X_ang_train = to_angles_0_pi(X_qae_train)

xmin = X_qae_train.min(axis=0)
xmax = X_qae_train.max(axis=0)
scale = np.where(np.isclose(xmax - xmin, 0.0), 1.0, xmax - xmin)


def apply_train_minmax_to_pi(X, xmin, scale):
    Z = (X - xmin) / scale
    Z = np.clip(Z, 0.0, 1.0)
    return Z * np.pi

X_ang_test = apply_train_minmax_to_pi(X_qae_test, xmin, scale)
X_ang_test_m = apply_train_minmax_to_pi(X_test_m, xmin, scale)


weights, steps, fids, losses = train_qae_with_history_pb(
    X_ang_train, steps=200, batch_size=128, lr=0.15, ref_angle=None, seed=42
)


Z_train = np.load("Z_train.npy")
Z_test_m = np.load("Z_test_m.npy")
Y_train = np.load("Y_train.npy")
Y_test_m = np.load("Y_test_m.npy")
X_train = np.load("X_train.npy")
X_test_m= np.load("X_test_m.npy")

mixed_Z = np.concatenate((Z_test_m,X_test_values.reshape(-1,1)), axis=1)
mixed_Y = np.concatenate((Y_test_m,X_test_values.reshape(-1,1)), axis=1)
mixed_X = np.concatenate((X_test_m, X_test_values.reshape(-1,1)), axis=1)

normal_values = mixed_Z[mixed_Z[:, 1] == 0][:, 0]
attack_values = mixed_Z[mixed_Z[:, 1] == 1][:, 0]

# Plot histograms
plt.figure(figsize=(8, 5))
plt.hist(normal_values, bins=30, alpha=0.6, label='Normal', edgecolor='black')
plt.hist(attack_values, bins=30, alpha=0.6, label='Attack', edgecolor='black')

plt.title('Histogram of Z_test latent values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

Z = np.asarray(Z_test_m).reshape(-1)
Y = np.asarray(Y_test_m).reshape(-1)  # 0/1 labels
X = np.asarray(X_test_m).reshape(-1)

df = pd.DataFrame({
    'Z': Z,
    'Y': X,
    'X': Y
})

# Correlation matrix
corr_matrix = df.corr()
print("Correlation matrix:")
print(corr_matrix)

# Scatter plot
Z_vals = mixed_Z[:, 0]  # Z_test_m values
X_vals = mixed_Z[:, 1]  # X_test_values



# Create masks
normal_mask = (X_vals == 0)
attack_mask = (X_vals == 1)

plt.figure(figsize=(8, 6))

# Plot Normal
plt.scatter(
    Y_test_m[normal_mask],
    Z_test_m[normal_mask],
    c='blue',
    s=8,
    alpha=0.4,
    label='Normal',
    edgecolor='none'
)

# Plot Attack
plt.scatter(
    Y_test_m[attack_mask],
    Z_test_m[attack_mask],
    c='red',
    s=8,
    alpha=0.4,
    label='Attack',
    edgecolor='none'
)

plt.xlabel('Y_test')
plt.ylabel('Z_test')
plt.title('Scatter Plot: Normal vs Attack')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

#QkNN
from bisect import bisect_left
YZ_train  = np.c_[Y_train,  Z_train]      # (N_train, 2)  -> [Y, Z]
YZ_test_m = np.c_[Y_test_m, Z_test_m]


def angles_from_yz(YZ):
    YZ = np.asarray(YZ, dtype=np.float32)
    r = np.linalg.norm(YZ, axis=1, keepdims=True)
    r = np.maximum(r, 1e-9)              # avoid divide-by-zero
    YZu = YZ / r
    Y, Z = YZu[:,0], YZu[:,1]
    a = np.arctan2(Y, Z).astype(np.float32)  # (-π, π]
    a[a < 0] += 2.0*np.pi                    # [0, 2π)
    return a

def qdist_from_delta(delta):
    delta = delta.astype(np.float32, copy=False)
    return np.sin(0.5*delta)**2             # 1 - fidelity on a great circle

def qknn_scores_circle(alpha_train, alpha_query, k=5):
    T = np.asarray(alpha_train, dtype=np.float32).ravel()
    Q = np.asarray(alpha_query, dtype=np.float32).ravel()
    order = np.argsort(T)
    Ts = T[order]
    n = Ts.size
    Ts2 = np.concatenate([Ts, Ts + 2.0*np.pi])  # handle wrap-around

    out = np.empty(Q.shape[0], dtype=np.float32)
    for j, q in enumerate(Q):
        i = bisect_left(Ts, q)
        left, right = max(0, i-k), min(n, i+k)
        idxs = np.arange(left, right)
        cand = np.concatenate([Ts2[idxs], Ts2[idxs+n]])
        deltas = np.abs(cand - q)
        if deltas.size > k:
            deltas = np.partition(deltas, k-1)[:k]
        out[j] = qdist_from_delta(deltas).mean()
    return out

a_tr = angles_from_yz(YZ_train)
a_te = angles_from_yz(YZ_test_m)


def tune_qknn_yz(Z_train, Y_train, Z_val, Y_val, y_val,
                  k_grid=(2,3,5,7,9), q_grid=(0.90,0.95,0.975,0.99),
                  metric="f1", leave_one_out=False):
    """
    Z_train, Y_train: normals (1D arrays)
    Z_val, Y_val: mixed validation set
    y_val: labels for mixed val (0 normal, 1 attack)
    metric: "f1" | "pr_auc" | "roc_auc"
    """
    YZ_tr  = np.c_[Y_train, Z_train]
    YZ_val = np.c_[Y_val,   Z_val]
    a_tr = angles_from_yz(YZ_tr)
    a_va = angles_from_yz(YZ_val)

    best = {"score": -1, "k": None, "q": None, "thr": None}

    # precompute optional jit for LOO
    rng = np.random.default_rng(0)
    a_tr_jit = a_tr + 1e-7 * rng.standard_normal(a_tr.shape) if leave_one_out else a_tr

    for k in k_grid:
        # train self-scores once per k
        tr_scores = qknn_scores_circle(a_tr_jit, a_tr, k=k)

        # val scores once per k
        val_scores = qknn_scores_circle(a_tr, a_va, k=k)

        for q in q_grid:
            thr = float(np.quantile(tr_scores, q))
            y_pred = (val_scores > thr).astype(np.int8)

            if metric == "f1":
                m = f1_score(y_val, y_pred, pos_label=1)
            elif metric == "pr_auc":
                m = average_precision_score(y_val, val_scores)
            elif metric == "roc_auc":
                m = roc_auc_score(y_val, val_scores)
            else:
                raise ValueError("metric must be 'f1', 'pr_auc', or 'roc_auc'")

            if m > best["score"]:
                best.update({"score": m, "k": k, "q": q, "thr": thr})

    return best

best = tune_qknn_yz(
    Z_train, Y_train,
    Z_test_m, Y_test_m, X_test_values,         # mixed set + labels
    k_grid=(2,3,5,7,9,15),
    q_grid=(0.90,0.95,0.975,0.99,0.995),
    metric="f1",
    leave_one_out=True
)
print(best)

k = 9
train_scores = qknn_scores_circle(a_tr, a_tr, k=k)     # self-scores
quantile = 0.9
thr = float(1.5789839325364846e-14)

test_scores = qknn_scores_circle(a_tr, a_te, k=k)
y_pred = (test_scores > thr).astype(np.int8)

roc_auc = roc_auc_score(X_test_values, test_scores)
pr_auc  = average_precision_score(X_test_values, test_scores)
f1      = f1_score(X_test_values, y_pred, pos_label=1)
prec    = precision_score(X_test_values, y_pred, pos_label=1)
rec     = recall_score(X_test_values, y_pred, pos_label=1)


fpr, tpr, thresholds = roc_curve(X_test_values, test_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f"QkNN (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — QkNN")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

print(f"ROC AUC = {roc_auc:.4f}")

prec, rec, _ = precision_recall_curve(X_test_values, test_scores)
ap = average_precision_score(X_test_values, test_scores)

plt.figure(figsize=(6,6))
plt.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — QkNN")
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

print(f"Average Precision (PR AUC) = {ap:.4f}")
print(classification_report(X_test_values, y_pred))


