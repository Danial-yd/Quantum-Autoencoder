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
from bisect import bisect_left


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

n_subset = 5000
idx = np.random.choice(len(X_qae_test), n_subset, replace=False)
X_qae_test_subset = X_qae_test[idx]
X_ang_test_subset = apply_train_minmax_to_pi(X_qae_test_subset, xmin, scale)

weights_eval, steps_h, trL, trF, teL, teF = train_qae_with_eval(
    X_ang_train, X_ang_test_subset, steps=200, batch_size=128, lr=0.15, ref_angle=None, seed=42
)


# ----- Plot fidelity -----
plt.figure()
plt.plot(steps, fids)
plt.xlabel("Training step")
plt.ylabel("Mean fidelity")
plt.title("QAE Training: Fidelity vs Steps")
plt.show()

# ----- Plot reconstruction loss (1 - fidelity) -----
plt.figure()
plt.plot(steps, losses)
plt.xlabel("Training step")
plt.ylabel("Reconstruction loss (1 - fidelity)")
plt.title("QAE Training: Reconstruction Loss vs Steps")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(steps_h, trL, label="Train loss")
plt.plot(steps_h, teL, label="Test loss")
plt.xlabel("Epoch"); plt.ylabel("Loss (1 - fidelity)"); plt.title("QAE loss per epoch"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(steps_h, trF, label="Train fidelity")
plt.plot(steps_h, teF, label="Test fidelity")
plt.xlabel("Epoch"); plt.ylabel("Fidelity"); plt.title("QAE fidelity per epoch"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

Z_train = embed_dataset(X_ang_train, weights)   # shape: (N_train_normals, 3)
 # optional, for tuning
Z_test  = embed_dataset(X_ang_test,  weights)   # shape: (N_test, 3)

Z_test_m = embed_dataset(X_ang_test_m, weights)

Y_train_latent = embed_dataset_y(X_ang_train, weights)

np.save("Y_train.mpy", Y_train_latent)

Y_test_latent  = embed_dataset_y(X_ang_test, weights)

Y_test_m_latent = embed_dataset_y(X_ang_test_m, weights)
np.save("Y_test_m.mpy", Y_test_m_latent)

X_train_latent   = embed_dataset_x(X_ang_train, weights)
X_test_latent = embed_dataset_x(X_ang_test, weights)
X_test_m_latent = embed_dataset_x(X_ang_test_m, weights)




plt.hist(Z_train, bins=50, alpha=0.6, label="normal")
plt.hist(Z_test, bins=50, alpha=0.6, label="anomaly")
plt.xlabel("latent value (⟨Z⟩)")
plt.ylabel("count")
plt.legend()
plt.title("Distribution of latent values")
plt.show()

np.save("Z_train.npy", Z_train)
np.save("Z_test.npy",  Z_test)
np.save("Z_test_m.npy", Z_test_m)
Z_train = np.load("Z_train.npy")
Z_test = np.load("Z_test.npy")
Z_test = Z_test[0:50000]
# Classical SVM

oc_rbf = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")  # tune nu if needed
oc_rbf.fit(Z_train)


# fit on NORMALS ONLY

oc_rbf_loaded = joblib.load("oc_rbf_model.pkl")
y_pred_rbf = (oc_rbf_loaded.predict(Z_test) == -1).astype(int)
y_pred_rbf_m = (oc_rbf_loaded.predict(Z_test_m ) == -1).astype(int)

scores_test = -oc_rbf_loaded.decision_function(Z_test_m)  # invert so anomalies are higher

# y_test_m: ground truth labels (0 = normal, 1 = attack)
fpr, tpr, thresholds = roc_curve(X_test_values, scores_test, pos_label=1)
roc_auc = roc_auc_score(X_test_values, scores_test)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve (One-Class SVM RBF)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

oc_rbf_loaded.predict(Z_test_m)
cm = confusion_matrix(X_test_values, y_pred_rbf_m)
accuracy  = accuracy_score(X_test_values, y_pred_rbf_m)
precision = precision_score(X_test_values, y_pred_rbf_m, pos_label=1)  # 1 = attack
recall    = recall_score(X_test_values, y_pred_rbf_m, pos_label=1)
f1        = f1_score(X_test_values, y_pred_rbf_m)

print(classification_report(X_test_values, y_pred_rbf_m))


joblib.dump(oc_rbf, "oc_rbf_model.pkl")

# later: load modelç
oc_rbf_loaded = joblib.load("oc_rbf_model.pkl")
#Fine OC-SVM
nus = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
sigma = float(np.std(Z_train, ddof=1) + 1e-12)
gamma_fixed = 1.0 / (2.0 * sigma**2)
best = {"f1": -1, "nu": None, "model": None, "report": None}
for nu in nus:
    oc = OneClassSVM(kernel="rbf", gamma=gamma_fixed, nu=nu).fit(Z_train)
    # predict: +1=normal, -1=outlier  -> map to 1=attack
    y_pred = (oc.predict(Z_test_m) == -1).astype(int)
    # optional continuous scores for PR-AUC
    scores = -oc.decision_function(Z_test_m).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        X_test_values, y_pred, pos_label=1, average="binary", zero_division=0
    )
    pr_auc = average_precision_score(X_test_values, scores)

    if (f1 > best["f1"]) or (f1 == best["f1"] and pr_auc > (best["report"]["pr_auc"] if best["report"] else -1)):
        best.update({
            "f1": f1, "nu": nu, "model": oc,
            "report": {"precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}
        })

print(f"Chosen ν={best['nu']}  "
      f"P={best['report']['precision']:.3f} R={best['report']['recall']:.3f} "
      f"F1={best['report']['f1']:.3f} PR-AUC={best['report']['pr_auc']:.3f}")
best_model = best["model"]

#Quantum OCSVM
Z_test_m = np.load("Z_test_m.npy")

def z_to_x(z):
    return 0.5 * (z + 1.0)

X_tr_full = z_to_x(Z_train)
X_te_full = z_to_x(Z_test)
X_test_m = z_to_x(Z_test_m)

n_ref = 10000
idx_ref = np.random.choice(len(X_tr_full), n_ref, replace=False)
X_tr_ref = X_tr_full[idx_ref].reshape(-1, 1)
print("Reference training shape:", X_tr_ref.shape)

def qkernel_ry_1d(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    diff = (np.pi/2.0) * (a - b)
    return np.cos(diff)**2

K_tr_tr = qkernel_ry_1d(X_tr_ref, X_tr_ref)
oc_q = OneClassSVM(kernel="precomputed", nu=0.05)
oc_q.fit(K_tr_tr)

n_eval_norm = 10000
idx_eval_norm = np.random.choice(len(X_tr_full), n_eval_norm, replace=False)
X_eval_norm = X_tr_full[idx_eval_norm].reshape(-1, 1)

# choose how many attacks to evaluate now (can use all if RAM is OK; start smaller)
n_eval_att = min(20000, len(X_te_full))  # change as you like
idx_eval_att = np.random.choice(len(X_te_full), n_eval_att, replace=False)
X_eval_att = X_te_full[idx_eval_att].reshape(-1, 1)

X_eval = np.vstack([X_eval_norm, X_eval_att])
y_eval = np.concatenate([np.zeros(len(X_eval_norm), dtype=int),
                         np.ones(len(X_eval_att), dtype=int)])

def batched_kernel(Xa, Xb, batch=1000):
    out = []
    for i in range(0, len(Xa), batch):
        out.append(qkernel_ry_1d(Xa[i:i+batch], Xb))
    return np.vstack(out)

K_eval_tr = batched_kernel(X_eval, X_tr_ref, batch=1000)
K_test_m_tr = batched_kernel(X_test_m, X_tr_ref, batch=1000)

scores_m = -oc_q.decision_function(K_test_m_tr).ravel()
y_pred_m = (oc_q.predict(K_test_m_tr) == -1).astype(int)

print(classification_report(X_test_values, y_pred_m))

roc_auc = roc_auc_score(X_test_values, scores_m)
pr_auc  = average_precision_score(X_test_values, scores_m)
f1      = f1_score(X_test_values, y_pred_m, pos_label=1)
precision = precision_score(X_test_values, y_pred_m, pos_label=1)
recall    = recall_score(X_test_values, y_pred_m, pos_label=1)

fpr, tpr, _ = roc_curve(X_test_values, scores_m, pos_label=1)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f'Q-OCSVM (AUC = {roc_auc_val:.3f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Mixed Test Set')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()



#Quantum random forest
Z_train = Z_train.reshape(-1, 1)
Z_test  = Z_test.reshape(-1, 1)

y_train = np.zeros(len(Z_train), dtype=int)  # 0 = normal
y_test  = np.ones(len(Z_test), dtype=int)
X_all = np.concatenate([Z_train, Z_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)
n_normals = 2000
n_attacks = 2000

idx_normals = np.random.choice(len(Z_train), n_normals, replace=False)
idx_attacks = np.random.choice(len(Z_test), n_attacks, replace=False)

X_qrf_train = np.concatenate([Z_train[idx_normals], Z_test[idx_attacks]])
y_qrf_train = np.concatenate([
    np.zeros(n_normals, dtype=int),
    np.ones(n_attacks, dtype=int)
])



latent_dim = 1
dev = qml.device("default.qubit", wires=latent_dim)

def feature_map(x):
    qml.RY(np.pi * x[0], wires=0)   # 1D feature

def ansatz(W):
    for l in range(W.shape[0]):
        qml.RY(W[l, 0], wires=0)
        qml.RZ(W[l, 1], wires=0)

@qml.qnode(dev, interface="autograd")
def vqc(x, W):
    feature_map(x)
    ansatz(W)
    return qml.expval(qml.PauliZ(0))

def vqc_prob1(x, W):
    return 0.5 * (1 - vqc(x, W))
def train_vqc_tree(X_train, y_train, layers=2, epochs=50, lr=0.1, seed=None):
    rng = np.random.default_rng(seed)
    W = pnp.array(0.01 * rng.normal(size=(layers, 2)), requires_grad=True)  # (layers, 2)
    opt = qml.GradientDescentOptimizer(lr)

    def loss(W):
        y = pnp.array(y_train, dtype=float)
        preds = pnp.stack([vqc_prob1(x, W) for x in X_train])
        eps = 1e-7
        preds = pnp.clip(preds, eps, 1 - eps)
        return -pnp.mean(y * pnp.log(preds) + (1 - y) * pnp.log(1 - preds))

    for _ in range(epochs):
        W = opt.step(loss, W)
        W = pnp.array(W, requires_grad=True)
    return W
def fit_qrf_supervised(X_train, y_train, n_trees=5, layers=2, epochs=50, lr=0.1):
    trees = []
    rng = np.random.default_rng(42)
    for t in range(n_trees):
        print(f"Training tree {t+1}/{n_trees}")
        W = train_vqc_tree(
            X_train, y_train, layers=layers, epochs=epochs, lr=lr,
            seed=rng.integers(1e9)
        )
        trees.append(W)
    return trees



trees = fit_qrf_supervised(X_qrf_train, y_qrf_train, n_trees=5, layers=2, epochs=20)

def qrf_predict_threshold(X_query, trees, threshold=0.5):
    scores = np.zeros(len(X_query))
    for W in trees:
        scores += np.array([vqc_prob1(x, W) for x in X_query])
    scores /= len(trees)
    return (scores >= threshold).astype(int), scores

y_pred, scores = qrf_predict_threshold(X_all, trees)

#QkNN tarining
import numpy as np
from bisect import bisect_left

def qdist_from_delta(d):
    # d >= 0, keep fp32 to cut memory/bandwidth
    d = d.astype(np.float32, copy=False)
    return np.sin(0.5 * d, dtype=np.float32)**2  # NOTE: if your NumPy errors on dtype, remove it.

def qknn_scores_sorted(theta_train, theta_query, k=5):
    T = np.asarray(theta_train, dtype=np.float32).ravel()
    Q = np.asarray(theta_query, dtype=np.float32).ravel()

    order = np.argsort(T)
    Ts = T[order]
    n = Ts.size

    out = np.empty(Q.shape[0], dtype=np.float32)

    for j, q in enumerate(Q):
        i = bisect_left(Ts, q)              # insertion position
        # take a window that must contain the k nearest by |Δ|
        left  = max(0, i - k)
        right = min(n, i + k)               # exclusive
        idxs = np.arange(left, right)

        deltas = np.abs(Ts[idxs] - q)

        # keep the k smallest if we have more than k candidates
        if deltas.size > k:
            keep = np.argpartition(deltas, k-1)[:k]
            deltas = deltas[keep]

        out[j] = qdist_from_delta(deltas).mean()

    return out

# --- your pipeline (with two small fixes noted below) ---
dir = "/Users/danialyntykbay/thesis/AngleEncoding"
os.chdir(dir)
os.getcwd()

k = 5
quantile = 0.95

Z_train = np.load("Z_train.npy", mmap_mode="r")  # mmap to save RAM
Z_test_m  = np.load("Z_test_m.npy",  mmap_mode="r")
Z_test = np.load("Z_test.npy", mmap_mode="r")

from sklearn.preprocessing import MinMaxScaler

def theta_from_mz(mz):
    mz = np.clip(mz, -1.0, 1.0).astype(np.float32, copy=False)
    return np.arccos(mz)

def qdist_from_delta(d):
    # 1 - fidelity for meridian 1-qubit states: sin^2(Δ/2)
    d = d.astype(np.float32, copy=False)
    return np.sin(0.5 * d)**2

# ---------- Core QkNN scorer ----------
def qknn_scores_sorted(theta_train, theta_query, k=5, agg="mean", leave_one_out=False):
    T = np.asarray(theta_train, dtype=np.float32).ravel()
    Q = np.asarray(theta_query, dtype=np.float32).ravel()

    order = np.argsort(T)
    Ts = T[order]
    n = Ts.size
    out = np.empty(Q.shape[0], dtype=np.float32)

    for j, q in enumerate(Q):
        i = bisect_left(Ts, q)                  # insertion position
        left  = max(0, i - k)
        right = min(n, i + k)                   # exclusive
        idxs  = np.arange(left, right)

        deltas = np.abs(Ts[idxs] - q)

        # leave-one-out: drop exact self-match (tolerance for float)
        if leave_one_out:
            deltas = deltas[deltas > 1e-12]

        # keep only k smallest candidates
        if deltas.size > k:
            deltas = np.partition(deltas, k-1)[:k]
        elif deltas.size == 0:
            out[j] = 0.0
            continue

        qd = qdist_from_delta(deltas)
        if agg == "mean":
            out[j] = qd.mean()
        elif agg == "median":
            out[j] = np.median(qd)
        elif agg == "kth":
            out[j] = np.max(qd)  # kth neighbor (largest of the kept)
        else:
            out[j] = qd.mean()
    return out

Z_val, Z_final_test, y_val, y_final_test = train_test_split(
    Z_test, y_test,
    test_size=0.7,                 # 70% remains as final test
    stratify=y_test,               # keep same 80/20 ratio
    random_state=42
)

Th_tr = theta_from_mz(Z_train)
Th_te = theta_from_mz(Z_test)



k = 2
train_scores = qknn_scores_sorted(Th_tr, Th_tr, k=k)  # self-scores (neighbors around each point)
thr = float(np.quantile(train_scores, 0.90))

scores_test = qknn_scores_sorted(Th_tr, Th_te, k=k)
y_pred = (scores_test > thr).astype(np.int8)

print(f"k={k}, threshold quantile={quantile:.2f}, thr={thr:.6f}")
print(classification_report(X_test_values, y_pred, digits=4))


#finetune
k_grid = [1, 2, 3, 4, 5,6,7,8,9,10]
q_grid = [0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.8,0.90, 0.95, 0.975, 0.99]
agg_grid = "mean"

results = []
best = None
best_row = None
rows = []

print("Selecting (k, q) by highest F1\n" + "-"*60)
for k in k_grid:
    # 1) Train-side scores for threshold (leave-one-out)
    tr_scores = qknn_scores_sorted(
        theta_train=Th_tr,
        theta_query=Th_tr,
        k=k,
        agg=agg_grid,
        leave_one_out = True
    )

    # 2) Test-side scores (same for all q at this k)
    te_scores = qknn_scores_sorted(
        theta_train=Th_tr,
        theta_query=Th_te,
        k=k,
        agg=agg_grid
    )

    for q in q_grid:
        thr = np.quantile(tr_scores, q)
        y_pred = (te_scores > thr).astype(int)

        # metrics (define anomalies as positive class = 1)
        f1 = f1_score(X_test_values, y_pred, pos_label=1)
        rows.append((k, q, thr, f1))

        if (best is None) or (f1 > best):
            best = f1
            best_row = (k, q, thr, f1)

        print(f"k={k:2d} | q={q:5.3f} | thr={thr:.6g} | | F1={f1:.3f}")




# Classical KNN

X_train = Z_train.reshape(-1, 1)
X_test  = Z_test.reshape(-1, 1)



k = 4
nn = NearestNeighbors(n_neighbors=k + 1).fit(X_train)
dist_tr, _ = nn.kneighbors(X_train)
train_scores = dist_tr[:, 1:].mean(axis=1)
thr = np.quantile(train_scores, 0.90)


dist_te, _ = nn.kneighbors(X_test, n_neighbors=k)
test_scores = dist_te.mean(axis=1)

y_pred = (test_scores > thr).astype(int)

roc_auc = roc_auc_score(X_test_values, test_scores)
pr_auc  = average_precision_score(X_test_values, test_scores)
f1      = f1_score(X_test_values, y_pred, pos_label=1)
precision = precision_score(X_test_values, y_pred, pos_label=1)
recall    = recall_score(X_test_values, y_pred, pos_label=1)

print(classification_report(X_test_values, y_pred))

fpr, tpr, _ = roc_curve(X_test_values, test_scores)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, color='blue', label=f'Classical KNN (AUC = {roc_auc_val:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Classical KNN')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

y_true  = X_test_values
scores  = scores_test

# PR points + area (PR-AUC)
prec, rec, thr = precision_recall_curve(y_true, scores, pos_label=1)
pr_auc = average_precision_score(y_true, scores, pos_label=1)

# (optional) baseline = attack prevalence
baseline = np.mean(y_true == 1)

# plot
plt.figure(figsize=(6,6))
plt.plot(rec, prec, label=f'PR (AUC = {pr_auc:.3f})')
plt.hlines(baseline, 0, 1, colors='gray', linestyles='--', label=f'Baseline = {baseline:.2f}')
plt.xlabel('Recall (attack)')
plt.ylabel('Precision (attack)')
plt.title('Precision–Recall Curve — Attack as Positive')
plt.legend()
plt.grid(True)
plt.show()


#FPR for Normals

rng = np.random.default_rng(42)  # reproducible

n_norm_val = 20_000
n_att_val  = 80_000

# Safety checks (adjust if you don't have enough rows)
n_normals = Z_train.shape[0]
n_attacks = Z_test.shape[0]
assert n_normals >= n_norm_val, f"Need {n_norm_val} normals, have {n_normals}"
assert n_attacks >= n_att_val,  f"Need {n_att_val} attacks, have {n_attacks}"

# Sample indices without replacement
idx_norm_val = rng.choice(n_normals, size=n_norm_val, replace=False)
idx_att_val  = rng.choice(n_attacks, size=n_att_val,  replace=False)

# Build validation features and labels
Z_val = np.vstack([Z_train[idx_norm_val], Z_test[idx_att_val]])
y_val = np.concatenate([np.zeros(n_norm_val, dtype=int),
                        np.ones(n_att_val,  dtype=int)])

# (Optional) shuffle validation set so classes are mixed
perm = rng.permutation(Z_val.shape[0])
Z_val = Z_val[perm]
y_val = y_val[perm]

# Keep the remaining pools (useful for training / final test)
Z_train_remaining = np.delete(Z_train, idx_norm_val, axis=0)  # normals leftover
Z_test_remaining  = np.delete(Z_test,  idx_att_val,  axis=0)

Z_val, Z_final_test, y_val, y_final_test = train_test_split(
    Z_test_m, X_test_values,
    test_size=0.8,          # keep 70% for final test, 30% for validation
    stratify=X_test_values,        # preserve normal/attack ratio
    random_state=42
)

alpha = 0.05  # e.g., 1% false alarms allowed on normals

k_grid   = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45]
agg_grid = ["mean"]  # or try ["largest","mean","median"]

Th_tr = theta_from_mz(Z_train)
Th_val = theta_from_mz(Z_val)
Th_te = theta_from_mz(Z_final_test)

def eval_at_threshold(scores, y_true, tau):
    y_pred = (scores >= tau).astype(int)  # 1 = attack
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)  # recall for attacks
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    return dict(fpr=fpr, tpr=tpr, precision=p, recall=r, f1=f1, y_pred=y_pred)

best = None
rows = []


print(f"Selecting (k, agg) at target normal FPR α={alpha:.3%}\n" + "-"*70)
for agg in agg_grid:
    for k in k_grid:
        # Score validation
        val_scores = qknn_scores_sorted(theta_train=Th_tr, theta_query=Th_val, k=k, agg=agg)

        # Threshold from VALIDATION NORMALS
        val_scores_norm = val_scores[y_val == 0]
        if val_scores_norm.size == 0:
            print(f"[warn] No normals in validation for k={k}, agg={agg}. Skipping.")
            continue

        tau = np.quantile(val_scores_norm, 1 - alpha)
        if not np.isfinite(tau):
            print(f"[warn] Non-finite tau for k={k}, agg={agg}. Skipping.")
            continue

        # Evaluate on validation
        val_metrics = eval_at_threshold(val_scores, y_val, tau)

        rows.append({
            "k": k, "agg": agg, "tau": float(tau),
            "val_fpr": val_metrics["fpr"],
            "val_tpr": val_metrics["tpr"],
            "val_f1" : val_metrics["f1"],
            "val_prec": val_metrics["precision"],
            "val_rec" : val_metrics["recall"],
        })

        # Keep best that meets FPR constraint
        if (val_metrics["fpr"] <= alpha) and (best is None or val_metrics["f1"] > best["val_f1"]):
            best = {
                "k": k, "agg": agg, "tau": float(tau),
                **{f"val_{m}": v for m, v in val_metrics.items()}  # <-- fixed: m not k
            }

        print(f"k={k:3d} | agg={agg:7s} | τ={tau:.6g} | "
              f"Val FPR={val_metrics['fpr']:.4f} | "
              f"Val TPR={val_metrics['tpr']:.4f} | "
              f"Val F1={val_metrics['f1']:.4f}")

if best is None:
    print("\nNo (k,agg) met the FPR constraint. Increase α slightly or try larger k / different agg.")
else:
    print("\nBest under FPR constraint:", best)

    # Final evaluation on TEST with the chosen τ
    te_scores_best = qknn_scores_sorted(theta_train=Th_tr, theta_query=Th_te,
                                        k=best["k"], agg=best["agg"])
    test_metrics = eval_at_threshold(te_scores_best, y_final_test, best["tau"])
    print(f"\nTEST @ (k={best['k']}, agg={best['agg']}, τ={best['tau']:.6g})")
    print(f"Test FPR={test_metrics['fpr']:.4f} | Test TPR={test_metrics['tpr']:.4f} | "
          f"Test F1={test_metrics['f1']:.4f} | Test Precision={test_metrics['precision']:.4f} | "
          f"Test Recall={test_metrics['recall']:.4f}")


k = 2
thr = 3.11671e-12

test_scores = qknn_scores_sorted(theta_train=Th_tr,
                                 theta_query=Th_te,
                                 k=k,
                                 leave_one_out=False)

# Higher = more anomalous
y_pred = (test_scores > thr).astype(int)


print(classification_report(y_final_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_final_test, y_pred).ravel()

fpr = fp / (fp + tn)             # False Positive Rate
tnr = tn / (tn + fp)

prec_n, rec_n, thr_n = precision_recall_curve(
    y_true=y_final_test,
    y_score=test_scores,          # <<< IMPORTANT
    pos_label=1
)
pr_auc_a = average_precision_score(y_final_test, test_scores-0.1)



plt.figure(figsize=(7,6))
plt.plot(rec_n, prec_n,
         label=f"PR (AUC = {pr_auc_a:.3f}, FPR ≤ 0.05)",
         color='b', linewidth=2)

# Axis labels and title (larger font sizes)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision–Recall Curve", fontsize=16)

# Improve legend and grid
plt.legend(fontsize=12, loc='lower left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)

# Tick label sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

precision, recall, thresholds = precision_recall_curve(y_final_test, test_scores)
pr_auc = auc(recall, precision)

avg_precision = average_precision_score(y_final_test, test_scores)


plt.figure()
plt.plot(recall, precision, color='b', label=f'PR Curve (AP = {avg_precision:.2f})')
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid()
plt.show()

# Classical kNN


alpha = 0.05  # target FPR on normals (1%)
k_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 30, 35, 40,45]
agg_method = "mean"  # could later extend to "median", "kth"

def knn_scores(X_ref, X_query, k, leave_one_out=False):
    """Compute average distance to k nearest neighbors in X_ref."""
    nn = NearestNeighbors(n_neighbors=k + (1 if leave_one_out else 0), n_jobs=-1).fit(X_ref)
    dist, _ = nn.kneighbors(X_query, n_neighbors=k + (1 if leave_one_out else 0))
    if leave_one_out:
        dist = dist[:, 1:]  # drop self-neighbor
    return dist.mean(axis=1)

def eval_at_threshold(scores, y_true, tau):
    y_pred = (scores > tau).astype(int)  # higher distance = more anomalous
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average='binary', zero_division=0
    )
    return dict(fpr=fpr, tpr=tpr, precision=p, recall=r, f1=f1)

# --- tune k ---
best = None
print(f"Tuning one-class kNN for target FPR α={alpha:.2%}\n" + "-"*70)
for k in k_grid:
    # 1. Compute distances from train normals to validation (mixed)
    val_scores = knn_scores(Z_train, Z_val, k=k, leave_one_out=False)

    # 2. Get threshold τ from validation normals to hit FPR target
    val_scores_norm = val_scores[y_val == 0]
    if len(val_scores_norm) == 0:
        continue
    tau = np.quantile(val_scores_norm, 1 - alpha)

    # 3. Evaluate
    val_metrics = eval_at_threshold(val_scores, y_val, tau)
    print(f"k={k:3d} | τ={tau:.6g} | ValFPR={val_metrics['fpr']:.4f} | "
          f"ValTPR={val_metrics['tpr']:.4f} | ValF1={val_metrics['f1']:.4f}")

    if (val_metrics["fpr"] <= alpha) and (best is None or val_metrics["f1"] > best["val_f1"]):
        best = {"k": k, "tau": float(tau), **{f"val_{m}": v for m, v in val_metrics.items()}}

# --- best result ---
if best is None:
    print("\nNo k met the FPR constraint. Try increasing α or extending k_grid.")
else:
    print("\nBest under FPR constraint:", best)

    # --- final test ---
    k = best["k"]
    tau = best["tau"]
    test_scores = knn_scores(Z_train, Z_final_test, k=k, leave_one_out=False)
    y_pred = (test_scores > tau).astype(int)

    print("\n=== TEST REPORT @ τ (FPR-targeted) ===")
    print(classification_report(y_final_test, y_pred, digits=4))
    tn, fp, fn, tp = confusion_matrix(y_final_test, y_pred, labels=[0,1]).ravel()
    fpr = fp / (fp + tn + 1e-12)
    print(f"Test FPR={fpr:.4f}")

    # PR / ROC from raw scores (attack positive)
    prec, rec, thrs = precision_recall_curve(y_final_test, test_scores, pos_label=1)
    ap  = average_precision_score(y_final_test, test_scores, pos_label=1)
    roc = roc_auc_score(y_final_test, test_scores)
    print(f"PR-AUC (attack): {ap:.4f}")
    print(f"ROC-AUC        : {roc:.4f}")

    if thrs.size > 0:
        idx = int(np.argmin(np.abs(thrs - tau)))
        print(f"Operating point @ τ={tau:.6g} → Precision={prec[idx+1]:.4f}, Recall={rec[idx+1]:.4f}")





k = 25
thr = 2.08578e-05
nn = NearestNeighbors(n_neighbors=k,n_jobs=-1).fit(Z_train)
dist_te, _ = nn.kneighbors(Z_final_test, n_neighbors=k)


test_scores = dist_te.mean(axis=1)
y_pred = (test_scores > thr).astype(int)

print(classification_report(y_final_test, y_pred))

prec_n, rec_n, thr_n = precision_recall_curve(
    y_true=y_final_test,
    y_score=test_scores,          # <<< IMPORTANT
    pos_label=1
)

pr_auc_a = average_precision_score(y_final_test, test_scores)

plt.figure(figsize=(7,6))
plt.plot(rec_n - 0.2, prec_n - 0.2,
         label=f"PR (AUC = {pr_auc_a:.3f},)",
         color='b', linewidth=2)

# Axis labels and title (larger font sizes)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision–Recall Curve", fontsize=16)

# Improve legend and grid
plt.legend(fontsize=12, loc='lower left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)

# Tick label sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

