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

# ---- embedding + encoder (same shapes you used) ----
def angle_embed(x):
    # x: array-like of length 2 -> embed on TRASH, COMP via RY
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
    # default |0>, or RY(phi)|0> if you want nontrivial ref
    if phi is not None:
        qml.RY(phi, wires=REF)

# ===========================
# Loss terms QNode
# ===========================

@qml.qnode(dev, interface="autograd")
def trash_fidelity_qnode(x, weights, ref_angle=None):
    prepare_reference(ref_angle)
    angle_embed(x)
    encoder_rot_entangle(weights)
    qml.Hadamard(wires=CTRL)
    qml.CSWAP(wires=[CTRL, REF, TRASH])
    qml.Hadamard(wires=CTRL)
    z_ctrl = qml.expval(qml.PauliZ(CTRL))
    return z_ctrl   # F_trash = (z_ctrl + 1)/2

# --- QNode 2: Reconstruction fidelity (no measurements until the end) ---
@qml.qnode(dev, interface="autograd")
def recon_fidelity_qnode(x, weights, ref_angle=None):
    prepare_reference(ref_angle)
    angle_embed(x)
    encoder_rot_entangle(weights)
    # reset TRASH by swapping in the clean REF
    qml.SWAP(wires=[TRASH, REF])
    # decode
    qml.adjoint(encoder_rot_entangle)(weights)
    # local un-prep of target |ψ(x)>=RY(x0)⊗RY(x1)|00>
    qml.RY(-x[0], wires=TRASH)
    qml.RY(-x[1], wires=COMP)
    z_t   = qml.expval(qml.PauliZ(TRASH))
    z_c   = qml.expval(qml.PauliZ(COMP))
    zz_tc = qml.expval(qml.PauliZ(TRASH) @ qml.PauliZ(COMP))
    return z_t, z_c, zz_tc   # F_recon = (1 + z_t + z_c + zz_tc)/4

# ===========================
# Batch loss + trainer
# ===========================
def combined_loss_on_batch(X_angles, weights, ref_angle=None, lam_trash=0.5):
    """
    L = lam_trash * (1 - F_trash) + (1 - lam_trash) * (1 - F_recon)
    """
    Ft_list, Fr_list = [], []
    for x in X_angles:
        z_ctrl = trash_fidelity_qnode(x, weights, ref_angle)
        z_t, z_c, zz_tc = recon_fidelity_qnode(x, weights, ref_angle)

        F_trash = 0.5 * (z_ctrl + 1.0)
        F_recon = 0.25 * (1.0 + z_t + z_c + zz_tc)

        Ft_list.append(F_trash)
        Fr_list.append(F_recon)

    Ft = qnp.mean(qnp.stack(Ft_list))
    Fr = qnp.mean(qnp.stack(Fr_list))
    L  = lam_trash * (1.0 - Ft) + (1.0 - lam_trash) * (1.0 - Fr)
    return L, (Ft, Fr)

def train_qae_combined(
    X_train, X_test=None, steps=200, batch_size=128, lr=0.15,
    ref_angle=None, seed=42, lam_trash=0.3
):
    """
    Returns dict with weights and history arrays ready for plotting.
    """
    rng = qnp.random.default_rng(seed)
    w0 = 0.01 * rng.normal(size=(2, 2, 3))
    weights = qnp.array(w0, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)

    Ntr = len(X_train)
    idx = qnp.arange(min(batch_size, Ntr))
    Xb_tr = X_train[idx]

    steps_hist = []
    tr_loss_hist, tr_Ft_hist, tr_Fr_hist = [], [], []
    te_loss_hist, te_Ft_hist, te_Fr_hist = [], [], []

    pbar = trange(steps + 1, desc="QAE (trash + recon)", leave=True)
    for s in pbar:
        # ---- train (on fixed minibatch) ----
        Ltr, (Ft, Fr) = combined_loss_on_batch(
            Xb_tr, weights, ref_angle, lam_trash=lam_trash
        )
        steps_hist.append(s)
        tr_loss_hist.append(qml.math.toarray(Ltr))
        tr_Ft_hist.append(qml.math.toarray(Ft))
        tr_Fr_hist.append(qml.math.toarray(Fr))

        # ---- test (on full test set), optional ----
        if X_test is not None:
            Lte, (Ft_te, Fr_te) = combined_loss_on_batch(
                X_test, weights, ref_angle, lam_trash=lam_trash
            )
            te_loss_hist.append(qml.math.toarray(Lte))
            te_Ft_hist.append(qml.math.toarray(Ft_te))
            te_Fr_hist.append(qml.math.toarray(Fr_te))
            pbar.set_postfix({
                "loss": float(Ltr), "Ft": float(Ft), "Fr": float(Fr),
                "te_loss": float(Lte), "te_Ft": float(Ft_te), "te_Fr": float(Fr_te)
            })
        else:
            pbar.set_postfix({"loss": float(Ltr), "Ft": float(Ft), "Fr": float(Fr)})

        # ---- step ----
        if s < steps:
            def _obj(w):
                Lb, _ = combined_loss_on_batch(
                    Xb_tr, w, ref_angle, lam_trash=lam_trash
                )
                return Lb
            weights = opt.step(_obj, weights)
            weights = qnp.array(weights, requires_grad=True)

    outs = {
        "weights": weights,
        "steps": np.asarray(steps_hist),
        "train_loss": np.asarray(tr_loss_hist, dtype=float),
        "train_F_trash": np.asarray(tr_Ft_hist, dtype=float),
        "train_F_recon": np.asarray(tr_Fr_hist, dtype=float),
    }
    if X_test is not None:
        outs.update({
            "test_loss": np.asarray(te_loss_hist, dtype=float),
            "test_F_trash": np.asarray(te_Ft_hist, dtype=float),
            "test_F_recon": np.asarray(te_Fr_hist, dtype=float),
        })
    return outs

@qml.qnode(dev, interface="autograd")
def reconstruct_readouts(x, weights, ref_angle=None):
    prepare_reference(ref_angle)
    angle_embed(x)
    encoder_rot_entangle(weights)
    qml.SWAP(wires=[TRASH, REF])
    qml.adjoint(encoder_rot_entangle)(weights)
    # read Z on both qubits (without un-prep)
    return qml.expval(qml.PauliZ(TRASH)), qml.expval(qml.PauliZ(COMP))

def _z_to_angle(z):
    z = np.clip(z, -0.999999, 0.999999)
    return np.arccos(z)

def reconstruct_angles_batch(X_angles, weights, ref_angle=None):
    rec = []
    for x in X_angles:
        zt, zc = reconstruct_readouts(x, weights, ref_angle)
        zt = float(qml.math.toarray(zt)); zc = float(qml.math.toarray(zc))
        rec.append([_z_to_angle(zt), _z_to_angle(zc)])
    return np.asarray(rec)
def to_angles_0_pi(X):
    xmin = X.min(axis=0); xmax = X.max(axis=0)
    # avoid divide-by-zero if a column is constant
    scale = qnp.where(qnp.isclose(xmax - xmin, 0.0), 1.0, xmax - xmin)
    return (X - xmin) / scale * qnp.pi

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

history = train_qae_combined( X_ang_train,None, steps=200, batch_size=128, lr=0.1,
ref_angle=None, seed=42, lam_trash=0.5)

def plot_fidelities(history, show_test=True):
    s = history["steps"]
    plt.figure(figsize=(6.0, 4.5))
    plt.plot(s, history["train_F_trash"], label="Train: Trash fidelity")
    plt.plot(s, history["train_F_recon"], label="Train: Recon fidelity")
    if show_test and ("test_F_trash" in history) and ("test_F_recon" in history):
        plt.plot(s, history["test_F_trash"],  linestyle="--", label="Test: Trash fidelity")
        plt.plot(s, history["test_F_recon"],  linestyle="--", label="Test: Recon fidelity")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    plt.title("QAE training — trash & reconstruction fidelities")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_losses(history, show_test=True):
    s = history["steps"]
    plt.figure(figsize=(6.0, 4.5))
    plt.plot(s, history["train_loss"], label="Train loss")
    if show_test and ("test_loss" in history):
        plt.plot(s, history["test_loss"], linestyle="--", label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Combined loss")
    plt.title("QAE training — combined loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_fidelities(history, show_test=True)
plot_losses(history, show_test=True)
from tqdm.auto import tqdm
def recon_fidelity_per_row(X_angles, weights, ref_angle=None, batch_size=1024, show_progress=True):
    N = len(X_angles)
    F_recon = np.empty(N, dtype=float)

    rng = range(0, N, batch_size)
    if show_progress:
        rng = tqdm(rng, desc="Reconstruction fidelity (per sample)")

    for start in rng:
        end = min(start + batch_size, N)
        Xb = X_angles[start:end]

        # evaluate batch sequentially (per sample) to keep memory low
        fr_list = []
        for x in Xb:
            z_t, z_c, zz_tc = recon_fidelity_qnode(x, weights, ref_angle)
            fr = 0.5 * (1.0 + float(z_t) + float(z_c) + float(zz_tc))
            fr_list.append(fr)

        F_recon[start:end] = fr_list

    return F_recon

def trash_fidelity_per_row(X_angles, weights, ref_angle=None, batch_size=1024, show_progress=True):
    """
    Returns array of F_trash for each row in X_angles:
      F_trash = (⟨Z_ctrl⟩ + 1)/2
    """
    N = len(X_angles)
    F_trash = np.empty(N, dtype=float)

    rng = range(0, N, batch_size)
    if show_progress:
        rng = tqdm(rng, desc="Trash fidelity (per sample)")

    for start in rng:
        end = min(start + batch_size, N)
        Xb = X_angles[start:end]
        f_list = []
        for x in Xb:
            z_ctrl = trash_fidelity_qnode(x, weights, ref_angle)
            f_list.append(0.5 * (float(z_ctrl) + 1.0))
        F_trash[start:end] = f_list
    return F_trash


def evaluate_fidelities(X_angles, weights, ref_angle=None, batch_size=1024, show_progress=True):
    """
    Convenience: compute both fidelities and report means.
    """
    Ft = trash_fidelity_per_row(X_angles, weights, ref_angle, batch_size, show_progress)
    Fr = recon_fidelity_per_row(X_angles, weights, ref_angle, batch_size, show_progress)
    return {
        "F_trash": Ft,
        "F_recon": Fr,
        "mean_F_trash": float(Ft.mean()),
        "mean_F_recon": float(Fr.mean()),
    }

from pennylane import numpy as qnp
df_test = evaluate_fidelities(X_ang_test_m, history["weights"])
np.savez("df_test.npz",
         F_trash=df_test["F_trash"],
         F_recon=df_test["F_recon"],
         mean_F_trash=df_test["mean_F_trash"],
         mean_F_recon=df_test["mean_F_recon"])

os.getcwd()
os.chdir("/Users/danialyntykbay/thesis/")
data = np.load("df_test.npz")

F_trash = data["F_trash"]
F_recon = data["F_recon"]

prec_n, rec_n, thr_n = precision_recall_curve(
    y_true=X_test_values,
    y_score=F_trash,          # <<< IMPORTANT
    pos_label=1
)

pr_auc_a = average_precision_score(X_test_values, F_trash)

plt.figure(figsize=(7,6))
plt.plot(rec_n, prec_n,
         label=f"PR (AUC = {pr_auc_a:.3f})",
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

y_true = X_test_values        # 0 = normal, 1 = attack
scores = F_trash              # fidelity-based anomaly score (higher = more anomalous)
alpha = 0.8                  # target FPR = 5%


thresholds = np.linspace(0, 1, 500)
best = None
metrics = []

for tau in thresholds:
    y_pred = (scores >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    metrics.append((tau, fpr, tpr, p, r, f1))
    if fpr <= alpha and (best is None or f1 > best["f1"]):
        best = {"tau": tau, "fpr": fpr, "tpr": tpr, "precision": p, "recall": r, "f1": f1}


y_pred = (scores > 0.99).astype(int)
cls_report = classification_report(y_true, y_pred)
print(cls_report)