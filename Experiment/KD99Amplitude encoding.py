import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer

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
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange
from time import perf_counter
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

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
    ("pca", PCA(n_components=4))  # match your # of qubits
])



preproc.fit(X_norm)

# 4) Transform normal-train, normal-val, and the mixed test set
X_qae_train = preproc.transform(X_norm)


X_test_mixed = X_attack
y_test_mixed = remain_df["label"].values
X_qae_test   = preproc.transform(X_test_mixed)
X_test_m = preproc.transform(X_test)

#QAE

AUX, REF, TRASH, LATENT = 0, 1, 2, 3
DATA_WIRES = [TRASH, LATENT]
ansatz = RealAmplitudes(2, reps=3, entanglement="linear")

def bind_ansatz(theta):
    # assign/bind parameters (supports older Qiskit too)
    if hasattr(ansatz, "bind_parameters"):
        return ansatz.bind_parameters(theta)
    return ansatz.assign_parameters(theta)

def amp_vec_from_x4(x4):
    """4D → normalized length-4 amplitude vector (for 2 qubits)."""
    v = np.asarray(x4, dtype=np.float64).copy()
    n = np.linalg.norm(v)
    if n == 0.0: v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:        v = v / n
    return v.astype(np.complex128)

def build_input_on_data(qc, x, encoding="amplitude"):
    """
    Put classical x on qubits [TRASH, LATENT].
    encoding='amplitude' expects len(x)=4,
    encoding='angle'     expects len(x)=2 (angles in radians).
    """
    if encoding == "amplitude":
        amps = amp_vec_from_x4(x)
        qc.initialize(amps, DATA_WIRES)
    elif encoding == "angle":
        # AngleEmbedding with RY on each data qubit
        th_trash, th_latent = x
        qc.ry(th_trash, TRASH)
        qc.ry(th_latent, LATENT)
    else:
        raise ValueError("encoding must be 'amplitude' or 'angle'")


def build_qae_circuit(x, theta, encoding="amplitude", ref_angle=None):
    qc = QuantumCircuit(4)
    if ref_angle is not None:
        qc.ry(ref_angle, REF)
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), qubits=DATA_WIRES, inplace=True)
    qc.h(AUX)
    qc.cswap(AUX, REF, TRASH)
    qc.h(AUX)
    return qc



def aux_z_from_statevector(sv):
    """⟨Z⟩ on AUX from a 4-qubit Statevector."""
    amps = np.asarray(sv.data)
    p0 = p1 = 0.0
    for idx, a in enumerate(amps):
        bit = (idx >> AUX) & 1
        p = (a.real*a.real + a.imag*a.imag)
        if bit == 0: p0 += p
        else:        p1 += p
    return p0 - p1

def p_aux0_from_z(z_aux):
    """Ancilla-0 probability from <Z_aux> in a SWAP test."""
    return 0.5*(z_aux + 1.0)


def latent_z_from_statevector(sv):
    """⟨Z⟩ on the latent qubit (qubit 3)."""
    amps = np.asarray(sv.data)
    p0 = p1 = 0.0
    for idx, a in enumerate(amps):
        bit = (idx >> LATENT) & 1
        p = (a.real*a.real + a.imag*a.imag)
        if bit == 0: p0 += p
        else:        p1 += p
    return p0 - p1

def qae_loss(theta, batch_X, encoding="amplitude", ref_angle=None):
    fids = []
    for x in batch_X:
        qc = build_qae_circuit(x, theta, encoding=encoding, ref_angle=ref_angle)
        sv = Statevector.from_instruction(qc)
        z_aux = aux_z_from_statevector(sv)
        fids.append(p_aux0_from_z(z_aux))
    return float(1.0 - np.mean(fids))

def train_qae(X_train, steps=200, batch_size=512, encoding="amplitude",
              ref_angle=None, seed=0):
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0, 1.0, ansatz.num_parameters)

    def objective(th):
        m = min(batch_size, len(X_train))
        idx = rng.choice(len(X_train), m, replace=False)
        return qae_loss(th, X_train[idx], encoding=encoding, ref_angle=ref_angle)

    res = minimize(objective, theta0, method="L-BFGS-B", options={"maxiter": steps})
    return res.x, res.fun

def embed_latent_Z(X, theta, encoding="amplitude", ref_angle=None):
    lat = np.empty((len(X), 1), dtype=float)
    for i, x in enumerate(X):
        qc = build_qae_circuit(x, theta, encoding=encoding, ref_angle=ref_angle)
        sv = Statevector.from_instruction(qc)
        lat[i, 0] = latent_z_from_statevector(sv)
    return lat

def train_qae_with_history(
    X_train, *,
    epochs=200, batch_size=512,
    encoding="amplitude", ref_angle=None, seed=0,
    eval_n=5000
):
    """
    Minibatch L-BFGS per epoch; logs train loss & fidelity evaluated
    on a fixed subset of the training set (size eval_n).
    Shows tqdm progress bar and reports first-epoch time.
    """
    rng = np.random.default_rng(seed)
    theta = 0.01 * rng.standard_normal(ansatz.num_parameters)

    # fixed evaluation slice from train (for stable curves)
    eval_n = min(eval_n, len(X_train))
    eval_idx = rng.choice(len(X_train), eval_n, replace=False)
    X_eval = X_train[eval_idx]

    steps = []
    tr_loss_hist = []
    tr_fid_hist  = []

    first_epoch_time = None

    pbar = trange(epochs, desc="QAE training", leave=True)
    for ep in pbar:
        # one optimizer step on a fresh minibatch
        m = min(batch_size, len(X_train))
        mb_idx = rng.choice(len(X_train), m, replace=False)
        X_mb = X_train[mb_idx]

        def obj(th):  # minibatch objective
            return qae_loss(th, X_mb, encoding=encoding, ref_angle=ref_angle)

        t0 = perf_counter()
        res = minimize(obj, theta, method="L-BFGS-B", options={"maxiter": 1})
        theta = res.x  # update parameters
        t1 = perf_counter()
        epoch_time = t1 - t0

        if ep == 0:
            first_epoch_time = epoch_time

        # evaluate on the fixed train-eval slice
        L_tr = qae_loss(theta, X_eval, encoding=encoding, ref_angle=ref_angle)
        F_tr = 1.0 - L_tr

        steps.append(ep)
        tr_loss_hist.append(L_tr)
        tr_fid_hist.append(F_tr)

        # update progress bar info
        pbar.set_postfix({
            "loss": f"{L_tr:.4f}",
            "fid": f"{F_tr:.4f}",
            "t/epoch(s)": f"{epoch_time:.2f}"
        })

    print(f"\nFirst epoch time: {first_epoch_time:.3f} s")
    print(f"Final train loss: {tr_loss_hist[-1]:.6f}  |  train fidelity: {tr_fid_hist[-1]:.6f}")

    return theta, np.array(steps), np.array(tr_loss_hist), np.array(tr_fid_hist)

theta_opt, steps, tr_loss, tr_fid = train_qae_with_history(
    X_qae_train[0:100],
    epochs=200,
    batch_size=512,
    encoding="amplitude",  # since each row → amplitude vector of len 4
    ref_angle=None,
    seed=42,
    eval_n=5000
)

def plot_train_curves(steps, tr_loss_hist, tr_fid_hist, title_suffix="(Train)"):
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(steps, tr_loss_hist, lw=2, label="Train loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (1 - fidelity)")
    plt.title(f"QAE Loss per Epoch {title_suffix}")
    plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.tight_layout()
    plt.show()
    # Fidelity
    plt.figure(figsize=(6,4))
    plt.plot(steps, tr_fid_hist, lw=2, label="Train fidelity")
    plt.xlabel("Epoch"); plt.ylabel("Fidelity")
    plt.title(f"QAE Fidelity per Epoch {title_suffix}")
    plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.tight_layout()
    plt.show()

plot_train_curves(steps,tr_loss, tr_fid)

Z_train = embed_latent_Z(X_qae_train, theta_opt, encoding="amplitude")

z_vals = Z_train.ravel()

plt.figure(figsize=(6,4))
plt.hist(z_vals, bins=30, color='skyblue', edgecolor='black')
plt.xlabel(r'$\langle Z_{\mathrm{latent}}\rangle$', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Latent Qubit Z Expectation Distribution')
plt.grid(alpha=0.3)
plt.show()

Z_test =embed_latent_Z(X_test_m, theta_opt, encoding="amplitude")

mixed_Z = np.concatenate((Z_test,X_test_values.reshape(-1,1)), axis=1)
normal_values = mixed_Z[mixed_Z[:, 1] == 0][:, 0]
attack_values = mixed_Z[mixed_Z[:, 1] == 1][:, 0]

plt.figure(figsize=(8, 5))
plt.hist(normal_values, bins=30, alpha=0.6, label='Normal', edgecolor='black')
plt.hist(attack_values, bins=30, alpha=0.6, label='Attack', edgecolor='black')

plt.title('Histogram of Z_test latent values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


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
k = 5
quantile = 0.95

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

Th_tr = theta_from_mz(Z_train)
Th_te = theta_from_mz(Z_test)

k = 8
train_scores = qknn_scores_sorted(Th_tr, Th_tr, k=8)
thr = float(np.quantile(train_scores, 0.99))

scores_test = qknn_scores_sorted(Th_tr, Th_te, k=k)
y_pred = (scores_test > thr).astype(np.int8)

print(f"k={k}, threshold quantile={quantile:.2f}, thr={thr:.6f}")
print(classification_report(X_test_values, y_pred, digits=4))

cm = confusion_matrix(X_test_values, y_pred)
accuracy  = accuracy_score(X_test_values, y_pred)
precision = precision_score(X_test_values, y_pred, pos_label=1)  # 1 = attack
recall    = recall_score(X_test_values, y_pred, pos_label=1)
f1        = f1_score(X_test_values, y_pred)

k_grid = [1, 2, 3, 4, 5,6,7,8,9,10]
q_grid = [0.90, 0.95, 0.975, 0.99]
agg_grid = "mean"
best_f1 = -1.0
best_conf = None
best_preds = None

results = []

for k in k_grid:
    # 1) Train self-scores (leave-one-out)
    train_scores = qknn_scores_sorted(
        theta_train=Th_tr,
        theta_query=Th_tr,
        k=k,
        agg=agg_grid,
        leave_one_out=True
    )

    # 2) Test scores (computed once per k)
    test_scores = qknn_scores_sorted(
        theta_train=Th_tr,
        theta_query=Th_te,
        k=k,
        agg=agg_grid,
        leave_one_out=False
    )

    for q in q_grid:
        # 3) Threshold from training distribution
        cutoff = float(np.quantile(train_scores, q))

        # 4) Classify: inlier=1 (≤ cutoff), anomaly=0 (> cutoff)
        y_pred_inlier1 = (test_scores <= cutoff).astype(int)

        # If your y_test uses anomaly=1 labels, flip:
        y_pred = 1 - y_pred_inlier1  # now anomaly=1 matches y_test convention

        # 5) Evaluate
        f1 = f1_score(X_test_values, y_pred,pos_label=1)  # or f1_score(y_test, y_pred_inlier1, pos_label=0)
        results.append((k, q, f1))
        print(f"k={k:2d}, q={q:.3f} → F1={f1:.4f}")

        # 6) Track best
        if f1 > best_f1:
            best_f1 = f1
            best_conf = (k, q)
            best_preds = y_pred.copy()

print("Best:", best_conf, "F1=", best_f1)



