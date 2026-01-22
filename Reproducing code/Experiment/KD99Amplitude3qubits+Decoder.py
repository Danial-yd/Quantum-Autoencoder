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
#normal_df = normal_df.sample(n=200_000, random_state=42)

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
    ("pca", PCA(n_components=8))  # match your # of qubits
])



preproc.fit(X_norm)

# 4) Transform normal-train, normal-val, and the mixed test set
X_qae_train = preproc.transform(X_norm)


X_test_mixed = X_attack
y_test_mixed = remain_df["label"].values
X_qae_test   = preproc.transform(X_test_mixed)
X_test_m = preproc.transform(X_test)

AUX, REF0, REF1, TRASH0, TRASH1, LATENT = 0, 1, 2, 3, 4, 5
AUX_REC, REF_DATA0, REF_DATA1, REF_DATA2 = 6, 7, 8, 9
DATA_WIRES = [TRASH0, TRASH1, LATENT]
REF_DATA_WIRES = [REF_DATA0, REF_DATA1, REF_DATA2]
TOTAL_QUBITS = 10

ansatz = RealAmplitudes(3, reps=6, entanglement="linear")

def bind_ansatz(theta):
    return ansatz.bind_parameters(theta) if hasattr(ansatz, "bind_parameters") else ansatz.assign_parameters(theta)

# --- Encoding helpers -------------------------------------------------------
def amp_vec_from_x8(x8):
    v = np.asarray(x8, dtype=np.float64).copy()
    n = np.linalg.norm(v)
    if n == 0.0: v = np.array([1.0]+[0.0]*7, dtype=np.float64)
    else:        v = v / n
    return v.astype(np.complex128)

def build_input_on_three(qc, wires, x, encoding="amplitude"):
    if encoding == "amplitude":
        qc.initialize(amp_vec_from_x8(x), wires)
    elif encoding == "angle":
        th0, th1, th2 = x
        for w, th in zip(wires, (th0, th1, th2)):
            qc.ry(th, w)
    else:
        raise ValueError("encoding must be 'amplitude' or 'angle'")

def build_input_on_data(qc, x, encoding="amplitude"):
    build_input_on_three(qc, DATA_WIRES, x, encoding=encoding)

# --- Utilities --------------------------------------------------------------
def z_expect_on_wire_from_statevector(sv, wire_index):
    amps = np.asarray(sv.data); p0 = p1 = 0.0
    for idx, a in enumerate(amps):
        bit = (idx >> wire_index) & 1
        p = (a.real*a.real + a.imag*a.imag)
        if bit == 0: p0 += p
        else:        p1 += p
    return p0 - p1

def p0_to_fidelity(p0):
    # True fidelity from SWAP test outcome
    return max(0.0, min(1.0, 2.0*p0 - 1.0))

def p_aux0_from_z(z_aux): return 0.5*(z_aux + 1.0)

# --- 1) TRASH fidelity circuit (unchanged idea, separate circuit) ----------
def build_trash_circuit_3q(x, theta, *, encoding="amplitude", ref_angles=None):
    """
    Prepare |x> on DATA, encode, SWAP-test {TRASH0,TRASH1} vs {REF0,REF1} using AUX.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    # prepare 2-qubit reference for trash (default |00>)
    if ref_angles is not None:
        a0, a1 = ref_angles
        qc.ry(a0, REF0); qc.ry(a1, REF1)
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)
    qc.h(AUX); qc.cswap(AUX, REF0, TRASH0); qc.cswap(AUX, REF1, TRASH1); qc.h(AUX)
    return qc

# --- 2) RECON fidelity circuit (swap in clean trash, then decode) ----------
def build_recon_circuit_3q(x, theta, *, encoding="amplitude", ref_angles=None):
    """
    Prepare |x>, encode with U(θ), swap in clean trash from REF0/REF1, decode with U(θ)†,
    prepare |x> on REF_DATA_[0..2], SWAP-test DATA vs REF_DATA using AUX_REC.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    # prepare clean trash refs (default |00>)
    if ref_angles is not None:
        a0, a1 = ref_angles
        qc.ry(a0, REF0); qc.ry(a1, REF1)
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    # >>> KEY STEP: replace trash with clean reference <<<
    qc.swap(TRASH0, REF0); qc.swap(TRASH1, REF1)

    # decode
    qc.compose(bind_ansatz(theta).inverse(), DATA_WIRES, inplace=True)

    # fresh |x> for comparison
    build_input_on_three(qc, REF_DATA_WIRES, x, encoding=encoding)

    # register SWAP test (3 cswaps)
    qc.h(AUX_REC)
    qc.cswap(AUX_REC, TRASH0, REF_DATA0)
    qc.cswap(AUX_REC, TRASH1, REF_DATA1)
    qc.cswap(AUX_REC, LATENT,  REF_DATA2)
    qc.h(AUX_REC)
    return qc

# --- Losses (use TRUE fidelities) ------------------------------------------
def qae_losses(theta, batch_X, *, encoding="amplitude", ref_angles=None):
    L_trash, L_recon = [], []
    for x in batch_X:
        sv_t = Statevector.from_instruction(build_trash_circuit_3q(x, theta, encoding=encoding, ref_angles=ref_angles))
        Ft = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_t, AUX)))  # in [0,1]

        sv_r = Statevector.from_instruction(build_recon_circuit_3q(x, theta, encoding=encoding, ref_angles=ref_angles))
        Fr = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_r, AUX_REC)))  # in [0,1]

        # maximize both fidelities -> minimize (1 - F)
        L_trash.append(1.0 - Ft)
        L_recon.append(1.0 - Fr)
    return float(np.mean(L_trash)), float(np.mean(L_recon))

def qae_combo_loss(theta, batch_X, *, lam=0.5, encoding="amplitude", ref_angles=None):
    L_t, L_r = qae_losses(theta, batch_X, encoding=encoding, ref_angles=ref_angles)
    return lam * L_t + (1.0 - lam) * L_r

def _wrap_angles(th):
    """
    Wrap parameters back into [-π, π] to keep angles bounded.
    This helps avoid numerical instability when training on periodic landscapes.
    """
    return (th + np.pi) % (2 * np.pi) - np.pi

def _fd_grad_combo(theta, X_batch, *, lam=0.5, encoding="amplitude", ref_angles=None, fd_eps=1e-3):
    theta = _wrap_angles(theta)
    base = qae_combo_loss(theta, X_batch, lam=lam, encoding=encoding, ref_angles=ref_angles)
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        th = theta.copy(); th[i] += fd_eps
        g[i] = (qae_combo_loss(th, X_batch, lam=lam, encoding=encoding, ref_angles=ref_angles) - base) / fd_eps
    return g, base

def train_qae_combo_with_history(
    X_train, *,
    lam=0.5,                     # weight between trash and reconstruction losses
    epochs=200,
    batch_size=512,
    encoding="amplitude",
    ref_angles=None,
    seed=0,
    lr=0.01,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    fd_eps=1e-3,
    steps_per_epoch=1
):
    rng = np.random.default_rng(seed)
    theta = 0.01 * rng.standard_normal(ansatz.num_parameters)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    steps, loss_total_hist, loss_trash_hist, loss_recon_hist, fid_trash_hist, fid_recon_hist = [], [], [], [], [], []
    first_epoch_time = None

    pbar = trange(epochs, desc="QAE training", leave=True)
    for ep in pbar:
        # sample minibatch
        mb_sz = min(batch_size, len(X_train))
        idx = rng.choice(len(X_train), mb_sz, replace=False)
        Xb = X_train[idx]

        t0 = perf_counter()
        # Do a few gradient steps per epoch
        for k in range(steps_per_epoch):
            g, L = _fd_grad_combo(
                theta, Xb, lam=lam, encoding=encoding, ref_angles=ref_angles, fd_eps=fd_eps
            )
            t = ep * max(1, steps_per_epoch) + (k + 1)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            theta = _wrap_angles(theta - lr * m_hat / (np.sqrt(v_hat) + eps))
        t1 = perf_counter()
        if ep == 0:
            first_epoch_time = t1 - t0

        # Evaluate on a random validation slice
        val_idx = rng.choice(len(X_train), min(2000, len(X_train)), replace=False)
        X_val = X_train[val_idx]
        L_t, L_r = qae_losses(theta, X_val, encoding=encoding, ref_angles=ref_angles)
        L_total = lam * L_t + (1 - lam) * L_r
        F_t, F_r = 1 - L_t, 1 - L_r

        # log
        steps.append(ep)
        loss_total_hist.append(L_total)
        loss_trash_hist.append(L_t)
        loss_recon_hist.append(L_r)
        fid_trash_hist.append(F_t)
        fid_recon_hist.append(F_r)

        pbar.set_postfix({
            "L_tot": f"{L_total:.4f}",
            "L_trash": f"{L_t:.4f}",
            "L_rec": f"{L_r:.4f}",
            "fid_t": f"{F_t:.3f}",
            "fid_r": f"{F_r:.3f}",
            "t/ep(s)": f"{(t1 - t0):.2f}"
        })

    print(f"\nFirst epoch time: {first_epoch_time:.3f} s")
    print(f"Final total loss: {loss_total_hist[-1]:.6f}")
    print(f"Final trash fidelity: {fid_trash_hist[-1]:.6f} | recon fidelity: {fid_recon_hist[-1]:.6f}")

    return (
        theta,
        np.array(steps),
        np.array(loss_total_hist),
        np.array(loss_trash_hist),
        np.array(loss_recon_hist),
        np.array(fid_trash_hist),
        np.array(fid_recon_hist)
    )

def latent_z_after_encoder_3q(x8, theta):
    qc = QuantumCircuit(3)
    qc.initialize(amp_vec_from_x8(x8), [0,1,2])      # data wires
    qc.compose(bind_ansatz(theta), [0,1,2], inplace=True)
    sv = Statevector.from_instruction(qc)
    return z_expect_on_wire_from_statevector(sv, 2)   # qubit index 2 = latent

def latent_z_after_decoder_3q(x8, theta):
    qc = QuantumCircuit(3)
    qc.initialize(amp_vec_from_x8(x8), [0,1,2])
    qc.compose(bind_ansatz(theta), [0,1,2], inplace=True)
    qc.compose(bind_ansatz(theta).inverse(), [0,1,2], inplace=True)
    sv = Statevector.from_instruction(qc)
    return z_expect_on_wire_from_statevector(sv, 2)

def embed_latent_Z_batch_3q(X8, theta, after_decoder=False):
    z = np.empty((len(X8), 1), float)
    fn = latent_z_after_decoder_3q if after_decoder else latent_z_after_encoder_3q
    for i, x in enumerate(X8):
        z[i, 0] = fn(x, theta)
    return z

print(len(ansatz.parameters))

LAM = 0.5                 # tilt toward reconstruction early on
EPOCHS = 50
BATCH = 64
STEPS_PER_EPOCH = 3       # a few Adam steps per epoch speeds learning
FD_EPS = 1e-3
SEED = 42


theta_opt, steps, L_tot, L_tr, L_rec, F_tr, F_rec = train_qae_combo_with_history(
    X_qae_train,
    lam=LAM,
    epochs=EPOCHS,
    batch_size=BATCH,
    steps_per_epoch=STEPS_PER_EPOCH,
    fd_eps=FD_EPS,
    seed=SEED
)
np.save("theta", theta_opt)


plt.figure(figsize=(8, 5))
plt.plot(steps, L_tot, label="Total loss", color='black', linewidth=2)
plt.plot(steps, L_tr, label="Trash loss", linestyle="--")
plt.plot(steps, L_rec, label="Recon loss", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("QAE Training Losses")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(steps, F_tr, label="Trash fidelity", linestyle="--")
plt.plot(steps, F_rec, label="Recon fidelity", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Fidelity")
plt.title("QAE Training Fidelities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# X_qae_test = pipe.transform(X_test_features)  # 8D vectors
Z_lat = embed_latent_Z_batch_3q(X_qae_train, theta_opt, after_decoder=False)
np.save("Z_train", Z_lat)
z_vals = Z_lat.ravel()# shape (N,1)
plt.figure(figsize=(6,4))
plt.hist(z_vals, bins=30, color='skyblue', edgecolor='black')
plt.xlabel(r'$\langle Z_{\mathrm{latent}}\rangle$', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Latent Qubit Z Expectation Distribution')
plt.grid(alpha=0.3)
plt.show()


Z_test = embed_latent_Z_batch_3q(X_test_m, theta_opt, after_decoder=False )
np.save("Z_test_m", Z_test)
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

dir = '/Users/danialyntykbay/thesis/AmplitudeEn_Decoder/3qubits/1latent/9layers'
os.chdir(dir)
os.getcwd()
theta_opt = np.load("theta.npy")
Z_train = np.load("Z_train.npy")
Z_test_m = np.load("Z_test_m.npy")


from bisect import bisect_left

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

def fit_threshold(theta_train, k, quantile, agg="mean"):
    # leave-one-out to avoid trivial zero from self-match
    train_scores = qknn_scores_sorted(theta_train, theta_train, k=k, agg=agg, leave_one_out=True)
    thr = float(np.quantile(train_scores, quantile))
    return thr, train_scores

def predict_labels(theta_train, theta_query, k, thr, agg="mean"):
    scores = qknn_scores_sorted(theta_train, theta_query, k=k, agg=agg, leave_one_out=False)
    y_pred = (scores > thr).astype(np.int8)
    return y_pred, scores



Z_val, Z_final_test, y_val, y_final_test = train_test_split(
    Z_1, X_test_values,
    test_size=0.8,          # keep 70% for final test, 30% for validation
    stratify=X_test_values,        # preserve normal/attack ratio
    random_state=42
)
alpha = 0.2  # e.g., 1% false alarms allowed on normals

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
thr = 1.32143e-12

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
pr_auc_a = average_precision_score(y_final_test, test_scores)



plt.figure(figsize=(7,6))
plt.plot(rec_n, prec_n,
         label=f"PR (AUC = {pr_auc_a:.3f}, FPR ≤ 0.2)",
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

# Classical kNN

alpha = 0.2  # target FPR on normals (1%)
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
print(f"Tuning kNN for target FPR α={alpha:.2%}\n" + "-"*70)
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





k = 2
thr = 1.51038e-06
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

def fidelities_per_row_3q(X_test, theta, *, encoding="amplitude", ref_angles=None):
    """
    Returns two arrays (Ft_per_row, Fr_per_row) with len == len(X_test).
      • Ft_per_row[i]: trash fidelity for sample i (AUX from build_trash_circuit_3q)
      • Fr_per_row[i]: reconstruction fidelity for sample i (AUX_REC from build_recon_circuit_3q)
    """
    Ft_list, Fr_list = [], []

    for x in X_test:
        # --- Trash fidelity ---
        sv_t = Statevector.from_instruction(
            build_trash_circuit_3q(x, theta, encoding=encoding, ref_angles=ref_angles)
        )
        Ft = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_t, AUX)))

        # --- Reconstruction fidelity ---
        sv_r = Statevector.from_instruction(
            build_recon_circuit_3q(x, theta, encoding=encoding, ref_angles=ref_angles)
        )
        Fr = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_r, AUX_REC)))

        Ft_list.append(Ft)
        Fr_list.append(Fr)

    return np.array(Ft_list, dtype=float), np.array(Fr_list, dtype=float)

Ft_rowss, Fr_rowss = fidelities_per_row_3q(
    X_test_m, theta_opt,
    encoding="amplitude",
    ref_angles=None     # or (0.0, 0.0) if you want explicit |00> as clean trash
)

prec_n, rec_n, thr_n = precision_recall_curve(
    y_true=X_test_values,
    y_score=Fr_rowss,          # <<< IMPORTANT
    pos_label=1
)

pr_auc_a = average_precision_score(X_test_values, Fr_rowss)

plt.figure(figsize=(7,6))
plt.plot(rec_n, prec_n,
         label=f"PR (AUC = {pr_auc_a:.3f})",
         color='b', linewidth=2)

# Axis labels and title (larger font sizes)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision–Recall Curve", fontsize=16)

plt.legend(fontsize=12, loc='lower left', frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)

# Tick label sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

y_true = X_test_values        # 0 = normal, 1 = attack
scores = Fr_rowss              # fidelity-based anomaly score (higher = more anomalous)
alpha = 0.50                  # target FPR = 5%


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


y_pred = (scores > thr).astype(int)

y_pred = (scores > 0.90).astype(int)
cls_report = classification_report(y_true, y_pred)
print(cls_report)
