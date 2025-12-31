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



# --- Qubit layout (6 total) -----------------------------------------------
# 0:AUX | 1:REF (trash ref) | 2:TRASH, 3:LATENT (data) | 4:REF_D0, 5:REF_D1 (recon refs)
AUX, REF, TRASH, LATENT, REF_D0, REF_D1 = 0, 1, 2, 3, 4, 5
DATA_WIRES = [TRASH, LATENT]
REF_DATA_WIRES = [REF_D0, REF_D1]
TOTAL_QUBITS = 6

# --- Ansatz (2 data qubits) ------------------------------------------------
ansatz = RealAmplitudes(2, reps=6, entanglement="linear")
def bind_ansatz(theta):
    return ansatz.bind_parameters(theta) if hasattr(ansatz, "bind_parameters") else ansatz.assign_parameters(theta)

# --- Encoding --------------------------------------------------------------
def amp_vec_from_x4(x4):
    v = np.asarray(x4, dtype=np.float64).copy()
    n = np.linalg.norm(v)
    v = np.array([1.0,0.0,0.0,0.0], dtype=np.float64) if n == 0.0 else (v / n)
    return v.astype(np.complex128)

def build_input_on_data(qc, x, encoding="amplitude"):
    if encoding == "amplitude":
        qc.initialize(amp_vec_from_x4(x), DATA_WIRES)
    elif encoding == "angle":
        th_trash, th_latent = x; qc.ry(th_trash, TRASH); qc.ry(th_latent, LATENT)
    else:
        raise ValueError("encoding must be 'amplitude' or 'angle'")

def build_input_on_two(qc, wires, x, encoding="amplitude"):
    if encoding == "amplitude":
        qc.initialize(amp_vec_from_x4(x), wires)
    elif encoding == "angle":
        th0, th1 = x; qc.ry(th0, wires[0]); qc.ry(th1, wires[1])
    else:
        raise ValueError("encoding must be 'amplitude' or 'angle'")

# --- Utilities -------------------------------------------------------------
def z_expect_on_wire_from_statevector(sv, wire_index):
    amps = np.asarray(sv.data); p0 = p1 = 0.0
    for idx, a in enumerate(amps):
        bit = (idx >> wire_index) & 1
        p = (a.real*a.real + a.imag*a.imag)
        if bit == 0: p0 += p
        else:        p1 += p
    return p0 - p1

def p_aux0_from_z(z_aux): return 0.5*(z_aux + 1.0)

def _wrap_angles(th):
    return (th + np.pi) % (2*np.pi) - np.pi

# --- Circuits to get EACH fidelity cleanly ---------------------------------
def build_trash_circuit_2q(x, theta, *, encoding="amplitude", ref_angle=None):
    """Prepare |x>, encode, SWAP-test TRASH vs REF (AUX used), stop."""
    qc = QuantumCircuit(TOTAL_QUBITS)
    if ref_angle is not None: qc.ry(ref_angle, REF)
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)
    qc.h(AUX); qc.cswap(AUX, REF, TRASH); qc.h(AUX)
    return qc

def build_recon_circuit_2q(x, theta, *, encoding="amplitude", ref_angle=None):
    """
    Prepare |x> on DATA, encode with U(θ),
    REPLACE TRASH with a clean reference |0> (or chosen angle),
    then decode with U(θ)†,
    prepare fresh |x> on REF_DATA, and SWAP-test DATA vs REF_DATA.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)

    # 1) Optional: set the single-qubit REF that will serve as clean trash
    if ref_angle is not None:
        qc.ry(ref_angle, REF)   # e.g., 0.0 for |0>, π for |1>
    # else REF is already |0>

    # 2) Prepare input on data and encode
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    # 3) Replace TRASH with clean reference
    #    (swap out the encoded trash; the REF now holds the old trash)
    qc.swap(TRASH, REF)

    # 4) Decode using the clean trash + latent
    qc.compose(bind_ansatz(theta).inverse(), DATA_WIRES, inplace=True)

    # 5) Prepare fresh |x> on the reference data register
    build_input_on_two(qc, REF_DATA_WIRES, x, encoding=encoding)

    # 6) Register SWAP test (two CSWAPs share the same ancilla)
    qc.h(AUX)
    qc.cswap(AUX, TRASH, REF_D0)
    qc.cswap(AUX, LATENT, REF_D1)
    qc.h(AUX)
    return qc

# --- Losses ----------------------------------------------------------------
def p0_to_fidelity(p0):  # p0 = (⟨Z⟩+1)/2 from the SWAP test
    return max(0.0, min(1.0, 2.0*p0 - 1.0))

def trash_recon_losses(theta, batch_X, *, encoding="amplitude", ref_angle=None):
    L_t, L_r = [], []
    for x in batch_X:
        # --- Trash fidelity (want high) ---
        sv_t = Statevector.from_instruction(
            build_trash_circuit_2q(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        Ft = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_t, AUX)))

        # --- Reconstruction fidelity (want high) ---
        sv_r = Statevector.from_instruction(
            build_recon_circuit_2q(x, theta, encoding=encoding)  # with the swap-in clean trash fix
        )
        Fr = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_r, AUX)))

        # Losses (minimize both): maximize Ft and Fr
        L_t.append(1.0 - Ft)   # maximize trash fidelity
        L_r.append(1.0 - Fr)   # maximize reconstruction fidelity

    return float(np.mean(L_t)), float(np.mean(L_r))

def combo_loss(theta, batch_X, *, lam=0.5, encoding="amplitude", ref_angle=None):
    Lt, Lr = trash_recon_losses(theta, batch_X, encoding=encoding, ref_angle=ref_angle)
    return lam*Lt + (1.0 - lam)*Lr

# --- Finite-difference gradient + Adam (small & simple) --------------------
def _fd_grad_combo(theta, Xb, *, lam=0.5, encoding="amplitude", ref_angle=None, fd_eps=1e-3):
    theta = _wrap_angles(theta)
    base = combo_loss(theta, Xb, lam=lam, encoding=encoding, ref_angle=ref_angle)
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        th = theta.copy(); th[i] += fd_eps
        g[i] = (combo_loss(th, Xb, lam=lam, encoding=encoding, ref_angle=ref_angle) - base) / fd_eps
    return g, base

def train_qae_2q_combo(
    X_train, *, lam=0.5, epochs=200, batch_size=128, encoding="amplitude", ref_angle=None,
    seed=0, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, fd_eps=1e-3, steps_per_epoch=1
):
    rng = np.random.default_rng(seed)
    theta = 0.01 * rng.standard_normal(ansatz.num_parameters)
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    from tqdm import trange; from time import perf_counter
    steps, L_hist, Lt_hist, Lr_hist, Ft_hist, Fr_hist = [], [], [], [], [], []

    pbar = trange(epochs, desc="2q QAE", leave=True)
    for ep in pbar:
        mb = min(batch_size, len(X_train))
        idx = rng.choice(len(X_train), mb, replace=False)
        Xb = X_train[idx]
        t0 = perf_counter()
        for k in range(steps_per_epoch):
            g, _ = _fd_grad_combo(theta, Xb, lam=lam, encoding=encoding, ref_angle=ref_angle, fd_eps=fd_eps)
            t = ep*max(1,steps_per_epoch) + (k+1)
            m = beta1*m + (1-beta1)*g
            v = beta2*v + (1-beta2)*(g*g)
            m_hat = m/(1-beta1**t); v_hat = v/(1-beta2**t)
            theta = _wrap_angles(theta - lr*m_hat/(np.sqrt(v_hat)+eps))
        # eval on a small random slice
        vidx = rng.choice(len(X_train), min(2000, len(X_train)), replace=False)
        L_t, L_r = trash_recon_losses(theta, X_train[vidx], encoding=encoding, ref_angle=ref_angle)
        L = lam*L_t + (1-lam)*L_r; Ft, Fr = 1-L_t, 1-L_r
        steps.append(ep); L_hist.append(L); Lt_hist.append(L_t); Lr_hist.append(L_r); Ft_hist.append(Ft); Fr_hist.append(Fr)
        pbar.set_postfix({"L": f"{L:.4f}", "Lt": f"{L_t:.4f}", "Lr": f"{L_r:.4f}", "Ft": f"{Ft:.3f}", "Fr": f"{Fr:.3f}"})
    return theta, np.array(steps), np.array(L_hist), np.array(Lt_hist), np.array(Lr_hist), np.array(Ft_hist), np.array(Fr_hist)

def latent_z_after_encoder_2q(x, theta, *, encoding="amplitude"):
    qc = QuantumCircuit(2)  # [trash=0, latent=1]
    if encoding == "amplitude":
        qc.initialize(amp_vec_from_x4(x), [0, 1])
    elif encoding == "angle":
        th0, th1 = x; qc.ry(th0, 0); qc.ry(th1, 1)
    qc.compose(bind_ansatz(theta), [0, 1], inplace=True)
    sv = Statevector.from_instruction(qc)
    return z_expect_on_wire_from_statevector(sv, 1)  # latent

def latent_z_after_decoder_2q(x, theta, *, encoding="amplitude"):
    qc = QuantumCircuit(2)
    if encoding == "amplitude":
        qc.initialize(amp_vec_from_x4(x), [0, 1])
    elif encoding == "angle":
        th0, th1 = x; qc.ry(th0, 0); qc.ry(th1, 1)
    U = bind_ansatz(theta)
    qc.compose(U, [0, 1], inplace=True)
    qc.compose(U.inverse(), [0, 1], inplace=True)
    sv = Statevector.from_instruction(qc)
    return z_expect_on_wire_from_statevector(sv, 1)

def embed_latent_Z_batch_2q(X, theta, *, after_decoder=False, encoding="amplitude"):
    fn = latent_z_after_decoder_2q if after_decoder else latent_z_after_encoder_2q
    z = np.empty((len(X), 1), float)
    for i, x in enumerate(X):
        z[i, 0] = fn(x, theta, encoding=encoding)
    return z

print(len(ansatz.parameters))


LAM = 0.5          # weight: 0=only reconstruction, 1=only trash
EPOCHS = 200
BATCH = 64         # 64–128 is a sweet spot
LR = 0.02
FD_EPS = 1e-3
SEED = 42

theta_opt, steps, L_hist, Lt_hist, Lr_hist, Ft_hist, Fr_hist = train_qae_2q_combo(
    X_qae_train,
    lam=LAM,
    epochs=EPOCHS,
    batch_size=BATCH,
    lr=LR,
    fd_eps=FD_EPS,
    seed=SEED,
    steps_per_epoch=1,     # try 1–3
)

np.save("theta", theta_opt)

plt.figure(figsize=(8, 5))
plt.plot(steps, L_hist, label="Total loss", color='black', linewidth=2)
plt.plot(steps, Lt_hist, label="Trash loss", linestyle="--")
plt.plot(steps, Lr_hist, label="Recon loss", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("QAE Training Losses")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(steps, Ft_hist, label="Trash fidelity", linestyle="--")
plt.plot(steps, Fr_hist, label="Recon fidelity", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Fidelity")
plt.title("QAE Training Fidelities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

Z_lat = embed_latent_Z_batch_2q(X_qae_train, theta_opt, after_decoder=False)
np.save("Z_train", Z_lat)
z_vals = Z_lat.ravel()# shape (N,1)
plt.figure(figsize=(6,4))
plt.hist(z_vals, bins=30, color='skyblue', edgecolor='black')
plt.ylabel('Count', fontsize=12)
plt.title('Latent Qubit Z Expectation Distribution')
plt.grid(alpha=0.3)
plt.show()

Z_test = embed_latent_Z_batch_2q(X_test_m, theta_opt, after_decoder=False )
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


dir = '/Users/danialyntykbay/thesis/AmplitudeEn_Decoder/2qubits/9layers'
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
            out[j] = np.max(qd)
        else:
            out[j] = qd.mean()
    return out

def fit_threshold(theta_train, k, quantile, agg="mean"):

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


k = 9
thr = 2.16876e-11

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
#Classical kNN

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
print(f"Tuning  kNN for target FPR α={alpha:.2%}\n" + "-"*70)
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





k = 3
thr = 2.00314e-06
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

def fidelities_per_row_2q(X_test, theta, *, encoding="amplitude", ref_angle=None):
    """
    Returns two arrays (Ft_per_row, Fr_per_row) with len == len(X_test).
      - Ft_per_row[i]: trash fidelity for sample i (via SWAP test on AUX)
      - Fr_per_row[i]: reconstruction fidelity for sample i (via SWAP test on AUX)
    Uses your build_trash_circuit_2q and build_recon_circuit_2q exactly as written.
    """
    Ft_list, Fr_list = [], []
    for x in X_test:
        # Trash fidelity
        sv_t = Statevector.from_instruction(
            build_trash_circuit_2q(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        Ft = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_t, AUX)))

        # Reconstruction fidelity
        sv_r = Statevector.from_instruction(
            build_recon_circuit_2q(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        Fr = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_r, AUX)))

        Ft_list.append(Ft)
        Fr_list.append(Fr)

    return np.array(Ft_list, dtype=float), np.array(Fr_list, dtype=float)


Ft_rows, Fr_rows = fidelities_per_row_2q(
    X_test_m, theta_opt,
    encoding="amplitude",
    ref_angle=None   # e.g., 0.0 to force |0> as the clean trash/reference if you want
)

prec_n, rec_n, thr_n = precision_recall_curve(
    y_true=X_test_values,
    y_score=Fr_rows,          # <<< IMPORTANT
    pos_label=1
)

pr_auc_a = average_precision_score(X_test_values, Fr_rows)

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
scores = Fr_rows              # fidelity-based anomaly score (higher = more anomalous)
alpha = 0.25                  # target FPR = 5%


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

y_pred = (scores > 0.97).astype(int)
cls_report = classification_report(y_true, y_pred)
print(cls_report)

