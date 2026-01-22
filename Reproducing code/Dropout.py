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

from Experiment.KD99Amplitudeencoding3qubits import Z_test

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
    ("pca", PCA(n_components=8))  # match your # of qubits
])



preproc.fit(X_norm)

# 4) Transform normal-train, normal-val, and the mixed test set
X_qae_train = preproc.transform(X_norm)


X_test_mixed = X_attack
y_test_mixed = remain_df["label"].values
X_qae_test   = preproc.transform(X_test_mixed)
X_test_m = preproc.transform(X_test)


# ============================================================
# 0. CONFIG
# ============================================================

USE_BRAINBOX = True
BATCH_SIZE   = 64
EPOCHS       = 200

lam          = 0.5
p_r          = None    # None = pure reference, full SWAP test
alpha_l2     = 0.0001    # L2 reg strength

layer_dropout = 0.2
gate_dropout  = 0.4


# ============================================================
# 1. QUBIT LAYOUT
# ============================================================

AUX, REF0, TRASH, LAT0, LAT1 = 0, 1, 2, 3, 4
AUX_REC, REF_DATA0, REF_DATA1, REF_DATA2 = 5, 6, 7, 8

DATA_WIRES     = [TRASH, LAT0, LAT1]
REF_DATA_WIRES = [REF_DATA0, REF_DATA1, REF_DATA2]
TOTAL_QUBITS   = 9


# ============================================================
# 2. BRAINBOX ANSATZ (only version, no duplicate)
# ============================================================

from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumCircuit
import numpy as np

# Encoder on 3 data qubits (NO dropout here)
ansatz_enc = RealAmplitudes(3, reps=3, entanglement="linear")

# Brainbox latent block on 2 qubits (dropout applies here)
ansatz_bb  = RealAmplitudes(2, reps=2, entanglement="linear")

N_ENC   = ansatz_enc.num_parameters
N_BB    = ansatz_bb.num_parameters
N_THETA = N_ENC + N_BB


def build_brainbox_ansatz_dropped(theta):
    """
    Brainbox ansatz with entangling dropout applied ONLY to
    the CNOTs inside ansatz_bb, mapped onto local qubits
    [0,1,2] = [TRASH,LAT0,LAT1].
    """

    theta  = np.asarray(theta, float)
    th_enc = theta[:N_ENC]
    th_bb  = theta[N_ENC:]

    # local 3-qubit circuit: [0,1,2] ≡ [TRASH, LAT0, LAT1]
    qc = QuantumCircuit(3)

    # ---------------------------------------------------------
    # Encoder on [0,1,2] (NO dropout)
    # ---------------------------------------------------------
    qc_enc = ansatz_enc.assign_parameters(th_enc)
    qc.compose(qc_enc, [0, 1, 2], inplace=True)

    # ---------------------------------------------------------
    # Brainbox on [LAT0,LAT1] ≡ [1,2] with CNOT dropout
    # ---------------------------------------------------------
    qc_bb = ansatz_bb.assign_parameters(th_bb)

    # qc_bb.data now holds CircuitInstruction objects
    for ci in qc_bb.data:
        inst   = ci.operation
        qargs  = ci.qubits
        cargs  = ci.clbits
        name   = inst.name

        # Map brainbox's 2-qubit register indices {0,1} → local qubits {1,2}
        mapped_qargs = []
        for q in qargs:
            # local index within brainbox circuit
            local_idx = qc_bb.qubits.index(q)   # 0 or 1
            mapped_qargs.append(qc.qubits[1 + local_idx])  # 0→1, 1→2

        if name in ("ry", "rz"):
            # single-qubit rotations: always keep
            qc.append(inst, mapped_qargs, [])
            continue

        if name == "cx":
            # Decide if this CNOT should be dropped (entangling dropout)
            drop_layer = (np.random.rand() < layer_dropout)

            if drop_layer and (np.random.rand() < gate_dropout):
                # DROP this CNOT → do nothing
                continue

            # keep CNOT
            qc.append(inst, mapped_qargs, [])
            continue

        # any other gates (not expected) → keep safely
        qc.append(inst, mapped_qargs, [])

    return qc




def bind_ansatz(theta):
    return build_brainbox_ansatz_dropped(theta)



# ============================================================
# 3. ENCODING UTILITIES
# ============================================================

from qiskit.quantum_info import Statevector

def amp_vec_from_x8(x8):
    v = np.asarray(x8, float)
    n = np.linalg.norm(v)
    if n == 0:
        v = np.array([1]+[0]*7, float)
    else:
        v = v/n
    return v.astype(complex)

def build_input_on_three(qc, wires, x, encoding="amplitude"):
    if encoding == "amplitude":
        qc.initialize(amp_vec_from_x8(x), wires)
    elif encoding == "angle":
        for w, th in zip(wires, x):
            qc.ry(th, w)
    else:
        raise ValueError("encoding must be 'amplitude' or 'angle'")


def build_input_on_data(qc, x, encoding="amplitude"):
    build_input_on_three(qc, DATA_WIRES, x, encoding)


# ============================================================
# 4. FIDELITY UTILITIES
# ============================================================

def z_expect_on_wire_from_statevector(sv, wire):
    amps = np.asarray(sv.data)
    p0 = p1 = 0.0
    for idx, a in enumerate(amps):
        bit = (idx >> wire) & 1
        p = (a.real*a.real + a.imag*a.imag)
        if bit == 0: p0 += p
        else:        p1 += p
    return (p0 - p1)

def p_aux0_from_z(z):
    return 0.5*(1+z)

def p0_to_fidelity(p0):
    return np.clip(2*p0 - 1, 0, 1)

def _wrap_angles(th):
    return (th + np.pi) % (2*np.pi) - np.pi


# ============================================================
# 5. SWAP-TEST CIRCUITS
# ============================================================

def build_trash_circuit_3to2(x, theta, *, encoding="amplitude", ref_angle=None):
    qc = QuantumCircuit(TOTAL_QUBITS)

    if ref_angle is not None:
        qc.ry(ref_angle, REF0)

    build_input_on_data(qc, x, encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    qc.h(AUX)
    qc.cswap(AUX, REF0, TRASH)
    qc.h(AUX)

    return qc


def build_recon_circuit_3to2(x, theta, *, encoding="amplitude", ref_angle=None):

    qc = QuantumCircuit(TOTAL_QUBITS)

    if ref_angle is not None:
        qc.ry(ref_angle, REF0)

    build_input_on_data(qc, x, encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    # bring clean REF0 into TRASH
    qc.swap(TRASH, REF0)

    # decode
    qc.compose(bind_ansatz(theta).inverse(), DATA_WIRES, inplace=True)

    # reference state |x> for swap test
    build_input_on_three(qc, REF_DATA_WIRES, x, encoding)

    qc.h(AUX_REC)
    qc.cswap(AUX_REC, TRASH, REF_DATA0)
    qc.cswap(AUX_REC, LAT0,  REF_DATA1)
    qc.cswap(AUX_REC, LAT1,  REF_DATA2)
    qc.h(AUX_REC)

    return qc


# ============================================================
# 6. LOSS FUNCTIONS (with L2)
# ============================================================

def qae_losses(theta, batch_X, *, encoding="amplitude", p_r=None, ref_angle=None):
    L_t, L_r = [], []

    for x in batch_X:

        # TRASH fidelity
        if p_r is None:
            svt = Statevector.from_instruction(
                build_trash_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
            )
            zt = z_expect_on_wire_from_statevector(svt, AUX)
            Ft = p0_to_fidelity(p_aux0_from_z(zt))
        else:
            raise NotImplementedError("mixed reference fast path omitted here")

        # RECON fidelity
        svr = Statevector.from_instruction(
            build_recon_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        zr = z_expect_on_wire_from_statevector(svr, AUX_REC)
        Fr = p0_to_fidelity(p_aux0_from_z(zr))

        L_t.append(1-Ft)
        L_r.append(1-Fr)

    return float(np.mean(L_t)), float(np.mean(L_r))


def qae_combo_loss(
    theta,
    batch_X,
    *,
    lam=0.5,
    encoding="amplitude",
    p_r=None,
    ref_angle=None,
    alpha_l2=0.0
):
    L_t, L_r = qae_losses(theta, batch_X, encoding=encoding, p_r=p_r, ref_angle=ref_angle)
    L = lam*L_t + (1-lam)*L_r
    if alpha_l2 > 0:
        L += alpha_l2 * float(np.sum(theta**2))
    return L


# ============================================================
# 7. FINITE-DIFFERENCE GRADIENT (dropout ON each evaluation)
# ============================================================

def _fd_grad_combo(
    theta,
    X_batch,
    *,
    lam=0.5,
    encoding="amplitude",
    p_r=None,
    ref_angle=None,
    fd_eps=1e-3,
    alpha_l2=0.0
):
    theta = _wrap_angles(theta)

    base = qae_combo_loss(theta, X_batch, lam=lam,
                          encoding=encoding, p_r=p_r,
                          ref_angle=ref_angle, alpha_l2=alpha_l2)

    g = np.zeros_like(theta)

    for i in range(len(theta)):
        th = theta.copy()
        th[i] += fd_eps

        Lp = qae_combo_loss(th, X_batch, lam=lam,
                            encoding=encoding, p_r=p_r,
                            ref_angle=ref_angle, alpha_l2=alpha_l2)

        g[i] = (Lp - base) / fd_eps

    return g, base

def build_latent_state_circuit(x, theta, encoding="amplitude"):
    """
    Prepare |x> on TRASH,LAT0,LAT1, apply (encoder + brainbox with dropout),
    and return the FULL 9-qubit statevector.
    No decoding. No SWAP test.
    We only care about LAT0 and LAT1.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)

    # prepare input
    build_input_on_data(qc, x, encoding)

    # apply encoder+brainbox (dropout inside bind_ansatz)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    return qc


def get_latent_Z_features(X, theta, encoding="amplitude"):
    """
    Returns array of shape (N, 2):
        [ <Z_LAT0>, <Z_LAT1> ] for each data sample.
    """
    feats = []

    for x in X:
        qc_lat = build_latent_state_circuit(x, theta, encoding)
        sv = Statevector.from_instruction(qc_lat)

        Z0 = z_expect_on_wire_from_statevector(sv, LAT0)
        Z1 = z_expect_on_wire_from_statevector(sv, LAT1)

        feats.append([Z0, Z1])

    return np.array(feats)


# ============================================================
# 8. TRAINER (Brainbox + dropout + L2)
# ============================================================

from tqdm import trange

def train_qae_brainbox_dropout(
    X_train,
    *,
    lam=0.6,
    epochs=80,
    batch_size=64,
    encoding="amplitude",
    p_r=None,
    ref_angle=None,
    seed=0,
    lr=0.01,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    fd_eps=1e-3,
    steps_per_epoch=1,
    alpha_l2=0.0
):
    X_train = np.asarray(X_train, float)
    rng = np.random.default_rng(seed)

    theta = 0.01 * rng.standard_normal(N_THETA)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    steps = []
    Lt_hist = []
    Lr_hist = []
    Ltot_hist = []
    Ft_hist = []
    Fr_hist = []

    pbar = trange(epochs, desc="Brainbox QAE Train", leave=True)

    for ep in pbar:

        idx = rng.choice(
            len(X_train), min(batch_size, len(X_train)), replace=False
        )
        Xb = X_train[idx]

        # Gradient steps
        for _ in range(steps_per_epoch):
            g,_ = _fd_grad_combo(
                theta, Xb,
                lam=lam,
                encoding=encoding,
                p_r=p_r,
                ref_angle=ref_angle,
                fd_eps=fd_eps,
                alpha_l2=alpha_l2
            )

            t = ep+1
            m = beta1*m + (1-beta1)*g
            v = beta2*v + (1-beta2)*(g*g)
            m_hat = m/(1-beta1**t)
            v_hat = v/(1-beta2**t)
            theta = _wrap_angles(theta - lr*m_hat/(np.sqrt(v_hat)+eps))

        # Validation subset
        vidx = rng.choice(len(X_train), min(300,len(X_train)), replace=False)
        Xt = X_train[vidx]

        Lt, Lr = qae_losses(theta, Xt, encoding=encoding, p_r=p_r, ref_angle=ref_angle)
        Ltot = lam*Lt + (1-lam)*Lr + alpha_l2*np.sum(theta**2)

        Ft, Fr = 1-Lt, 1-Lr

        steps.append(ep)
        Lt_hist.append(Lt)
        Lr_hist.append(Lr)
        Ltot_hist.append(Ltot)
        Ft_hist.append(Ft)
        Fr_hist.append(Fr)

        pbar.set_postfix({"L": f"{Ltot:.3f}", "Ft": f"{Ft:.3f}", "Fr": f"{Fr:.3f}"})

    return (
        theta,
        np.array(steps),
        np.array(Ltot_hist),
        np.array(Lt_hist),
        np.array(Lr_hist),
        np.array(Ft_hist),
        np.array(Fr_hist),
    )


theta_best_, steps_, Ltot_hist_, Lt_hist_, Lr_hist_, Ft_hist_, Fr_hist_ = (
    train_qae_brainbox_dropout(
        X_qae_train,
        lam=0.5,
        epochs=200,
        batch_size=64,
        lr=0.001,
        fd_eps=1e-3,
        alpha_l2=0,
        encoding="amplitude",
        p_r=None,
        seed=123
    )
)

plt.figure(figsize=(8, 5))
plt.plot(steps_, Ltot_hist_, label="Total loss", color='black', linewidth=2)
plt.plot(steps_, Lt_hist_, label="Trash loss", linestyle="--")
plt.plot(steps_, Lr_hist_, label="Recon loss", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("QAE Training Losses")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(steps_, Ft_hist_, label="Trash fidelity", linestyle="--")
plt.plot(steps_, Fr_hist_, label="Recon fidelity", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Fidelity")
plt.title("QAE Training Fidelities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

Z_2 = get_latent_Z_features(X_test_m, theta_best_)
Z_train_2 = get_latent_Z_features(X_qae_train, theta_best_)

mixed_Z = np.concatenate((Z_2 ,X_test_values.reshape(-1,1)), axis=1)


normal_values = mixed_Z[mixed_Z[:, 2] == 0][:, 0:2]
attack_values = mixed_Z[mixed_Z[:, 2] == 1][:, 0:2]


plt.figure(figsize=(8,6))

# Scatter the two classes
plt.scatter(normal_values[:,0], normal_values[:,1],
            c='dodgerblue', label='Normal', alpha=0.7, edgecolors='k')
plt.scatter(attack_values[:,0], attack_values[:,1],
            c='crimson', label='Attack', alpha=0.7, edgecolors='k')

# Labels and formatting
plt.xlabel("⟨Z⟩ of LAT0", fontsize=12)
plt.ylabel("⟨Z⟩ of LAT1", fontsize=12)
plt.title("QAE Latent Space (3→2 compression)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




# Fidelity based test

def fidelities_per_row(X_test, theta, *, encoding="amplitude", ref_angle=None):
    """
    Returns two arrays (Ft_per_row, Fr_per_row) for every row in X_test:
      - Ft_per_row[i] = trash fidelity for sample i
      - Fr_per_row[i] = reconstruction fidelity for sample i

    Uses pure-reference SWAP test (p_r is assumed None).
    Dropout is active inside bind_ansatz(theta).
    """
    Ft_list, Fr_list = [], []

    for x in X_test:
        # ----- Trash fidelity via SWAP test -----
        sv_t = Statevector.from_instruction(
            build_trash_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        z_aux = z_expect_on_wire_from_statevector(sv_t, AUX)
        Ft = p0_to_fidelity(p_aux0_from_z(z_aux))

        # ----- Reconstruction fidelity via SWAP test -----
        sv_r = Statevector.from_instruction(
            build_recon_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        z_aux_rec = z_expect_on_wire_from_statevector(sv_r, AUX_REC)
        Fr = p0_to_fidelity(p_aux0_from_z(z_aux_rec))

        Ft_list.append(Ft)
        Fr_list.append(Fr)

    return np.array(Ft_list, dtype=float), np.array(Fr_list, dtype=float)


Ft_rows, Fr_rows = fidelities_per_row(
    X_test_m,
    theta_best_,           # or theta_opt, depending on your variable
    encoding="amplitude",
    ref_angle=None         # or 0.0 if you explicitly RY the reference
)

prec_n, rec_n, thr_n = precision_recall_curve(
    y_true=X_test_values,
    y_score=Fr_rows,          # <<< IMPORTANT
    pos_label=1
)

mask = rec_n > 0
prec = prec_n[mask]
rec  = rec_n[mask]

pr_auc_a = average_precision_score(X_test_values, Fr_rows)

plt.figure(figsize=(7,6))
plt.plot(rec, prec,
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
alpha = 0.60                  # target FPR = 5%


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


y_pred = (scores > (0.8458116232464929)).astype(int)

cls_report = classification_report(y_true, y_pred)
print(cls_report)

#kNN
Z_val, Z_final_test, y_val, y_final_test = train_test_split(
    Z_2, X_test_values,
    test_size=0.8,          # keep 70% for final test, 30% for validation
    stratify=X_test_values,        # preserve normal/attack ratio
    random_state=42
)

alpha = 0.15  # target FPR on normals (1%)
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
    val_scores = knn_scores(Z_train_2, Z_val, k=k, leave_one_out=False)

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
    test_scores = knn_scores(Z_train_2, Z_final_test, k=k, leave_one_out=False)
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
thr = 0.00109304
nn = NearestNeighbors(n_neighbors=k,n_jobs=-1).fit(Z_train_2)
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

mask = rec_n > 0
prec = prec_n[mask]
rec  = rec_n[mask]

plt.figure(figsize=(7,6))
plt.plot(rec, prec,
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