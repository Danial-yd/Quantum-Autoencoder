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
    ("pca", PCA(n_components=8))  # match your # of qubits
])



preproc.fit(X_norm)

# 4) Transform normal-train, normal-val, and the mixed test set
X_qae_train = preproc.transform(X_norm)


X_test_mixed = X_attack
y_test_mixed = remain_df["label"].values
X_qae_test   = preproc.transform(X_test_mixed)
X_test_m = preproc.transform(X_test)


AUX, REF0, TRASH, LAT0, LAT1 = 0, 1, 2, 3, 4
AUX_REC, REF_DATA0, REF_DATA1, REF_DATA2 = 5, 6, 7, 8

DATA_WIRES     = [TRASH, LAT0, LAT1]         # 3 data qubits total
REF_DATA_WIRES = [REF_DATA0, REF_DATA1, REF_DATA2]
TOTAL_QUBITS   = 9

# -----------------
# Ansatz
# -----------------
ansatz = RealAmplitudes(3, reps=6, entanglement="linear")  # circular is often more expressive than linear

def bind_ansatz(theta):
    return ansatz.bind_parameters(theta) if hasattr(ansatz, "bind_parameters") else ansatz.assign_parameters(theta)

# -----------------
# Encoding helpers
# -----------------
def amp_vec_from_x8(x8):
    v = np.asarray(x8, dtype=np.float64).copy()
    n = np.linalg.norm(v)
    if n == 0.0: v = np.array([1.0] + [0.0]*7, dtype=np.float64)
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

# -----------------
# Utilities
# -----------------
def z_expect_on_wire_from_statevector(sv, wire_index):
    amps = np.asarray(sv.data); p0 = p1 = 0.0
    for idx, a in enumerate(amps):
        bit = (idx >> wire_index) & 1
        p = (a.real*a.real + a.imag*a.imag)
        if bit == 0: p0 += p
        else:        p1 += p
    return p0 - p1

def p_aux0_from_z(z_aux): return 0.5*(z_aux + 1.0)

def p0_to_fidelity(p0):
    # SWAP-test fidelity from ancilla-0 prob
    return max(0.0, min(1.0, 2.0*p0 - 1.0))

def _wrap_angles(th):
    return (th + np.pi) % (2*np.pi) - np.pi

# -----------------
# Fast mixed-reference Ft (1 trash qubit)
# Ft = p_r * P(trash=0) + (1-p_r)/2
# -----------------
def P_trash0_after_encode(x8, theta):
    # 3-qubit statevector: map [TRASH, LAT0, LAT1] → [0,1,2]
    qc = QuantumCircuit(3)
    qc.initialize(amp_vec_from_x8(x8), [0,1,2])
    qc.compose(bind_ansatz(theta), [0,1,2], inplace=True)
    sv = Statevector.from_instruction(qc).data  # length 8
    # trash bit (qubit 0 in this local circuit) == 0 → indices {0,2,4,6}
    return float(np.sum(np.abs(sv[[0,2,4,6]])**2).real)

def Ft_mixed_1trash_fast(x8, theta, p_r):
    return float(p_r * P_trash0_after_encode(x8, theta) + (1.0 - p_r)/2.0)

# -----------------
# Circuits
# -----------------
def build_trash_circuit_3to2(x, theta, *, encoding="amplitude", ref_angle=None):
    """
    Pure-reference SWAP test: compare TRASH vs REF0 using AUX.
    Use this only when p_r is None (pure |0> reference).
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    if ref_angle is not None:
        qc.ry(ref_angle, REF0)   # else default |0>
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)
    qc.h(AUX); qc.cswap(AUX, REF0, TRASH); qc.h(AUX)
    return qc

def build_recon_circuit_3to2(x, theta, *, encoding="amplitude", ref_angle=None):
    """
    Prepare |x>, encode U(θ), swap in clean trash from REF0, decode U(θ)†.
    Then SWAP-test DATA vs fresh reference |x> using AUX_REC (3 cswaps).
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    if ref_angle is not None:
        qc.ry(ref_angle, REF0)    # default |0>
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    # Replace TRASH by a clean reference, then decode
    qc.swap(TRASH, REF0)
    qc.compose(bind_ansatz(theta).inverse(), DATA_WIRES, inplace=True)

    # Fresh |x> on reference data and SWAP-test
    build_input_on_three(qc, REF_DATA_WIRES, x, encoding=encoding)
    qc.h(AUX_REC)
    qc.cswap(AUX_REC, TRASH, REF_DATA0)
    qc.cswap(AUX_REC, LAT0,  REF_DATA1)
    qc.cswap(AUX_REC, LAT1,  REF_DATA2)
    qc.h(AUX_REC)
    return qc

# -----------------
# Losses
# -----------------
def qae_losses(theta, batch_X, *, encoding="amplitude", p_r=None, ref_angle=None):
    """
    Returns (L_trash, L_recon) where L = 1 - F (so minimizing L maximizes fidelity).
    If p_r is not None, uses the fast mixed-reference Ft; else uses pure-reference SWAP test.
    """
    L_trash, L_recon = [], []
    for x in batch_X:
        # Trash fidelity
        if p_r is not None:
            Ft = Ft_mixed_1trash_fast(x, theta, p_r)   # mixed ref (fast)
        else:
            sv_t = Statevector.from_instruction(
                build_trash_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
            )
            Ft = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_t, AUX)))

        # Reconstruction fidelity via SWAP test
        sv_r = Statevector.from_instruction(
            build_recon_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        Fr = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_r, AUX_REC)))

        L_trash.append(1.0 - Ft)
        L_recon.append(1.0 - Fr)

    return float(np.mean(L_trash)), float(np.mean(L_recon))

def qae_combo_loss(theta, batch_X, *, lam=0.5, encoding="amplitude", p_r=None, ref_angle=None):
    L_t, L_r = qae_losses(theta, batch_X, encoding=encoding, p_r=p_r, ref_angle=ref_angle)
    return lam * L_t + (1.0 - lam) * L_r

# -----------------
# Finite-difference + Adam
# -----------------
def _fd_grad_combo(theta, X_batch, *, lam=0.5, encoding="amplitude", p_r=None, ref_angle=None, fd_eps=1e-3):
    theta = _wrap_angles(theta)
    base = qae_combo_loss(theta, X_batch, lam=lam, encoding=encoding, p_r=p_r, ref_angle=ref_angle)
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        th = theta.copy(); th[i] += fd_eps
        g[i] = (qae_combo_loss(th, X_batch, lam=lam, encoding=encoding, p_r=p_r, ref_angle=ref_angle) - base) / fd_eps
    return g, base

def train_qae_combo_with_history(
    X_train, *,
    lam=0.6,                  # start compression-heavy; later you can anneal to 0.5
    epochs=100,
    batch_size=64,
    encoding="amplitude",
    p_r=None,                 # set e.g. 0.8 for mixed ref; None for pure |0>
    ref_angle=None,           # e.g., 0.0 for |0> (pure case)
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
    m = np.zeros_like(theta); v = np.zeros_like(theta)

    steps, loss_total_hist, loss_trash_hist, loss_recon_hist, fid_trash_hist, fid_recon_hist = [], [], [], [], [], []
    first_epoch_time = None

    pbar = trange(epochs, desc="QAE 3→2", leave=True)
    for ep in pbar:
        mb = min(batch_size, len(X_train))
        Xb = X_train[rng.choice(len(X_train), mb, replace=False)]

        t0 = perf_counter()
        for k in range(steps_per_epoch):
            g, _ = _fd_grad_combo(theta, Xb, lam=lam, encoding=encoding, p_r=p_r, ref_angle=ref_angle, fd_eps=fd_eps)
            t = ep * max(1, steps_per_epoch) + (k+1)
            m = beta1*m + (1-beta1)*g
            v = beta2*v + (1-beta2)*(g*g)
            m_hat = m/(1-beta1**t); v_hat = v/(1-beta2**t)
            theta = _wrap_angles(theta - lr * m_hat / (np.sqrt(v_hat) + eps))
        t1 = perf_counter()
        if ep == 0: first_epoch_time = t1 - t0

        # validation
        vidx = rng.choice(len(X_train), min(300, len(X_train)), replace=False)
        L_t, L_r = qae_losses(theta, X_train[vidx], encoding=encoding, p_r=p_r, ref_angle=ref_angle)
        L_total = lam*L_t + (1-lam)*L_r
        F_t, F_r = 1-L_t, 1-L_r

        steps.append(ep)
        loss_total_hist.append(L_total); loss_trash_hist.append(L_t); loss_recon_hist.append(L_r)
        fid_trash_hist.append(F_t); fid_recon_hist.append(F_r)

        pbar.set_postfix({"L": f"{L_total:.4f}", "Lt": f"{L_t:.4f}", "Lr": f"{L_r:.4f}",
                          "Ft": f"{F_t:.3f}", "Fr": f"{F_r:.3f}", "t/ep(s)": f"{(t1-t0):.2f}"})

    print(f"\nFirst epoch time: {first_epoch_time:.3f} s")
    print(f"Final total loss: {loss_total_hist[-1]:.6f}")
    print(f"Final Ft: {fid_trash_hist[-1]:.6f} | Fr: {fid_recon_hist[-1]:.6f}")

    return (
        theta,
        np.array(steps),
        np.array(loss_total_hist),
        np.array(loss_trash_hist),
        np.array(loss_recon_hist),
        np.array(fid_trash_hist),
        np.array(fid_recon_hist),
    )

# -----------------
# Optional diagnostics: latent Z (not used in loss)
# -----------------
def latent_z_after_encoder_3q(x8, theta):
    qc = QuantumCircuit(3)
    qc.initialize(amp_vec_from_x8(x8), [0,1,2])      # [TRASH, LAT0, LAT1] locally
    qc.compose(bind_ansatz(theta), [0,1,2], inplace=True)
    sv = Statevector.from_instruction(qc)
    return z_expect_on_wire_from_statevector(sv, 1)  # pick LAT0 (local index 1)

def latent_z_after_decoder_3q(x8, theta):
    qc = QuantumCircuit(3)
    qc.initialize(amp_vec_from_x8(x8), [0,1,2])
    qc.compose(bind_ansatz(theta), [0,1,2], inplace=True)
    qc.compose(bind_ansatz(theta).inverse(), [0,1,2], inplace=True)  # identity diagnostic
    sv = Statevector.from_instruction(qc)
    return z_expect_on_wire_from_statevector(sv, 1)

def embed_latent_Z_batch_3q(X8, theta, after_decoder=False):
    z = np.empty((len(X8), 1), float)
    fn = latent_z_after_decoder_3q if after_decoder else latent_z_after_encoder_3q
    for i, x in enumerate(X8):
        z[i, 0] = fn(x, theta)
    return z

def latent_z_vector_after_encoder_3to2(x8, theta):
    """
    Run encoder U(θ) on input |x>, return Z expectation values on the 2 latent qubits.
    Returns np.array([<Z_L0>, <Z_L1>]).
    """
    qc = QuantumCircuit(3)
    qc.initialize(amp_vec_from_x8(x8), [0,1,2])    # local wires: [TRASH, LAT0, LAT1]
    qc.compose(bind_ansatz(theta), [0,1,2], inplace=True)
    sv = Statevector.from_instruction(qc)

    z_lat0 = z_expect_on_wire_from_statevector(sv, 1)  # local qubit 1 = LAT0
    z_lat1 = z_expect_on_wire_from_statevector(sv, 2)  # local qubit 2 = LAT1
    return np.array([z_lat0, z_lat1])

def embed_latent_Z_batch_3to2(X8, theta):
    """
    Compute latent Z vectors for all samples in X8.
    Returns array of shape (N, 2) where N = len(X8).
    """
    Z_lat = np.empty((len(X8), 2), float)
    for i, x in enumerate(X8):
        Z_lat[i, :] = latent_z_vector_after_encoder_3to2(x, theta)
    return Z_lat

print(len(ansatz.parameters))

# -----------------
# Minimal usage example
# -----------------
# Prepare some toy data (8-amplitude vectors). Replace with your real X_qae_train.
def random_amp_vectors(n, rng=np.random.default_rng(0)):
    X = rng.standard_normal((n, 8))
    # (normalization done inside amp_vec_from_x8)
    return X


LAM = 0.6
EPOCHS = 200
BATCH = 64
SEED = 42

theta_opt, steps, L_tot, L_tr, L_rec, F_tr, F_rec = train_qae_combo_with_history(
    X_qae_train,
    lam=LAM,
    epochs=EPOCHS,
    batch_size=BATCH,
    p_r=0.8,          # mixed reference; try 1.0, 0.8, 0.6
    ref_angle=None,   # or 0.0 if you run pure ref (p_r=None)
    seed=SEED,
    steps_per_epoch=1,
    fd_eps=1e-3,
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

plt.figure(figsize=(8, 5))
plt.plot(steps, F_tr, label="Trash fidelity", linestyle="--")
plt.plot(steps, F_rec, label="Recon fidelity", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Fidelity")
plt.title("QAE Training Fidelities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



Z_train = embed_latent_Z_batch_3to2(X_qae_train, theta_opt)
np.save("Z_train", Z_train)

Z_test = embed_latent_Z_batch_3to2(X_test_m, theta_opt)
np.save("Z_test", Z_test)

mixed_Z = np.concatenate((Z_test,X_test_values.reshape(-1,1)), axis=1)


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



mixed = np.concatenate((X_test_m[:,0:2],X_test_values.reshape(-1,1)), axis=1)

normal_values = mixed[mixed[:, 2] == 0][:, 0:2]
attack_values = mixed[mixed[:, 2] == 1][:, 0:2]

plt.figure(figsize=(6, 5))
plt.scatter(normal_values[:, 0], normal_values[:, 1], s=10, alpha=0.7, label="Normal")
plt.scatter(attack_values[:, 0], attack_values[:, 1], s=10, alpha=0.7, label="Attack")

plt.title("PCA Projection (X_qae_test)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.show()

dir = '/Users/danialyntykbay/thesis/AmplitudeEn_Decoder/3qubits/2latent/3layer'
os.chdir(dir)
os.getcwd()
theta_opt = np.load("theta.npy")
Z_train = np.load("Z_train.npy")
Z_test_m = np.load("Z_test.npy")

ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
ocsvm.fit(Z_train)

scores = ocsvm.decision_function(Z_test_m)
y_pred = ocsvm.predict(Z_test_m)
pred   = ocsvm.predict(Z_test_m)
pred01 = (pred == -1).astype(int)

auc  = roc_auc_score(X_test_values, -scores)           # negate so higher = more anomalous
aupr = average_precision_score(X_test_values, -scores)
f1   = f1_score(X_test_values, pred01)

precision = precision_score(X_test_values, pred01)
recall    = recall_score(X_test_values, pred01)
f1        = f1_score(X_test_values, pred01)
cm        = confusion_matrix(X_test_values, pred01)

print(classification_report(X_test_values, pred01))

def angles_from_yz(YZ):
    YZ = np.asarray(YZ, dtype=np.float32)
    r = np.linalg.norm(YZ, axis=1, keepdims=True)
    r = np.maximum(r, 1e-9)
    YZu = YZ / r
    Y, Z = YZu[:, 0], YZu[:, 1]
    a = np.arctan2(Y, Z).astype(np.float32)   # (-π, π]
    a[a < 0] += 2.0 * np.pi                   # [0, 2π)
    return a

def qdist_from_delta(delta):
    delta = delta.astype(np.float32, copy=False)
    return np.sin(0.5 * delta)**2

def _circ_abs_diff(a, b):
    """Smallest absolute angular difference on [0, 2π)."""
    return np.abs(((a - b + np.pi) % (2*np.pi)) - np.pi)

def qknn_scores_circle(alpha_train, alpha_query, k=5, agg="mean", leave_one_out=False):
    T = np.asarray(alpha_train, dtype=np.float32).ravel()
    Q = np.asarray(alpha_query, dtype=np.float32).ravel()

    order = np.argsort(T)
    Ts = T[order]
    n = Ts.size
    if n == 0:
        return np.zeros_like(Q, dtype=np.float32)

    # guard k (subtract one if we plan to drop exact self match)
    k_max = max(1, n - (1 if leave_one_out else 0))
    k = max(1, min(int(k), k_max))

    # Duplicate once and index in the middle copy so we can take a centered window
    Ts2 = np.concatenate([Ts, Ts + 2*np.pi, Ts + 4*np.pi])   # length 3n
    out = np.empty(Q.shape[0], dtype=np.float32)

    for j, q in enumerate(Q):
        # map q into the middle band so neighbors exist on both sides
        q_mid = q + 2*np.pi
        i = bisect_left(Ts2, q_mid)   # position in the 3n array

        # take a symmetric 2k window around i to ensure >= k candidates
        left  = max(0, i - k)
        right = min(Ts2.size, i + k)
        cand = Ts2[left:right]

        deltas = _circ_abs_diff(cand, q_mid)

        if leave_one_out:
            # drop exact zeros (same angle); tolerance for float
            deltas = deltas[deltas > 1e-12]

        if deltas.size == 0:
            out[j] = 0.0
            continue

        # keep k smallest
        if deltas.size > k:
            deltas = np.partition(deltas, k-1)[:k]

        qd = qdist_from_delta(deltas)
        if   agg == "median": out[j] = np.median(qd)
        elif agg == "kth":    out[j] = np.max(qd)  # kth neighbor
        else:                 out[j] = qd.mean()
    return out

A_tr = angles_from_yz(Z_train)     # shape (n,)
A_te = angles_from_yz(Z_test_m)

k = 5
train_scores = qknn_scores_circle(A_tr, A_tr, k=k, agg="mean", leave_one_out=True)
thr = float(np.quantile(train_scores, 0.4))

scores_test = qknn_scores_circle(A_tr, A_te, k=k, agg="mean", leave_one_out=False)
y_pred = (scores_test > thr).astype(np.int8)

print(classification_report(X_test_values, y_pred, digits=4))
f1 = f1_score(X_test_values, y_pred)

fpr, tpr, _ = roc_curve(X_test_values, int(scores_test))
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


# Classical kNN
dir = '/Users/danialyntykbay/thesis/AmplitudeEn_Decoder/3qubits/2latent/6layer'
os.chdir(dir)
os.getcwd()
theta_opt = np.load("theta.npy")
Z_train = np.load("Z_train.npy")
Z_test_m = np.load("Z_test.npy")

Z_val, Z_final_test, y_val, y_final_test = train_test_split(
    Z_test_m, X_test_values,
    test_size=0.8,          # keep 70% for final test, 30% for validation
    stratify=X_test_values,        # preserve normal/attack ratio
    random_state=42
)

alpha = 0.1  # target FPR on normals (1%)
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





k = 25
thr = 0.00393317
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
plt.plot(rec_n , prec_n,
         label=f"PR (AUC = {pr_auc_a:.3f}, FPR ≤ 0.1)",
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


def fidelities_per_row(X_test, theta, *, encoding="amplitude", p_r=None, ref_angle=None):
    """
    Returns two arrays (Ft_per_row, Fr_per_row) with the same length as X_test.
      - Ft_per_row[i] = trash fidelity for sample i
      - Fr_per_row[i] = reconstruction fidelity for sample i
    If p_r is not None, uses the mixed-reference fast path for Ft (your Ft_mixed_1trash_fast).
    Otherwise, uses the pure-reference SWAP test circuit.
    """
    Ft_list, Fr_list = [], []
    for x in X_test:
        # Trash fidelity
        if p_r is not None:
            Ft = Ft_mixed_1trash_fast(x, theta, p_r)
        else:
            sv_t = Statevector.from_instruction(
                build_trash_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
            )
            Ft = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_t, AUX)))

        # Reconstruction fidelity
        sv_r = Statevector.from_instruction(
            build_recon_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        Fr = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_r, AUX_REC)))

        Ft_list.append(Ft)
        Fr_list.append(Fr)

    return np.array(Ft_list, dtype=float), np.array(Fr_list, dtype=float)



Ft_rows, Fr_rows = fidelities_per_row(
    X_test_m, theta_opt,
    encoding="amplitude",
    p_r=None,        # or None if you're using pure |0> reference
    ref_angle=None  # e.g., 0.0 if p_r is None and you want explicit |0>
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

y_pred = (scores > 0.567).astype(int)
cls_report = classification_report(y_true, y_pred)
print(cls_report)

