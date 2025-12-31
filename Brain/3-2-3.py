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
    ("pca", PCA(n_components=16))  # match your # of qubits
])




preproc.fit(X_norm)

# 4) Transform normal-train, normal-val, and the mixed test set
X_qae_train = preproc.transform(X_norm)


X_test_mixed = X_attack
y_test_mixed = remain_df["label"].values
X_qae_test   = preproc.transform(X_test_mixed)
X_test_m = preproc.transform(X_test)

# ============================================================
# 0. Config
# ============================================================

USE_BRAINBOX = True   # toggle baseline vs 3-2-3 brainbox

lam = 0.5             # weight between trash + reconstruction losses
p_r = None            # if not None, use fast mixed-reference trash fidelity

# ============================================================
# 1. Qubit layout (4 data qubits)
# ============================================================

# Global indices
AUX, REF0, TRASH, LAT0, LAT1, LAT2 = 0, 1, 2, 3, 4, 5
AUX_REC, REF_DATA0, REF_DATA1, REF_DATA2, REF_DATA3 = 6, 7, 8, 9, 10

DATA_WIRES     = [TRASH, LAT0, LAT1, LAT2]              # 4 data qubits total
REF_DATA_WIRES = [REF_DATA0, REF_DATA1, REF_DATA2, REF_DATA3]
TOTAL_QUBITS   = 11

# ============================================================
# 2. Ansatz / brainbox (4 data qubits, 3–2–3 on latents)
# ============================================================

if not USE_BRAINBOX:
    # --- Baseline: single RealAmplitudes on 4 data qubits ---
    ansatz = RealAmplitudes(4, reps=4, entanglement="linear")
    N_THETA = ansatz.num_parameters

    def bind_ansatz(theta):
        theta = np.asarray(theta, dtype=float)
        return ansatz.assign_parameters(theta)

else:
    # --- Brainbox version: encoder(4q) + 3–2–3 BB on (LAT0,LAT1,LAT2) ---
    #
    # Local layout for ansatz: [0,1,2,3] = [TRASH, LAT0, LAT1, LAT2]
    #
    #   U_total(theta) =
    #       U_enc   (4q) on [0,1,2,3]
    #     ∘ U_bb3_a(3q) on [1,2,3]      (first "3")
    #     ∘ U_bb2   (2q) on [1,2]       (the "2", approx 3→2)
    #     ∘ U_bb3_b(3q) on [1,2,3]      (final "3")
    #
    # LAT0,LAT1,LAT2 form the 3-qubit latent subsystem.

    ansatz_enc = RealAmplitudes(4, reps=3, entanglement="linear")
    bb3_a      = RealAmplitudes(3, reps=1, entanglement="linear")
    bb2        = RealAmplitudes(2, reps=1, entanglement="linear")
    bb3_b      = RealAmplitudes(3, reps=1, entanglement="linear")

    N_ENC   = ansatz_enc.num_parameters
    N_BB3_A = bb3_a.num_parameters
    N_BB2   = bb2.num_parameters
    N_BB3_B = bb3_b.num_parameters

    N_THETA = N_ENC + N_BB3_A + N_BB2 + N_BB3_B

    def build_brainbox_ansatz(theta):
        """
        4-qubit local circuit on [0,1,2,3] = [TRASH,LAT0,LAT1,LAT2].

        U_total(theta) =
            encoder(4q) on [0,1,2,3]
          ∘ bb3_a(3q)  on [1,2,3]
          ∘ bb2(2q)    on [1,2]
          ∘ bb3_b(3q)  on [1,2,3]
        """
        theta = np.asarray(theta, dtype=float)
        assert len(theta) == N_THETA, f"Expected {N_THETA} params, got {len(theta)}"

        theta_enc   = theta[:N_ENC]
        theta_bb3_a = theta[N_ENC:N_ENC + N_BB3_A]
        theta_bb2   = theta[N_ENC + N_BB3_A : N_ENC + N_BB3_A + N_BB2]
        theta_bb3_b = theta[N_ENC + N_BB3_A + N_BB2:]

        qc = QuantumCircuit(4)

        # Encoder on [0,1,2,3] = [TRASH,LAT0,LAT1,LAT2]
        enc_bound = ansatz_enc.assign_parameters(theta_enc)
        qc.compose(enc_bound, [0,1,2,3], inplace=True)

        # First 3q BB on (LAT0,LAT1,LAT2) = (1,2,3)
        bb3a_bound = bb3_a.assign_parameters(theta_bb3_a)
        qc.compose(bb3a_bound, [1,2,3], inplace=True)

        # 2q BB on (LAT0,LAT1) = (1,2)
        bb2_bound = bb2.assign_parameters(theta_bb2)
        qc.compose(bb2_bound, [1,2], inplace=True)

        # Final 3q BB on (LAT0,LAT1,LAT2) = (1,2,3)
        bb3b_bound = bb3_b.assign_parameters(theta_bb3_b)
        qc.compose(bb3b_bound, [1,2,3], inplace=True)

        return qc

    def bind_ansatz(theta):
        return build_brainbox_ansatz(theta)

# ============================================================
# 3. Encoding helpers
# ============================================================

def amp_vec_from_x16(x16):
    v = np.asarray(x16, dtype=np.float64).copy()
    n = np.linalg.norm(v)
    if n == 0.0:
        v = np.zeros(16, dtype=np.float64)
        v[0] = 1.0
    else:
        v = v / n
    return v.astype(np.complex128)

def build_input_on_four(qc, wires, x, encoding="amplitude"):
    if encoding == "amplitude":
        qc.initialize(amp_vec_from_x16(x), wires)
    elif encoding == "angle":
        th0, th1, th2, th3 = x
        for w, th in zip(wires, (th0, th1, th2, th3)):
            qc.ry(th, w)
    else:
        raise ValueError("encoding must be 'amplitude' or 'angle'")

def build_input_on_data(qc, x, encoding="amplitude"):
    build_input_on_four(qc, DATA_WIRES, x, encoding=encoding)

# ============================================================
# 4. Utilities
# ============================================================

def z_expect_on_wire_from_statevector(sv, wire_index):
    # sv can be a Statevector or just an object with .data
    amps = np.asarray(sv.data)
    p0 = 0.0
    p1 = 0.0
    for idx, a in enumerate(amps):
        bit = (idx >> wire_index) & 1
        p = (a.real*a.real + a.imag*a.imag)
        if bit == 0:
            p0 += p
        else:
            p1 += p
    return p0 - p1

def p_aux0_from_z(z_aux):
    return 0.5*(z_aux + 1.0)

def p0_to_fidelity(p0):
    # SWAP-test fidelity from ancilla-0 prob
    return max(0.0, min(1.0, 2.0*p0 - 1.0))

def _wrap_angles(th):
    return (th + np.pi) % (2*np.pi) - np.pi

# ============================================================
# 5. Trash fidelity (fast mixed-reference version)
# ============================================================

def P_trash0_after_encode(x16, theta):
    """
    4-qubit local circuit with layout [0,1,2,3] = [TRASH,LAT0,LAT1,LAT2].
    """
    qc = QuantumCircuit(4)
    qc.initialize(amp_vec_from_x16(x16), [0,1,2,3])
    qc.compose(bind_ansatz(theta), [0,1,2,3], inplace=True)
    sv = Statevector.from_instruction(qc)

    # Probability TRASH (local qubit 0) = 0:
    z_trash = z_expect_on_wire_from_statevector(sv, 0)
    p0_trash = p_aux0_from_z(z_trash)
    return float(p0_trash)

def Ft_mixed_1trash_fast(x16, theta, p_r):
    return float(p_r * P_trash0_after_encode(x16, theta) + (1.0 - p_r)/2.0)

# ============================================================
# 6. Circuits for losses (4→3 compression)
# ============================================================

def build_trash_circuit_4to3(x, theta, *, encoding="amplitude", ref_angle=None):
    """
    Pure-reference SWAP test: compare TRASH vs REF0 using AUX.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    if ref_angle is not None:
        qc.ry(ref_angle, REF0)   # else default |0>

    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    qc.h(AUX)
    qc.cswap(AUX, REF0, TRASH)
    qc.h(AUX)
    return qc

def build_recon_circuit_4to3(x, theta, *, encoding="amplitude", ref_angle=None):
    """
    Prepare |x>, encode U(θ), swap in clean trash from REF0, decode U(θ)†.
    Then SWAP-test DATA vs fresh reference |x> using AUX_REC (4 cswaps).
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    if ref_angle is not None:
        qc.ry(ref_angle, REF0)    # default |0>
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    # Replace TRASH by a clean reference, then decode
    qc.swap(TRASH, REF0)
    qc.compose(bind_ansatz(theta).inverse(), DATA_WIRES, inplace=True)

    # Fresh |x> on reference data and SWAP-test (4 qubits)
    build_input_on_four(qc, REF_DATA_WIRES, x, encoding=encoding)

    qc.h(AUX_REC)
    qc.cswap(AUX_REC, TRASH, REF_DATA0)
    qc.cswap(AUX_REC, LAT0,  REF_DATA1)
    qc.cswap(AUX_REC, LAT1,  REF_DATA2)
    qc.cswap(AUX_REC, LAT2,  REF_DATA3)
    qc.h(AUX_REC)
    return qc

# ============================================================
# 7. Losses
# ============================================================

def qae_losses(theta, batch_X, *, encoding="amplitude", p_r=None, ref_angle=None):
    """
    Returns (L_trash, L_recon) where L = 1 - F.
    """
    L_trash, L_recon = [], []
    for x in batch_X:
        if p_r is not None:
            Ft = Ft_mixed_1trash_fast(x, theta, p_r)
        else:
            sv_t = Statevector.from_instruction(
                build_trash_circuit_4to3(x, theta, encoding=encoding, ref_angle=ref_angle)
            )
            z_aux = z_expect_on_wire_from_statevector(sv_t, AUX)
            Ft = p0_to_fidelity(p_aux0_from_z(z_aux))

        sv_r = Statevector.from_instruction(
            build_recon_circuit_4to3(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        z_aux_rec = z_expect_on_wire_from_statevector(sv_r, AUX_REC)
        Fr = p0_to_fidelity(p_aux0_from_z(z_aux_rec))

        L_trash.append(1.0 - Ft)
        L_recon.append(1.0 - Fr)

    return float(np.mean(L_trash)), float(np.mean(L_recon))

def qae_combo_loss(theta, batch_X, *, lam=0.5, encoding="amplitude", p_r=None, ref_angle=None):
    L_t, L_r = qae_losses(theta, batch_X, encoding=encoding, p_r=p_r, ref_angle=ref_angle)
    return lam * L_t + (1.0 - lam) * L_r

# ============================================================
# 8. Finite-difference + training
# ============================================================

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
    lam=0.6,
    epochs=50,
    batch_size=64,
    encoding="amplitude",
    p_r=None,
    ref_angle=None,
    seed=0,
    lr=0.01,
    fd_eps=1e-3,
    steps_per_epoch=1
):
    """
    Trains the QAE combo loss and records loss/fidelity history.
    """

    X_train = np.asarray(X_train, dtype=float)
    n_samples = X_train.shape[0]

    rng = np.random.default_rng(seed)
    theta = 0.01 * rng.standard_normal(N_THETA)

    steps = []
    loss_total_hist = []
    loss_trash_hist = []
    loss_recon_hist = []
    fid_trash_hist = []
    fid_recon_hist = []

    first_epoch_time = None

    pbar = trange(
        epochs,
        desc="QAE 4→3 (3-2-3 BB)" if USE_BRAINBOX else "QAE 4→3 baseline",
        leave=True
    )
    for ep in pbar:
        mb = min(batch_size, n_samples)
        idx_mb = rng.choice(n_samples, mb, replace=False)
        Xb = X_train[idx_mb]

        t0 = perf_counter()
        for k in range(max(1, steps_per_epoch)):
            g, _ = _fd_grad_combo(
                theta,
                Xb,
                lam=lam,
                encoding=encoding,
                p_r=p_r,
                ref_angle=ref_angle,
                fd_eps=fd_eps,
            )
            # simple SGD step (you can plug Adam if you like)
            theta = _wrap_angles(theta - lr * g)

        t1 = perf_counter()
        if ep == 0:
            first_epoch_time = t1 - t0

        # validation on random subset of train
        v_mb = min(300, n_samples)
        vidx = rng.choice(n_samples, v_mb, replace=False)
        Xv = X_train[vidx]

        L_t, L_r = qae_losses(
            theta,
            Xv,
            encoding=encoding,
            p_r=p_r,
            ref_angle=ref_angle,
        )
        L_total = lam * L_t + (1 - lam) * L_r
        F_t, F_r = 1 - L_t, 1 - L_r

        steps.append(ep)
        loss_total_hist.append(L_total)
        loss_trash_hist.append(L_t)
        loss_recon_hist.append(L_r)
        fid_trash_hist.append(F_t)
        fid_recon_hist.append(F_r)

        pbar.set_postfix({
            "L":   f"{L_total:.4f}",
            "Lt":  f"{L_t:.4f}",
            "Lr":  f"{L_r:.4f}",
            "Ft":  f"{F_t:.3f}",
            "Fr":  f"{F_r:.3f}",
            "t/ep(s)": f"{(t1 - t0):.2f}",
        })

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

# ============================================================
# 9. Latent extraction (from LAT0,LAT1,LAT2 after full U_total)
# ============================================================

def build_latent_state_circuit(x, theta, encoding="amplitude"):
    qc = QuantumCircuit(TOTAL_QUBITS)
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)
    return qc

def get_latent_Z_features(X, theta, encoding="amplitude"):
    feats = []
    for x in X:
        qc_lat = build_latent_state_circuit(x, theta, encoding=encoding)
        sv = Statevector.from_instruction(qc_lat)
        Z0 = z_expect_on_wire_from_statevector(sv, LAT0)
        Z1 = z_expect_on_wire_from_statevector(sv, LAT1)
        Z2 = z_expect_on_wire_from_statevector(sv, LAT2)
        feats.append([Z0, Z1, Z2])
    return np.array(feats)

def get_bb_bottleneck_latent_Z(X, theta, encoding="amplitude"):
    """
    True 2-qubit bottleneck latent for 3-2-3 BB:
    - Apply encoder(4q) + bb3_a(3q) + bb2(2q)
    - STOP before bb3_b
    - Return [⟨Z_LAT0⟩, ⟨Z_LAT1⟩] at this bottleneck.

    Shape: (N_samples, 2)
    """

    theta = np.asarray(theta, float)

    # Slice theta exactly like in build_brainbox_ansatz
    theta_enc   = theta[:N_ENC]
    theta_bb3_a = theta[N_ENC:N_ENC + N_BB3_A]
    theta_bb2   = theta[N_ENC + N_BB3_A : N_ENC + N_BB3_A + N_BB2]

    # Bind subcircuits once
    enc_bound   = ansatz_enc.assign_parameters(theta_enc)
    bb3a_bound  = bb3_a.assign_parameters(theta_bb3_a)
    bb2_bound   = bb2.assign_parameters(theta_bb2)

    feats = []

    for x in X:
        qc = QuantumCircuit(TOTAL_QUBITS)

        # 1) load input on TRASH,LAT0,LAT1,LAT2
        build_input_on_data(qc, x, encoding=encoding)

        # 2) encoder on DATA_WIRES = [TRASH,LAT0,LAT1,LAT2]
        qc.compose(enc_bound, DATA_WIRES, inplace=True)

        # 3) first 3q BB on (LAT0,LAT1,LAT2)
        qc.compose(bb3a_bound, [LAT0, LAT1, LAT2], inplace=True)

        # 4) 2q BB on (LAT0,LAT1)  → this defines the 3→2 bottleneck
        qc.compose(bb2_bound, [LAT0, LAT1], inplace=True)

        # Now read Z on LAT0 and LAT1 at the bottleneck
        sv = Statevector.from_instruction(qc)
        Z0 = z_expect_on_wire_from_statevector(sv, LAT0)
        Z1 = z_expect_on_wire_from_statevector(sv, LAT1)
        feats.append([Z0, Z1])

    return np.array(feats)


theta_, steps_, L_tot_, L_t_, L_r_, Ft_hist_, Fr_hist_ = train_qae_combo_with_history(
    X_qae_train,
    lam=0.5,          # weight between trash & recon loss
    epochs=50,        # adjust as you like
    batch_size=64,
    encoding="amplitude",
    p_r=None,         # None = pure |0> reference for SWAP test
    ref_angle=None,   # None = |0> as ref
    seed=42,
    lr=0.01,          # learning rate
    fd_eps=1e-3,      # finite diff step
    steps_per_epoch=1 # increase for more FD steps per epoch
)

plt.figure(figsize=(8, 5))
plt.plot(steps_, L_tot_, label="Total loss", color='black', linewidth=2)
plt.plot(steps_, L_t_, label="Trash loss", linestyle="--")
plt.plot(steps_, L_r_, label="Recon loss", linestyle=":")
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

z_latent_ = get_bb_bottleneck_latent_Z(X_test_m, theta_)

Z_train_ = get_bb_bottleneck_latent_Z(X_qae_train, theta_)
mixed_Z = np.concatenate((z_latent_ ,X_test_values.reshape(-1,1)), axis=1)



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


Z_val, Z_final_test, y_val, y_final_test = train_test_split(
    z_latent_, X_test_values,
    test_size=0.8,          # keep 70% for final test, 30% for validation
    stratify=X_test_values,        # preserve normal/attack ratio
    random_state=42
)

alpha = 0.09
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
    val_scores = knn_scores(Z_train_, Z_val, k=k, leave_one_out=False)

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
    test_scores = knn_scores(Z_train_, Z_final_test, k=k, leave_one_out=False)
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
thr = 0.00131419
nn = NearestNeighbors(n_neighbors=k,n_jobs=-1).fit(Z_train_)
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

#Quantum kNN

from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# -------------------------
# 2D -> angles in [0, pi]^2
# -------------------------
Z_train = np.asarray(Z_train_, dtype=np.float32)
Z_val = np.asarray(Z_val, dtype=np.float32)
Z_final_test = np.asarray(Z_final_test, dtype=np.float32)


mn = Z_train.min(axis=0)
mx = Z_train.max(axis=0)
span = mx - mn
span = np.where(span < 1e-12, 1.0, span).astype(np.float32)

def to_angles(X):
    Xn = (X - mn) / span
    Xn = np.clip(Xn, 0.0, 1.0)
    return (Xn * np.pi).astype(np.float32)

Phi_tr = to_angles(Z_train)
Phi_val = to_angles(Z_val)
Phi_te  = to_angles(Z_final_test)

# -------------------------
# 3) Build KD-tree on TRAIN angles (2D)
# -------------------------
tree = cKDTree(Phi_tr)

# -------------------------
# 4) Quantum distance function (vectorized on candidates)
# d = 1 - Π cos^2(|Δ|/2)
# -------------------------
def qdist_candidates(cand_pts, q_pt):
    d = np.abs(cand_pts - q_pt)              # (cand,2)
    c = np.cos(0.5 * d)
    K = np.prod(c * c, axis=1)               # (cand,)
    return 1.0 - K                           # (cand,)

# -------------------------
# 5) Scoring: mean distance to k nearest (quantum rerank within candidates)
# -------------------------
def qknn_scores_2d(Phi_train, Phi_query, k, cand=200, agg="mean"):
    # Euclidean candidate retrieval in angle space
    _, idxs = tree.query(Phi_query, k=cand)
    if cand == 1:
        idxs = idxs[:, None]

    scores = np.empty(Phi_query.shape[0], dtype=np.float32)

    for j in range(Phi_query.shape[0]):
        q = Phi_query[j]
        cand_idx = idxs[j]
        cand_pts = Phi_train[cand_idx]              # (cand,2)

        qd = qdist_candidates(cand_pts, q)          # (cand,)

        # keep k smallest distances
        if qd.size > k:
            keep = np.argpartition(qd, k - 1)[:k]
            qd = qd[keep]

        if agg == "mean":
            scores[j] = float(qd.mean())
        elif agg == "median":
            scores[j] = float(np.median(qd))
        elif agg == "kth":
            scores[j] = float(np.max(qd))
        else:
            scores[j] = float(qd.mean())

    return scores

# -------------------------
# 6) Same eval helper you had
# -------------------------
def eval_at_threshold(scores, y_true, tau):
    y_pred = (scores >= tau).astype(int)  # 1 = attack
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)  # recall for attacks
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    return dict(fpr=fpr, tpr=tpr, precision=p, recall=r, f1=f1, y_pred=y_pred)

# -------------------------
# 7) Tune k and τ at target FPR alpha (NO wrapper function)
# -------------------------
alpha = 0.11
k_grid = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45]
agg = "mean"      # or "median" or "kth"
cand = 200        # candidate pool size (increase for better accuracy; decrease for speed)

best = None
print(f"Tuning  Quantum-kNN (2D) for target FPR α={alpha:.2%} | cand={cand} | agg={agg}\n" + "-"*80)

for k in k_grid:
    # 1) Score validation points using TRAIN as reference
    val_scores = qknn_scores_2d(Phi_tr, Phi_val, k=k, cand=cand, agg=agg)

    # 2) Threshold τ from VALIDATION NORMALS to hit FPR target
    val_scores_norm = val_scores[np.asarray(y_val) == 0]
    if val_scores_norm.size == 0:
        continue
    tau = float(np.quantile(val_scores_norm, 1 - alpha))

    # 3) Evaluate on validation
    val_metrics = eval_at_threshold(val_scores, y_val, tau)

    print(f"k={k:3d} | τ={tau:.6g} | ValFPR={val_metrics['fpr']:.4f} | "
          f"ValTPR={val_metrics['tpr']:.4f} | ValF1={val_metrics['f1']:.4f}")

    if (val_metrics["fpr"] <= alpha) and (best is None or val_metrics["f1"] > best["val_f1"]):
        best = {"k": k, "tau": tau, **{f"val_{m}": v for m, v in val_metrics.items()}}


k = 2
thr = 9.53674e-07

test_scores = qknn_scores_2d(Phi_tr, Phi_te, k=k, cand=cand, agg=agg)

# 2) Predict: 1 = attack, 0 = normal
y_pred = (test_scores > thr).astype(int)

print("\n=== TEST REPORT @ tuned threshold ===")
print(classification_report(y_final_test, y_pred, digits=4))


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

