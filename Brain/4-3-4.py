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

from Experiment.KD99Amplitudeencoding3qubits import Z_test, Z_train

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
    ])),  # match your # of qubits
])



preproc.fit(X_norm)

# 4) Transform normal-train, normal-val, and the mixed test set
X_qae_train = preproc.transform(X_norm)


X_test_mixed = X_attack
y_test_mixed = remain_df["label"].values
X_qae_test   = preproc.transform(X_test_mixed)
X_test_m = preproc.transform(X_test)




import numpy as np
from time import perf_counter
from tqdm.auto import trange

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RealAmplitudes

# ============================================================
# 0. Config
# ============================================================

USE_BRAINBOX = True   # <<< MUST be True to use 4-3-4 BB

lam = 0.5             # weight between trash + reconstruction losses
p_r = None            # if not None, mixed-reference version (not used here)

# ============================================================
# 1. Qubit layout (5 data qubits: 1 trash + 4 latent)
# ============================================================

# Global indices
AUX, REF0, TRASH, LAT0, LAT1, LAT2, LAT3 = 0, 1, 2, 3, 4, 5, 6
AUX_REC, REF_DATA0, REF_DATA1, REF_DATA2, REF_DATA3, REF_DATA4 = 7, 8, 9, 10, 11, 12

DATA_WIRES     = [TRASH, LAT0, LAT1, LAT2, LAT3]        # 5 data qubits total
REF_DATA_WIRES = [REF_DATA0, REF_DATA1, REF_DATA2, REF_DATA3, REF_DATA4]
TOTAL_QUBITS   = 13

# ============================================================
# 2. Ansatz / brainbox (5 data qubits, 4–3–4 on latents)
# ============================================================

if not USE_BRAINBOX:
    # --- Baseline: single RealAmplitudes on 5 data qubits ---
    ansatz = RealAmplitudes(5, reps=4, entanglement="linear")
    N_THETA = ansatz.num_parameters

    def bind_ansatz(theta):
        theta = np.asarray(theta, dtype=float)
        return ansatz.assign_parameters(theta)

else:
    # --- Brainbox version: encoder(5q) + 4–3–4 BB on (LAT0..LAT3) ---
    #
    # Local layout for ansatz: [0,1,2,3,4] = [TRASH, LAT0, LAT1, LAT2, LAT3]
    #
    #   U_total(theta) =
    #       U_enc   (5q) on [0,1,2,3,4]
    #     ∘ U_bb4_a(4q) on [1,2,3,4]     (first "4")
    #     ∘ U_bb3  (3q) on [1,2,3]       (the "3" bottleneck)
    #     ∘ U_bb4_b(4q) on [1,2,3,4]     (final "4")
    #
    # LAT0..LAT3 form the 4-qubit latent subsystem.

    ansatz_enc = RealAmplitudes(5, reps=3, entanglement="linear")
    bb4_a      = RealAmplitudes(4, reps=1, entanglement="linear")
    bb3        = RealAmplitudes(3, reps=1, entanglement="linear")
    bb4_b      = RealAmplitudes(4, reps=1, entanglement="linear")

    N_ENC   = ansatz_enc.num_parameters
    N_BB4_A = bb4_a.num_parameters
    N_BB3   = bb3.num_parameters
    N_BB4_B = bb4_b.num_parameters

    N_THETA = N_ENC + N_BB4_A + N_BB3 + N_BB4_B

    def build_brainbox_ansatz(theta):
        """
        5-qubit local circuit on [0,1,2,3,4] = [TRASH,LAT0,LAT1,LAT2,LAT3].

        U_total(theta) =
            encoder(5q) on [0,1,2,3,4]
          ∘ bb4_a(4q)  on [1,2,3,4]
          ∘ bb3(3q)    on [1,2,3]
          ∘ bb4_b(4q)  on [1,2,3,4]
        """
        theta = np.asarray(theta, dtype=float)
        assert len(theta) == N_THETA, f"Expected {N_THETA} params, got {len(theta)}"

        theta_enc   = theta[:N_ENC]
        theta_bb4_a = theta[N_ENC:N_ENC + N_BB4_A]
        theta_bb3   = theta[N_ENC + N_BB4_A : N_ENC + N_BB4_A + N_BB3]
        theta_bb4_b = theta[N_ENC + N_BB4_A + N_BB3:]

        qc = QuantumCircuit(5)

        # Encoder on [0,1,2,3,4] = [TRASH,LAT0,LAT1,LAT2,LAT3]
        enc_bound = ansatz_enc.assign_parameters(theta_enc)
        qc.compose(enc_bound, [0,1,2,3,4], inplace=True)

        # First 4q BB on (LAT0..LAT3) = (1,2,3,4)
        bb4a_bound = bb4_a.assign_parameters(theta_bb4_a)
        qc.compose(bb4a_bound, [1,2,3,4], inplace=True)

        # 3q BB on (LAT0,LAT1,LAT2) = (1,2,3)  → 4→3 bottleneck
        bb3_bound = bb3.assign_parameters(theta_bb3)
        qc.compose(bb3_bound, [1,2,3], inplace=True)

        # Final 4q BB on (LAT0..LAT3) = (1,2,3,4)
        bb4b_bound = bb4_b.assign_parameters(theta_bb4_b)
        qc.compose(bb4b_bound, [1,2,3,4], inplace=True)

        return qc

    def bind_ansatz(theta):
        return build_brainbox_ansatz(theta)

# ============================================================
# 3. Encoding helpers
# ============================================================

def amp_vec_from_x32(x32):
    v = np.asarray(x32, dtype=np.float64).copy()
    n = np.linalg.norm(v)
    if n == 0.0:
        v = np.zeros(32, dtype=np.float64)
        v[0] = 1.0
    else:
        v = v / n
    return v.astype(np.complex128)

def build_input_on_five(qc, wires, x, encoding="amplitude"):
    if encoding == "amplitude":
        qc.initialize(amp_vec_from_x32(x), wires)
    elif encoding == "angle":
        th0, th1, th2, th3, th4 = x
        for w, th in zip(wires, (th0, th1, th2, th3, th4)):
            qc.ry(th, w)
    else:
        raise ValueError("encoding must be 'amplitude' or 'angle'")

def build_input_on_data(qc, x, encoding="amplitude"):
    build_input_on_five(qc, DATA_WIRES, x, encoding=encoding)

# ============================================================
# 4. Utilities
# ============================================================

def z_expect_on_wire_from_statevector(sv, wire_index):
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
    return max(0.0, min(1.0, 2.0*p0 - 1.0))

def _wrap_angles(th):
    return (th + np.pi) % (2*np.pi) - np.pi

# ============================================================
# 5. SWAP-test circuits for losses (5→4 compression)
# ============================================================

def build_trash_circuit_5to4(x, theta, *, encoding="amplitude", ref_angle=None):
    """
    SWAP test: compare TRASH vs REF0 using AUX.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    if ref_angle is not None:
        qc.ry(ref_angle, REF0)

    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    qc.h(AUX)
    qc.cswap(AUX, REF0, TRASH)
    qc.h(AUX)
    return qc

def build_recon_circuit_5to4(x, theta, *, encoding="amplitude", ref_angle=None):
    """
    Prepare |x>, encode U(θ), swap in clean trash from REF0, decode U(θ)†.
    SWAP-test all 5 data qubits vs fresh reference |x>.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    if ref_angle is not None:
        qc.ry(ref_angle, REF0)
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    # swap in clean trash
    qc.swap(TRASH, REF0)
    qc.compose(bind_ansatz(theta).inverse(), DATA_WIRES, inplace=True)

    # reference |x> on 5 qubits
    build_input_on_five(qc, REF_DATA_WIRES, x, encoding=encoding)

    qc.h(AUX_REC)
    qc.cswap(AUX_REC, TRASH, REF_DATA0)
    qc.cswap(AUX_REC, LAT0,  REF_DATA1)
    qc.cswap(AUX_REC, LAT1,  REF_DATA2)
    qc.cswap(AUX_REC, LAT2,  REF_DATA3)
    qc.cswap(AUX_REC, LAT3,  REF_DATA4)
    qc.h(AUX_REC)
    return qc

# ============================================================
# 6. Losses
# ============================================================

def qae_losses(theta, batch_X, *, encoding="amplitude", p_r=None, ref_angle=None):
    L_trash, L_recon = [], []
    for x in batch_X:
        # Trash fidelity via SWAP test
        sv_t = Statevector.from_instruction(
            build_trash_circuit_5to4(x, theta, encoding=encoding, ref_angle=ref_angle)
        )
        z_aux = z_expect_on_wire_from_statevector(sv_t, AUX)
        Ft = p0_to_fidelity(p_aux0_from_z(z_aux))

        # Reconstruction fidelity via SWAP test
        sv_r = Statevector.from_instruction(
            build_recon_circuit_5to4(x, theta, encoding=encoding, ref_angle=ref_angle)
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
# 7. Finite-difference + Adam training
# ============================================================

def _fd_grad_combo(theta, X_batch, *, lam=0.5, encoding="amplitude",
                   p_r=None, ref_angle=None, fd_eps=1e-3):
    theta = _wrap_angles(theta)
    base = qae_combo_loss(theta, X_batch, lam=lam, encoding=encoding, p_r=p_r, ref_angle=ref_angle)
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        th = theta.copy(); th[i] += fd_eps
        g[i] = (qae_combo_loss(th, X_batch, lam=lam, encoding=encoding, p_r=p_r, ref_angle=ref_angle) - base) / fd_eps
    return g, base

def train_qae_combo_with_history(
    X_train, *,
    lam=0.5,
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
    X_train = np.asarray(X_train, dtype=float)
    n_samples = X_train.shape[0]

    rng = np.random.default_rng(seed)
    theta = 0.01 * rng.standard_normal(N_THETA)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    steps = []
    loss_total_hist = []
    loss_trash_hist = []
    loss_recon_hist = []
    fid_trash_hist = []
    fid_recon_hist = []

    first_epoch_time = None

    pbar = trange(
        epochs,
        desc="QAE 5→4 (4-3-4 BB)" if USE_BRAINBOX else "QAE 5→4 baseline",
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
            t = ep * max(1, steps_per_epoch) + (k + 1)

            # Adam update
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            theta = _wrap_angles(theta - lr * m_hat / (np.sqrt(v_hat) + eps))

        t1 = perf_counter()
        if ep == 0:
            first_epoch_time = t1 - t0

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
# 8. 3D bottleneck latent extraction (the "3" in 4-3-4)
# ============================================================

if USE_BRAINBOX:
    def get_bb_bottleneck_latent_Z(X, theta, encoding="amplitude"):
        """
        True 3-qubit bottleneck latent for 4-3-4 BB:
        - Apply encoder(5q) + bb4_a(4q) + bb3(3q)
        - STOP before bb4_b
        - Return [⟨Z_LAT0⟩, ⟨Z_LAT1⟩, ⟨Z_LAT2⟩] at the bottleneck.

        Shape: (N_samples, 3)
        """

        theta = np.asarray(theta, float)

        # Slice parameters exactly like in build_brainbox_ansatz
        theta_enc   = theta[:N_ENC]
        theta_bb4_a = theta[N_ENC : N_ENC + N_BB4_A]
        theta_bb3   = theta[N_ENC + N_BB4_A : N_ENC + N_BB4_A + N_BB3]

        enc_bound  = ansatz_enc.assign_parameters(theta_enc)
        bb4a_bound = bb4_a.assign_parameters(theta_bb4_a)
        bb3_bound  = bb3.assign_parameters(theta_bb3)

        feats = []

        for x in X:
            qc = QuantumCircuit(TOTAL_QUBITS)

            # 1) input on TRASH,LAT0,LAT1,LAT2,LAT3
            build_input_on_data(qc, x, encoding=encoding)

            # 2) encoder on 5 data qubits
            qc.compose(enc_bound, DATA_WIRES, inplace=True)

            # 3) first 4q BB on (LAT0..LAT3)
            qc.compose(bb4a_bound, [LAT0, LAT1, LAT2, LAT3], inplace=True)

            # 4) 3q BB on (LAT0,LAT1,LAT2) → bottleneck
            qc.compose(bb3_bound, [LAT0, LAT1, LAT2], inplace=True)

            # measure Z on the 3-qubit bottleneck
            sv = Statevector.from_instruction(qc)
            Z0 = z_expect_on_wire_from_statevector(sv, LAT0)
            Z1 = z_expect_on_wire_from_statevector(sv, LAT1)
            Z2 = z_expect_on_wire_from_statevector(sv, LAT2)

            feats.append([Z0, Z1, Z2])

        return np.array(feats)


theta__, steps__, L_tot__, L_t__, L_r__, Ft_hist__, Fr_hist__ = train_qae_combo_with_history(
    X_qae_train[:,0:32],
    lam=0.5,
    epochs=50,
    batch_size=32,
    encoding="amplitude",
    p_r=None,
    ref_angle=None,
    seed=42,
    lr=0.01,
    fd_eps=1e-3,
    steps_per_epoch=1,
)

plt.figure(figsize=(8, 5))
plt.plot(steps__, L_tot__, label="Total loss", color='black', linewidth=2)
plt.plot(steps__, L_t__, label="Trash loss", linestyle="--")
plt.plot(steps__, L_r__, label="Recon loss", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("QAE Training Losses")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(steps__, Ft_hist__, label="Trash fidelity", linestyle="--")
plt.plot(steps__, Fr_hist__, label="Recon fidelity", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Fidelity")
plt.title("QAE Training Fidelities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3D bottleneck latent
Z_latent__ = get_bb_bottleneck_latent_Z(X_test_m, theta__)

Z_train__ = get_bb_bottleneck_latent_Z(X_qae_train, theta__)

mixed_Z = np.concatenate((Z_latent__ ,X_test_values.reshape(-1,1)), axis=1)



normal_values = mixed_Z[mixed_Z[:, 3] == 0][:, 0:3]
attack_values = mixed_Z[mixed_Z[:, 3] == 1][:, 0:3]


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


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Separate classes
normal_values = mixed_Z[mixed_Z[:, 3] == 0][:, 0:3]
attack_values = mixed_Z[mixed_Z[:, 3] == 1][:, 0:3]

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Normal (now less transparent)
ax.scatter(
    normal_values[:, 0], normal_values[:, 1], normal_values[:, 2],
    c='dodgerblue', label='Normal',
    alpha=0.6, s=40, depthshade=False
)

# Attack (opaque + larger + on top)
ax.scatter(
    attack_values[:, 0], attack_values[:, 1], attack_values[:, 2],
    c='crimson', label='Attack',
    alpha=0.5, s=40, edgecolors='k',
    depthshade=False
)

# Labels
ax.set_xlabel("⟨Z⟩ of LAT0", fontsize=12)
ax.set_ylabel("⟨Z⟩ of LAT1", fontsize=12)
ax.set_zlabel("⟨Z⟩ of LAT2", fontsize=12)

ax.set_title("QAE Latent Space (3D)", fontsize=14)
ax.legend()

ax.grid(True, alpha=0.3)

# Camera angle
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def encode_data(qc, qubit, theta):
    qc.ry(theta, qubit)  # your meridian encoding

def swap_test_circuit(theta_train, theta_query):
    anc = QuantumRegister(1, "anc")
    data = QuantumRegister(2, "data")  # data[0] = train, data[1] = query
    c = ClassicalRegister(1, "c")
    qc = QuantumCircuit(anc, data, c)

    # 1) Prepare ancilla |0> (already) and data states:
    encode_data(qc, data[0], theta_train)
    encode_data(qc, data[1], theta_query)

    # 2) Hadamard on ancilla
    qc.h(anc[0])

    # 3) Controlled-SWAP between data[0] and data[1], controlled on anc
    qc.cswap(anc[0], data[0], data[1])

    # 4) Second Hadamard on ancilla
    qc.h(anc[0])

    # 5) Measure ancilla
    qc.measure(anc[0], c[0])

    return qc

