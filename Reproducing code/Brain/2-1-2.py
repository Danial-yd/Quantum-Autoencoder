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
# 0. Config
# ============================================================

USE_BRAINBOX = True   # <<< toggle between baseline False / brainbox True

lam = 0.5             # weight between trash + reconstruction losses
p_r = None            # if not None, use fast mixed-reference trash fidelity

# ============================================================
# 1. Qubit layout
# ============================================================

AUX, REF0, TRASH, LAT0, LAT1 = 0, 1, 2, 3, 4
AUX_REC, REF_DATA0, REF_DATA1, REF_DATA2 = 5, 6, 7, 8

DATA_WIRES     = [TRASH, LAT0, LAT1]         # 3 data qubits total
REF_DATA_WIRES = [REF_DATA0, REF_DATA1, REF_DATA2]
TOTAL_QUBITS   = 9

# ============================================================
# 2. Ansatz / brainbox
# ============================================================

if not USE_BRAINBOX:
    # --- Baseline: single RealAmplitudes on 3 data qubits ---
    ansatz = RealAmplitudes(3, reps=4, entanglement="linear")
    N_THETA = ansatz.num_parameters

    def bind_ansatz(theta):
        theta = np.asarray(theta, dtype=float)
        return ansatz.assign_parameters(theta)

else:
    # --- Brainbox version: approximate 2–1–2 on LAT0,LAT1 / LAT0 ---
    #
    # Inside local 3-qubit system [0,1,2] = [TRASH, LAT0, LAT1]:
    #
    #   U_total(theta) =
    #       U_enc(theta_enc)       on [0,1,2]    (encoder, 3 qubits)
    #     ∘ U_bb1(theta_bb1)       on [1,2]      (2-qubit BB layer)
    #     ∘ U_bb2(theta_bb2)       on [1]        (1-qubit BB layer on LAT0)
    #     ∘ U_bb3(theta_bb3)       on [1,2]      (2-qubit BB layer)
    #
    # This gives a deeper "2 → 1 → 2"-flavored brainbox with LAT0 as core latent.

    # Encoder on all 3 data qubits
    ansatz_enc = RealAmplitudes(3, reps=3, entanglement="linear")

    # BB layer 1: 2-qubit block on (LAT0, LAT1)
    bb_21_a = RealAmplitudes(2, reps=1, entanglement="linear")

    # BB layer 2: 1-qubit block on LAT0 only
    bb_1    = RealAmplitudes(1, reps=1, entanglement="linear")

    # BB layer 3: another 2-qubit block on (LAT0, LAT1)
    bb_21_b = RealAmplitudes(2, reps=1, entanglement="linear")

    N_ENC   = ansatz_enc.num_parameters
    N_B21_A = bb_21_a.num_parameters
    N_B1    = bb_1.num_parameters
    N_B21_B = bb_21_b.num_parameters

    N_THETA = N_ENC + N_B21_A + N_B1 + N_B21_B

    def build_brainbox_ansatz(theta):
        """
        3-qubit local circuit on indices [0,1,2] = [TRASH, LAT0, LAT1].

        U_total(theta) =
            encoder(3q) on [0,1,2]
          ∘ BB_21_a(2q) on [1,2]
          ∘ BB_1(1q)    on [1]
          ∘ BB_21_b(2q) on [1,2]

        This approximates a 2–1–2-style brainbox with LAT0 as main latent qubit.
        """
        theta = np.asarray(theta, dtype=float)
        assert len(theta) == N_THETA, f"Expected {N_THETA} params, got {len(theta)}"

        theta_enc   = theta[:N_ENC]
        theta_b21_a = theta[N_ENC:N_ENC + N_B21_A]
        theta_b1    = theta[N_ENC + N_B21_A:N_ENC + N_B21_A + N_B1]
        theta_b21_b = theta[N_ENC + N_B21_A + N_B1:]

        qc = QuantumCircuit(3)

        # 1) Encoder on [0,1,2] = [TRASH,LAT0,LAT1]
        enc_bound = ansatz_enc.assign_parameters(theta_enc)
        qc.compose(enc_bound, [0, 1, 2], inplace=True)

        # 2) First 2-qubit BB layer on (LAT0,LAT1) = (1,2)
        b21a_bound = bb_21_a.assign_parameters(theta_b21_a)
        qc.compose(b21a_bound, [1, 2], inplace=True)

        # 3) 1-qubit BB layer on LAT0 = 1
        b1_bound = bb_1.assign_parameters(theta_b1)
        qc.compose(b1_bound, [1], inplace=True)

        # 4) Second 2-qubit BB layer on (LAT0,LAT1) = (1,2)
        b21b_bound = bb_21_b.assign_parameters(theta_b21_b)
        qc.compose(b21b_bound, [1, 2], inplace=True)

        return qc

    def bind_ansatz(theta):
        return build_brainbox_ansatz(theta)

# ============================================================
# 3. Encoding helpers
# ============================================================

def amp_vec_from_x8(x8):
    v = np.asarray(x8, dtype=np.float64).copy()
    n = np.linalg.norm(v)
    if n == 0.0:
        v = np.array([1.0] + [0.0]*7, dtype=np.float64)
    else:
        v = v / n
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
    # SWAP-test fidelity from ancilla-0 prob
    return max(0.0, min(1.0, 2.0*p0 - 1.0))

def _wrap_angles(th):
    return (th + np.pi) % (2*np.pi) - np.pi

# ============================================================
# 5. Trash fidelity (fast mixed-reference version)
# ============================================================

def P_trash0_after_encode(x8, theta):
    """
    3-qubit local circuit with layout [0,1,2] = [TRASH,LAT0,LAT1].
    """
    qc = QuantumCircuit(3)
    qc.initialize(amp_vec_from_x8(x8), [0,1,2])
    qc.compose(bind_ansatz(theta), [0,1,2], inplace=True)
    sv = Statevector.from_instruction(qc).data  # length 8
    # trash bit (qubit 0 in this local circuit) == 0 → indices {0,2,4,6}
    return float(np.sum(np.abs(sv[[0,2,4,6]])**2).real)

def Ft_mixed_1trash_fast(x8, theta, p_r):
    return float(p_r * P_trash0_after_encode(x8, theta) + (1.0 - p_r)/2.0)

# ============================================================
# 6. Circuits for losses
# ============================================================

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
    qc.h(AUX)
    qc.cswap(AUX, REF0, TRASH)
    qc.h(AUX)
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

# ============================================================
# 7. Losses
# ============================================================

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
            z_aux = z_expect_on_wire_from_statevector(sv_t, AUX)
            Ft = p0_to_fidelity(p_aux0_from_z(z_aux))

        # Reconstruction fidelity via SWAP test
        sv_r = Statevector.from_instruction(
            build_recon_circuit_3to2(x, theta, encoding=encoding, ref_angle=ref_angle)
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
# 8. Finite-difference + Adam
# ============================================================

def _fd_grad_combo(theta, X_batch, *, lam=0.5, encoding="amplitude", p_r=None, ref_angle=None, fd_eps=1e-3):
    theta = _wrap_angles(theta)
    base = qae_combo_loss(theta, X_batch, lam=lam, encoding=encoding, p_r=p_r, ref_angle=ref_angle)
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        th = theta.copy(); th[i] += fd_eps
        g[i] = (qae_combo_loss(th, X_batch, lam=lam, encoding=encoding, p_r=p_r, ref_angle=ref_angle) - base) / fd_eps
    return g, base

def build_latent_state_circuit(x, theta, encoding="amplitude"):
    """
    Prepare |x>, apply encoder+brainbox, return statevector of ALL qubits.
    No decoding, no SWAP test. We only want the latent qubits' state.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)
    return qc

def get_latent_Z_features(X, theta, encoding="amplitude"):
    """
    For 2–1–2, LAT0 is the main compressed qubit; LAT1 is auxiliary.
    Here we return both (Z_LAT0, Z_LAT1) so you can choose 1D or 2D latent.
    """
    feats = []
    for x in X:
        qc_lat = build_latent_state_circuit(x, theta, encoding=encoding)
        sv = Statevector.from_instruction(qc_lat)
        Z0 = z_expect_on_wire_from_statevector(sv, LAT0)
        Z1 = z_expect_on_wire_from_statevector(sv, LAT1)
        feats.append([Z0, Z1])
    return np.array(feats)

def train_qae_combo_with_history(
    X_train, *,
    lam=0.6,
    epochs=100,
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
    steps_per_epoch=1
):
    """
    Trains the QAE combo loss and records loss/fidelity history.
    """

    global N_THETA  # ensure we see the right N_THETA

    X_train = np.asarray(X_train, dtype=float)
    n_samples = X_train.shape[0]

    rng = np.random.default_rng(seed)
    theta = 0.01 * rng.standard_normal(N_THETA)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    steps = []
    loss_total_hist = []
    loss_trash_hist = []
    loss_recon_hist = []
    fid_trash_hist = []
    fid_recon_hist = []

    first_epoch_time = None

    pbar = trange(
        epochs,
        desc="QAE 3→2 (2-1-2 BB)" if USE_BRAINBOX else "QAE 3→2 baseline",
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

def get_latent_Z_features(X, theta, encoding="amplitude"):
    feats = []
    for x in X:
        qc_lat = build_latent_state_circuit(x, theta, encoding=encoding)
        sv = Statevector.from_instruction(qc_lat)

        Z0 = z_expect_on_wire_from_statevector(sv, LAT0)
        Z1 = z_expect_on_wire_from_statevector(sv, LAT1)
        feats.append([Z0, Z1])
    return np.array(feats)

def get_1d_latent_Z(X, theta, encoding="amplitude"):
    """
    1D latent: expectation value of Pauli-Z on LAT0
    after the full encoder+2-1-2 brainbox U_total.
    """
    feats = []
    for x in X:
        qc_lat = build_latent_state_circuit(x, theta, encoding=encoding)
        sv = Statevector.from_instruction(qc_lat)
        Z0 = z_expect_on_wire_from_statevector(sv, LAT0)
        feats.append(Z0)
    return np.array(feats)
def get_bottleneck_latent_Z(X, theta, encoding="amplitude"):
    """
    True 1-qubit latent for 2–1–2 BB:
    Z-expectation on LAT0 *after* the 1-qubit BB layer,
    but *before* the final 2-qubit BB layer.

    Returns: np.array of shape (len(X),)
    """

    # Make sure theta is a flat float array
    theta = np.asarray(theta, float)

    # Slice parameters same way as in build_brainbox_ansatz
    theta_enc   = theta[:N_ENC]
    theta_b21_a = theta[N_ENC:N_ENC + N_B21_A]
    theta_b1    = theta[N_ENC + N_B21_A:N_ENC + N_B21_A + N_B1]

    # Bind the subcircuits once
    enc_bound  = ansatz_enc.assign_parameters(theta_enc)
    b21a_bound = bb_21_a.assign_parameters(theta_b21_a)
    b1_bound   = bb_1.assign_parameters(theta_b1)

    feats = []

    for x in X:
        qc = QuantumCircuit(TOTAL_QUBITS)

        # 1) load input on TRASH,LAT0,LAT1
        build_input_on_data(qc, x, encoding=encoding)

        # 2) encoder on DATA_WIRES = [TRASH,LAT0,LAT1]
        qc.compose(enc_bound, DATA_WIRES, inplace=True)

        # 3) first 2-qubit BB on (LAT0,LAT1)
        qc.compose(b21a_bound, [LAT0, LAT1], inplace=True)

        # 4) 1-qubit BB on LAT0 (this defines the bottleneck)
        qc.compose(b1_bound, [LAT0], inplace=True)

        # Now measure Z on LAT0
        sv = Statevector.from_instruction(qc)
        Z_lat0 = z_expect_on_wire_from_statevector(sv, LAT0)
        feats.append(Z_lat0)

    return np.array(feats)




theta, steps, L_tot, L_t, L_r, Ft_hist, Fr_hist = train_qae_combo_with_history(
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
plt.plot(steps, L_tot, label="Total loss", color='black', linewidth=2)
plt.plot(steps, L_t, label="Trash loss", linestyle="--")
plt.plot(steps, L_r, label="Recon loss", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("QAE Training Losses")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(steps, Ft_hist, label="Trash fidelity", linestyle="--")
plt.plot(steps, Fr_hist, label="Recon fidelity", linestyle=":")
plt.xlabel("Epoch")
plt.ylabel("Fidelity")
plt.title("QAE Training Fidelities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

z_latent = get_bottleneck_latent_Z(X_test_m, theta)


mixed_Z = np.concatenate((z_latent.reshape(-1,1) ,X_test_values.reshape(-1,1)), axis=1)


normal_values = mixed_Z[mixed_Z[:, 1] == 0][:, 0:1]
attack_values = mixed_Z[mixed_Z[:, 1] == 1][:, 0:1]

plt.figure(figsize=(8, 5))
plt.hist(normal_values, bins=30, alpha=0.6, label='Normal', edgecolor='black')
plt.hist(attack_values, bins=30, alpha=0.6, label='Attack', edgecolor='black')

plt.title('Histogram of latent values', fontsize=16)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.legend(fontsize=14)

# increase tick label size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

