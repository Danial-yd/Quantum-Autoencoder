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
from qiskit.quantum_info import DensityMatrix, partial_trace

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

ansatz = RealAmplitudes(3, reps=3, entanglement="linear")

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

def rho_ref_mixed_2q(p_r: float) -> np.ndarray:
    """ρ_ref = p_r |00><00| + (1-p_r) I/4   (2-qubit, 4x4)."""
    rho00 = np.zeros((4,4), dtype=np.complex128); rho00[0,0] = 1.0
    I4 = np.eye(4, dtype=np.complex128) / 4.0
    return p_r * rho00 + (1.0 - p_r) * I4

def trash_fidelity_mixed_density(x, theta, *, p_r=0.8, encoding="amplitude"):
    """Encode |x>, extract ρ_T on {TRASH0,TRASH1}, return Tr(ρ_T ρ_ref)."""
    qc = QuantumCircuit(TOTAL_QUBITS)
    build_input_on_data(qc, x, encoding=encoding)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)
    rho_full = DensityMatrix.from_instruction(qc)  # full density matrix
    # keep only the trash subsystem (TRASH0, TRASH1)
    trace_out = [q for q in range(TOTAL_QUBITS) if q not in (TRASH0, TRASH1)]
    rho_T = partial_trace(rho_full, trace_out).data  # 4x4
    rho_ref = rho_ref_mixed_2q(p_r)
    Ft = float(np.real(np.trace(rho_T @ rho_ref)))
    return Ft

# --- Losses (use TRUE fidelities) ------------------------------------------
def qae_losses(theta, batch_X, *, encoding="amplitude", p_r=None, ref_angles=None):
    L_trash, L_recon = [], []
    for x in batch_X:
        # --- Trash fidelity (mixed if p_r is given, else your current SWAP test) ---
        if p_r is not None:
            Ft = trash_fidelity_mixed_density(x, theta, p_r=p_r, encoding=encoding)
        else:
            sv_t = Statevector.from_instruction(
                build_trash_circuit_3q(x, theta, encoding=encoding, ref_angles=ref_angles)
            )
            Ft = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_t, AUX)))

        # --- Recon fidelity (unchanged; true fidelity via SWAP test) ---
        sv_r = Statevector.from_instruction(
            build_recon_circuit_3q(x, theta, encoding=encoding, ref_angles=ref_angles)
        )
        Fr = p0_to_fidelity(p_aux0_from_z(z_expect_on_wire_from_statevector(sv_r, AUX_REC)))

        L_trash.append(1.0 - Ft)   # maximize Ft
        L_recon.append(1.0 - Fr)   # maximize Fr
    return float(np.mean(L_trash)), float(np.mean(L_recon))

def qae_combo_loss(theta, batch_X, *, lam=0.5, encoding="amplitude", p_r=None, ref_angles=None):
    L_t, L_r = qae_losses(theta, batch_X, encoding=encoding, p_r=p_r, ref_angles=ref_angles)
    return lam * L_t + (1.0 - lam) * L_r


def _wrap_angles(th):
    """
    Wrap parameters back into [-π, π] to keep angles bounded.
    This helps avoid numerical instability when training on periodic landscapes.
    """
    return (th + np.pi) % (2 * np.pi) - np.pi

def _fd_grad_combo(theta, X_batch, *, lam=0.5, encoding="amplitude", p_r=None, ref_angles=None, fd_eps=1e-3):
    theta = _wrap_angles(theta)
    base = qae_combo_loss(theta, X_batch, lam=lam, encoding=encoding, p_r=p_r, ref_angles=ref_angles)
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        th = theta.copy(); th[i] += fd_eps
        g[i] = (qae_combo_loss(th, X_batch, lam=lam, encoding=encoding, p_r=p_r, ref_angles=ref_angles) - base) / fd_eps
    return g, base

def train_qae_combo_with_history(
    X_train, *,
    lam=0.5,                     # weight between trash and reconstruction losses
    epochs=200,
    batch_size=512,
    encoding="amplitude",
    p_r=None,
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
                theta, Xb, lam=lam, encoding=encoding,p_r=p_r, ref_angles=ref_angles, fd_eps=fd_eps
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
        L_t, L_r = qae_losses(theta, X_val, encoding=encoding, p_r=p_r, ref_angles=ref_angles)
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
    seed=SEED,
    p_r=0.8,               # <- choose mixing ratio (try 1.0, 0.8, 0.6, 0.5)
)