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

AUX, REF0, REF1, TRASH0, TRASH1, LATENT = 0, 1, 2, 3, 4, 5
DATA_WIRES = [TRASH0, TRASH1, LATENT]

# 3-qubit encoder ansatz (acts only on DATA_WIRES)
ansatz = RealAmplitudes(3, reps=3, entanglement="linear")

def bind_ansatz(theta):
    # assign/bind parameters (supports older Qiskit too)
    if hasattr(ansatz, "bind_parameters"):
        return ansatz.bind_parameters(theta)
    return ansatz.assign_parameters(theta)

# ---- Encoding helpers -------------------------------------------------------
def amp_vec_from_x8(x8):
    """8D → normalized length-8 amplitude vector (for 3 data qubits)."""
    v = np.asarray(x8, dtype=np.float64).copy()
    n = np.linalg.norm(v)
    if n == 0.0:
        v = np.array([1.0] + [0.0]*7, dtype=np.float64)
    else:
        v = v / n
    return v.astype(np.complex128)

def build_input_on_data(qc, x, encoding="amplitude"):
    """
    Put classical x on qubits [TRASH0, TRASH1, LATENT].
    encoding='amplitude' expects len(x)=8,
    encoding='angle'     expects len(x)=3 (angles in radians).
    """
    if encoding == "amplitude":
        amps = amp_vec_from_x8(x)
        qc.initialize(amps, DATA_WIRES)
    elif encoding == "angle":
        th0, th1, th2 = x
        qc.ry(th0, TRASH0)
        qc.ry(th1, TRASH1)
        qc.ry(th2, LATENT)
    else:
        raise ValueError("encoding must be 'amplitude' or 'angle'")

# ---- Circuit builder --------------------------------------------------------
def build_qae_circuit(x, theta, *, encoding="amplitude", ref_angles=None):
    """
    8D input on 3 data qubits -> compress to 1 latent (LATENT).
    Two trash qubits (TRASH0/1) are SWAP-tested against a 2-qubit REF (REF0/1).
    ref_angles: None => |00> reference; or tuple(a0,a1) to set RY on REF0/REF1.
    """
    qc = QuantumCircuit(6)

    # (1) Prepare 2-qubit reference
    if ref_angles is not None:
        a0, a1 = ref_angles
        qc.ry(a0, REF0)
        qc.ry(a1, REF1)

    # (2) Encode classical x on data
    build_input_on_data(qc, x, encoding=encoding)

    # (3) Variational encoder on the data register
    qc.compose(bind_ansatz(theta), qubits=DATA_WIRES, inplace=True)

    # (4) Multi-qubit SWAP test: two CSWAPs under the same AUX
    qc.h(AUX)
    qc.cswap(AUX, REF0, TRASH0)
    qc.cswap(AUX, REF1, TRASH1)
    qc.h(AUX)
    return qc

# ---- Observables / utilities ------------------------------------------------
def z_expect_on_wire_from_statevector(sv, wire_index):
    """⟨Z⟩ on a given wire from a full Statevector."""
    amps = np.asarray(sv.data)
    p0 = p1 = 0.0
    for idx, a in enumerate(amps):
        bit = (idx >> wire_index) & 1
        p = (a.real*a.real + a.imag*a.imag)
        if bit == 0: p0 += p
        else:        p1 += p
    return p0 - p1

def aux_z_from_statevector(sv):
    """⟨Z⟩ on AUX from a 6-qubit Statevector."""
    return z_expect_on_wire_from_statevector(sv, AUX)

def latent_z_from_statevector(sv):
    """⟨Z⟩ on the latent qubit."""
    return z_expect_on_wire_from_statevector(sv, LATENT)

def p_aux0_from_z(z_aux):
    """Ancilla-0 probability from <Z_aux> in a SWAP test."""
    return 0.5*(z_aux + 1.0)

# ---- Loss / training --------------------------------------------------------
def qae_loss(theta, batch_X, *, encoding="amplitude", ref_angles=None):
    """
    Loss = 1 - mean fidelity from SWAP test (AUX in |0>) across the batch.
    With two CSWAPs, fidelity increases only when BOTH trash qubits match REF.
    """
    fids = []
    for x in batch_X:
        qc = build_qae_circuit(x, theta, encoding=encoding, ref_angles=ref_angles)
        sv = Statevector.from_instruction(qc)
        z_aux = aux_z_from_statevector(sv)
        fids.append(p_aux0_from_z(z_aux))
    return float(1.0 - np.mean(fids))

def train_qae(X_train, *, steps=200, batch_size=512,
              encoding="amplitude", ref_angles=None, seed=0):
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(0, 1.0, ansatz.num_parameters)

    def objective(th):
        m = min(batch_size, len(X_train))
        idx = rng.choice(len(X_train), m, replace=False)
        return qae_loss(th, X_train[idx], encoding=encoding, ref_angles=ref_angles)

    res = minimize(objective, theta0, method="COBYLA",
                   options={"maxiter": steps, "rhobeg": 0.5, "tol": 1e-6})
    return res.x, res.fun

def embed_latent_Z(X, theta, *, encoding="amplitude", ref_angles=None):
    """
    Extract a 1D latent feature per sample: ⟨Z⟩ on LATENT after the encoder.
    """
    lat = np.empty((len(X), 1), dtype=float)
    for i, x in enumerate(X):
        qc = build_qae_circuit(x, theta, encoding=encoding, ref_angles=ref_angles)
        sv = Statevector.from_instruction(qc)
        lat[i, 0] = latent_z_from_statevector(sv)
    return lat

def train_qae_with_history(
    X_train, *,
    epochs=200, batch_size=512,
    encoding="amplitude", ref_angles=None, seed=0,
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
            return qae_loss(th, X_mb, encoding=encoding, ref_angles=ref_angles)

        t0 = perf_counter()
        res = minimize(obj, theta, method="COBYLA",
                       options={"maxiter": 5, "rhobeg": 0.5, "tol": 1e-6})

        theta = res.x  # update parameters
        t1 = perf_counter()
        epoch_time = t1 - t0

        if ep == 0:
            first_epoch_time = epoch_time

        # evaluate on the fixed train-eval slice
        L_tr = qae_loss(theta, X_eval, encoding=encoding, ref_angles=ref_angles)
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
    X_qae_train,
    epochs=200,
    batch_size=512,
    encoding="amplitude",  # since each row → amplitude vector of len 4
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

Z_test =embed_latent_Z(X_test_m, theta_opt, encoding="amplitude")
z_vals = Z_train.ravel()
plt.figure(figsize=(6,4))
plt.hist(z_vals, bins=30, color='skyblue', edgecolor='black')
plt.xlabel(r'$\langle Z_{\mathrm{latent}}\rangle$', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Latent Qubit Z Expectation Distribution')
plt.grid(alpha=0.3)
plt.show()

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

x = np.random.rand(8)

# Initialize random encoder parameters
theta = np.random.randn(ansatz.num_parameters)

# Build the full circuit
qc = build_qae_circuit(x, theta, encoding="amplitude")

# Show the circuit
print(qc.draw())
def train_qae_with_history_and_test(
    X_train, X_test, *,
    epochs=200, batch_size=512,
    encoding="amplitude", ref_angles=None, seed=0,
    eval_n=5000,               # fixed train-eval slice size (same idea as yours)
    test_full=True,            # True: use full test set each epoch; False: sample a test minibatch
    test_batch_size=2048       # used only if test_full is False
):
    """
    Same training dynamics as your train_qae_with_history (1 L-BFGS-B step/epoch on a fresh minibatch),
    but ALSO computes and returns test (attack) mean fidelity/loss each epoch for comparison.
    """
    rng = np.random.default_rng(seed)
    theta = 0.01 * rng.standard_normal(ansatz.num_parameters)

    # fixed evaluation slice from TRAIN (for stable train curves)
    eval_n = min(eval_n, len(X_train))
    eval_idx = rng.choice(len(X_train), eval_n, replace=False)
    X_eval = X_train[eval_idx]

    steps = []
    tr_loss_hist = []
    tr_fid_hist  = []
    te_loss_hist = []
    te_fid_hist  = []

    first_epoch_time = None

    pbar = trange(epochs, desc="QAE training (train & test eval)", leave=True)
    for ep in pbar:
        # one optimizer step on a fresh TRAIN minibatch (identical to your loop)
        m = min(batch_size, len(X_train))
        mb_idx = rng.choice(len(X_train), m, replace=False)
        X_mb = X_train[mb_idx]

        def obj(th):  # minibatch objective (train)
            return qae_loss(th, X_mb, encoding=encoding, ref_angles=ref_angles)

        t0 = perf_counter()
        res = minimize(obj, theta, method="L-BFGS-B", options={"maxiter": 1})
        theta = res.x  # update parameters
        t1 = perf_counter()
        epoch_time = t1 - t0

        if ep == 0:
            first_epoch_time = epoch_time

        # ---- Evaluate TRAIN (fixed eval slice)
        L_tr = qae_loss(theta, X_eval, encoding=encoding, ref_angles=ref_angles)
        F_tr = 1.0 - L_tr

        # ---- Evaluate TEST (attack) mean fidelity
        if test_full or len(X_test) <= test_batch_size:
            X_te_eval = X_test
        else:
            te_idx = rng.choice(len(X_test), test_batch_size, replace=False)
            X_te_eval = X_test[te_idx]
        L_te = qae_loss(theta, X_te_eval, encoding=encoding, ref_angles=ref_angles)
        F_te = 1.0 - L_te

        # ---- log
        steps.append(ep)
        tr_loss_hist.append(L_tr)
        tr_fid_hist.append(F_tr)
        te_loss_hist.append(L_te)
        te_fid_hist.append(F_te)

        pbar.set_postfix({
            "loss_tr": f"{L_tr:.4f}", "fid_tr": f"{F_tr:.4f}",
            "loss_te": f"{L_te:.4f}", "fid_te": f"{F_te:.4f}",
            "t/ep(s)": f"{epoch_time:.2f}"
        })

    print(f"\nFirst epoch time: {first_epoch_time:.3f} s")
    print(f"Final TRAIN: loss={tr_loss_hist[-1]:.6f} | fid={tr_fid_hist[-1]:.6f}")
    print(f"Final  TEST: loss={te_loss_hist[-1]:.6f} | fid={te_fid_hist[-1]:.6f}")

    return (
        theta,
        np.array(steps),
        np.array(tr_loss_hist, dtype=float),
        np.array(tr_fid_hist,  dtype=float),
        np.array(te_loss_hist, dtype=float),
        np.array(te_fid_hist,  dtype=float),
    )


n_subset = 5000
idx = np.random.choice(len(X_qae_test), n_subset, replace=False)
X_qae_test_subset = X_qae_test[idx]

theta_opt, steps_hist, tr_loss, tr_fid, te_loss, te_fid = train_qae_with_history_and_test(
    X_qae_train,            # your normal set (already preprocessed to 8D)
    X_qae_test_subset,     # your attack-only test set (same preproc!)
    epochs=100,
    batch_size=512,
    encoding="amplitude",
    ref_angles=None,
    seed=42,
    eval_n=min(5000, len(X_qae_train)),
    test_full=True          # set False if test set is huge
)




