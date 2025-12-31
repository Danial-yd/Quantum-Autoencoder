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


USE_BRAINBOX = True   # <<< toggle between baseline False / brainbox True

     # number of test samples
BATCH_SIZE = 8
EPOCHS = 200

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
    ansatz = RealAmplitudes(3, reps=6, entanglement="linear")
    N_THETA = ansatz.num_parameters

    def bind_ansatz(theta):
        theta = np.asarray(theta, dtype=float)
        return ansatz.assign_parameters(theta)

else:
    # --- Brainbox version ---
    # Encoder on all 3 data qubits (TRASH, LAT0, LAT1)
    ansatz_enc = RealAmplitudes(3, reps=3, entanglement="linear")
    # Brainbox on LAT0,LAT1 only
    ansatz_bb  = RealAmplitudes(2, reps=2, entanglement="linear")

    N_ENC   = ansatz_enc.num_parameters
    N_BB    = ansatz_bb.num_parameters
    N_THETA = N_ENC + N_BB

    def build_brainbox_ansatz(theta):
        """
        3-qubit local circuit:
          U_total(theta) = U_enc(theta_enc) on [0,1,2]
                            followed by
                            U_bb(theta_bb)  on [1,2]
        where [0,1,2] correspond to [TRASH,LAT0,LAT1] when composed
        on DATA_WIRES.
        """
        theta = np.asarray(theta, dtype=float)
        assert len(theta) == N_THETA, f"Expected {N_THETA} params, got {len(theta)}"

        theta_enc = theta[:N_ENC]
        theta_bb  = theta[N_ENC:]

        qc = QuantumCircuit(3)

        enc_bound = ansatz_enc.assign_parameters(theta_enc)
        qc.compose(enc_bound, [0, 1, 2], inplace=True)

        bb_bound = ansatz_bb.assign_parameters(theta_bb)
        qc.compose(bb_bound, [1, 2], inplace=True)

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


def qae_combo_loss(
        theta,
        batch_X,
        *,
        lam=0.5,
        encoding="amplitude",
        p_r=None,
        ref_angle=None,
        alpha_l1=0.0    # <<< L1 strength
):
    # original QAE losses
    L_t, L_r = qae_losses(
        theta,
        batch_X,
        encoding=encoding,
        p_r=p_r,
        ref_angle=ref_angle,
    )

    # main QAE loss
    L_total = lam * L_t + (1 - lam) * L_r

    # ---- L1 REGULARIZATION ONLY ----
    if alpha_l1 > 0.0:
        L_total += alpha_l1 * float(np.sum(np.abs(theta)))

    return L_total

# ============================================================
# 8. Finite-difference + Adam
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
        alpha_l1=0.0
):
    theta = _wrap_angles(theta)

    base = qae_combo_loss(
        theta,
        X_batch,
        lam=lam,
        encoding=encoding,
        p_r=p_r,
        ref_angle=ref_angle,
        alpha_l1=alpha_l1,
    )

    g = np.zeros_like(theta)

    for i in range(len(theta)):
        th = theta.copy()
        th[i] += fd_eps

        L_pert = qae_combo_loss(
            th,
            X_batch,
            lam=lam,
            encoding=encoding,
            p_r=p_r,
            ref_angle=ref_angle,
            alpha_l1=alpha_l1,
        )
        g[i] = (L_pert - base) / fd_eps

    return g, base


def adam_update(theta, grad, m, v, t, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad * grad)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v

def build_latent_state_circuit(x, theta, encoding="amplitude"):
    """
    Prepare |x>, apply encoder+brainbox, return statevector of ALL qubits.
    No decoding, no SWAP test. We only want the latent qubits' state.
    """
    qc = QuantumCircuit(TOTAL_QUBITS)

    # load input state on TRASH,LAT0,LAT1
    build_input_on_data(qc, x, encoding=encoding)

    # apply encoder (baseline or brainbox chosen automatically by bind_ansatz)
    qc.compose(bind_ansatz(theta), DATA_WIRES, inplace=True)

    return qc

def get_latent_Z_features(X, theta, encoding="amplitude"):
    feats = []
    for x in X:
        qc_lat = build_latent_state_circuit(x, theta, encoding=encoding)
        sv = Statevector.from_instruction(qc_lat)

        Z0 = z_expect_on_wire_from_statevector(sv, LAT0)
        Z1 = z_expect_on_wire_from_statevector(sv, LAT1)
        feats.append([Z0, Z1])
    return np.array(feats)

def train_qae_combo_with_history(
    X_train,
    *,
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
    steps_per_epoch=1,
    alpha_l1=0.0    # <<< L1 only
):
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
    pbar = trange(epochs, desc="QAE Train", leave=True)

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
                alpha_l1=alpha_l1,
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

        # validation on subset of training set
        vidx = rng.choice(n_samples, min(300, n_samples), replace=False)
        Xv = X_train[vidx]

        L_t, L_r = qae_losses(theta, Xv, encoding=encoding, p_r=p_r, ref_angle=ref_angle)
        L_total = lam * L_t + (1 - lam) * L_r + alpha_l1 * np.sum(np.abs(theta))

        F_t, F_r = 1 - L_t, 1 - L_r

        steps.append(ep)
        loss_total_hist.append(L_total)
        loss_trash_hist.append(L_t)
        loss_recon_hist.append(L_r)
        fid_trash_hist.append(F_t)
        fid_recon_hist.append(F_r)

        pbar.set_postfix({
            "L": f"{L_total:.4f}",
            "Lt": f"{L_t:.4f}",
            "Lr": f"{L_r:.4f}",
            "Ft": f"{F_t:.3f}",
            "Fr": f"{F_r:.3f}",
            "t/ep(s)": f"{(t1 - t0):.2f}",
        })

    print(f"\nFirst epoch time: {first_epoch_time:.3f} s")
    print(f"Final total loss: {loss_total_hist[-1]:.6f}")
    print(f"Final Ft={fid_trash_hist[-1]:.6f}, Fr={fid_recon_hist[-1]:.6f}")

    return (
        theta,
        np.array(steps),
        np.array(loss_total_hist),
        np.array(loss_trash_hist),
        np.array(loss_recon_hist),
        np.array(fid_trash_hist),
        np.array(fid_recon_hist),
    )


theta_, steps, Ltot, Ltrash, Lrecon, Ftrash, Frecon = (
    train_qae_combo_with_history(
        X_qae_train,
        lam=0.5,
        epochs=120,
        batch_size=64,
        alpha_l2=3e-4,     # <<< L2 REG HERE
        lr=0.01,
        fd_eps=1e-3,
    )
)


import numpy as np

# ============================================================
# Cross-validation for L2 (alpha_l2)
# ============================================================

alpha_list = [
    0,
    5e-5,
    1e-4,
    2e-4,
    3e-4,
    5e-4,
    7e-4,
    1e-3,
    2e-3,
]


from tqdm import tqdm

from tqdm import trange

def cross_validate_l1_qae(
    X_all,
    *,
    lam=0.5,
    alpha_list=(0.0, 1e-4, 3e-4, 1e-3),
    k_folds=2,
    epochs=12,          # keep small for CV!
    batch_size=64,
    encoding="amplitude",
    p_r=None,
    ref_angle=None,
    base_seed=123,
    lr=0.01,
    fd_eps=1e-3,
    steps_per_epoch=1,
    val_max_samples=300,   # <<< NEW: limit validation cost
):
    """
    Lightweight k-fold cross-validation over alpha_l2.

    - Uses a single progress bar over all (alpha, fold) pairs.
    - Critically: evaluates validation fidelity on a RANDOM SUBSET
      of each validation fold (up to val_max_samples) so it doesn't
      get stuck on huge datasets.
    """

    X_all = np.asarray(X_all, dtype=float)
    N = len(X_all)
    assert N >= k_folds, "Not enough samples for the chosen number of folds."

    # ---- build folds ----
    rng = np.random.default_rng(base_seed)
    indices = np.arange(N)
    rng.shuffle(indices)

    fold_sizes = np.full(k_folds, N // k_folds, dtype=int)
    fold_sizes[: (N % k_folds)] += 1

    folds = []
    start = 0
    for fs in fold_sizes:
        folds.append(indices[start:start+fs])
        start += fs

    cv_results = []

    total_runs = len(alpha_list) * k_folds
    print(f"\n=== CV over alpha_l1={alpha_list}, k={k_folds}, total runs={total_runs} ===\n")

    run_id = 0
    pbar = trange(total_runs, desc="CV runs", leave=True)
    for alpha in alpha_list:
        fold_val_fids = []

        for k in range(k_folds):
            run_id += 1
            pbar.set_description(f"CV runs (α={alpha}, fold={k+1}/{k_folds})")

            val_idx   = folds[k]
            train_idx = np.hstack([folds[j] for j in range(k_folds) if j != k])

            X_train_fold = X_all[train_idx]
            X_val_fold   = X_all[val_idx]

            print(
                f"\n[Run {run_id}/{total_runs}] α={alpha}, fold {k+1}/{k_folds}, "
                f"train={len(X_train_fold)}, val={len(X_val_fold)}",
                flush=True,
            )

            theta_k, *_ = train_qae_combo_with_history(
                X_train_fold,
                lam=lam,
                epochs=epochs,
                batch_size=min(batch_size, len(X_train_fold)),
                encoding=encoding,
                p_r=p_r,
                ref_angle=ref_angle,
                seed=base_seed + k,
                lr=lr,
                fd_eps=fd_eps,
                steps_per_epoch=steps_per_epoch,
                alpha_l1=alpha,
            )

            # --- VALIDATION ON A SUBSET ---
            n_val = len(X_val_fold)
            n_eval = min(val_max_samples, n_val)
            sub_idx = rng.choice(n_val, n_eval, replace=False)
            X_val_eval = X_val_fold[sub_idx]

            L_t_val, L_r_val = qae_losses(
                theta_k,
                X_val_eval,
                encoding=encoding,
                p_r=p_r,
                ref_angle=ref_angle,
            )
            F_r_val = 1.0 - L_r_val
            fold_val_fids.append(F_r_val)

            print(f"    → F_recon_val (fold {k+1}, {n_eval} samples) = {F_r_val:.4f}", flush=True)

            pbar.update(1)

        mean_F = float(np.mean(fold_val_fids))
        std_F  = float(np.std(fold_val_fids))
        cv_results.append((alpha, mean_F, std_F))

        print(f"\nα = {alpha}: mean F_recon_val = {mean_F:.4f} ± {std_F:.4f}\n", flush=True)

    pbar.close()

    # pick best
    best_alpha, best_mean, best_std = max(cv_results, key=lambda x: x[1])

    print("\n=== CV summary ===")
    for alpha, mean_F, std_F in cv_results:
        print(f"alpha_l1={alpha:8g} → F_recon_val = {mean_F:.4f} ± {std_F:.4f}")
    print(
        f"\n>>> Best alpha_l1 = {best_alpha} "
        f"(mean F_recon_val = {best_mean:.4f} ± {best_std:.4f})"
    )

    return best_alpha, cv_results

alpha_grid = [
    0,
    5e-5,
    1e-4,
    2e-4,
    3e-4,
    5e-4,
    7e-4,
    1e-3,
    2e-3,
]
best_alpha1, cv_results1 = cross_validate_l1_qae(
    X_qae_train,
    lam=0.5,
    alpha_list=alpha_grid,
    k_folds=3,
    epochs=30,          # very small, just to see it go through
    batch_size=32,
    lr=0.01,
    fd_eps=1e-3,
    val_max_samples=500,
)

# ============================================================
# Example usage
# ============================================================


# 3) retrain final model on all X_qae_train with best_alpha
theta_best, steps, Ltot_hist, Lt_hist, Lr_hist, Ft_hist, Fr_hist = (
    train_qae_combo_with_history(
        X_qae_train,
        lam=0.5,
        epochs=200,          # longer final training
        batch_size=64,
        encoding="amplitude",
        p_r=None,
        ref_angle=None,
        seed=0,
        lr=0.001,
        fd_eps=1e-3,
        steps_per_epoch=1,
        alpha_l1=0.0001,
    )
)

plt.figure(figsize=(8, 5))
plt.plot(steps, Ltot_hist, label="Total loss", color='black', linewidth=2)
plt.plot(steps, Lt_hist, label="Trash loss", linestyle="--")
plt.plot(steps, Lr_hist, label="Recon loss", linestyle=":")
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

print("\nFinal model trained with alpha_l2 =", best_alpha)
print("Final train F_recon =", Fr_hist[-1])

Z = get_latent_Z_features(X_test_m, theta_best)
np.save("Z_test", Z)
np.save("theta", theta_best)

mixed_Z = np.concatenate((Z ,X_test_values.reshape(-1,1)), axis=1)


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

alphas = [t[0] for t in cv_results]
means  = [np.mean(t[1]) for t in cv_results]
stds   = [np.std(t[1]) for t in cv_results]

plt.figure(figsize=(9, 6))

# Mean validation loss curve
plt.plot(alphas, means, label="Mean CV loss", linestyle="--", linewidth=2)

# Std error bars
plt.errorbar(alphas, means, yerr=stds, fmt="o", capsize=5, markersize=8)

plt.xscale("log")
plt.xlabel("Regularization 1 α", fontsize=14)
plt.ylabel("Validation loss", fontsize=14)
plt.title("QAE Cross-Validation Loss vs α", fontsize=16)
plt.legend(fontsize=13)

# Tick sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
