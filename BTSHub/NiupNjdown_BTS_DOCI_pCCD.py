#!/usr/bin/env python3
"""

Raw opposite-spin density correlation for the 10-site half-filled Hubbard ring:

    C_ij = < n_{i,up} n_{j,down} >

This script computes THREE separate results at one user-selected U:
    1. UBTS
    2. DOCI in the frozen UBTS orbitals
    3. pCCD in the frozen UBTS-derived orbitals used by the existing pCCD run

It writes THREE separate text files, one for each method.

"""

import os
import re
import itertools
import numpy as np
from scipy.linalg import expm


# =============================================================================
# USER CONFIGURATION
# =============================================================================

HERE = os.path.dirname(os.path.abspath("/BTS/"))

N = 10
NPAIRS = 5
T_HOP = 1.0

# ------------------------------------------------------------
# Choose U value
# ------------------------------------------------------------
TARGET_U = 5.0

# Files from the REPULSIVE (+10 -> 0) UBTS/pCCD scans
REP_ETA_FILE = os.path.join(
    HERE, "EtaUnHubBTSreal_10x1_PBC.txt"
)
REP_U_FILE = os.path.join(
    HERE, "UmatrixUnHubBTSreal_10x1_PBC.txt"
)
REP_PCCD_T_FILE = os.path.join(
    HERE, "UPCCD_amplitudes_10.txt"
)
REP_PCCD_ENERGY_FILE = os.path.join(
    HERE, "UPCCD_from_BTS_10.txt"
)

# Files from the ATTRACTIVE (-10 -> 0) UBTS/pCCD scans
ATT_ETA_FILE = os.path.join(
    HERE, "EtaUnHubBTS2real_10x1_PBC.txt"
)
ATT_U_FILE = os.path.join(
    HERE, "UmatrixUnHubBTS2real_10x1_PBC.txt"
)
ATT_PCCD_T_FILE = os.path.join(
    HERE, "UPCCD_amplitudes_10at.txt"
)
ATT_PCCD_ENERGY_FILE = os.path.join(
    HERE, "UPCCD_from_BTS_10at.txt"
)

OUTPUT_DIR = os.path.join(HERE, "NiupNjdown_results")
U_MATCH_TOL = 1.0e-8


# =============================================================================
# 1. BASIC UTILITIES
# =============================================================================

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_BTS_STEP = re.compile(r"^Step\s+(\d+),\s*Gval\s+(" + _FLOAT + r")")
_PCCD_T_STEP = re.compile(
    r"^#\s*step\s+(\d+)\s+U\s*=\s*(" + _FLOAT + r")\s+t\[i,a\]"
)


def hubbard_1d_pbc_hopping(N, t=1.0):
    h = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        h[i, i + 1] = -t
        h[i + 1, i] = -t
    if N > 2:
        h[0, N - 1] = -t
        h[N - 1, 0] = -t
    return h


def _parse_julia_complex(token):
    token = token.strip()
    return complex(token.replace("im", "j"))


def _find_U_record(records, target_U, tol=U_MATCH_TOL):
    matches = [r for r in records if abs(float(r["U"]) - target_U) <= tol]
    if not matches:
        vals = [float(r["U"]) for r in records]
        raise ValueError(
            f"U={target_U} not found. Available U values are:\n{vals}"
        )
    return matches[0]


def _safe_U_tag(U):
    if abs(U) < 5.0e-13:
        return "0.000000"
    return ("p" if U > 0 else "m") + f"{abs(U):.6f}"


# =============================================================================
# 2. READ THE SAVED UBTS ORBITALS
# =============================================================================

def load_bts_U_matrices(filename, N):
    """
    Read blocks of the form

        Step k, Gval U:
        Energy = ...
        U_up:
        10x10 Matrix{ComplexF64}:
        U_down:
        10x10 Matrix{ComplexF64}:

    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    lines = open(filename, "r").read().splitlines()
    records = []
    i = 0

    while i < len(lines):
        m = _BTS_STEP.match(lines[i].strip())
        if not m:
            i += 1
            continue

        step = int(m.group(1))
        U = float(m.group(2))
        U_up = None
        U_down = None

        j = i + 1
        while j < len(lines) and not _BTS_STEP.match(lines[j].strip()):
            s = lines[j].strip()

            if s in ("U_up:", "U_down:"):
                label = s[:-1]
                k = j + 1

                while k < len(lines) and (
                    lines[k].strip() == "" or "Matrix" in lines[k]
                ):
                    k += 1

                rows = []
                for rr in range(N):
                    if k + rr >= len(lines):
                        raise ValueError(
                            f"Unexpected end of file while reading {label} "
                            f"at step {step}."
                        )
                    toks = lines[k + rr].split()
                    if len(toks) != N:
                        raise ValueError(
                            f"{label}, step {step}: expected {N} entries "
                            f"in row {rr}, got {len(toks)}."
                        )
                    rows.append([_parse_julia_complex(x) for x in toks])

                M = np.array(rows, dtype=complex)
                if label == "U_up":
                    U_up = M
                else:
                    U_down = M

                j = k + N
                continue

            j += 1

        if U_up is None or U_down is None:
            raise ValueError(
                f"Could not read both U_up and U_down at step {step}, U={U}."
            )

        records.append(
            dict(step=step, U=U, U_up=U_up, U_down=U_down)
        )
        i = j

    if not records:
        raise ValueError(f"No UBTS orbital blocks parsed from {filename}")

    return records


def raw_pair_orbitals_from_saved_U(U_up, U_down):
    """
    BTS convention used in the existing pCCD reader:
        C_alpha = U_up^T
        C_beta  = U_down^T

    """
    Ca = U_up.T.copy()
    Cb = U_down.T.copy()
    return Ca, Cb


# =============================================================================
# 3. READ THE SAVED BTS ETA PARAMETERS
# =============================================================================

def load_bts_eta_vectors(filename):
    """
    Read the full-precision eta_vector from every saved BTS step.

    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    lines = open(filename, "r").read().splitlines()
    records = []
    i = 0

    while i < len(lines):
        m = _BTS_STEP.match(lines[i].strip())
        if not m:
            i += 1
            continue

        step = int(m.group(1))
        U = float(m.group(2))
        eta_vec = None

        j = i + 1
        while j < len(lines) and not _BTS_STEP.match(lines[j].strip()):
            if lines[j].strip() == "eta_vector:":
                if j + 1 >= len(lines):
                    raise ValueError(
                        f"Missing eta_vector data at step {step}."
                    )
                s = lines[j + 1].strip()
                if not (s.startswith("[") and s.endswith("]")):
                    raise ValueError(
                        f"Could not parse eta_vector at step {step}."
                    )
                eta_vec = np.array(
                    [float(x.strip()) for x in s[1:-1].split(",")],
                    dtype=float,
                )
                break
            j += 1

        if eta_vec is None:
            raise ValueError(
                f"eta_vector not found at step {step}, U={U}."
            )

        records.append(dict(step=step, U=U, eta_vector=eta_vec))
        i = j + 1

    if not records:
        raise ValueError(f"No BTS eta blocks parsed from {filename}")

    return records


def bts_vector_to_matrix(eta_vector, m, n):
    """
    Reconstruct the n x m BTS eta matrix from the free vector.
    For one-based column p:
        row_min = max(1, p + n - m)
        row_max = min(p, n)
    """
    eta = np.zeros((n, m), dtype=float)
    q = 0

    for p1 in range(1, m + 1):
        row_min = max(1, p1 + n - m)
        row_max = min(p1, n)

        for k1 in range(row_min, row_max + 1):
            if q >= len(eta_vector):
                raise ValueError("eta_vector ended too early.")
            eta[k1 - 1, p1 - 1] = eta_vector[q]
            q += 1

    if q != len(eta_vector):
        raise ValueError(
            f"Used {q} eta parameters, but vector has {len(eta_vector)}."
        )

    return eta


# =============================================================================
# 4. READ THE SAVED pCCD t AMPLITUDES AND ENERGY/METHOD
# =============================================================================

def load_pccd_t_amplitudes(filename, nocc):
    """
    Read the existing pCCD amplitude file:

        # step k  U = ...  t[i,a]  (shape 5 x 5)
        row 1
        ...
        row 5
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    lines = open(filename, "r").read().splitlines()
    records = []
    i = 0

    while i < len(lines):
        m = _PCCD_T_STEP.match(lines[i].strip())
        if not m:
            i += 1
            continue

        step = int(m.group(1))
        U = float(m.group(2))
        rows = []

        for k in range(1, nocc + 1):
            if i + k >= len(lines):
                raise ValueError(
                    f"Unexpected EOF while reading pCCD t at U={U}."
                )
            vals = [float(x) for x in lines[i + k].split()]
            if len(vals) != nocc:
                raise ValueError(
                    f"Expected a {nocc}x{nocc} t matrix at U={U}; "
                    f"row has {len(vals)} entries."
                )
            rows.append(vals)

        records.append(
            dict(step=step, U=U, t=np.array(rows, dtype=float))
        )
        i += nocc + 1

    if not records:
        raise ValueError(f"No pCCD t blocks parsed from {filename}")

    return records


def load_pccd_energy_table(filename):
    """
    Read the pCCD energy file.  The last column is METHOD.

    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    records = []
    for line in open(filename, "r"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.split()
        if len(parts) < 10:
            continue

        try:
            step = int(parts[0])
            U = float(parts[1])
            energy = float(parts[2])
            e_ref = float(parts[3])
        except ValueError:
            continue

        method = parts[-1]
        records.append(
            dict(
                step=step,
                U=U,
                energy=energy,
                e_ref=e_ref,
                method=method,
            )
        )

    if not records:
        raise ValueError(f"No pCCD energy rows parsed from {filename}")

    return records


# =============================================================================
# 5. PAIR BASIS
# =============================================================================

def make_pair_basis(norb, npairs):
    basis = [tuple(c) for c in itertools.combinations(range(norb), npairs)]
    index = {cfg: k for k, cfg in enumerate(basis)}
    return basis, index


# =============================================================================
# 6. FROZEN-ORBITAL SENIORITY-ZERO HUBBARD HAMILTONIAN
# =============================================================================

def pair_hamiltonian_tensors_complex(h, U, Ca, Cb):

    ha = Ca.conj().T @ h @ Ca
    hb = Cb.conj().T @ h @ Cb

    A2 = np.abs(Ca) ** 2
    B2 = np.abs(Cb) ** 2
    AB = Ca * Cb

    J = U * (A2.T @ B2)
    L = U * (AB.conj().T @ AB)
    W = J + J.T

    eps = np.diag(ha) + np.diag(hb) + np.diag(J)

    return eps, J, L, W


def build_pair_hamiltonian(h, U, Ca, Cb, basis, index):
    """
    Matrix of the seniority-zero projected Hubbard Hamiltonian in the
    pair-determinant basis.
    """
    eps, J, L, W = pair_hamiltonian_tensors_complex(h, U, Ca, Cb)

    dim = len(basis)
    norb = Ca.shape[1]
    H = np.zeros((dim, dim), dtype=complex)

    for col, cfg in enumerate(basis):
        occ = set(cfg)

        # diagonal
        val = np.sum(eps[list(cfg)])
        for p, q in itertools.combinations(cfg, 2):
            val += W[p, q]
        H[col, col] = val

        # pair hopping: q -> p
        for q in cfg:
            for p in range(norb):
                if p in occ:
                    continue
                new_cfg = tuple(sorted((occ - {q}) | {p}))
                row = index[new_cfg]
                H[row, col] += L[p, q]

    return H


# =============================================================================
# 7. OPERATOR  n_{i up} n_{j down}  IN THE PAIR BASIS
# =============================================================================

def build_Niup_Njdown_operator(Ca, Cb, site_i, site_j, basis, index):

    dim = len(basis)
    norb = Ca.shape[1]
    O = np.zeros((dim, dim), dtype=complex)

    A = np.outer(np.conj(Ca[site_i, :]), Ca[site_i, :])
    B = np.outer(np.conj(Cb[site_j, :]), Cb[site_j, :])

    for col, cfg in enumerate(basis):
        occ = set(cfg)
        occ_list = list(cfg)

        # diagonal piece:
        # <S|n_i_up|S> * <S|n_j_down|S>
        ni = np.sum(np.abs(Ca[site_i, occ_list]) ** 2)
        nj = np.sum(np.abs(Cb[site_j, occ_list]) ** 2)
        O[col, col] = ni * nj

        # off-diagonal pair replacement q -> p
        for q in cfg:
            for p in range(norb):
                if p in occ:
                    continue
                new_cfg = tuple(sorted((occ - {q}) | {p}))
                row = index[new_cfg]
                O[row, col] += A[p, q] * B[p, q]

    return O


def correlation_matrix_hermitian_state(Ca, Cb, basis, index, psi):
    """
    C_ij = <psi|n_i_up n_j_down|psi>/<psi|psi>
    for BTS or DOCI.
    """
    norm = np.vdot(psi, psi)
    if abs(norm) < 1.0e-15:
        raise ValueError("Wavefunction has zero norm.")

    C = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            O = build_Niup_Njdown_operator(
                Ca, Cb, i, j, basis, index
            )
            C[i, j] = np.vdot(psi, O @ psi) / norm

    return np.real_if_close(C, tol=1000)


def correlation_matrix_biorthogonal(Ca, Cb, basis, index, left, right):
    """
    C_ij = <L|n_i_up n_j_down|R>
    for pCCD.

    """
    overlap = left @ right
    if abs(overlap - 1.0) > 1.0e-8:
        left = left / overlap

    C = np.zeros((N, N), dtype=complex)

    for i in range(N):
        for j in range(N):
            O = build_Niup_Njdown_operator(
                Ca, Cb, i, j, basis, index
            )
            C[i, j] = left @ (O @ right)

    return np.real_if_close(C, tol=1000)


def pbc_distance_average(C):

    Nloc = C.shape[0]
    out = np.zeros(Nloc, dtype=complex)

    for r in range(Nloc):
        out[r] = np.mean(
            [C[i, (i + r) % Nloc] for i in range(Nloc)]
        )

    return np.real_if_close(out, tol=1000)


# =============================================================================
# 8. BTS CORRELATION
# =============================================================================

def bts_coefficients(eta, basis):

    coeff = np.zeros(len(basis), dtype=complex)

    for kcfg, cfg in enumerate(basis):
        x = 1.0
        for k, p in enumerate(cfg):
            x *= eta[k, p]
        coeff[kcfg] = x

    return coeff


def calculate_BTS(U, h, eta_file, U_file, basis, index):
    eta_rec = _find_U_record(load_bts_eta_vectors(eta_file), U)
    orb_rec = _find_U_record(load_bts_U_matrices(U_file, N), U)

    eta = bts_vector_to_matrix(
        eta_rec["eta_vector"], m=N, n=NPAIRS
    )
    Ca, Cb = raw_pair_orbitals_from_saved_U(
        orb_rec["U_up"], orb_rec["U_down"]
    )

    psi = bts_coefficients(eta, basis)

    H = build_pair_hamiltonian(
        h, U, Ca, Cb, basis, index
    )
    E = np.vdot(psi, H @ psi) / np.vdot(psi, psi)

    C = correlation_matrix_hermitian_state(
        Ca, Cb, basis, index, psi
    )
    Cr = pbc_distance_average(C)

    return dict(
        method="BTS",
        energy=float(np.real(E)),
        C=C,
        Cr=Cr,
        Ca=Ca,
        Cb=Cb,
    )


# =============================================================================
# 9. DOCI CORRELATION IN THE SAME FROZEN BTS ORBITALS
# =============================================================================

def calculate_DOCI(U, h, U_file, basis, index):
    orb_rec = _find_U_record(load_bts_U_matrices(U_file, N), U)

    Ca, Cb = raw_pair_orbitals_from_saved_U(
        orb_rec["U_up"], orb_rec["U_down"]
    )

    H = build_pair_hamiltonian(
        h, U, Ca, Cb, basis, index
    )

    evals, evecs = np.linalg.eigh(H)
    E = float(np.real(evals[0]))
    psi = evecs[:, 0]

    C = correlation_matrix_hermitian_state(
        Ca, Cb, basis, index, psi
    )
    Cr = pbc_distance_average(C)

    return dict(
        method="DOCI",
        energy=E,
        C=C,
        Cr=Cr,
        Ca=Ca,
        Cb=Cb,
    )


# =============================================================================
# 10. pCCD ORBITAL PREPROCESSING -- EXACTLY MATCH THE OLD pCCD RUN
# =============================================================================

def pccd_preprocess_bts_orbitals(U_up, U_down):

    Ca = np.real(U_up.T)
    Cb = np.real(U_down.T)

    Ca = np.linalg.qr(Ca)[0]
    Cb = np.linalg.qr(Cb)[0]

    return Ca, Cb


def pccd_pair_energy_order(h, U, Ca, Cb):

    J = U * ((Ca ** 2).T @ (Cb ** 2))
    eps = (
        np.diag(Ca.T @ h @ Ca)
        + np.diag(Cb.T @ h @ Cb)
        + np.diag(J)
    )
    return np.argsort(eps)


def pccd_orbitals_in_used_order(h, U, U_up, U_down, method):

    Ca, Cb = pccd_preprocess_bts_orbitals(U_up, U_down)

    if str(method).endswith("*"):
        order = np.arange(N, dtype=int)
    else:
        order = pccd_pair_energy_order(h, U, Ca, Cb)

    return Ca[:, order], Cb[:, order], order


# =============================================================================
# 11. pCCD T OPERATOR AND LAMBDA STATE 
# =============================================================================

def build_pccd_T_matrix(t, basis, index, nocc):

    dim = len(basis)
    norb = 2 * nocc
    Tmat = np.zeros((dim, dim), dtype=complex)

    for col, cfg in enumerate(basis):
        occ = set(cfg)

        for i in range(nocc):
            if i not in occ:
                continue

            for av, a in enumerate(range(nocc, norb)):
                if a in occ:
                    continue

                new_cfg = tuple(sorted((occ - {i}) | {a}))
                row = index[new_cfg]
                Tmat[row, col] += t[i, av]

    return Tmat


def pccd_left_and_right_states(H, t, basis, index, nocc):

    ref_cfg = tuple(range(nocc))
    ref = index[ref_cfg]

    Tmat = build_pccd_T_matrix(
        t, basis, index, nocc
    )

    expT = expm(Tmat)
    expmT = expm(-Tmat)
    Hbar = expmT @ H @ expT

    Ecc = Hbar[ref, ref]

    # Pair-single excitations of the reference, ordered exactly as t[i,a].
    singles = []
    for i in range(nocc):
        for a in range(nocc, 2 * nocc):
            cfg = tuple(
                sorted((set(ref_cfg) - {i}) | {a})
            )
            singles.append(index[cfg])

    singles = np.array(singles, dtype=int)


    right_resid = np.max(
        np.abs(Hbar[singles, ref])
    )

    # Lambda equation:
    Hss = Hbar[np.ix_(singles, singles)]
    rhs = -Hbar[ref, singles]
    A = Hss.T - Ecc * np.eye(len(singles), dtype=complex)
    lam = np.linalg.solve(A, rhs)

    left_residual_vector = (
        Hbar[ref, singles]
        + lam @ Hss
        - Ecc * lam
    )
    left_resid = np.max(np.abs(left_residual_vector))

    # similarity-transformed representation
    lbar = np.zeros(len(basis), dtype=complex)
    lbar[ref] = 1.0
    lbar[singles] = lam

    # Physical biorthogonal left/right states
    right = expT[:, ref]
    left = lbar @ expmT

    overlap = left @ right

    return dict(
        energy=Ecc,
        right=right,
        left=left,
        lambda_vector=lam,
        right_residual=float(right_resid),
        left_residual=float(left_resid),
        overlap=overlap,
    )


# =============================================================================
# 12. pCCD CORRELATION
# =============================================================================

def calculate_pCCD(
    U, h, U_file, pccd_t_file, pccd_energy_file, basis, index
):
    orb_rec = _find_U_record(
        load_bts_U_matrices(U_file, N), U
    )
    t_rec = _find_U_record(
        load_pccd_t_amplitudes(pccd_t_file, NPAIRS), U
    )
    e_rec = _find_U_record(
        load_pccd_energy_table(pccd_energy_file), U
    )

    Ca, Cb, order = pccd_orbitals_in_used_order(
        h,
        U,
        orb_rec["U_up"],
        orb_rec["U_down"],
        e_rec["method"],
    )

    H = build_pair_hamiltonian(
        h, U, Ca, Cb, basis, index
    )

    cc = pccd_left_and_right_states(
        H,
        t_rec["t"],
        basis,
        index,
        NPAIRS,
    )

    Ecc = float(np.real(cc["energy"]))
    E_saved = float(e_rec["energy"])
    energy_error = abs(Ecc - E_saved)

    if energy_error > 1.0e-7:
        raise RuntimeError(
            "Reconstructed pCCD energy does not match saved pCCD energy:\n"
            f"  reconstructed = {Ecc:.12f}\n"
            f"  saved         = {E_saved:.12f}\n"
            f"  |difference|  = {energy_error:.3e}\n"
            "Check the orbital file, amplitude file, and energy file all "
            "belong to the same pCCD run."
        )

    if cc["right_residual"] > 1.0e-7:
        raise RuntimeError(
            f"pCCD right residual in pair-space reconstruction is too large: "
            f"{cc['right_residual']:.3e}"
        )

    if cc["left_residual"] > 1.0e-9:
        raise RuntimeError(
            f"pCCD Lambda residual is too large: "
            f"{cc['left_residual']:.3e}"
        )

    C = correlation_matrix_biorthogonal(
        Ca,
        Cb,
        basis,
        index,
        cc["left"],
        cc["right"],
    )
    Cr = pbc_distance_average(C)

    return dict(
        method="pCCD",
        energy=Ecc,
        saved_energy=E_saved,
        energy_check=energy_error,
        C=C,
        Cr=Cr,
        Ca=Ca,
        Cb=Cb,
        order=order,
        lambda_vector=cc["lambda_vector"],
        right_residual=cc["right_residual"],
        left_residual=cc["left_residual"],
        overlap=cc["overlap"],
    )


# =============================================================================
# 13. OUTPUT
# =============================================================================

def _as_real_array(x, name, tol=1.0e-8):
    x = np.asarray(x)
    if np.iscomplexobj(x):
        imag = np.max(np.abs(np.imag(x)))
        if imag > tol:
            raise RuntimeError(
                f"{name} has an unexpectedly large imaginary part: {imag:.3e}"
            )
        x = np.real(x)
    return np.asarray(x, dtype=float)


def write_correlation_file(filename, result, U):
    """
    Each method gets one independent text file containing
    """
    C = _as_real_array(result["C"], "C_ij")
    Cr = _as_real_array(result["Cr"], "C(r)")

    with open(filename, "w") as f:
        f.write("# Raw opposite-spin density correlation\n")
        f.write("#\n")
        f.write("# C_ij = < n_{i,up} n_{j,down} >\n")
        f.write(
            "# C_updown(r) = (1/N) sum_i "
            "< n_{i,up} n_{i+r,down} >, PBC\n"
        )
        f.write(f"# method = {result['method']}\n")
        f.write(f"# N = {N}, npairs = {NPAIRS}, PBC\n")
        f.write(f"# U/t = {U:.12f}\n")
        f.write(f"# energy/t = {result['energy']:.15f}\n")

        if result["method"] == "pCCD":
            f.write(
                f"# saved pCCD energy/t = "
                f"{result['saved_energy']:.15f}\n"
            )
            f.write(
                f"# pCCD energy reconstruction error = "
                f"{result['energy_check']:.6e}\n"
            )
            f.write(
                f"# right pCCD residual = "
                f"{result['right_residual']:.6e}\n"
            )
            f.write(
                f"# Lambda residual = "
                f"{result['left_residual']:.6e}\n"
            )
            f.write(
                f"# <L|R> = "
                f"{complex(result['overlap']).real:.15f}\n"
            )

        f.write("\n")
        f.write("# ---------------------------------------------------------\n")
        f.write("# PBC DISTANCE-AVERAGED RAW CORRELATION\n")
        f.write("# r    C_updown(r)\n")
        f.write("# ---------------------------------------------------------\n")
        for r, val in enumerate(Cr):
            f.write(f"{r:3d}  {val:.15e}\n")

        f.write("\n")
        f.write("# ---------------------------------------------------------\n")
        f.write("# FULL C_ij MATRIX\n")
        f.write("# rows    = up-spin site i = 1,...,10\n")
        f.write("# columns = down-spin site j = 1,...,10\n")
        f.write("# ---------------------------------------------------------\n")
        f.write(
            "# i\\j  " + "  ".join(f"{j:>18d}" for j in range(1, N + 1))
            + "\n"
        )

        for i in range(N):
            f.write(
                f"{i+1:4d}  "
                + "  ".join(f"{C[i,j]: .15e}" for j in range(N))
                + "\n"
            )

        # exact-number sum rule:
        f.write("\n")
        f.write(
            "# sum_r C_updown(r) = "
            f"{np.sum(Cr):.15e}\n"
        )
        f.write(
            "# expected fixed-number sum rule = "
            f"{(NPAIRS * NPAIRS / N):.15e}\n"
        )


# =============================================================================
# 14. MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if TARGET_U >= 0.0:
        eta_file = REP_ETA_FILE
        U_file = REP_U_FILE
        pccd_t_file = REP_PCCD_T_FILE
        pccd_energy_file = REP_PCCD_ENERGY_FILE
        branch = "repulsive"
    else:
        eta_file = ATT_ETA_FILE
        U_file = ATT_U_FILE
        pccd_t_file = ATT_PCCD_T_FILE
        pccd_energy_file = ATT_PCCD_ENERGY_FILE
        branch = "attractive"

    print("=" * 78)
    print("10-site PBC raw <n_i_up n_j_down> correlation")
    print(f"TARGET U/t = {TARGET_U}")
    print(f"branch     = {branch}")
    print("=" * 78)

    h = hubbard_1d_pbc_hopping(N, T_HOP)
    basis, index = make_pair_basis(N, NPAIRS)

    print(f"Pair basis dimension = C({N},{NPAIRS}) = {len(basis)}")

    # -------------------------------------------------------------------------
    # BTS
    # -------------------------------------------------------------------------
    print("\n[1/3] BTS ...")
    bts = calculate_BTS(
        TARGET_U, h, eta_file, U_file, basis, index
    )
    print(f"  E_BTS  = {bts['energy']:.15f}")

    # -------------------------------------------------------------------------
    # DOCI on frozen BTS orbitals
    # -------------------------------------------------------------------------
    print("\n[2/3] DOCI on frozen BTS orbitals ...")
    doci = calculate_DOCI(
        TARGET_U, h, U_file, basis, index
    )
    print(f"  E_DOCI = {doci['energy']:.15f}")

    # -------------------------------------------------------------------------
    # pCCD with Lambda
    # -------------------------------------------------------------------------
    print("\n[3/3] pCCD + Lambda ...")
    pccd = calculate_pCCD(
        TARGET_U,
        h,
        U_file,
        pccd_t_file,
        pccd_energy_file,
        basis,
        index,
    )
    print(f"  E_pCCD = {pccd['energy']:.15f}")
    print(
        f"  saved-energy check = {pccd['energy_check']:.3e}"
    )
    print(
        f"  right residual      = {pccd['right_residual']:.3e}"
    )
    print(
        f"  Lambda residual     = {pccd['left_residual']:.3e}"
    )

    # -------------------------------------------------------------------------
    # Write three independent files
    # -------------------------------------------------------------------------
    tag = _safe_U_tag(TARGET_U)

    bts_file = os.path.join(
        OUTPUT_DIR, f"NiupNjdown_BTS_U_{tag}real.txt"
    )
    doci_file = os.path.join(
        OUTPUT_DIR, f"NiupNjdown_DOCI_U_{tag}real.txt"
    )
    pccd_file = os.path.join(
        OUTPUT_DIR, f"NiupNjdown_pCCD_U_{tag}real.txt"
    )

    write_correlation_file(bts_file, bts, TARGET_U)
    write_correlation_file(doci_file, doci, TARGET_U)
    write_correlation_file(pccd_file, pccd, TARGET_U)

    print("\nSaved:")
    print(" ", bts_file)
    print(" ", doci_file)
    print(" ", pccd_file)

    # Print C(r) to screen as a quick sanity check
    print("\nRaw PBC C_updown(r):")
    print(" r          BTS                 DOCI                pCCD")
    for r in range(N):
        vb = float(np.real(bts["Cr"][r]))
        vd = float(np.real(doci["Cr"][r]))
        vp = float(np.real(pccd["Cr"][r]))
        print(f"{r:2d}  {vb: .12e}  {vd: .12e}  {vp: .12e}")

    print("\nNumber sum-rule checks:")
    target = NPAIRS * NPAIRS / N
    print(f" expected sum_r C(r) = {target:.12f}")
    print(
        f" BTS  = {np.sum(_as_real_array(bts['Cr'], 'BTS Cr')):.12f}"
    )
    print(
        f" DOCI = {np.sum(_as_real_array(doci['Cr'], 'DOCI Cr')):.12f}"
    )
    print(
        f" pCCD = {np.sum(_as_real_array(pccd['Cr'], 'pCCD Cr')):.12f}"
    )


if __name__ == "__main__":
    main()
