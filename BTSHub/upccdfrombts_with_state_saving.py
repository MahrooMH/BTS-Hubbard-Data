"""
================================================================================
 Unrestricted pair Coupled-Cluster Doubles (U-pCCD) from BTS orbitals
 -- frozen orbitals: solve ONLY the t amplitudes, report energies.
================================================================================

"""

import os
import re
import numpy as np
from scipy.optimize import root

HERE = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
#  1. LATTICE 
# ==============================================================================

def hubbard_2d_hopping(Lx, Ly, t=1.0, periodic_x=False, periodic_y=False):
    """m-by-n Hubbard hopping matrix.  Site index (0-based): x*Ly + y."""
    n = Lx * Ly
    h = np.zeros((n, n))
    tt = -float(t)

    def site(x, y):
        return x * Ly + y

    for x in range(Lx):
        for y in range(Ly):
            i = site(x, y)
            if x + 1 < Lx:
                j = site(x + 1, y); h[i, j] = tt; h[j, i] = tt
            elif periodic_x and Lx > 2:
                j = site(0, y);     h[i, j] = tt; h[j, i] = tt
            if y + 1 < Ly:
                j = site(x, y + 1); h[i, j] = tt; h[j, i] = tt
            elif periodic_y and Ly > 2:
                j = site(x, 0);     h[i, j] = tt; h[j, i] = tt
    return h


def hubbard_1d_hopping(L, t=1.0, periodic=False):
    """1D chain hopping matrix."""
    h = np.zeros((L, L))
    tt = -float(t)
    for i in range(L - 1):
        h[i, i + 1] = tt; h[i + 1, i] = tt
    if periodic and L > 2:
        h[0, L - 1] = tt; h[L - 1, 0] = tt
    return h


# ==============================================================================
#  2. BTS FILE PARSER  
# ==============================================================================
_FLOAT = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'  
_CPLX = re.compile(rf'^({_FLOAT})({_FLOAT})im$')
_STEP = re.compile(r'^Step\s+(\d+),\s*Gval\s+(' + _FLOAT + r')')


def _parse_complex_token(tok):
    tok = tok.strip()
    try:                                   
        return complex(tok.replace('im', 'j'))
    except ValueError:
        pass
    m = _CPLX.match(tok)
    if m:
        return complex(float(m.group(1)), float(m.group(2)))
    if tok.endswith('im'):
        return complex(0.0, float(tok[:-2]))
    return complex(float(tok), 0.0)


def _tokens_to_complex(tokens):
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if i + 2 < len(tokens) and tokens[i + 1] in ('+', '-') and tokens[i + 2].endswith('im'):
            sgn = 1.0 if tokens[i + 1] == '+' else -1.0
            out.append(complex(float(tok), sgn * float(tokens[i + 2][:-2]))); i += 3; continue
        out.append(_parse_complex_token(tok)); i += 1
    return out


def _parse_singleline_matrix(s, N):
    s = s.strip()
    s = re.sub(r'^[A-Za-z0-9{},]+\[', '', s)      # strip "ComplexF64["
    s = re.sub(r'\]$', '', s).strip()             # strip trailing "]"
    rows = re.split(r'\s*;\s*', s)
    entries = []
    for rs in rows:
        entries += _tokens_to_complex(rs.split())
    if len(entries) != N * N:
        raise ValueError(f"Expected {N*N} entries for {N}x{N}, got {len(entries)}. "
                         f"Check N matches the lattice size.")
    return np.array(entries).reshape(N, N)        


def _parse_grid_matrix(lines, start, N):
    i = start
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    if i < len(lines) and 'Matrix' in lines[i]:   
        i += 1
    rows = []
    while i < len(lines) and len(rows) < N:
        line = lines[i]
        if line.strip() == '':
            i += 1; continue
        if (':' in line) or ('=' in line) or ('[' in line) or ('Matrix' in line):
            break
        if any(ch in line for ch in '…⋯⋮'):
            raise ValueError("Julia truncated the matrix on write (found an ellipsis). "
                             "Re-save with show(IOContext(io,:limit=>false),\"text/plain\",M).")
        toks = line.split()
        if len(toks) != N:
            raise ValueError(f"Matrix row has {len(toks)} entries, expected {N}:\n  {line[:70]}")
        rows.append([_parse_complex_token(t) for t in toks])
        i += 1
    if len(rows) != N:
        raise ValueError(f"Expected {N} matrix rows, parsed {len(rows)}.")
    return np.array(rows, dtype=complex), i


def load_bts_U_matrices(filename, N, transpose_U=True):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    lines = open(filename).read().splitlines()
    results = []
    i = 0
    while i < len(lines):
        m = _STEP.match(lines[i].strip())
        if not m:
            i += 1; continue
        step = int(m.group(1)); U_value = float(m.group(2))
        U_up = U_dn = None
        j = i + 1
        while j < len(lines) and not _STEP.match(lines[j].strip()):
            l = lines[j].strip()
            if l.startswith('U_up:') or l.startswith('U_down:'):
                label, _, rest = l.partition(':')
                if rest.strip():                       # layout (A): matrix on same line
                    M = _parse_singleline_matrix(rest.strip(), N); nxt = j + 1
                else:                                  # layout (B): grid on following lines
                    M, nxt = _parse_grid_matrix(lines, j + 1, N)
                if label.strip() == 'U_up':
                    U_up = M
                else:
                    U_dn = M
                j = nxt; continue
            j += 1                                     # ignore 'Energy = ...' etc.
        if U_up is None or U_dn is None:
            print(f"  [warn] could not find U_up/U_down for step {step} (Gval={U_value}); skipping")
            i = j; continue
        Ca = np.real(U_up.T) if transpose_U else np.real(U_up)
        Cb = np.real(U_dn.T) if transpose_U else np.real(U_dn)
        # re-orthonormalise (QR); orbital signs are irrelevant to pCCD energies.
        Ca = np.linalg.qr(Ca)[0]
        Cb = np.linalg.qr(Cb)[0]
        results.append(dict(step=step, U_value=U_value, Ca=Ca, Cb=Cb))
        i = j
    if not results:
        raise ValueError(f"Parsed 0 valid BTS steps from {filename} -- check the file format.")
    print(f"Loaded {len(results)} BTS steps from {os.path.basename(filename)}")
    return results


# ==============================================================================
#  3. UNRESTRICTED pair (seniority-zero) integrals from Cα, Cβ
# ==============================================================================

def build_pair_integrals(h, U, Ca, Cb, nocc):
    """
    Build everything the pCCD t-equation needs, from unrestricted orbitals.
    
    """
    norb = h.shape[0]
    ha = Ca.T @ h @ Ca
    hb = Cb.T @ h @ Cb
    epsa = np.diag(ha).copy()
    epsb = np.diag(hb).copy()

    A2 = Ca ** 2
    B2 = Cb ** 2
    AB = Ca * Cb
    J = U * (A2.T @ B2)      
    L = U * (AB.T @ AB)      
    W = J + J.T             
    eps = epsa + epsb + np.diag(J)   

    o = np.arange(nocc)
    v = np.arange(nocc, norb)
    oi, ai = o[:, None], v[None, :]

    S = dict(
        viiaa=L[oi, ai],                   
        vaaii=L[ai, oi],                    
        viaia=0.5 * W[oi, ai],             
        voo=L[o[:, None], o[None, :]],      
        vvv=L[v[:, None], v[None, :]],     
    )

    # Fock diagonal 
    fdiag = 0.5 * eps - 0.25 * np.diag(W) + 0.5 * W[:, o].sum(axis=1)

    Woo = W[np.ix_(o, o)]
    e_ref = eps[o].sum() + 0.5 * (Woo.sum() - np.trace(Woo))

    return S, fdiag, e_ref


# ==============================================================================
#  4. DIIS accelerator (Pulay) 
# ==============================================================================

class DIIS:
    def __init__(self, max_vec=8):
        self.max_vec = max_vec
        self.params, self.errors = [], []

    def update(self, param, error):
        self.params.append(param.copy()); self.errors.append(error.copy())
        if len(self.params) > self.max_vec:
            self.params.pop(0); self.errors.pop(0)
        nv = len(self.params)
        if nv == 1:
            return param
        B = np.zeros((nv + 1, nv + 1))
        B[-1, :] = B[:, -1] = -1.0
        B[-1, -1] = 0.0
        for i in range(nv):
            for jj in range(nv):
                B[i, jj] = np.dot(self.errors[i].ravel(), self.errors[jj].ravel())
        rhs = np.zeros(nv + 1); rhs[-1] = -1.0
        try:
            c = np.linalg.solve(B, rhs)[:-1]
        except np.linalg.LinAlgError:
            return param
        out = np.zeros_like(param)
        for i in range(nv):
            out += c[i] * self.params[i]
        return out


# ==============================================================================
#  5. pCCD t-amplitude residual, energy, and solver
# ==============================================================================

def _residual(t, S):
    """pCCD t residual (your Eq. 12b), returned flat.  t shape (nocc, nvirt)."""
    viiaa, vaaii = S['viiaa'], S['vaaii']
    viaia, viaai = S['viaia'], S['viaai']
    voo, vvv = S['voo'], S['vvv']
    s1 = np.einsum('ja,ja->a', viiaa, t)
    s2 = np.einsum('ib,ib->i', viiaa, t)
    B = S['_fdiff'] - s1[None, :] - s2[:, None]
    M = np.einsum('jb,ib->ji', viiaa, t)
    R = (vaaii
         + 2.0 * B * t
         - 2.0 * (2.0 * viaia - viaai - viiaa * t) * t
         + np.einsum('ab,ib->ia', vvv, t)
         + np.einsum('ji,ja->ia', voo, t)
         + np.einsum('ja,ji->ia', t, M))
    return R


def solve_t(S, fdiag, nocc, norb, t0=None, tol=1e-11, maxit=4000,
            denom_shift=1e-6, damp=0.0):
    """
    Solve the pCCD t amplitudes at fixed orbitals.

    """
    nvirt = norb - nocc
    fo, fv = fdiag[:nocc], fdiag[nocc:]
    fdiff = fv[None, :] - fo[:, None]
    denom = 2.0 * fdiff
    denom = np.where(np.abs(denom) < denom_shift,
                     np.sign(denom + 1e-30) * denom_shift, denom)
    S = dict(S); S['_fdiff'] = fdiff 

    t = (-S['vaaii'] / denom) if t0 is None else t0.copy()

    diis = DIIS(max_vec=6)
    converged = False
    resid = np.inf
    it = 0
    for it in range(maxit):
        R = _residual(t, S)
        resid = np.max(np.abs(R))
        if not np.isfinite(resid):
            break
        if resid < tol:
            converged = True
            break
        tnew = t - R / denom
        if damp:
            tnew = (1.0 - damp) * tnew + damp * t
        if it > 10:
            tnew = diis.update(tnew, tnew - t)
        t = tnew

    if converged:
        return t, it, resid, True, "diis"

    # ---- Newton fallback ----
    g0 = (-S['vaaii'] / denom).ravel() if t0 is None else t0.ravel()

    def f_flat(x):
        return _residual(x.reshape(nocc, nvirt), S).ravel()

    sol = root(f_flat, g0, method='hybr', tol=1e-12, options=dict(maxfev=20000))
    t = sol.x.reshape(nocc, nvirt)
    resid = np.max(np.abs(f_flat(t.ravel())))
    return t, sol.nfev, resid, (resid < 1e-8), "newton"


def pccd_energy(S, e_ref, t):
    """E = E_ref + sum_{ia} t_i^a V_{ia}."""
    return e_ref + np.sum(t * S['viiaa'])


# ==============================================================================
#  5.5 save everything needed for Lambda/RDM work later
# ==============================================================================

def _pair_energy_order_for_saving(h, U, Ca, Cb):
    J = U * ((Ca ** 2).T @ (Cb ** 2))
    eps = (np.diag(Ca.T @ h @ Ca)
           + np.diag(Cb.T @ h @ Cb)
           + np.diag(J))
    return np.argsort(eps)


def _pccd_used_orbitals_for_saving(h, U, Ca, Cb, reorder_by_energy, method):
    N = Ca.shape[1]
    identity = np.arange(N, dtype=int)
    energy_order = _pair_energy_order_for_saving(h, U, Ca, Cb)
    fallback_used = str(method).endswith("*")

    if reorder_by_energy:
        order_used = identity if fallback_used else energy_order
    else:
        order_used = energy_order if fallback_used else identity

    return Ca[:, order_used].copy(), Cb[:, order_used].copy(), order_used


def _full_pair_tensors_for_saving(h, U, Ca, Cb):
    ha = Ca.T @ h @ Ca
    hb = Cb.T @ h @ Cb
    epsa = np.diag(ha).copy()
    epsb = np.diag(hb).copy()

    A2 = Ca ** 2
    B2 = Cb ** 2
    AB = Ca * Cb

    J = U * (A2.T @ B2)
    L = U * (AB.T @ AB)
    W = J + J.T
    pair_eps = epsa + epsb + np.diag(J)

    return pair_eps, J, L, W


def _u_filename_tag(U):
    return ("p" if U >= 0 else "m") + f"{abs(float(U)):.6f}"


def save_pccd_state_for_later_rdm(
        state_dir, h, step, U, Ca_input, Cb_input, nocc, res,
        reorder_by_energy=True, transpose_U=True):

    os.makedirs(state_dir, exist_ok=True)

    Ca_used, Cb_used, order_used = _pccd_used_orbitals_for_saving(
        h, U, Ca_input, Cb_input, reorder_by_energy, res['method']
    )

    S, fdiag, e_ref_check = build_pair_integrals(
        h, U, Ca_used, Cb_used, nocc
    )
    pair_eps, J, L, W = _full_pair_tensors_for_saving(
        h, U, Ca_used, Cb_used
    )

    if not np.isclose(e_ref_check, res['e_ref'], rtol=1e-10, atol=1e-10):
        print(
            f"  [warn] saved-state E_ref check differs at U={U}: "
            f"{e_ref_check:.12e} vs {res['e_ref']:.12e}"
        )

    fn = os.path.join(
        state_dir,
        f"UPCCD_state_step{int(step):03d}_U_{_u_filename_tag(U)}.npz"
    )

    np.savez_compressed(
        fn,

        # Identification / metadata
        step=np.array(int(step)),
        U=np.array(float(U)),
        nocc=np.array(int(nocc)),
        norb=np.array(int(h.shape[0])),
        reorder_by_energy=np.array(bool(reorder_by_energy)),
        transpose_U=np.array(bool(transpose_U)),
        method=np.array(str(res['method'])),
        converged=np.array(bool(res['converged'])),
        iters=np.array(int(res['iters'])),
        resid=np.array(float(res['resid'])),

        # Energies
        energy=np.array(float(res['energy'])),
        e_ref=np.array(float(res['e_ref'])),

        # RIGHT pCCD state 
        t=np.asarray(res['t']).copy(),

        Ca_bts_input=np.asarray(Ca_input).copy(),
        Cb_bts_input=np.asarray(Cb_input).copy(),

        order_used=np.asarray(order_used, dtype=int),
        Ca_used=np.asarray(Ca_used).copy(),
        Cb_used=np.asarray(Cb_used).copy(),

        h=np.asarray(h).copy(),

        # Full pair Hamiltonian
        pair_eps=np.asarray(pair_eps).copy(),
        J=np.asarray(J).copy(),
        L=np.asarray(L).copy(),
        W=np.asarray(W).copy(),

        fdiag=np.asarray(fdiag).copy(),
        viiaa=np.asarray(S['viiaa']).copy(),
        vaaii=np.asarray(S['vaaii']).copy(),
        viaia=np.asarray(S['viaia']).copy(),
        viaai=np.asarray(S['viaai']).copy(),
        voo=np.asarray(S['voo']).copy(),
        vvv=np.asarray(S['vvv']).copy(),
    )

    return fn


# ==============================================================================
#  6. Solve one BTS step (fixed orbitals)
# ==============================================================================

def upccd_fixed_orbitals(h, U, Ca, Cb, nocc, t0=None,
                         reorder_by_energy=True, tol=1e-11):
    """
    Frozen-orbital U-pCCD for one set of (Ca, Cb).
    """
    norb = h.shape[0]

    def _order(Ca, Cb):
        J = U * ((Ca ** 2).T @ (Cb ** 2))
        eps = np.diag(Ca.T @ h @ Ca) + np.diag(Cb.T @ h @ Cb) + np.diag(J)
        return np.argsort(eps)

    def _run(Ca, Cb, t0):
        S, fd, er = build_pair_integrals(h, U, Ca, Cb, nocc)
        t, it, r, ok, how = solve_t(S, fd, nocc, norb, t0=t0, tol=tol)
        E = pccd_energy(S, er, t)
        return E, t, it, r, ok, how, er

    if reorder_by_energy:
        p = _order(Ca, Cb); CaO, CbO = Ca[:, p], Cb[:, p]
    else:
        CaO, CbO = Ca, Cb

    E, t, it, r, ok, how, er = _run(CaO, CbO, t0)
    if not ok:
        # retry with the opposite ordering, from a cold MP2 guess
        if reorder_by_energy:
            CaO, CbO = Ca, Cb
        else:
            p = _order(Ca, Cb); CaO, CbO = Ca[:, p], Cb[:, p]
        E2, t2, it2, r2, ok2, how2, er2 = _run(CaO, CbO, None)
        if ok2 or r2 < r:
            E, t, it, r, ok, how, er = E2, t2, it2, r2, ok2, how2 + "*", er2

    return dict(energy=E, e_ref=er, t=t, iters=it, resid=r,
                converged=ok, method=how)


# ==============================================================================
#  7. Driver: read BTS file, scan steps, write energies (flushed)
# ==============================================================================

def run(config):
    h = config['h']
    N = h.shape[0]
    nocc = config['npairs']
    if not (0 < nocc < N):
        raise ValueError("npairs must satisfy 0 < npairs < number of sites")

    steps = load_bts_U_matrices(config['U_file'], N,
                                transpose_U=config.get('transpose_U', True))

    reorder = config.get('reorder_by_energy', True)
    continuation = config.get('continuation', True)
    tol = config.get('tol', 1e-11)
    save_amps = config.get('amplitudes_file', None)

    state_dir = config.get('states_dir', None)
    state_index_file = config.get('states_index_file', None)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    fE = open(config['energies_file'], 'w')
    header = (f"# U-pCCD from BTS orbitals (frozen orbitals; t amplitudes only)\n"
              f"# source: {config['U_file']}\n"
              f"# N={N} sites  npairs={nocc} ({2*nocc} electrons)  "
              f"reorder_by_energy={reorder}\n")
    fE.write(header)
    fE.write("# STEP      U_VALUE            E_UPCCD                 E_REF"
             "               E_CORR         MAXABS_T     |RESID|   ITERS  CONV  METHOD\n")
    fE.flush()

    fT = open(save_amps, 'w') if save_amps else None
    if fT:
        fT.write(header); fT.flush()

    fState = open(state_index_file, 'w') if state_index_file else None
    if fState:
        fState.write("# STEP  U_VALUE  STATE_FILE\n")
        fState.flush()

    t_guess = None
    print()
    for st in steps:
        step, U = st['step'], st['U_value']
        res = upccd_fixed_orbitals(h, U, st['Ca'], st['Cb'], nocc,
                                   t0=t_guess, reorder_by_energy=reorder, tol=tol)
        e_corr = res['energy'] - res['e_ref']
        maxt = np.max(np.abs(res['t'])) if res['t'].size else 0.0

        print(f"Step {step:3d} | U = {U:7.4f} | E_pCCD = {res['energy']:16.10f} | "
              f"E_corr = {e_corr:13.8f} | max|t| = {maxt:.4f} | "
              f"resid = {res['resid']:.1e} | {res['method']:7s} | conv={res['converged']}")

        fE.write(f"{step:6d}  {U:12.6f}  {res['energy']:22.12f}  {res['e_ref']:20.12f}  "
                 f"{e_corr:16.10f}  {maxt:11.6f}  {res['resid']:.2e}  {res['iters']:5d}   "
                 f"{int(res['converged'])}   {res['method']}\n")
        fE.flush()

        if fT:
            fT.write(f"\n# step {step}  U = {U:.6f}  t[i,a]  (shape {nocc} x {N-nocc})\n")
            np.savetxt(fT, res['t'], fmt="% .12e"); fT.flush()

        if state_dir:
            state_path = save_pccd_state_for_later_rdm(
                state_dir=state_dir,
                h=h,
                step=step,
                U=U,
                Ca_input=st['Ca'],
                Cb_input=st['Cb'],
                nocc=nocc,
                res=res,
                reorder_by_energy=reorder,
                transpose_U=config.get('transpose_U', True),
            )
            if fState:
                fState.write(f"{step:6d}  {U:12.6f}  {state_path}\n")
                fState.flush()

        if continuation and res['converged']:
            t_guess = res['t']

    fE.close()
    if fT:
        fT.close()
    if fState:
        fState.close()

    print(f"\nEnergies written to: {config['energies_file']}")
    if save_amps:
        print(f"Amplitudes written to: {save_amps}")
    if state_dir:
        print(f"Complete pCCD states written to: {state_dir}")
    if state_index_file:
        print(f"State index written to: {state_index_file}")


# ==============================================================================
#  8. CONFIG 
# ==============================================================================

if __name__ == "__main__":

    # ---- lattice ----
    Lx, Ly = 3, 2         
    t_hop = 1.0
    periodic_x = False    
    periodic_y = False

    h = hubbard_2d_hopping(Lx, Ly, t_hop, periodic_x=periodic_x, periodic_y=periodic_y)
    # 1D example instead:
    # h = hubbard_1d_hopping(6, t_hop, periodic=False)

    config = {
        'h': h,
        'npairs': 3,                 

        # ---- BTS input file ----
        'U_file': "UmatrixUnHubBTSreal_2x3.txt",
        'transpose_U': True,         

        # ---- solver options ----
        'reorder_by_energy': True,   
        'continuation': True,        # warm-start t amplitudes across steps
        'tol': 1e-11,

        # ---- original outputs ----
        'energies_file':   os.path.join(HERE, 'UPCCD_from_BTS_32.txt'),
        'amplitudes_file': os.path.join(HERE, 'UPCCD_amplitudes_32.txt'), 

        # ---- everything needed later for Lambda/RDMs ----
        'states_dir': os.path.join(HERE, 'UPCCD_states_32'),
        'states_index_file': os.path.join(HERE, 'UPCCD_states_32.txt'),
    }

    run(config)
