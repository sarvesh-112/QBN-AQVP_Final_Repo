"""QBN-AQVP core: data, graph MDP ground truth, scores (circuit / Bayesian network / classical),
online estimators and metrics.  Pure NumPy.

Conventions follow the manuscript:
  * x_i(t) in [0,1] normalised speed, U_i(t) = x_i(t)
  * circuit: RY(pi x_i + theta_i) on every qubit, then CRY(pi w_ij / 2) for every edge (i<j),
    control i, target j, lexicographic order; Q_i = P_i(0); centred score s_i = <Z_i> = 2Q_i - 1
  * estimator: V_i(t) = U_i(t) + lam_t s_i(t) + gamma max_{j in N+(i)} V_j(t-1)
"""
import os, pickle
import numpy as np
import h5py

DATA_DIR = os.environ.get("QBN_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "data"))


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def _ffill(v):
    v = v.copy()
    v[v <= 0] = np.nan
    for j in range(v.shape[1]):
        col = v[:, j]
        mask = np.isnan(col)
        if mask.all():
            v[:, j] = 0.0
            continue
        idx = np.where(~mask, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col = col[idx]
        # leading NaNs: back-fill with first valid value
        first = np.argmax(~np.isnan(col))
        col[:first] = col[first]
        v[:, j] = col
    return v


def load_dataset(name="metr-la", vmax=70.0):
    """Returns X (T, n) in [0,1] and symmetric normalised adjacency W (n, n), zero diagonal."""
    if name == "metr-la":
        f, key, adjf = "metr-la.h5", "df", "adj_mx.pkl"
    elif name == "pems-bay":
        f, key, adjf = "pems-bay.h5", "speed", "adj_mx_bay.pkl"
    else:
        raise ValueError(name)
    h = h5py.File(os.path.join(DATA_DIR, f), "r")[key]
    vals = h["block0_values"][:]
    cols = [c.decode() if isinstance(c, bytes) else str(c) for c in h["block0_items"][:]]
    sensor_ids, id_to_ind, adj = pickle.load(open(os.path.join(DATA_DIR, adjf), "rb"), encoding="latin1")
    order = [id_to_ind[str(c)] for c in cols]
    A = adj[np.ix_(order, order)].astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    A = A / A.max()
    X = np.minimum(_ffill(vals) / vmax, 1.0)
    return X, A


# ----------------------------------------------------------------------------------------------
# subgraphs and windows
# ----------------------------------------------------------------------------------------------
def greedy_subgraphs(A, k=5, size=10, max_shared=2, seed=0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    out = []
    tries = 0
    while len(out) < k and tries < 10000:
        tries += 1
        s = int(rng.integers(n))
        S = [s]
        ok = True
        while len(S) < size:
            sub = A[S]                       # (|S|, n)
            strength = sub.max(axis=0)
            strength[S] = 0
            if strength.max() <= 0:
                ok = False
                break
            S.append(int(np.argmax(strength)))
        if not ok:
            continue
        S = sorted(S)
        if any(len(set(S) & set(T)) > max_shared for T in out):
            continue
        out.append(S)
    return out


def draw_windows(Ttot, n_val=3, n_test=6, T=288, extra=60, split=0.7, seed=0):
    rng = np.random.default_rng(seed)
    L = T + extra
    cut = int(split * Ttot)
    val = sorted(int(v) for v in rng.integers(0, cut - L, size=n_val))
    test = sorted(int(v) for v in rng.integers(cut, Ttot - L, size=n_test))
    return val, test


def extra_steps(gamma, tol=2e-3):
    return int(np.ceil(np.log(tol) / np.log(gamma)))


# ----------------------------------------------------------------------------------------------
# graph MDP helpers
# ----------------------------------------------------------------------------------------------
def nb_mask(W):
    M = W > 0
    np.fill_diagonal(M, True)
    return M


def maxnb(V, M):
    """max_{j in N+(i)} V_j for V of shape (n,) or (B, n)."""
    if V.ndim == 1:
        return np.max(np.where(M, V[None, :], -np.inf), axis=1)
    return np.max(np.where(M[None], V[:, None, :], -np.inf), axis=2)


def vstar(U, M, gamma):
    L, n = U.shape
    V = np.empty_like(U)
    V[-1] = U[-1] / (1 - gamma)
    for t in range(L - 2, -1, -1):
        V[t] = U[t] + gamma * maxnb(V[t + 1], M)
    return V


def argmax_nb(V, M):
    """index of best successor for each node; V (T, n) -> (T, n)."""
    Z = np.where(M[None], V[:, None, :], -np.inf)
    return np.argmax(Z, axis=2)


# ----------------------------------------------------------------------------------------------
# exact circuit scores via the Bayesian-network form (Theorem 2), batched over angle vectors
# ----------------------------------------------------------------------------------------------
class BNCircuit:
    """Exact Z-basis marginals of RY + edge-wise CRY circuit for n <= ~14 by enumeration."""

    def __init__(self, W):
        self.n = n = W.shape[0]
        self.Beta = np.triu(np.pi * W / 2.0, k=1)              # control i < target j
        self.B = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        self.parent_shift = self.B @ self.Beta                   # (2^n, n): sum_c beta_ct b_c
        self.nedges = int((self.Beta > 0).sum())

    def Q(self, alpha):
        """alpha (K, n) -> Q (K, n) = P(b_k = 0)."""
        a = alpha[:, None, :] + self.parent_shift[None]          # (K, 2^n, n)
        p1 = np.sin(a / 2) ** 2
        cond = np.where(self.B[None] > 0, p1, 1.0 - p1)
        P = np.prod(cond, axis=2)                                # (K, 2^n)
        return P @ (1.0 - self.B)

    def QJ(self, alpha):
        """exact Q (n,) and Jacobian J[k, m] = dQ_k/dalpha_m (n, n) from one enumeration."""
        a = alpha[None, :] + self.parent_shift                   # (2^n, n)
        p1 = np.sin(a / 2) ** 2
        cond = np.where(self.B > 0, p1, 1.0 - p1)
        dcond = np.where(self.B > 0, 0.5, -0.5) * np.sin(a)
        ones = np.ones((cond.shape[0], 1))
        pre = np.cumprod(np.hstack([ones, cond[:, :-1]]), axis=1)
        suf = np.cumprod(np.hstack([ones, cond[:, :0:-1]]), axis=1)[:, ::-1]
        dP = pre * suf * dcond                                   # (2^n, n): dP/dalpha_m
        P = pre[:, -1] * cond[:, -1]
        Ind0 = 1.0 - self.B
        return P @ Ind0, Ind0.T @ dP


def shifted_alphas(alpha):
    n = alpha.shape[0]
    E = np.eye(n) * (np.pi / 2)
    return np.vstack([alpha[None], alpha[None] + E, alpha[None] - E])  # (2n+1, n)


# ----------------------------------------------------------------------------------------------
# real-amplitude state-vector simulator (used for M5 and for verification)
# ----------------------------------------------------------------------------------------------
class StateVector:
    def __init__(self, W):
        self.n = n = W.shape[0]
        self.N = 2 ** n
        self.edges = [(i, j, np.pi * W[i, j] / 2) for i in range(n) for j in range(i + 1, n) if W[i, j] > 0]
        idx = np.arange(self.N)
        self.bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1)
        self.zsign = 1.0 - 2.0 * self.bits                      # (N, n)
        self.q_idx = []
        for q in range(n):
            i0 = idx[(idx >> q) & 1 == 0]
            self.q_idx.append((i0, i0 | (1 << q)))
        self.e_idx = []
        for (c, t, b) in self.edges:
            i0 = idx[((idx >> c) & 1 == 1) & ((idx >> t) & 1 == 0)]
            self.e_idx.append((i0, i0 | (1 << t), b))

    def expZ(self, alpha, phi=None):
        """alpha (K, n), phi (K, n) or None -> <Z> (K, n)."""
        K = alpha.shape[0]
        c, s = np.cos(alpha / 2), np.sin(alpha / 2)
        amp = np.where(self.bits[None] > 0, s[:, None, :], c[:, None, :]).prod(axis=2)  # (K, N)
        for (i0, i1, b) in self.e_idx:
            cb, sb = np.cos(b / 2), np.sin(b / 2)
            a0, a1 = amp[:, i0], amp[:, i1]
            amp[:, i0], amp[:, i1] = cb * a0 - sb * a1, sb * a0 + cb * a1
        if phi is not None:
            for q in range(self.n):
                i0, i1 = self.q_idx[q]
                cq, sq = np.cos(phi[:, q] / 2)[:, None], np.sin(phi[:, q] / 2)[:, None]
                a0, a1 = amp[:, i0], amp[:, i1]
                amp[:, i0], amp[:, i1] = cq * a0 - sq * a1, sq * a0 + cq * a1
        return (amp ** 2) @ self.zsign


# ----------------------------------------------------------------------------------------------
# score models: each returns s (n,) and ds/dparams (n, P)
# ----------------------------------------------------------------------------------------------
class Score:
    P = 0
    batchable = False

    def init(self, rng):
        self.params = rng.uniform(-0.1, 0.1, self.P) if self.P else np.zeros(0)

    def eval(self, ctx, t):
        raise NotImplementedError


class CircuitScore(Score):
    """Entangled circuit, exact. centred=True: s=<Z>, else s=Q (v1).  Optional noise."""

    def __init__(self, W, centred=True, shots=None, depol=0.0, rng=None, affine=None):
        self.bn = BNCircuit(W)
        self.affine = affine                                     # (c_k, d_k): <Z> -> c <Z> + d
        self.n = self.P = W.shape[0]
        self.centred = centred
        self.shots, self.depol = shots, depol
        self.G = self.n + self.bn.nedges
        self.nrng = rng

    def _noisy(self, Q):
        if self.depol > 0:
            Q = 0.5 + (1 - self.depol) ** self.G * (Q - 0.5)
        if self.affine is not None:
            c, d = self.affine
            Q = np.clip(0.5 + 0.5 * (c * (2 * Q - 1) + d), 0, 1)
        if self.shots:
            Q = self.nrng.binomial(self.shots, np.clip(Q, 0, 1)) / self.shots
        return Q

    def eval(self, ctx, t):
        alpha = np.pi * ctx.F[t] + self.params
        if self.shots or self.depol > 0 or self.affine is not None:   # parameter shift on noisy outputs
            Qs = self._noisy(self.bn.Q(shifted_alphas(alpha)))
            Q, Qp, Qm = Qs[0], Qs[1:1 + self.n], Qs[1 + self.n:]
            J = 0.5 * (Qp - Qm).T                               # J[k, m] = dQ_k / dtheta_m
        else:                                                    # exact (identical to parameter shift)
            Q, J = self.bn.QJ(alpha)
        if self.centred:
            return 2 * Q - 1, 2 * J
        return Q, J

    def eval_batch(self, F, params):
        Q = self.bn.Q(np.pi * F + params[None])
        return 2 * Q - 1 if self.centred else Q


class ReadoutScore(Score):
    """M5: entangled circuit + trainable RY readout layer (state-vector simulation)."""

    def __init__(self, W):
        self.sv = StateVector(W)
        self.n = W.shape[0]
        self.P = 2 * self.n

    def eval(self, ctx, t):
        n = self.n
        th, ph = self.params[:n], self.params[n:]
        alpha = np.pi * ctx.F[t] + th
        A = np.repeat(alpha[None], 4 * n + 1, axis=0)
        Ph = np.repeat(ph[None], 4 * n + 1, axis=0)
        E = np.eye(n) * (np.pi / 2)
        A[1:1 + n] += E
        A[1 + n:1 + 2 * n] -= E
        Ph[1 + 2 * n:1 + 3 * n] += E
        Ph[1 + 3 * n:] -= E
        Z = self.sv.expZ(A, Ph)
        s = Z[0]
        J = np.hstack([0.5 * (Z[1:1 + n] - Z[1 + n:1 + 2 * n]).T, 0.5 * (Z[1 + 2 * n:1 + 3 * n] - Z[1 + 3 * n:]).T])
        return s, J


class ProductScore(Score):
    """C1 (centred) / V1-C (non-negative): CRY-free limit. mix=True gives C2 (graph-mixing input)."""

    def __init__(self, W, centred=True, mix=False):
        self.n = self.P = W.shape[0]
        self.centred, self.mix = centred, mix
        self.Wn = W

    def feat(self, F):
        if not self.mix:
            return F
        return (F + F @ self.Wn.T) / (1 + self.Wn.sum(1))[None] if F.ndim == 2 else (F + self.Wn @ F) / (1 + self.Wn.sum(1))

    def eval(self, ctx, t):
        f = self.feat(ctx.F[t])
        a = np.pi * f + self.params
        if self.centred:
            return np.cos(a), np.diag(-np.sin(a))
        return (1 + np.cos(a)) / 2, np.diag(-np.sin(a) / 2)

    def eval_batch(self, F, params):
        a = np.pi * self.feat(F) + params[None]
        return np.cos(a) if self.centred else (1 + np.cos(a)) / 2


class MLPScore(Score):
    """C4: learned classical correction. Node-shared MLP on rich local features
    (speed, neighbourhood mean, 1/6-step changes, 1-h mean, time-of-day), tanh output in [-1,1]."""

    def __init__(self, W, hidden=8):
        self.n = W.shape[0]
        self.Wn = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
        self.h = hidden
        self.d = 7
        self.P = self.d * hidden + hidden + hidden + 1

    def init(self, rng):
        d, h = self.d, self.h
        W1 = rng.normal(0, 1 / np.sqrt(d), (d, h))
        w2 = rng.normal(0, 0.1 / np.sqrt(h), h)
        self.params = np.concatenate([W1.ravel(), np.zeros(h), w2, [0.0]])

    def features(self, ctx, t):
        F = ctx.F
        x = F[t]
        g = lambda k: F[max(t - k, 0)]
        hist = F[max(t - 11, 0):t + 1].mean(0)
        tod = ctx.tod[t]
        return np.stack([x, self.Wn @ x, x - g(1), x - g(6), hist,
                         np.full(self.n, np.sin(2 * np.pi * tod)), np.full(self.n, np.cos(2 * np.pi * tod))], 1)

    def eval(self, ctx, t):
        d, h = self.d, self.h
        p = self.params
        W1 = p[:d * h].reshape(d, h); b1 = p[d * h:d * h + h]; w2 = p[d * h + h:d * h + 2 * h]; b2 = p[-1]
        X = self.features(ctx, t)
        H = np.tanh(X @ W1 + b1)
        s = np.tanh(H @ w2 + b2)
        ds = 1 - s ** 2                                          # (n,)
        dH = (1 - H ** 2) * w2[None]                              # (n, h)
        gW1 = (ds[:, None, None] * X[:, :, None] * dH[:, None, :]).reshape(self.n, -1)
        gb1 = ds[:, None] * dH
        gw2 = ds[:, None] * H
        gb2 = ds[:, None]
        return s, np.hstack([gW1, gb1, gw2, gb2])


# ----------------------------------------------------------------------------------------------
# online estimator
# ----------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, U, F, M, gamma, tod=None):
        self.U, self.F, self.M, self.gamma = U, F, M, gamma
        self.tod = tod if tod is not None else np.zeros(len(U))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run_estimator(ctx, T, score=None, lam_rule="none", theta_rule="none", lam0=0.0, eta_lam=0.0,
                  eta_theta=0.0, seed=0, offset=None, rho=0.1, beta=0.1, kappa_sig=1.0, track=False,
                  record_grads=False):
    """Online QBN-AQVP. Returns dict with V (T, n), lam (T,), mean |delta| (T-1,), qterm (T, n)."""
    U, M, g = ctx.U, ctx.M, ctx.gamma
    n = U.shape[1]
    rng = np.random.default_rng(seed)
    if score is not None:
        score.init(rng)
    V = np.empty((T, n))
    lam_hist = np.zeros(T)
    qterm = np.zeros((T, n))
    mtd = np.zeros(T - 1)
    cos_hist = []
    Vprev = U[0] / (1 - g)
    lam = lam0
    cache = None
    for t in range(T):
        if score is not None:
            s, J = score.eval(ctx, t)
        else:
            s, J = np.zeros(n), None
        q = lam * s
        if offset is not None:
            q = q + offset
        Vt = U[t] + q + g * maxnb(Vprev, M)
        V[t] = Vt
        qterm[t] = q
        lam_hist[t] = lam
        if t >= 1:
            s0, J0, V0, lam_old = cache
            delta = U[t - 1] + g * maxnb(Vt, M) - V0
            mtd[t - 1] = np.abs(delta).mean()
            if score is not None:
                k = int(np.argmax(V0))                          # tracked node: highest-valued
                if record_grads:                                 # direction only (lambda cancels)
                    g_full = J0.T @ delta
                    g_diag = delta[k] * np.diag(J0)
                    ng, nd = np.linalg.norm(g_full), np.linalg.norm(g_diag)
                    if ng > 0 and nd > 0:
                        cos_hist.append(g_full @ g_diag / (ng * nd))
                if theta_rule == "full" and eta_theta > 0:
                    grad = lam_old * (J0.T @ delta) / n
                    score.params = score.params + eta_theta * grad
                elif theta_rule == "diag" and eta_theta > 0:
                    score.params = score.params + eta_theta * lam_old * delta[k] * np.diag(J0)
                dbar = np.abs(delta[k]) if track else np.abs(delta).mean()
                if lam_rule == "signed":
                    lam = float(np.clip(lam + eta_lam * (delta @ s0) / n, 0, 1))
                elif lam_rule == "mono":
                    lam = float(np.clip(lam + eta_lam * dbar, 0, 1))
                elif lam_rule == "leaky":
                    lam = float(np.clip((1 - rho) * lam + eta_lam * dbar, 0, 1))
                elif lam_rule == "soft":
                    lam = float((1 - beta) * lam + beta * sigmoid(kappa_sig * dbar))
        cache = (s, J, Vt, lam_hist[t])
        Vprev = Vt
    out = dict(V=V, lam=lam_hist, mtd=mtd, qterm=qterm)
    if record_grads:
        out["cos"] = np.mean(cos_hist) if cos_hist else np.nan
    return out


def forecast_estimator(ctx, T, xhat):
    """F1: one-step forecast-augmented Bellman estimator.
    V_i(t) = U_i(t) + g max_{j in N+(i)} [ xhat_j(t+1) + g max_{k in N+(j)} V_k(t-1) ]."""
    U, M, g = ctx.U, ctx.M, ctx.gamma
    n = U.shape[1]
    V = np.empty((T, n))
    Vprev = U[0] / (1 - g)
    for t in range(T):
        inner = xhat[t] + g * maxnb(Vprev, M)
        V[t] = U[t] + g * maxnb(inner, M)
        Vprev = V[t]
    mtd = np.array([np.abs(U[t] + g * maxnb(V[t + 1], M) - V[t]).mean() for t in range(T - 1)])
    return dict(V=V, lam=np.zeros(T), mtd=mtd, qterm=np.zeros((T, n)))


def metrics(V, Vs, M, burn=12, mtd=None, lam=None):
    e = V[burn:] - Vs[burn:len(V)]
    a_hat = argmax_nb(V[burn:], M)
    a_star = argmax_nb(Vs[burn:len(V)], M)
    out = dict(mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
               policy_acc=(a_hat == a_star).mean())
    if mtd is not None:
        out["mtd"] = mtd[burn:].mean()
    if lam is not None:
        out["lammean"] = lam[burn:].mean()
        out["lamfinal"] = lam[-1]
        out["sat"] = float((lam >= 1 - 1e-12).any())
    return out
