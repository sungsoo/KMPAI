"""Decision rules for C7 (shared by the pilot and, once locked, by the C7 runner)."""
import numpy as np
from scipy.stats import beta as _beta


def betting_ucb(x, eps=0.05, grid=2001):
    """One-sided (1 - eps) upper confidence bound for the mean of x in [0, 1] by betting (Waudby-Smith and Ramdas,
    JRSS B 2024): for each m, the capital prod_t (1 + lam_t (m - x_t)) with predictable lam_t in [0, 0.5 / (1 - m)] (factors stay positive) is a
    nonnegative supermartingale under H0: mean >= m; H0_m is rejected when the capital reaches 1 / eps (Ville).
    The bound is the smallest grid m that is rejected (1.0 if none)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0); n = len(x)
    if n == 0:
        return 1.0
    # predictable plug-in variance and bet size (fixed in advance, no look-ahead)
    mu_hat = np.concatenate([[0.5], np.cumsum(x)[:-1] / np.arange(1, n)])
    var_hat = np.concatenate([[0.25], np.array([np.var(x[:t]) if t > 1 else 0.25 for t in range(1, n)])]) + 1e-4
    lam_base = np.sqrt(2 * np.log(1 / eps) / (n * var_hat))
    for m in np.linspace(0.0, 1.0, grid)[1:]:
        lam = np.minimum(lam_base, 0.5 / max(1 - m, 1e-12))
        if np.prod(1 + lam * (m - x)) >= 1 / eps:
            return float(m)
    return 1.0


def cp_upper(k, n, eps=0.05):
    return 1.0 if k >= n else float(_beta.ppf(1 - eps, k + 1, n - k))


def magnitude_gate(ratios, tau=0.0, kappa=0.10, beta=0.10, eps=0.05, clip=2.0):
    """Rule (1): accept iff UCB of mean(min(ratio, clip) / clip) <= (1 + tau) / clip  AND
    CP upper bound of P(ratio > 1 + kappa) <= beta (tail condition). Returns (accept, details)."""
    r = np.asarray(ratios, dtype=float)
    ucb = betting_ucb(np.minimum(r, clip) / clip, eps)
    k_tail = int((r > 1 + kappa).sum()); u_tail = cp_upper(k_tail, len(r), eps)
    ok = (ucb <= (1 + tau) / clip) and (u_tail <= beta)
    return ok, {'mean_ratio_ucb': ucb * clip, 'k_tail': k_tail, 'tail_upper': u_tail}
