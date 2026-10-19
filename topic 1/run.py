"""CE 465 project - Topic 1: river rating curve.

Model:  Q = a * (h - h0)^b      fitted as   ln Q = ln a + b * ln(h - h0)

Usage:
    python run.py --topic 1 --data <folder containing topic1.csv> --out <folder>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

FLOOD_STAGE = 5.0        # m, the design flood stage H
SCAN_STEP = 0.001        # m, spacing of the h0 scan grid
GN_MAX_ITER = 100        # Gauss-Newton iteration limit
GN_STEP_TOL = 1e-10      # stop when the step actually taken is shorter than this
H0_MARGIN = 1e-6         # h0 must stay below min(h) - H0_MARGIN
MAX_HALVINGS = 60        # safety cap so step halving can never loop forever


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

def load_data(data_folder):
    """Read topic1.csv and return stage h and discharge Q as numpy arrays."""
    path = os.path.join(data_folder, "topic1.csv")
    table = pd.read_csv(path)
    table = table.sort_values("h_m")
    h = table["h_m"].to_numpy(dtype=float)
    Q = table["Q_m3s"].to_numpy(dtype=float)
    return h, Q


def fit_line(h, Q, h0):
    """For a FIXED h0, fit  ln Q = ln a + b * ln(h - h0)  by least squares.

    Returns (ln_a, b, SSE).
    """
    x = np.log(h - h0)
    y = np.log(Q)

    # Design matrix for the straight line y = ln_a * 1 + b * x
    A = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    ln_a = coeffs[0]
    b = coeffs[1]

    residuals = y - ln_a - b * x
    sse = np.sum(residuals ** 2)
    return ln_a, b, sse


def flood_discharge(ln_a, b, h0):
    """Q5 = a * (H - h0)^b."""
    a = np.exp(ln_a)
    return a * (FLOOD_STAGE - h0) ** b


def predict(ln_a, b, h0, h_values):
    """Predicted discharge at the given stages."""
    a = np.exp(ln_a)
    return a * (h_values - h0) ** b


# ---------------------------------------------------------------------------
# Part (a): naive fit with h0 = 0
# ---------------------------------------------------------------------------

def naive_fit(h, Q):
    """Fit with h0 = 0. Returns (ln_a, b)."""
    ln_a, b, _ = fit_line(h, Q, 0.0)
    return ln_a, b


# ---------------------------------------------------------------------------
# Part (b): scan h0 over a grid
# ---------------------------------------------------------------------------

def scan_h0(h, Q):
    """Try every h0 on the grid 0, 0.001, 0.002, ... <= min(h) - 0.001.

    Returns (best_h0, best_ln_a, best_b, best_SSE).
    """
    h_min = h.min()

    # Count grid points with integers, to avoid floating point surprises.
    # The tiny 1e-9 protects against e.g. 2.9999999999 being floored to 2.
    largest_index = int(np.floor((h_min - SCAN_STEP) / SCAN_STEP + 1e-9))

    best_h0 = None
    best_ln_a = None
    best_b = None
    best_sse = np.inf

    for k in range(largest_index + 1):
        h0 = k * SCAN_STEP
        ln_a, b, sse = fit_line(h, Q, h0)
        if sse < best_sse:
            best_sse = sse
            best_h0 = h0
            best_ln_a = ln_a
            best_b = b

    return best_h0, best_ln_a, best_b, best_sse


# ---------------------------------------------------------------------------
# Part (c): Gauss-Newton
# ---------------------------------------------------------------------------

def residuals_and_sse(h, Q, p):
    """Residual vector and SSE for parameters p = (ln_a, b, h0).

    The caller must make sure h0 < min(h), so the log is defined.
    """
    ln_a, b, h0 = p
    r = np.log(Q) - ln_a - b * np.log(h - h0)
    sse = np.sum(r ** 2)
    return r, sse


def jacobian(h, p):
    """15 x 3 matrix of derivatives of the residuals w.r.t. (ln_a, b, h0)."""
    ln_a, b, h0 = p
    d = h - h0
    col_ln_a = -np.ones(len(h))
    col_b = -np.log(d)
    col_h0 = b / d
    return np.column_stack([col_ln_a, col_b, col_h0])


def gauss_newton(h, Q, p_start, use_halving):
    """Gauss-Newton fit of (ln_a, b, h0).

    Returns a dictionary with:
        p          final parameters (numpy array of 3)
        iterations number of iterations performed
        step_norms length of the step actually taken, per iteration
        failed     True if the run left the allowed region, produced
                   non-finite numbers, or did not converge
    """
    h_min = h.min()
    h0_limit = h_min - H0_MARGIN

    p = np.array(p_start, dtype=float)
    step_norms = []
    converged = False
    failed = False

    # The starting point must be valid, otherwise we cannot even begin.
    if p[2] >= h_min:
        return {"p": p, "iterations": 0, "step_norms": step_norms, "failed": True}

    r, sse = residuals_and_sse(h, Q, p)

    for iteration in range(GN_MAX_ITER):
        J = jacobian(h, p)

        # Step delta: least-squares solution of J * delta = -r
        delta, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)

        if use_halving:
            # Halve the step while h0 would be too high or SSE would grow.
            for _ in range(MAX_HALVINGS):
                p_new = p + delta
                if p_new[2] > h0_limit:
                    delta = delta / 2.0
                    continue
                _, sse_new = residuals_and_sse(h, Q, p_new)
                if sse_new > sse:
                    delta = delta / 2.0
                    continue
                break

        # Take the step.
        p = p + delta
        step_norms.append(float(np.linalg.norm(delta)))

        # Without halving, the step can leave the allowed region.
        if p[2] >= h_min or not np.all(np.isfinite(p)):
            failed = True
            break

        r, sse = residuals_and_sse(h, Q, p)
        if not np.isfinite(sse):
            failed = True
            break

        # Stopping rule: length of the step actually taken.
        if step_norms[-1] < GN_STEP_TOL:
            converged = True
            break

    if not converged:
        failed = True

    return {
        "p": p,
        "iterations": len(step_norms),
        "step_norms": step_norms,
        "failed": failed,
    }


def run_gn_safely(h, Q, p_start, use_halving):
    """Run Gauss-Newton but never let an exception escape (used for R3)."""
    try:
        with np.errstate(all="ignore"):
            return gauss_newton(h, Q, p_start, use_halving)
    except Exception:
        return {"p": np.array(p_start, dtype=float), "iterations": 0,
                "step_norms": [], "failed": True}


# ---------------------------------------------------------------------------
# The "full method": scan, then refine with Gauss-Newton (R1)
# ---------------------------------------------------------------------------

def full_method(h, Q):
    """Scan h0, then run R1 from the scan result.

    Returns a dictionary with the scan result and the R1 result.
    """
    h0_scan, ln_a_scan, b_scan, sse_scan = scan_h0(h, Q)
    start = [ln_a_scan, b_scan, h0_scan]
    r1 = gauss_newton(h, Q, start, use_halving=True)
    return {
        "scan": (h0_scan, ln_a_scan, b_scan, sse_scan),
        "r1": r1,
    }


# ---------------------------------------------------------------------------
# Part (d): hold back the three highest readings
# ---------------------------------------------------------------------------

def holdout_check(h, Q):
    """Fit on all but the 3 highest stages, predict those 3."""
    order = np.argsort(h)
    h = h[order]
    Q = Q[order]

    h_train, Q_train = h[:-3], Q[:-3]
    h_hold, Q_hold = h[-3:], Q[-3:]

    # Naive model (h0 = 0)
    ln_a_n, b_n = naive_fit(h_train, Q_train)
    pred_naive = predict(ln_a_n, b_n, 0.0, h_hold)

    # Full model (scan, then R1)
    result = full_method(h_train, Q_train)
    ln_a_f, b_f, h0_f = result["r1"]["p"]
    pred_fit = predict(ln_a_f, b_f, h0_f, h_hold)

    rmse_naive = np.sqrt(np.mean((pred_naive - Q_hold) ** 2))
    rmse_fit = np.sqrt(np.mean((pred_fit - Q_hold) ** 2))

    if rmse_fit < rmse_naive:
        winner = "fitted"
    else:
        winner = "naive"

    return {
        "pred_naive": pred_naive,
        "pred_fit": pred_fit,
        "rmse_naive": rmse_naive,
        "rmse_fit": rmse_fit,
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Part (e): leave-one-out error bar on Q5
# ---------------------------------------------------------------------------

def leave_one_out_error(h, Q):
    """Refit 15 times, each time without one reading, and collect Q5."""
    n = len(h)
    q5_values = []

    for i in range(n):
        keep = np.arange(n) != i
        h_sub = h[keep]
        Q_sub = Q[keep]

        result = full_method(h_sub, Q_sub)
        ln_a, b, h0 = result["r1"]["p"]
        q5_values.append(flood_discharge(ln_a, b, h0))

    q5_values = np.array(q5_values)
    q_mean = q5_values.mean()
    sum_sq = np.sum((q5_values - q_mean) ** 2)

    # Factor (n-1)/n = 14/15 as defined on the project page.
    error_bar = np.sqrt((n - 1) / n * sum_sq)
    return error_bar


# ---------------------------------------------------------------------------
# Putting it all together
# ---------------------------------------------------------------------------

def topic1(data_folder):
    h, Q = load_data(data_folder)

    # Part (a): naive fit
    ln_a_naive, b_naive = naive_fit(h, Q)
    a_naive = np.exp(ln_a_naive)
    q5_naive = flood_discharge(ln_a_naive, b_naive, 0.0)

    # Part (b): scan
    h0_scan, ln_a_scan, b_scan, sse_scan = scan_h0(h, Q)

    # Part (c): the three Gauss-Newton runs
    start_scan = [ln_a_scan, b_scan, h0_scan]
    start_default = [0.0, 1.0, 0.0]

    r1 = run_gn_safely(h, Q, start_scan, use_halving=True)
    r2 = run_gn_safely(h, Q, start_default, use_halving=True)
    r3 = run_gn_safely(h, Q, start_default, use_halving=False)  # may fail, not marked

    ln_a_r1, b_r1, h0_r1 = r1["p"]
    h0_r2 = r2["p"][2]

    # Part (d): holdout check
    hold = holdout_check(h, Q)

    # Part (e): flood discharge and leave-one-out error bar
    q5 = flood_discharge(ln_a_r1, b_r1, h0_r1)
    q5_err = leave_one_out_error(h, Q)

    fields = {
        "a_naive": float(a_naive),
        "b_naive": float(b_naive),
        "Q5_naive": float(q5_naive),
        "h0_scan": float(h0_scan),
        "SSE_scan": float(sse_scan),
        "lna_R1": float(ln_a_r1),
        "b_R1": float(b_r1),
        "h0_R1": float(h0_r1),
        "h0_R2": float(h0_r2),
        "Q_hold_naive": [float(v) for v in hold["pred_naive"]],
        "Q_hold_fit": [float(v) for v in hold["pred_fit"]],
        "holdout_winner": hold["winner"],
        "Q5": float(q5),
        "Q5_err": float(q5_err),
    }

    diagnostics = {
        "iterations_R1": int(r1["iterations"]),
        "iterations_R2": int(r2["iterations"]),
        "R3_failed": bool(r3["failed"]),
        "step_norms_R1": r1["step_norms"],
        "holdout_rmse_naive": float(hold["rmse_naive"]),
        "holdout_rmse_fit": float(hold["rmse_fit"]),
    }

    return {"schema_version": "T1-1.0", "topic": 1,
            "fields": fields, "diagnostics": diagnostics}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    result = {1: topic1}[a.topic](a.data)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"results_T{a.topic}.json"), "w") as f:
        json.dump(result, f, indent=2)