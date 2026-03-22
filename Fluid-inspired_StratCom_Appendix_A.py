from __future__ import annotations

"""
Appendix A companion simulation code for the paper:

    Fluid-inspired Strategic Communications Modelling

Author: Yukai Zeng
Affiliation: Singapore Ministry of Defence Communications Organisation
Email: Zeng_Yukai@defence.gov.sg

Purpose
-------
This script reproduces the paper's illustrative Figure 1 calibration and prints
summary diagnostics for Scenarios A, B, and C.

What the script does
--------------------
1. Builds a stylised two-block interaction graph.
2. Defines three scenario-specific parameter schedules.
3. Integrates the coupled state equations with fourth-order Runge-Kutta (RK4).
4. Computes effective flow, turbulence risk, and related diagnostics.
5. Renders the two-panel figure used in the manuscript.

Model equations implemented here
--------------------------------
(1)  d rho / dt = -N(t) L rho + [A(t)R(t) - D(t) - F(t)] rho + P(t) s
(2)  dM / dt     = alpha A(t) mean(rho) - [mu(t) + D(t)] M
(3)  Phi(t)      = P(t) R(t) A(t) N(t) M(t) / [mu(t) + D(t) + F(t)]
(4)  Re_c(t)     = V(t) M(t) / [mu(t) + D(t)]
(5)  Stability:  A(t)R(t) - D(t) - F(t) < N(t) lambda_2(L)
(6)  TR(t)       = max(0, A(t)R(t) - D(t) - F(t) - N(t)lambda_2(L)) / C
(7)  F(t)        = D_KL(p_hat(t) || p_hat(t0))

Notes
-----
- This is an illustrative research script rather than a production tool.
- Parameter schedules are analytically defined for reproducibility.
- The figure generated is intended to support the paper's narrative comparison
  of defended, siloed, and contested communication environments.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def bump(t: float | np.ndarray, peak: float, width: float = 0.18, floor: float = 0.05) -> np.ndarray:
    """
    Create a smooth pulse-shaped schedule centred at `peak`.

    This helper is used to build time-varying amplification, receptivity, and
    injection schedules without introducing discontinuities that would make the
    simulated trajectories look artificial.
    """
    t = np.asarray(t)
    return np.exp(-width * np.abs(t - peak) ** 1.3) + floor


def make_source_weighted(weights: list[float]) -> np.ndarray:
    """
    Convert a list of non-negative source weights into a normalised source vector.

    The source vector `s` identifies which nodes are seeded more strongly at the
    start of the campaign and by the continuous source-injection term P(t)s.
    """
    s = np.array(weights, dtype=float)
    s = np.clip(s, 0.0, None)
    s /= s.sum()
    return s


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(p || q).

    A small epsilon is used so the divergence remains numerically stable even
    when one distribution is sparse or contains near-zero entries.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def two_block_graph(n: int, beta: float, scale: float = 0.02, intra: float = 1.0):
    """
    Construct a weighted two-block graph and return its key spectral quantities.

    Parameters
    ----------
    n : int
        Number of nodes. Must be even so the graph can be divided into two blocks.
    beta : float
        Relative cross-block permeability. Smaller beta means stronger siloing.
    scale : float
        Overall edge-weight scaling.
    intra : float
        Within-block weight multiplier.

    Returns
    -------
    W : ndarray
        Weighted adjacency matrix.
    L : ndarray
        Graph Laplacian.
    lambda_2 : float
        Fiedler eigenvalue; captures the graph's weakest connectivity mode.
    lambda_n : float
        Largest Laplacian eigenvalue.
    H : float
        Simple heterogeneity / fragmentation-style index used in the paper.
    """
    if n % 2 != 0:
        raise ValueError("n must be even for the two-block graph.")

    m = n // 2
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            same_block = (i < m and j < m) or (i >= m and j >= m)
            weight = scale * (intra if same_block else beta)
            W[i, j] = W[j, i] = weight

    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals = np.sort(np.linalg.eigvalsh(L))
    lambda_2 = float(eigvals[1])
    lambda_n = float(eigvals[-1])
    H = 1.0 - lambda_2 / lambda_n
    return W, L, lambda_2, lambda_n, H


@dataclass
class Scenario:
    """
    Container for one communication environment / scenario.
    """
    name: str
    color: str
    linestyle: str
    beta: float
    C: float
    graph_scale: float
    source: np.ndarray
    P_fn: callable
    R_fn: callable
    A_fn: callable
    N_fn: callable
    mu_fn: callable
    D_fn: callable


@dataclass
class SimulationResult:
    """
    Store full trajectories and derived diagnostics from one model run.
    """
    t: np.ndarray
    rho: np.ndarray
    M: np.ndarray
    Phi: np.ndarray
    TR: np.ndarray
    margin: np.ndarray
    F: np.ndarray
    V: np.ndarray
    Rec: np.ndarray
    lambda_2: float
    lambda_n: float
    H: float


def simulate_scenario(
    sc: Scenario,
    *,
    alpha: float = 5.2,
    T_end: float = 28.0,
    n_steps: int = 1401,
) -> SimulationResult:
    """
    Simulate one scenario using fourth-order Runge-Kutta integration.

    State vector
    ------------
    y = [rho_1, ..., rho_n, M]

    where:
    - rho_i is narrative density at node i
    - M is the aggregate memory / momentum state used by the effective-flow term
    """
    n = len(sc.source)

    # Build the scenario-specific graph and extract its spectral quantities.
    _, L, lambda_2, lambda_n, H = two_block_graph(
        n=n,
        beta=sc.beta,
        scale=sc.graph_scale,
    )

    # Time grid used in the paper's figure generation.
    t = np.linspace(0.0, T_end, n_steps)
    dt = t[1] - t[0]

    # Initialise the state close to zero, but with a tiny source-weighted seed.
    y = np.zeros(n + 1, dtype=float)
    y[:n] = 1e-6 * sc.source

    # Allocate arrays for the main trajectories and diagnostics.
    rho_hist = np.zeros((n_steps, n), dtype=float)
    M_hist = np.zeros(n_steps, dtype=float)
    Phi = np.zeros(n_steps, dtype=float)
    TR = np.zeros(n_steps, dtype=float)
    margin = np.zeros(n_steps, dtype=float)
    F = np.zeros(n_steps, dtype=float)

    def rhs(y_state: np.ndarray, tt: float) -> np.ndarray:
        """
        Right-hand side of the coupled ODE system at time `tt`.
        """
        rho = np.clip(y_state[:n], 0.0, None)
        M = max(float(y_state[n]), 0.0)

        # Convert node-level density into a probability distribution so the
        # fidelity-loss term can be computed as KL divergence.
        rho_tot = rho.sum()
        p_hat = rho / rho_tot if rho_tot > 1e-12 else sc.source
        F_now = kl_divergence(p_hat, sc.source)

        # Evaluate the time-varying scenario schedules at the current time.
        P = float(sc.P_fn(tt))
        R = float(sc.R_fn(tt))
        A = float(sc.A_fn(tt))
        N = float(sc.N_fn(tt))
        mu = float(sc.mu_fn(tt))
        D = float(sc.D_fn(tt))

        # Equation (1): narrative-density evolution on the graph.
        drho = -N * (L @ rho) + (A * R - D - F_now) * rho + P * sc.source

        # Equation (2): memory / momentum evolution.
        dM = alpha * A * np.mean(rho) - (mu + D) * M

        return np.concatenate([drho, [dM]])

    # March forward in time and record both state variables and diagnostics.
    for i, tt in enumerate(t):
        rho = np.clip(y[:n], 0.0, None)
        M = max(float(y[n]), 0.0)
        rho_hist[i] = rho
        M_hist[i] = M

        # Fidelity-loss term F(t), evaluated against the source distribution.
        rho_tot = rho.sum()
        p_hat = rho / rho_tot if rho_tot > 1e-12 else sc.source
        F[i] = kl_divergence(p_hat, sc.source)

        # Current schedule values.
        P = float(sc.P_fn(tt))
        R = float(sc.R_fn(tt))
        A = float(sc.A_fn(tt))
        N = float(sc.N_fn(tt))
        mu = float(sc.mu_fn(tt))
        D = float(sc.D_fn(tt))

        # Equation (3): effective communicative flow.
        Phi[i] = (P * R * A * N * M) / max(mu + D + F[i], 1e-12)

        # Stability margin relative to the Fiedler-mode threshold.
        margin[i] = A * R - D - F[i] - N * lambda_2

        # Equation (6): turbulence risk, scaled by scenario-specific C.
        TR[i] = max(0.0, margin[i]) / sc.C

        # Classic RK4 integration step.
        if i < n_steps - 1:
            k1 = rhs(y, tt)
            k2 = rhs(y + 0.5 * dt * k1, tt + 0.5 * dt)
            k3 = rhs(y + 0.5 * dt * k2, tt + 0.5 * dt)
            k4 = rhs(y + dt * k3, tt + dt)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Keep all states non-negative because negative density / memory
            # would be physically and communicatively unintuitive here.
            y[:n] = np.clip(y[:n], 0.0, None)
            y[n] = max(0.0, y[n])

    # Derived totals used to compute V(t) and Re_c(t).
    rho_tot = rho_hist.sum(axis=1)
    drho_tot = np.gradient(rho_tot, t)
    V = np.divide(drho_tot, np.maximum(rho_tot, 1e-12))
    mu_arr = np.array([sc.mu_fn(tt) for tt in t])
    D_arr = np.array([sc.D_fn(tt) for tt in t])
    Rec = (V * M_hist) / np.maximum(mu_arr + D_arr, 1e-12)

    return SimulationResult(
        t=t,
        rho=rho_hist,
        M=M_hist,
        Phi=Phi,
        TR=TR,
        margin=margin,
        F=F,
        V=V,
        Rec=Rec,
        lambda_2=lambda_2,
        lambda_n=lambda_n,
        H=H,
    )


def build_scenarios() -> list[Scenario]:
    """
    Define the three paper scenarios:

    A. Defended posture with strong trusted amplification
    B. Siloed institutional campaign
    C. Contested open-platform ecosystem
    """
    beta_A = 2.2717717717717716
    beta_B = 0.4075075075075075
    beta_C = 0.12402402402402403

    # Source vectors encode where the initial and ongoing injections are focused.
    source_A = make_source_weighted([1] * 7 + [0.12] * 13)
    source_B = make_source_weighted([1] * 8 + [0.25] * 12)
    source_C = make_source_weighted([1] * 11 + [0.50] * 9)

    scenario_A = Scenario(
        name="A: CDC/WHO + health influencers",
        color="#1a6faf",
        linestyle="-",
        beta=beta_A,
        C=0.78,
        graph_scale=0.0105,
        source=source_A,
        P_fn=lambda t: bump(t, 2.0, width=0.12, floor=0.08) * 0.82 + 0.11,
        R_fn=lambda t: np.clip(0.70 + 0.07 * np.sin(0.32 * t) * np.exp(-0.02 * t), 0.0, 1.0),
        A_fn=lambda t: bump(t, 5.3, width=0.16, floor=0.08) * 0.88 + 0.05
        + bump(t, 10.8, width=0.20, floor=0.01) * 0.22,
        N_fn=lambda t: np.clip(0.72 + 0.11 * np.tanh(t - 4.5), 0.05, 1.0),
        mu_fn=lambda t: 0.19 + 0.03 * np.exp(-0.08 * t),
        D_fn=lambda t: 0.075 + 0.06 * t / 28.0,
    )

    scenario_B = Scenario(
        name="B: Hospital internal campaign",
        color="#2ca02c",
        linestyle="--",
        beta=beta_B,
        C=0.44,
        graph_scale=0.0030,
        source=source_B,
        P_fn=lambda t: bump(t, 2.2, width=0.13, floor=0.08) * 1.00 + 0.11,
        R_fn=lambda t: np.clip(
            0.60 + 0.06 * np.sin(0.25 * t) + 0.07 * bump(t, 8.6, width=0.24, floor=0.0),
            0.0,
            1.0,
        ),
        A_fn=lambda t: np.clip(
            0.90 * (bump(t, 8.7, width=0.20, floor=0.09) * 0.92 + 0.08)
            + 0.07 * bump(t, 10.8, width=0.23, floor=0.0),
            0.0,
            1.25,
        ),
        N_fn=lambda t: np.clip(0.48 + 0.08 * np.tanh(t - 7.3), 0.05, 1.0),
        mu_fn=lambda t: 0.39 + 0.05 * np.exp(-0.05 * t),
        D_fn=lambda t: 0.19 + 0.07 * t / 28.0,
    )

    scenario_C = Scenario(
        name="C: Open Twitter ecosystem",
        color="#d62728",
        linestyle=":",
        beta=beta_C,
        C=0.22,
        graph_scale=0.025,
        source=source_C,
        P_fn=lambda t: bump(t, 2.2, width=0.12, floor=0.08) * 0.75 + 0.12,
        R_fn=lambda t: np.clip(0.56 + 0.25 * np.sin(0.95 * t) * np.exp(-0.022 * t), 0.10, 1.0),
        A_fn=lambda t: bump(t, 3.0, width=0.22, floor=0.05) * 0.52 + 0.04
        + bump(t, 7.0, width=0.18, floor=0.02) * 0.90,
        N_fn=lambda t: np.clip(0.60 + 0.16 * np.sin(0.56 * t), 0.08, 1.0),
        mu_fn=lambda t: 0.30 + 0.15 * np.abs(np.sin(0.42 * t + 0.8)),
        D_fn=lambda t: 0.18 + 0.11 * t / 28.0,
    )

    return [scenario_A, scenario_B, scenario_C]


def make_figure(results: dict[str, SimulationResult], out_path: Path):
    """
    Render the two-panel figure used in the appendix / paper.

    Left panel:
        Normalised effective communicative flow.

    Right panel:
        Turbulence risk with an illustrative threshold line.
    """
    A = results["A"]
    B = results["B"]
    C = results["C"]
    t = A.t

    # Normalise effective flow to Scenario A's peak so the three curves can be
    # compared on a common visual scale in the manuscript.
    phi_norm = max(A.Phi.max(), 1e-12)
    Phi_A = A.Phi / phi_norm
    Phi_B = B.Phi / phi_norm
    Phi_C = C.Phi / phi_norm

    tr_top = max(A.TR.max(), B.TR.max(), C.TR.max())
    tr_ylim = max(1.1, tr_top * 1.08)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.18, wspace=0.34)

    # Panel (a): effective communicative flow.
    for y, color, ls, label in [
        (Phi_A, "#1a6faf", "-", "A: CDC/WHO + health influencers"),
        (Phi_B, "#2ca02c", "--", "B: Hospital internal campaign"),
        (Phi_C, "#d62728", ":", "C: Open Twitter ecosystem"),
    ]:
        ax1.plot(t, y, color=color, lw=2.5, ls=ls, label=label)
        ax1.fill_between(t, 0, y, color=color, alpha=0.07)

    ax1.axvline(0.0, color="#aaaaaa", lw=0.9, ls="-.", zorder=0)
    ax1.axvline(14.0, color="#aaaaaa", lw=0.9, ls="-.", zorder=0)
    ax1.text(0.45, 1.12, "Rollout\nannounced", fontsize=6.8, color="#555555", va="top")
    ax1.text(14.25, 1.12, "Booster\nnews", fontsize=6.8, color="#555555", va="top")

    ix = np.argmin(np.abs(t - 7.0))
    ax1.annotate(
        "Counter-narrative\nsurge",
        xy=(t[ix], Phi_C[ix]),
        xytext=(11.4, 0.50),
        fontsize=7.0,
        color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9),
    )

    ax1.set_xlabel("Days post-rollout announcement", fontsize=9)
    ax1.set_ylabel("Effective Communicative Flow (norm.)", fontsize=9)
    ax1.set_title("(a) Eq. (3) - Effective Flow $\\Phi(t)$", fontsize=9.8, fontweight="bold")
    ax1.legend(fontsize=7.8, loc="upper right", framealpha=0.90)
    ax1.set_xlim(0.0, 28.0)
    ax1.set_ylim(0.0, 1.25)
    ax1.grid(True, alpha=0.20, lw=0.5)
    ax1.tick_params(labelsize=8)

    # Panel (b): turbulence risk.
    for y, color, ls, label in [
        (A.TR, "#1a6faf", "-", "A"),
        (B.TR, "#2ca02c", "--", "B"),
        (C.TR, "#d62728", ":", "C"),
    ]:
        ax2.plot(t, y, color=color, lw=2.5, ls=ls, label=label)
        ax2.fill_between(t, 0, y, color=color, alpha=0.06)

    T_threshold = 0.35
    ax2.axhline(T_threshold, color="#777777", lw=1.1, ls="-.", alpha=0.85)
    ax2.text(0.5, T_threshold + 0.03, "Turbulence threshold (T)", fontsize=7.2, color="#555555")

    unstable = C.margin > 0.0
    ax2.fill_between(
        t,
        T_threshold,
        C.TR,
        where=unstable & (C.TR > T_threshold),
        alpha=0.14,
        color="#d62728",
    )

    unstable_idx = np.where(unstable & (C.TR > T_threshold))[0]
    if len(unstable_idx) > 0:
        j = unstable_idx[len(unstable_idx) // 2]
        ax2.annotate(
            "Narrative\ninstability\nwindow",
            xy=(t[j], C.TR[j]),
            xytext=(t[j] + 3.6, min(C.TR[j] * 0.85, tr_ylim * 0.82)),
            fontsize=6.8,
            color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9),
        )

    ax2.set_xlabel("Days post-rollout announcement", fontsize=9)
    ax2.set_ylabel("Turbulence Risk (norm.)", fontsize=9)
    ax2.set_title("(b) Eq. (6) - Turbulence Risk $TR(t)$", fontsize=9.8, fontweight="bold")
    ax2.legend(fontsize=7.8, loc="upper right", framealpha=0.90)
    ax2.set_xlim(0.0, 28.0)
    ax2.set_ylim(0.0, tr_ylim)
    ax2.grid(True, alpha=0.20, lw=0.5)
    ax2.tick_params(labelsize=8)

    fig.text(
        0.5,
        0.03,
        "Pronounced illustrative calibration of the paper's equations: all three scenarios now show stronger flow and risk signatures.",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#444444",
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    """
    Command-line entry point.

    Example
    -------
    python "Fluid-inspired StratCom_Appendix A_annotated.py" --out figure.png
    """
    parser = argparse.ArgumentParser(
        description="Generate the appendix figure and summary diagnostics for the StratCom model."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/mnt/data/Fluid-inspired_StratCom_Appendix_A_figure.png",
        help="Output PNG path for the generated figure.",
    )
    args = parser.parse_args()

    # Simulate the three scenarios in the same order used in the paper.
    scenarios = build_scenarios()
    keys = ["A", "B", "C"]
    results = {k: simulate_scenario(sc) for k, sc in zip(keys, scenarios)}

    # Save the figure.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(results, out_path)

    # Print a compact console summary for quick verification and manuscript checks.
    print(f"Saved figure to: {out_path}")
    for k in keys:
        res = results[k]
        peak_i = int(np.argmax(res.Phi))
        unstable = np.where(res.margin > 0.0)[0]
        if len(unstable) > 0:
            unstable_window = f"{res.t[unstable[0]]:.2f}-{res.t[unstable[-1]]:.2f} d"
        else:
            unstable_window = "none"
        print(
            f"{k}: H={res.H:.3f}, lambda2={res.lambda_2:.4f}, "
            f"peak Phi={res.Phi[peak_i]:.5f} at day {res.t[peak_i]:.2f}, "
            f"peak TR={res.TR.max():.4f}, instability window={unstable_window}"
        )


if __name__ == "__main__":
    main()
