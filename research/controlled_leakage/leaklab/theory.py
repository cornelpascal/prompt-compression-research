"""Finite fixed-decoder convex program. Results include numerical gap bounds."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog, minimize


@dataclass
class FiniteProblem:
    probabilities: np.ndarray
    secrets: np.ndarray
    public: np.ndarray
    lengths: np.ndarray
    distortion: np.ndarray

    def __post_init__(self):
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        self.secrets, self.public = np.asarray(self.secrets), np.asarray(self.public)
        self.lengths, self.distortion = np.asarray(self.lengths, dtype=float), np.asarray(self.distortion, dtype=float)
        n = len(self.probabilities)
        if self.probabilities.ndim != 1 or np.any(self.probabilities <= 0) or not np.isclose(self.probabilities.sum(), 1):
            raise ValueError("Source probabilities must be positive and sum to one")
        if self.secrets.shape != (n,) or self.public.shape != (n,) or self.distortion.shape != (n, len(self.lengths)):
            raise ValueError("Finite-source dimensions disagree")
        if not np.isfinite(self.distortion).all() or not np.isfinite(self.lengths).all() or np.any(self.lengths < 0) or np.any(self.distortion < 0):
            raise ValueError("Lengths and distortions must be finite and nonnegative")


def leakage(problem, channel, gradient=False):
    p, s, w = problem.probabilities, problem.secrets, problem.public
    c = np.asarray(channel).reshape(len(p), -1)
    value, grad = 0., np.zeros_like(c)
    for wi in np.unique(w):
        wi_mask = w == wi
        pw = p[wi_mask].sum()
        pwz = (p[wi_mask, None] * c[wi_mask]).sum(0)
        for si in np.unique(s[wi_mask]):
            mask = wi_mask & (s == si)
            psw = p[mask].sum()
            joint = (p[mask, None] * c[mask]).sum(0)
            valid = (joint > 0) & (pwz > 0)
            logratio = np.zeros(c.shape[1])
            logratio[valid] = np.log2(joint[valid] * pw / (psw * pwz[valid]))
            value += float(np.sum(joint[valid] * logratio[valid]))
            grad[mask] = p[mask, None] * logratio
    return (value, grad.ravel()) if gradient else value


def solve(problem, length_budget, error_budget, tolerance=1e-7):
    if length_budget < 0 or error_budget < 0:
        raise ValueError("Budgets must be nonnegative")
    n, m = problem.distortion.shape
    # Merge symbols with identical decoder distortion into the shortest one.
    # This preserves error, weakens the rate constraint and cannot increase
    # leakage by data processing, so the restricted optimum equals the original.
    # It also removes redundant unused symbols that weaken numerical tangents.
    kept = []
    for j in sorted(range(m), key=lambda j:(problem.lengths[j],j)):
        if not any(np.array_equal(problem.distortion[:,j],problem.distortion[:,k]) for k in kept):
            kept.append(j)
    if len(kept) < m:
        reduced = FiniteProblem(problem.probabilities,problem.secrets,problem.public,
            problem.lengths[kept],problem.distortion[:,kept])
        result = solve(reduced,length_budget,error_budget,tolerance)
        if "channel" in result:
            channel = np.zeros((n,m))
            channel[:,kept] = np.asarray(result["channel"])
            result["channel"] = channel.tolist()
        result["equivalent_decoder_symbols_removed"] = m-len(kept)
        return result
    aeq = np.kron(np.eye(n), np.ones((1, m)))
    costs = np.stack([(problem.probabilities[:, None] * np.broadcast_to(problem.lengths, (n, m))).ravel(),
                      (problem.probabilities[:, None] * problem.distortion).ravel()])
    budgets = np.array([length_budget, error_budget])
    linear = dict(A_ub=costs, b_ub=budgets, A_eq=aeq, b_eq=np.ones(n), bounds=(0, 1), method="highs")
    feasible = linprog(np.zeros(n*m), **linear)
    if not feasible.success:
        if feasible.status == 2:
            return {"status": "infeasible", "length_budget": length_budget, "error_budget": error_budget}
        raise RuntimeError("Feasibility solver failed: " + feasible.message)
    # Smooth only the optimization objective. Report actual unsmoothed leakage.
    eps = 1e-10
    def objective(x):
        value, grad = leakage(problem, (1-eps)*x.reshape(n,m)+eps/m, True)
        return value, (1-eps)*grad
    constraints = [{"type": "eq", "fun": lambda x: aeq@x-1, "jac": lambda x: aeq},
                   {"type": "ineq", "fun": lambda x: budgets-costs@x, "jac": lambda x: -costs}]
    # A relative-interior feasible start avoids arbitrary unused symbols at
    # boundary optima, where a smoothed tangent can give a needlessly loose gap.
    starts = [feasible.x]
    for coordinate in range(n*m):
        direction = np.zeros(n*m)
        direction[coordinate] = -1
        vertex = linprog(direction, **linear)
        if vertex.success:
            starts.append(vertex.x)
    initial = np.mean(starts, axis=0)
    solution = minimize(objective, initial, jac=True, method="SLSQP", bounds=[(0,1)]*(n*m),
        constraints=constraints, options={"maxiter": 3000, "ftol": 1e-12})
    candidate = solution.x
    residual = max(float(np.max(np.abs(aeq@candidate-1))), float(np.max(costs@candidate-budgets)),
                   float(-candidate.min()), float(candidate.max()-1))
    if residual > tolerance or not np.isfinite(candidate).all():
        raise RuntimeError("No sufficiently feasible numerical solution")
    channel = candidate.reshape(n,m)
    upper = max(0., leakage(problem, channel))
    # A tangent plane of convex I(S;Z|W) globally lower-bounds the objective.
    # Use an interior point so every positive source has differentiable cells.
    interior = (1-1e-7)*channel + 1e-7/m
    f0, g = leakage(problem, interior, True)
    dual_lp = linprog(g, **linear)
    if not dual_lp.success:
        raise RuntimeError("Supporting-plane lower-bound solve failed")
    lower = max(0., float(f0 + dual_lp.fun - g @ interior.ravel()))
    if lower > upper+tolerance*10:
        raise RuntimeError("Numerical lower bound exceeds the feasible upper bound")
    gap = max(0., upper-lower)
    return {"status": "numerically_certified" if gap < tolerance*10 else "feasible_gap_remaining",
        "length_budget": length_budget, "error_budget": error_budget,
        "leakage_bits": upper, "lower_bound_bits": lower, "numerical_gap_bits": gap,
        "expected_length": float(costs[0]@candidate), "task_error": float(costs[1]@candidate),
        "constraint_residual": residual, "optimizer_success": bool(solution.success),
        "channel": channel.tolist(), "certificate_scope": "Floating-point finite-source/fixed-decoder problem only"}


def binary_problem(regime, correlation=.9):
    if regime not in ("independent", "correlated", "essential") or not .5 <= correlation < 1:
        raise ValueError("Invalid binary regime/correlation")
    rows = [(t,s) for t in (0,1) for s in (0,1) if regime != "essential" or t==s]
    probabilities = [.5 if regime == "essential" else .25 if regime == "independent" else
                     .5*(correlation if t==s else 1-correlation) for t,s in rows]
    # z = empty, "0", "1"; the fixed decoder predicts 0,0,1 respectively.
    return FiniteProblem(np.array(probabilities), np.array([s for t,s in rows]), np.zeros(len(rows), dtype=int),
        np.array([0.,1.,1.]), np.array([[float(t!=y) for y in (0,0,1)] for t,s in rows]))


def frontier():
    return [{"regime": regime, **solve(binary_problem(regime), rate, distortion)}
        for regime in ("independent", "correlated", "essential")
        for rate in (0., .25, .5, 1.) for distortion in (0., .1, .25, .5)]


def output_leakage(problem, compression_channel, output_channel):
    c, o = np.asarray(compression_channel), np.asarray(output_channel)
    if o.ndim != 2 or o.shape[0] != c.shape[1] or np.any(o < 0) or not np.allclose(o.sum(1), 1):
        raise ValueError("Output channel must be row stochastic")
    return leakage(problem, c @ o)
