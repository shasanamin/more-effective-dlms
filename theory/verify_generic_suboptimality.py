#!/usr/bin/env python3
"""Exact finite-state checks for the paper's finite-step theory.

Checks (all by exhaustive enumeration of reveal assignments and sequences):
  1. Accounting identity KL(P x nu || Q) = B + C and KL(P || pi) = B + C - R.
  2. Generic suboptimality of the Bayes temperature: for random full-support
     targets, d/d lambda KL(P || pi_lambda) at lambda = 1 is nonzero, and the
     analytical covariance formula matches finite differences.
  3. The V-ary pair witness: closed-form output KL and its negative derivative.
  4. Ranking reversal: an overconfident denoiser with larger posterior loss is a
     better finite-step generator than the Bayes denoiser at lambda = 1.
  5. One-at-a-time (collision-free) reveal reproduces P exactly at lambda = 1.
  6. The two-point Pareto property of (log gPPL, mean within-sample entropy).
No model weights or network access are used. This is not a formal proof.
Run: python theory/verify_generic_suboptimality.py [--output results.json]
"""
from __future__ import annotations

import argparse
import itertools as it
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize_scalar

RNG = np.random.default_rng(20260922)
RESULT: dict = {"checks": {}, "max_residual": 0.0, "summaries": {}}


def check(name: str, lhs: float, rhs: float, tol: float = 1e-9) -> None:
    err = abs(float(lhs) - float(rhs))
    if not np.isfinite(err) or err > tol:
        raise AssertionError(f"{name}: {lhs} vs {rhs} (residual {err})")
    RESULT["checks"][name] = RESULT["checks"].get(name, 0) + 1
    RESULT["max_residual"] = max(RESULT["max_residual"], err)


def require(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(name)
    RESULT["checks"][name] = RESULT["checks"].get(name, 0) + 1


class Model:
    """Exact MDLM-style parallel sampler on V^L with independent reveal rounds."""

    def __init__(self, P: np.ndarray, V: int, L: int, delta: np.ndarray):
        self.P, self.V, self.L, self.delta = P, V, L, np.asarray(delta, float)
        self.N = len(self.delta)
        self.seqs = list(it.product(range(V), repeat=L))
        self.index = {x: k for k, x in enumerate(self.seqs)}
        self.assignments = list(it.product(range(self.N), repeat=L))
        self.nu = np.array([np.prod([self.delta[k] for k in K]) for K in self.assignments])
        self._cond_cache: dict = {}

    def cond(self, i: int, A: tuple[int, ...], x: tuple[int, ...]) -> np.ndarray:
        """Bayes conditional P(X_i = . | X_A = x_A)."""
        key = (i, A, tuple(x[j] for j in A))
        if key not in self._cond_cache:
            w = np.zeros(self.V)
            for y, p in zip(self.seqs, self.P):
                if all(y[j] == x[j] for j in A):
                    w[y[i]] += p
            self._cond_cache[key] = w / w.sum()
        return self._cond_cache[key]

    @staticmethod
    def temper(q: np.ndarray, lam: float) -> np.ndarray:
        z = np.exp(lam * np.log(q))
        return z / z.sum()

    def path_law(self, K: tuple[int, ...], lam: float, denoiser=None) -> np.ndarray:
        denoiser = denoiser or self.cond
        out = np.empty(len(self.seqs))
        for n, x in enumerate(self.seqs):
            val = 1.0
            for i in range(self.L):
                A = tuple(j for j in range(self.L) if K[j] < K[i])
                val *= self.temper(denoiser(i, A, x), lam)[x[i]]
            out[n] = val
        return out

    def laws(self, lam: float, denoiser=None) -> np.ndarray:
        return np.array([self.path_law(K, lam, denoiser) for K in self.assignments])

    def output(self, lam: float, denoiser=None) -> np.ndarray:
        return self.nu @ self.laws(lam, denoiser)

    def kl_output(self, lam: float, denoiser=None) -> float:
        pi = self.output(lam, denoiser)
        m = self.P > 0
        return float(np.sum(self.P[m] * np.log(self.P[m] / pi[m])))


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def accounting(model: Model, lam: float) -> None:
    Q = model.laws(lam)                          # [assignments, seqs]
    P = model.P
    joint_kl = float(np.sum(model.nu[:, None] * P[None, :] * np.log(P[None, :] / Q)))
    # B: expected conditional total correlation inside blocks (Bayes, lambda = 1).
    Q1 = model.laws(1.0)
    b = float(np.sum(model.nu[:, None] * P[None, :] * np.log(P[None, :] / Q1)))
    c = joint_kl - b
    pi = model.nu @ Q
    post = model.nu[:, None] * Q / pi[None, :]
    r = float(np.sum(P * np.sum(model.nu[:, None] * np.log(model.nu[:, None] / post), axis=0)))
    kl_out = float(np.sum(P * np.log(P / pi)))
    check("marginal identity KL(P||pi)=B+C-R", kl_out, b + c - r)
    require("B >= 0", b >= -1e-12)
    require("C >= 0", c >= -1e-12)
    require("R >= 0", r >= -1e-12)


def derivative_formula(model: Model) -> float:
    """Cov_{P x nu}(w, sigma - eta) at lambda = 1 for the Bayes denoiser."""
    Q = model.laws(1.0)
    pi = model.nu @ Q
    total = 0.0
    for a, K in enumerate(model.assignments):
        for n, x in enumerate(model.seqs):
            sigma = eta = 0.0
            for i in range(model.L):
                A = tuple(j for j in range(model.L) if K[j] < K[i])
                q = model.cond(i, A, x)
                sigma -= math.log(q[x[i]])
                eta += entropy(q)
            w = Q[a, n] / pi[n]
            total += model.nu[a] * model.P[n] * (w - 1.0) * (sigma - eta)
    return total


def random_grid(N: int) -> np.ndarray:
    d = RNG.dirichlet(np.ones(N))
    return d


def generic_suboptimality(trials: int) -> None:
    configs = [(2, 2, 2), (2, 3, 2), (2, 2, 3), (3, 2, 2), (3, 2, 3), (2, 4, 2), (3, 3, 2)]
    sharpen = flatten = 0
    min_abs = math.inf
    lam_opts, rel_gain = [], []
    entropy_gap_pos = 0
    count = 0
    for (L, V, N) in configs:
        for _ in range(trials):
            alpha = RNG.choice([0.3, 1.0, 3.0])
            P = RNG.dirichlet(alpha * np.ones(V ** L))
            P = np.maximum(P, 1e-9)
            P /= P.sum()
            model = Model(P, V, L, random_grid(N))
            accounting(model, float(RNG.uniform(0.5, 2.5)))
            h = 1e-5
            fd = (model.kl_output(1 + h) - model.kl_output(1 - h)) / (2 * h)
            an = derivative_formula(model)
            check("derivative covariance formula", an, fd, tol=2e-6 * max(1.0, abs(fd)))
            min_abs = min(min_abs, abs(an))
            require("Bayes temperature not stationary", abs(an) > 1e-10)
            if an < 0:
                sharpen += 1
            else:
                flatten += 1
            k1 = model.kl_output(1.0)
            curv = (model.kl_output(1 + 1e-3) - 2 * k1 + model.kl_output(1 - 1e-3)) / 1e-6
            step = min(1e-3, 0.5 * abs(an) / max(abs(curv), 1e-8)) * (-1.0 if an > 0 else 1.0)
            if abs(step) > 1e-7:
                require("small move in the descent direction improves output KL",
                        model.kl_output(1.0 + step) < k1)
            grid = np.exp(np.linspace(np.log(0.1), np.log(20.0), 121))
            vals = [model.kl_output(g) for g in grid]
            k = int(np.argmin(vals))
            lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
            res = minimize_scalar(model.kl_output, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-7})
            require("optimal temperature strictly better than lambda=1",
                    min(res.fun, vals[k]) < model.kl_output(1.0))
            best = min(res.fun, vals[k])
            lam_opts.append(float(res.x if res.fun <= vals[k] else grid[k]))
            rel_gain.append(float((k1 - best) / k1))
            if entropy(model.output(1.0)) > entropy(P):
                entropy_gap_pos += 1
            if L == 2:
                c = float(np.sum(model.delta ** 2))
                joint = P.reshape(V, V)
                mi = entropy(joint.sum(1)) + entropy(joint.sum(0)) - entropy(P)
                require("pair entropy inflation H(pi) >= H(P) + c I",
                        entropy(model.output(1.0)) >= entropy(P) + c * mi - 1e-12)
            count += 1
    RESULT["summaries"]["generic"] = {
        "targets": count,
        "sharpening_improves": sharpen,
        "smoothing_improves": flatten,
        "min_abs_derivative": min_abs,
        "median_optimal_inverse_temperature": float(np.median(lam_opts)),
        "median_relative_output_kl_reduction": float(np.median(rel_gain)),
        "output_entropy_exceeds_target_entropy": entropy_gap_pos,
        "optimal_inverse_temperatures": [round(x, 5) for x in lam_opts],
        "relative_output_kl_reductions": [round(x, 5) for x in rel_gain],
    }


def pair_witness() -> None:
    for V in [2, 3, 5]:
        for s in [0.05, 0.3, 0.7, 0.95]:
            if abs(s - 1 / V) < 1e-9:
                continue
            for L in [2, 3]:
                P = np.zeros(V ** L)
                seqs = list(it.product(range(V), repeat=L))
                for n, x in enumerate(seqs):
                    pair = (1 / V) * (s if x[0] == x[1] else (1 - s) / (V - 1))
                    P[n] = pair * (1 / V) ** (L - 2)
                delta = random_grid(3)
                model = Model(P, V, L, delta)
                c = float(np.sum(delta ** 2))
                for lam in [0.5, 1.0, 1.7, 3.0]:
                    num = s ** lam
                    u = num / (num + (V - 1) * ((1 - s) / (V - 1)) ** lam)
                    t = c / V + (1 - c) * u
                    closed = s * math.log(s / t) + (1 - s) * math.log((1 - s) / (1 - t))
                    check("V-ary pair output KL closed form", model.kl_output(lam), closed)
                h = 1e-5
                fd = (model.kl_output(1 + h) - model.kl_output(1 - h)) / (2 * h)
                require("pair witness: sharpening strictly improves", fd < 0)


def ranking_reversal() -> None:
    s, delta = 0.7, np.array([0.5, 0.5])
    P = np.array([s / 2, (1 - s) / 2, (1 - s) / 2, s / 2])
    model = Model(P, 2, 2, delta)
    c = 0.5
    u_star = (s - c / 2) / (1 - c)

    def make(u):
        def den(i, A, x):
            if len(A) == 0:
                return np.array([0.5, 0.5])
            other = x[A[0]]
            q = np.full(2, 1 - u)
            q[other] = u
            return q
        return den

    kl_bayes = model.kl_output(1.0)
    for u in [0.75, 0.8, 0.85, u_star]:
        post_loss = (1 - c) * (s * math.log(s / u) + (1 - s) * math.log((1 - s) / (1 - u)))
        require("overconfident denoiser has larger posterior loss", post_loss > 0)
        require("overconfident denoiser is the better generator",
                model.kl_output(1.0, make(u)) < kl_bayes)
    check("exact repair at u*", model.kl_output(1.0, make(u_star)), 0.0, tol=1e-12)
    lam_star = math.log(u_star / (1 - u_star)) / math.log(s / (1 - s))
    check("exact repair by tempering Bayes", model.kl_output(lam_star), 0.0, tol=1e-12)
    RESULT["summaries"]["pair_example"] = {"s": s, "c": c, "u_star": u_star,
                                            "lambda_star": lam_star,
                                            "bayes_output_kl": kl_bayes}


def sequential_exact() -> None:
    for (L, V) in [(2, 3), (3, 2)]:
        P = RNG.dirichlet(np.ones(V ** L))
        seqs = list(it.product(range(V), repeat=L))
        model = Model(P, V, L, np.array([1.0]))
        for order in it.permutations(range(L)):
            K = tuple(order.index(i) for i in range(L))
            law = model.path_law(K, 1.0)
            check("one-at-a-time reveal reproduces P", float(np.abs(law - P).max()), 0.0, tol=1e-12)
        _ = seqs


def pareto_two_points(trials: int) -> None:
    for _ in range(trials):
        n = int(RNG.integers(5, 40))
        ell = RNG.uniform(1.0, 8.0, n)        # per-string log evaluator perplexity
        ent = RNG.uniform(3.0, 7.0, n)        # per-string within-sample entropy
        h0 = float(np.quantile(ent, RNG.uniform(0.1, 0.9)))
        res = linprog(ell, A_ub=[-ent], b_ub=[-h0], A_eq=[np.ones(n)], b_eq=[1.0],
                      bounds=[(0, None)] * n, method="highs-ds")
        require("entropy-constrained gPPL LP solved", res.status == 0)
        require("optimal generator uses at most two strings", int(np.sum(res.x > 1e-9)) <= 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generic_suboptimality(args.trials)
    pair_witness()
    ranking_reversal()
    sequential_exact()
    pareto_two_points(200)
    RESULT["total_assertions"] = int(sum(RESULT["checks"].values()))
    text = json.dumps(RESULT, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
