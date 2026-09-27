#!/usr/bin/env python3
"""Build every figure in the paper from archived aggregates, in one house style.

No generation, inference, or judge call is performed. Inputs are portable CSV/TSV/JSON
aggregates in the supplementary release; the theory figure uses the closed form of
Proposition 2 and the exact finite-state results written by
theory/verify_generic_suboptimality.py.
Usage: python analysis/make_iclr_figures.py
By default, figures are written to build/figures inside this release. Use
--output-dir to choose another destination, or --paper-dir to export to a
LaTeX source tree's figs directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C = {  # semantic palette: blue = the unchanged model, sharpened; grey = its standard sampler
    "sharp": "#0F4D92", "sharp2": "#6A9BD1", "std": "#8C8C8C", "dark": "#272727",
    "duo": "#DE8C86", "dcd": "#B64342", "didi": "#42949E",
    "didi2": "#1F5F66", "ochre": "#C99A2E", "shade": "#EEF3FA",
    "warn": "#FBEDEC",
}
BLUES = LinearSegmentedColormap.from_list("house", ["#FFFFFF", C["sharp2"], C["sharp"]])
RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 8.0,
    "xtick.labelsize": 6.9, "ytick.labelsize": 6.9, "legend.fontsize": 6.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.titleweight": "bold", "axes.titlepad": 4.0,
}
STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def step_axis(ax, ticks=(1, 4, 16, 64, 256, 1024), label="Sampling steps") -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(ticks), [str(t) for t in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlabel(label)


def sweep() -> dict:
    out = {}
    for r in rows(ROOT / "paper_data/mdlm_sampling_grid.csv"):
        out[(r["ckpt"], int(r["N"]), round(float(r["tau"]), 3), r["prec"])] = float(r["gPPL"])
    return out


def panel_successors(ax) -> None:
    data: dict[str, dict[int, float]] = {}
    for r in rows(ROOT / "paper_data/cross_model_gp_may05.csv"):
        data.setdefault(r["model"], {})[int(r["steps"])] = float(r["gen_ppl"])
    ms = [8, 16, 32, 64, 128, 256, 512, 1024]
    style = [
        ("DUO-Distilled", "distilled DUO", C["dcd"], "-", 1.1, "D"),
        ("DiDi-Instruct", "DiDi-Instruct", C["didi"], "-", 1.1, "^"),
        ("MDLM", "MDLM, standard", C["std"], "--", 1.3, "o"),
        ("MDLM calibrated", r"MDLM$^\sharp$, same weights", C["sharp"], "-", 2.0, "o"),
    ]
    for key, label, color, ls, lw, mk in style:
        y = [data[key][m] for m in ms]
        ax.plot(ms, y, ls=ls, lw=lw, color=color, marker=mk, ms=2.4 if lw < 2 else 3.0,
                label=label, zorder=4 if key == "MDLM calibrated" else 3,
                mec="white", mew=0.3)
    ax.set(xscale="log", yscale="log", xlabel="Sampling steps", ylabel="GPT-2-Large gPPL ↓")
    ax.set_xticks([8, 16, 64, 256, 1024], ["8", "16", "64", "256", "1024"])
    ax.set_xticks([], minor=True)
    ax.set_yticks([20, 50, 100, 200, 500, 1000], ["20", "50", "100", "200", "500", "1000"])
    ax.set_yticks([], minor=True)
    ax.set_ylim(14, 2600)
    best = min(data["DUO-Distilled"][m] for m in ms)
    ax.axhline(best, color=C["dcd"], lw=0.7, ls=":", zorder=1, label="best distilled DUO")
    ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1.0), fontsize=5.8, handlelength=1.4,
              labelspacing=0.18, borderaxespad=0.05, handletextpad=0.35)
    ax.set_title("(a) Same weights, better sampler", loc="left")


def panel_groupeval(ax) -> None:
    # one row per budget, one inset per score: the (GEQ, GED) scatter was too crowded to pair points
    r = rows(ROOT / "group-eval/results/mdlm_pilot_gpt_oss_120b_20260725/pilot_results.csv")
    cells = {(int(x["updates_M"]), x["operating_point"]): x for x in r}
    ms = [16, 64, 256, 1024]
    ax.set_axis_off()
    ax.set_title("(b) Better and more diverse", loc="left")
    specs = [("geq", "GEQ ↑", (3.5, 6.3), [4, 5, 6], [0.0, 0.0, 0.44, 1.0]),
             ("ged", "GED ↑", (5.6, 9.4), [6, 7, 8, 9], [0.56, 0.0, 0.44, 1.0])]
    for i, (key, label, xlim, ticks, rect) in enumerate(specs):
        sub = ax.inset_axes(rect)
        for row, m in enumerate(ms):
            x0 = float(cells[(m, "reference")][key])
            x1 = float(cells[(m, "selected")][key])
            sub.axhline(row, color="#EDEDED", lw=0.5, zorder=0)
            sub.annotate("", xy=(x1, row), xytext=(x0, row), zorder=2,
                         arrowprops=dict(arrowstyle="-|>", color=C["sharp"], lw=0.9,
                                         shrinkA=2.6, shrinkB=2.6, mutation_scale=5.5))
            sub.scatter([x0], [row], s=14, color=C["std"], zorder=3, edgecolor="white", lw=0.5)
            sub.scatter([x1], [row], s=18, color=C["sharp"], zorder=4, edgecolor="white", lw=0.5)
        sub.set(xlim=xlim, ylim=(len(ms) - 0.45, -0.55), xlabel=label)
        sub.set_xticks(ticks)
        sub.set_yticks(range(len(ms)), [str(m) for m in ms] if i == 0 else [])
        if i == 0:
            sub.set_ylabel("Sampling steps")
        else:
            sub.tick_params(axis="y", length=0)
            sub.scatter([], [], s=14, color=C["std"], label="standard")
            sub.scatter([], [], s=18, color=C["sharp"], label=r"MDLM$^\sharp$")
            sub.legend(loc="lower left", fontsize=5.8, handletextpad=0.1, borderaxespad=0.2,
                       labelspacing=0.25)


def panel_fingerprint(ax, gppl: dict) -> None:
    cert = rows(ROOT / "results/calibration_certificate/summary/temperature_certificate.csv")
    names = {"public_mdlm_small": ("public", "public", "o"), "early_mdlm_small": ("early", "early", "s"),
             "late_mdlm_small": ("late", "late", "^")}
    ax.axhspan(0.2, 1.0, xmin=0.4, xmax=1.0, color=C["shade"], zorder=0)
    ax.axvline(0, color=C["std"], lw=0.6)
    ax.axhline(1, color=C["std"], lw=0.6)
    xs, ys = [], []
    for ck, (key, label, mk) in names.items():
        for r in cert:
            if r["checkpoint"] != ck:
                continue
            m = int(r["configured_updates"])
            xs.append(float(r["delta_cross_entropy_vs_tau1"]))
            ys.append(gppl[(key, m, 0.9, "fp64")] / gppl[(key, m, 1.0, "fp64")])
    ax.scatter(xs, ys, s=13, marker="o", color=C["sharp"], alpha=0.8, edgecolor="white",
               lw=0.35, zorder=3)
    ax.set(yscale="log", xlim=(-0.06, 0.09), ylim=(0.2, 2.0),
           xlabel="Δ posterior loss (nats/token)",
           ylabel="gPPL ratio, τ = .9 vs. 1")
    ax.set_yticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
    ax.set_yticks([], minor=True)
    ax.set_xticks([-0.05, 0, 0.05], ["−0.05", "0", "+0.05"])
    ax.text(0.085, 0.88, "all 15 cells:\nworse posterior,\nbetter samples",
            fontsize=5.9, ha="right", va="top", color=C["sharp"])
    ax.text(-0.055, 0.88, "calibration\nwould land\nhere", fontsize=5.9,
            ha="left", va="top", color=C["std"])
    ax.set_title("(c) Not a calibration fix", loc="left")


def overview(path: Path) -> None:
    g = sweep()
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2))
        fig.subplots_adjust(left=0.087, right=0.995, bottom=0.18, top=0.89, wspace=0.42)
        panel_successors(axes[0])
        panel_groupeval(axes[1])
        panel_fingerprint(axes[2], g)
        save(fig, path)


def less_is_more(path: Path) -> None:
    g = sweep()
    settings = [((1.0, "fp64"), "standard", C["std"], "--", 1.3),
                ((0.9, "fp32"), r"MDLM$^\sharp$", C["sharp"], "-", 1.6),
                ((0.8, "fp32"), "sharper, τ = .8", C["sharp2"], ":", 1.3)]
    pairs = [("late", "early", "(a) 44× more training: 915k vs. 20.8k steps"),
             ("late", "tiny", "(b) 5× more parameters: 170M vs. 34M")]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.78), sharey=True)
        fig.subplots_adjust(left=0.09, right=0.995, bottom=0.22, top=0.87, wspace=0.12)
        for ax, (big, small, title) in zip(axes, pairs):
            ax.axhspan(1.0, 1.6, color=C["warn"], zorder=0)
            ax.axhline(1.0, color=C["dark"], lw=0.6)
            for (tau, prec), label, color, ls, lw in settings:
                y = [g[(big, m, tau, prec)] / g[(small, m, tau, prec)] for m in STEPS]
                ax.plot(STEPS, y, ls=ls, color=color, lw=lw, marker="o", ms=2.0, label=label)
            ax.set(ylim=(0.35, 1.55))
            step_axis(ax)
            ax.set_title(title, loc="left")
            ax.text(0.98, 0.96, "less is more", transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.2, color=C["dcd"])
        axes[0].set_ylabel("gPPL ratio, more / less")
        axes[0].legend(loc="lower left", fontsize=6.0, handlelength=1.8, labelspacing=0.25)
        save(fig, path)


def _pair_matrix(agree: float) -> np.ndarray:
    return np.array([[agree / 2, (1 - agree) / 2], [(1 - agree) / 2, agree / 2]])


def _heat(fig, rect, mat, xt=False, yt=False) -> None:
    ax = fig.add_axes(rect)
    ax.imshow(mat, cmap=BLUES, vmin=0, vmax=0.5)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=5.9,
                color="white" if v >= 0.3 else C["dark"])
    ax.set_xticks([0, 1], ["0", "1"] if xt else ["", ""])
    ax.set_yticks([0, 1], ["0", "1"] if yt else ["", ""])
    ax.tick_params(length=0, pad=1.6, labelsize=5.6)
    if xt:
        ax.set_xlabel(r"$x_2$", fontsize=5.8, labelpad=0.5)
    if yt:
        ax.set_ylabel(r"$x_1$", fontsize=5.8, labelpad=1.5, rotation=0, va="center")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color(C["std"])


def theory(path: Path) -> None:
    s, c = 0.7, 0.5
    logit_s = math.log(s / (1 - s))
    lam = np.linspace(0.3, 3.4, 600)
    u = 1 / (1 + np.exp(-lam * logit_s))
    t = c / 2 + (1 - c) * u

    def kl(a, b):
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    post, out = (1 - c) * kl(s, u), kl(s, t)
    u_star = (s - c / 2) / (1 - c)
    lam_star = math.log(u_star / (1 - u_star)) / logit_s
    t_one = c / 2 + (1 - c) * s
    res = json.loads((ROOT / "theory/generic_suboptimality_results.json").read_text())
    lam_opt = np.array(res["summaries"]["generic"]["optimal_inverse_temperatures"])
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(5.5, 3.02))
        # (a) the mixture over reveal paths, drawn as 2x2 joint laws of (x1, x2)
        fig.text(0.0, 0.985, "(a) The sampler is a mixture over reveal paths", fontsize=8.0,
                 fontweight="bold", va="top")
        hw, hh = 0.37 / 5.5, 0.37 / 3.02
        xs = [0.185, 0.335, 0.485, 0.66]
        heads = [("same round (½)", "independent draws"),
                 ("separate rounds (½)", "one given the other"),
                 (r"sampler output $\pi_\lambda$", ""), (r"target $P_s$", "")]
        for x, (h1, h2) in zip(xs, heads):
            fig.text(x + hw / 2, 0.912, h1, ha="center", va="baseline", fontsize=6.0,
                     color=C["dark"])
            if h2:
                fig.text(x + hw / 2, 0.884, h2, ha="center", va="baseline", fontsize=5.6,
                         color=C["std"])
        rows_ = [(s, t_one, 0.785, r"$\lambda = 1$", "Bayes posterior", "≠",
                  f"too random: agree {t_one:.2f}\nposterior loss 0\noutput KL "
                  f"{kl(s, t_one):.3f}".replace("0.", ".")),
                 (u_star, s, 0.625, rf"$\lambda_\star = {lam_star:.2f}$", "sharpened", "=",
                  f"exact: agree {s:.2f}\nposterior loss "
                  f"{(1 - c) * kl(s, u_star):.3f}\noutput KL 0".replace("0.", "."))]
        for i, (agree_seq, agree_out, yc, name, sub, rel, note) in enumerate(rows_):
            y0 = yc - hh / 2
            fig.text(0.0, yc + 0.012, name, fontsize=7.0, fontweight="bold", va="bottom",
                     color=C["sharp"] if i else C["std"])
            fig.text(0.0, yc + 0.006, sub, fontsize=6.0, va="top", color=C["dark"])
            last = i == 1
            _heat(fig, [xs[0], y0, hw, hh], _pair_matrix(0.5), xt=last, yt=True)
            _heat(fig, [xs[1], y0, hw, hh], _pair_matrix(agree_seq), xt=last)
            _heat(fig, [xs[2], y0, hw, hh], _pair_matrix(agree_out), xt=last)
            _heat(fig, [xs[3], y0, hw, hh], _pair_matrix(s), xt=last)
            for xa, sym in [((xs[0] + hw + xs[1]) / 2, "+"), ((xs[1] + hw + xs[2]) / 2, "="),
                            ((xs[2] + hw + xs[3]) / 2, rel)]:
                fig.text(xa, yc, sym, ha="center", va="center", fontsize=9.5,
                         color=C["dcd"] if sym == "≠" else C["dark"])
            fig.text(xs[3] + hw + 0.035, yc, note, fontsize=6.0, va="center", ha="left",
                     color=C["sharp"] if i else C["dcd"], linespacing=1.25)
        # (b) two objectives, (c) generic suboptimality
        ax = fig.add_axes([0.075, 0.105, 0.43, 0.30])
        ax.plot(lam, post, color=C["std"], lw=1.5)
        ax.plot(lam, out, color=C["sharp"], lw=1.9)
        ax.scatter([1], [0], color=C["std"], s=16, zorder=4)
        ax.scatter([lam_star], [0], color=C["sharp"], s=16, zorder=4)
        ax.axvline(1, color=C["std"], lw=0.5, ls=":")
        ax.axvline(lam_star, color=C["sharp"], lw=0.5, ls=":")
        ax.text(0.96, 0.147, "Bayes\nposterior", fontsize=5.9, color=C["std"], va="top",
                ha="right")
        ax.text(lam_star - 0.05, 0.147, "exact\ngenerator", fontsize=5.9, color=C["sharp"],
                va="top", ha="right")
        ax.text(3.38, 0.075, "posterior\nloss $C_N$\n(training)", fontsize=5.9,
                color=C["std"], ha="right", va="top")
        ax.text(0.37, 0.064, "output error\n(generation)",
                fontsize=5.9, color=C["sharp"], ha="left", va="bottom")
        ax.set(xlabel=r"Inverse temperature $\lambda = 1/\tau$", ylabel="Excess loss (nats)",
               ylim=(-0.008, 0.15), xlim=(0.3, 3.4))
        ax.set_xticks([0.5, 1, 2, lam_star, 3], ["0.5", "1", "2", f"{lam_star:.2f}", "3"])
        ax.set_title("(b) Two objectives, two optima", loc="left")
        ax = fig.add_axes([0.63, 0.105, 0.365, 0.30])
        bins = 1.05 ** np.arange(-5, 25)
        ax.hist(np.clip(lam_opt, 0.8, 3.2), bins=bins, color=C["sharp"], alpha=0.85,
                edgecolor="white", lw=0.4)
        ax.axvline(1, color=C["dark"], lw=0.8, ls="--")
        g = res["summaries"]["generic"]
        top = ax.get_ylim()[1]
        ax.text(3.1, top * 0.93, f"sharpening improves\n{g['sharpening_improves']} of {g['targets']} targets",
                fontsize=6.0, color=C["sharp"], va="top", ha="right")
        ax.text(0.97, top * 0.93, "λ = 1", fontsize=6.0, ha="right", va="top")
        ax.set(xscale="log", xlabel=r"Output-optimal $\lambda_\star$ of a Bayes denoiser",
               ylabel="Random targets", xlim=(0.8, 3.2))
        ax.set_xticks([1, 1.5, 2, 3], ["1", "1.5", "2", "3"])
        ax.set_xticks([], minor=True)
        ax.set_title("(c) Generic, not a special case", loc="left")
        save(fig, path)


# ---------------------------------------------------------------- appendix figures

def tiny_steps(path: Path) -> None:
    g = sweep()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.0, 1.85))
        ax.plot(STEPS, [g[("tiny", m, 1.0, "fp64")] for m in STEPS], color=C["std"], ls="--",
                lw=1.3, marker="o", ms=2.2, label="standard (τ = 1, fp64)")
        ax.plot(STEPS, [g[("tiny", m, 0.9, "fp32")] for m in STEPS], color=C["sharp"], lw=1.7,
                marker="o", ms=2.4, label="sharpened (τ = .9, fp32)")
        step_axis(ax)
        ax.set_yscale("log")
        ax.set_yticks([50, 100, 200, 500, 1000, 2000, 5000],
                      ["50", "100", "200", "500", "1000", "2000", "5000"])
        ax.set_yticks([], minor=True)
        ax.set_ylabel("GPT-2 gPPL ↓")
        ax.legend(loc="upper right")
        save(fig, path)


def tau_frontier(path: Path) -> None:
    taus = [0.5, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    data = {}
    for r in rows(ROOT / "paper_data/mdlm_sampling_grid.csv"):
        if r["ckpt"] == "public":
            data[(int(r["N"]), round(float(r["tau"]), 3), r["prec"])] = (
                float(r["gPPL"]), float(r["entropy"]))
    budgets = {8: "#9DB9D8", 128: C["sharp2"], 1024: C["sharp"]}
    with plt.rc_context(RC):
        fig, (qa, ea) = plt.subplots(2, 1, figsize=(2.64, 3.0), sharex=True,
                                     gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.08})
        for m, color in budgets.items():
            for prec, ls in (("fp64", "-"), ("fp32", "--")):
                pts = [(tau, *data[(m, tau, prec)]) for tau in taus]
                tt = np.array([p[0] for p in pts])
                gp = np.array([p[1] for p in pts])
                hh = np.array([p[2] for p in pts])
                ok = hh >= 5.0
                for ax, y in ((qa, gp), (ea, hh)):
                    ax.plot(tt, y, color=color, ls=ls, lw=1.2)
                    ax.scatter(tt[ok], y[ok], color=color, s=9, zorder=3)
                    ax.scatter(tt[~ok], y[~ok], facecolor="white", edgecolor=color, lw=0.8,
                               s=11, zorder=3)
        for ax in (qa, ea):
            ax.axvline(1.0, color=C["std"], lw=0.6, ls=":")
            ax.axvline(0.9, color=C["sharp"], lw=0.6, ls=":")
        qa.set_yscale("log")
        qa.set_yticks([10, 100, 1000, 10000], ["10", "100", "1k", "10k"])
        qa.set_ylabel("GPT-2 gPPL ↓")
        ea.text(0.89, 1.62, "MDLM$^\\sharp$", fontsize=5.9, color=C["sharp"], ha="right",
                va="bottom")
        ea.text(1.01, 1.62, "standard", fontsize=5.9, color=C["std"], ha="left", va="bottom")
        ea.axhline(5.0, color=C["dcd"], lw=0.7, ls="--")
        ea.text(0.52, 5.08, "H = 5 guard", fontsize=5.9, color=C["dcd"], va="bottom")
        ea.set(ylim=(1.5, 7.0), xlabel="Sampling temperature τ", ylabel="Token entropy H")
        ea.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                      ["0.5", "0.6", "0.7", "0.8", "0.9", "1", "1.1", "1.2"])
        handles = [plt.Line2D([], [], color=col, lw=1.2, label=f"M = {m}") for m, col in budgets.items()]
        handles += [plt.Line2D([], [], color=C["dark"], lw=1.0, ls="-", label="fp64"),
                    plt.Line2D([], [], color=C["dark"], lw=1.0, ls="--", label="fp32"),
                    plt.Line2D([], [], color=C["dark"], marker="o", ls="none", ms=3.2,
                               mfc="white", label="H < 5")]
        qa.legend(handles=handles, loc="upper left", ncol=2, fontsize=5.9, columnspacing=0.8,
                  handlelength=1.6, labelspacing=0.2)
        fig.align_ylabels([qa, ea])
        save(fig, path)


def sampling_cost(path: Path) -> None:
    data: dict[str, dict[int, tuple[float, float]]] = {}
    for r in rows(ROOT / "results/sampling_cost/summary_mdlm.tsv", delimiter="\t"):
        data.setdefault(r["label"], {})[int(r["steps"])] = (
            float(r["peak_allocated_mib"]) / 1024.0, float(r["wall_seconds_per_generation"]))
    series = [("mdlm_small_fp64", "standard (τ = 1, fp64)", C["std"], "--", 1.3),
              ("mdlm_small_tau0p9_fp32", r"MDLM$^\sharp$ (τ = .9, fp32)", C["sharp"], "-", 1.7)]
    with plt.rc_context(RC):
        fig, (ma, ta) = plt.subplots(2, 1, figsize=(2.64, 3.0), sharex=True,
                                     gridspec_kw={"height_ratios": [1.0, 1.35], "hspace": 0.08})
        for key, label, color, ls, lw in series:
            ms = sorted(data[key])
            ma.plot(ms, [data[key][m][0] for m in ms], color=color, ls=ls, lw=lw, marker="o",
                    ms=2.2, label=label)
            ta.plot(ms, [data[key][m][1] for m in ms], color=color, ls=ls, lw=lw, marker="o",
                    ms=2.2, label=label)
        ma.set(ylim=(0, 15), ylabel="Memory (GiB) ↓")
        ma.set_yticks([0, 5, 10, 15])
        ma.legend(loc="lower right", fontsize=6.0)
        ta.set_yscale("log")
        ta.set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
        ta.set_ylabel("Seconds / sequence ↓")
        step_axis(ta, label="Sampling steps M")
        fig.align_ylabels([ma, ta])
        save(fig, path)


def cross_model_entropy(path: Path) -> None:
    data: dict[str, dict[int, float]] = {}
    for r in rows(ROOT / "paper_data/cross_model_gp_may05.csv"):
        data.setdefault(r["model"], {})[int(r["steps"])] = float(r["entropy"])
    style = [("MDLM", "MDLM, standard", C["std"], "--", 1.2, "o"),
             ("MDLM calibrated", r"MDLM$^\sharp$", C["sharp"], "-", 1.8, "o"),
             ("DUO", "DUO", C["duo"], "-", 1.0, "s"),
             ("DUO-Distilled", "DUO-DCD", C["dcd"], "-", 1.0, "D"),
             ("SEDD", "SEDD", C["ochre"], "-", 1.0, "v"),
             ("DiDi-Instruct", "DiDi-Instruct", C["didi"], "-", 1.0, "^"),
             ("DiDi-Instruct calibrated", "DiDi-Instruct, sharpened", C["didi2"], "-", 1.6, "^"),
             ("Autoregressive", "GPT-2 generator", C["dark"], ":", 0.9, None)]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.4, 1.9))
        for key, label, color, ls, lw, mk in style:
            ms = [m for m in STEPS if m in data[key]]
            ax.plot(ms, [data[key][m] for m in ms], color=color, ls=ls, lw=lw, marker=mk,
                    ms=2.4, mec="white", mew=0.3, label=label,
                    zorder=4 if key == "MDLM calibrated" else 3)
        ax.axhline(5.0, color=C["dcd"], lw=0.7, ls="--", zorder=1)
        ax.text(1.05, 5.012, "H = 5 guard", fontsize=5.9, color=C["dcd"], va="bottom")
        step_axis(ax, label="Sampling steps M")
        ax.set(ylabel="Token entropy H (nats)", ylim=(4.85, 6.02))
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.2, handlelength=1.8)
        save(fig, path)


def _task_panels(path: Path, csv_path: Path, models: list[tuple[str, str, str, str]],
                 legend_below: bool) -> None:
    agg: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for r in rows(csv_path):
        agg.setdefault((r["model"], r["task"]), []).append(
            (float(r["temperature"]), float(r["accuracy_pct"]), float(r["seed_sem_pct"])))
    tasks = [("gsm8k", "(a) GSM8K exact match"), ("humaneval", "(b) HumanEval pass@1")]
    ticks = {0.0: "0", 0.25: ".25", 0.5: ".5", 0.75: ".75", 1.0: "1", 1.5: "1.5", 2.0: "2"}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.85))
        fig.subplots_adjust(wspace=0.22)
        for ax, (task, title) in zip(axes, tasks):
            for key, label, color, mk in models:
                pts = sorted(agg[(key, task)])
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                            color=color, marker=mk, ms=2.8, lw=1.2, capsize=1.6, elinewidth=0.8,
                            label=label)
            top = max(p[0] for key, *_ in models for p in agg[(key, task)])
            ax.set_xticks([x for x in ticks if x <= top], [v for x, v in ticks.items() if x <= top])
            ax.set_xlim(-0.06, top + 0.06)
            ax.set(xlabel="Sampling temperature τ (0 = greedy)")
            ax.grid(axis="y", color="#E4E4E4", lw=0.5)
            ax.set_title(title, loc="left")
        axes[0].set_ylabel("Accuracy (%)")
        if legend_below:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(models),
                       bbox_to_anchor=(0.5, -0.21), fontsize=6.4)
        else:
            axes[0].legend(loc="lower left", fontsize=6.4)
        save(fig, path)


def large_model_stress(path: Path) -> None:
    _task_panels(path, ROOT / "paper_data/large_model/diffusion_temperature.csv",
                 [("llada", "LLaDA-8B-Instruct", C["sharp"], "o"),
                  ("dream", "Dream-v0-Base-7B", C["dcd"], "s")], legend_below=False)


def autoregressive_stress(path: Path) -> None:
    _task_panels(path, ROOT / "paper_data/large_model/autoregressive_temperature.csv",
                 [("llama31_8b", "Llama-3.1-8B", C["sharp"], "o"),
                  ("ministral3_8b", "Ministral-3-8B", C["dcd"], "s"),
                  ("qwen35_9b", "Qwen3.5-9B", C["didi"], "D"),
                  ("gemma4_12b", "Gemma-4-12B", C["ochre"], "^")], legend_below=True)


FIGURES = {
    "overview.pdf": overview, "less_is_more.pdf": less_is_more, "theory.pdf": theory,
    "tiny_calibration_vs_steps.pdf": tiny_steps, "tau_frontier_sweep.pdf": tau_frontier,
    "sampling_cost_combined.pdf": sampling_cost,
    "cross_model_entropy_vs_steps.pdf": cross_model_entropy,
    "large_model_temperature_stress.pdf": large_model_stress,
    "autoregressive_temperature_stress.pdf": autoregressive_stress,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    destination = p.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path,
                             help="output directory (default: build/figures in this release)")
    destination.add_argument("--paper-dir", type=Path,
                             help="export figures to <paper-dir>/figs")
    args = p.parse_args()
    figs = (args.paper_dir.resolve() / "figs" if args.paper_dir else
            (args.output_dir or ROOT / "build/figures").resolve())
    figs.mkdir(parents=True, exist_ok=True)
    for name, build in FIGURES.items():
        build(figs / name)
    print(f"wrote {len(FIGURES)} figures to {figs}: " + ", ".join(FIGURES))


if __name__ == "__main__":
    main()
