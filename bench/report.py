"""Render ``results/*.json`` into docs/RESULTS.md, the README key-results table and plots.

Usage: ``uv run python -m bench.report``

This script never computes a benchmark number: it only formats values that already exist in the
result files (each of which carries hardware info and the git commit). Anything missing is shown
as ``TBD (not yet measured)``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bench.common import RESULTS_DIR, ROOT

TBD = "TBD (not yet measured)"
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

# Validated categorical palette (light mode), fixed slot order: blue, orange, aqua, violet.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

Docs = dict[str, dict[str, Any] | None]
RESULT_NAMES = (
    "lfw_verification",
    "open_set",
    "latency",
    "robustness",
    "quality_calibration",
    "index_scaling",
    "index_real",
    "precision",
)


def load_results(directory: Path = RESULTS_DIR) -> Docs:
    docs: Docs = {}
    for name in RESULT_NAMES:
        path = directory / f"{name}.json"
        docs[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return docs


def get(doc: dict[str, Any] | None, *keys: str | int) -> Any:
    """Nested lookup that returns ``None`` instead of raising when a key is absent."""
    cur: Any = doc
    for key in keys:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def pct(x: float | None, digits: int = 2) -> str:
    return TBD if x is None else f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 3) -> str:
    return TBD if x is None else f"{x:.{digits}f}"


def ms(x: float | None) -> str:
    return TBD if x is None else f"{x:.1f} ms"


def provenance(doc: dict[str, Any] | None) -> str:
    if doc is None:
        return TBD
    git = get(doc, "env", "git") or {}
    commit = (git.get("commit") or "unknown")[:8]
    dirty = " (uncommitted changes)" if git.get("dirty") else ""
    return f"`{commit}`{dirty}, {get(doc, 'env', 'timestamp_utc')}"


# --- key results ---------------------------------------------------------------------------------


def key_results_table(docs: Docs) -> str:
    """The README table: one row per headline number, each traceable to a result file."""
    ver, ops, lat = docs["lfw_verification"], docs["open_set"], docs["latency"]
    rows: list[tuple[str, str, str]] = []

    acc, std = get(ver, "result", "accuracy_mean"), get(ver, "result", "accuracy_std")
    rows.append(
        (
            "LFW verification accuracy (10-fold, threshold from the other 9 folds)",
            TBD if acc is None else f"{100 * acc:.2f}% +/- {100 * std:.2f}% (std over folds)",
            "`results/lfw_verification.json`",
        )
    )
    rows.append(("LFW EER / AUC", _eer_auc(ver), "same"))
    for far in ("0.001", "0.0001"):
        ext = get(ver, "result", "tar_at_far_extended_impostors", "targets", far)
        std_t = get(ver, "result", "tar_at_far_standard_protocol", "targets", far)
        floor = get(ver, "result", "tar_at_far_standard_protocol", "far_resolution_floor")
        std_text = TBD
        if std_t is not None and floor is not None:
            std_text = (
                f"{pct(std_t['tar_mean'])}"
                if float(far) >= floor
                else f"not resolvable (needs FAR < {floor:.1e})"
            )
        ext_text = (
            TBD
            if ext is None
            else f"{pct(ext['tar_mean'])} (FAR measured {ext['far_measured_mean']:.1e})"
        )
        rows.append(
            (
                f"TAR @ FAR={float(far):g}: standard protocol / extended impostors",
                f"{std_text} / {ext_text}",
                "same",
            )
        )
    rows.append(
        (
            "Face detection failure rate (LFW images)",
            _failure_rate(ver),
            "same",
        )
    )
    exp = get(ops, "result", "experiments", "enroll1_max", "summary")
    rows.append(
        (
            "Open-set rank-1 (closed-set) vs FNIR @ FPIR=1e-1 / 1e-2",
            TBD
            if exp is None
            else f"{pct(exp['rank1_closed_set']['mean'])} vs {pct(exp['fnir_at_fpir']['0.1']['mean'])} / {pct(exp['fnir_at_fpir']['0.01']['mean'])}",
            "`results/open_set.json`",
        )
    )
    warm = get(lat, "result", "scenarios", "lfw_250x250")
    rows.append(
        (
            "CPU latency p50 / p95, 250x250 image: detect",
            TBD
            if warm is None
            else f"{ms(warm['detect']['p50_ms'])} / {ms(warm['detect']['p95_ms'])}",
            "`results/latency.json`",
        )
    )
    rows.append(
        (
            "CPU latency p50 / p95: embed (batch 1)",
            TBD
            if warm is None
            else f"{ms(warm['embed_single']['p50_ms'])} / {ms(warm['embed_single']['p95_ms'])}",
            "same",
        )
    )
    rows.append(
        (
            "GPU latency",
            TBD if get(lat, "result", "gpu") is None else str(get(lat, "result", "gpu")),
            "same",
        )
    )
    return "\n".join(
        [
            "| Metric | Result | Source |",
            "|---|---|---|",
            *[f"| {a} | {b} | {c} |" for a, b, c in rows],
        ]
    )


def _eer_auc(ver: dict[str, Any] | None) -> str:
    eer, auc = get(ver, "result", "pooled", "eer"), get(ver, "result", "pooled", "auc")
    return TBD if eer is None else f"{pct(eer)} / {auc:.4f}"


def _failure_rate(ver: dict[str, Any] | None) -> str:
    total, failed = (
        get(ver, "result", "detection", "images_total"),
        get(ver, "result", "detection", "images_failed"),
    )
    if total is None:
        return TBD
    return f"{failed} of {total} images ({pct(failed / total)}); failed pairs are counted as errors"


# --- plots ---------------------------------------------------------------------------------------


def _style(ax: Axes, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_roc(docs: Docs, out: Path) -> Path | None:
    roc = get(docs["lfw_verification"], "result", "pooled", "roc")
    if roc is None:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(roc["far"], roc["tar"], color=BLUE, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0.9, 1.001)
    _style(ax, "False accept rate (log scale)", "True accept rate")
    ax.set_title("LFW verification ROC (all 6000 pairs pooled)", color=INK, fontsize=11, loc="left")
    ax.annotate("ArcFace R50 (this repo)", (1e-3, 0.97), color=INK2, fontsize=9)
    return _save(fig, out)


def plot_det(docs: Docs, out: Path) -> Path | None:
    runs = get(docs["open_set"], "result", "experiments", "enroll1_max", "runs")
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    det = runs[0]["det"]
    ax.plot(det["fpir"], det["fnir"], color=BLUE, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1)
    _style(ax, "FPIR (non-enrolled probes matched)", "FNIR (enrolled probes missed)")
    ax.set_title(
        "Open-set identification DET (seed 0, one enrollment image)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    return _save(fig, out)


def plot_robustness(docs: Docs, out: Path) -> Path | None:
    settings = get(docs["robustness"], "result", "settings")
    if not settings:
        return None
    families = sorted({v["family"] for k, v in settings.items() if k != "clean"})
    cols = 4
    rows = (len(families) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    clean = get(settings, "clean", "test", "accuracy")
    for ax, family in zip(axes.ravel(), families, strict=False):
        ax.set_visible(True)
        items = sorted(
            (
                (v["level"], v)
                for k, v in settings.items()
                if k != "clean" and v["family"] == family
            ),
            key=lambda t: t[0],
        )
        xs = [lvl for lvl, _ in items]
        for split, color, label in (("test", BLUE, "held-out"), ("val", ORANGE, "validation")):
            ys = [v[split]["accuracy"] for _, v in items if split in v]
            if ys:
                ax.plot(
                    xs[: len(ys)],
                    ys,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=label,
                )
        if clean is not None:
            ax.axhline(clean, color=MUTED, linewidth=1, linestyle=":")
        _style(ax, family, "accuracy")
        ax.set_ylim(0.4, 1.02)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Accuracy vs degradation severity (fixed threshold; small samples, see CIs in RESULTS.md)",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    return _save(fig, out)


def plot_quality_curves(docs: Docs, out: Path) -> Path | None:
    curves = get(docs["quality_calibration"], "result", "curves_test")
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    palette = [BLUE, ORANGE, AQUA, VIOLET]
    for color, (name, rows) in zip(palette, list(curves.items())[:4], strict=False):
        ax.plot(
            [r["reject_rate"] for r in rows],
            [r["accuracy_kept"] for r in rows],
            color=color,
            linewidth=2,
            label=name,
        )
    _style(ax, "Rejected fraction of held-out pairs", "Accuracy of the pairs kept")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reject rate vs accuracy (held-out)", color=INK, fontsize=11, loc="left")
    return _save(fig, out)


def plot_latency(docs: Docs, out: Path) -> Path | None:
    scenarios = get(docs["latency"], "result", "scenarios")
    if not scenarios:
        return None
    stages = ["decode", "detect", "align", "embed_single"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    width = 0.38
    for i, (key, color, label) in enumerate(
        (("lfw_250x250", BLUE, "250x250 image"), ("upload_1920x1080", ORANGE, "1920x1080 upload"))
    ):
        if key not in scenarios:
            continue
        p50 = [scenarios[key][s]["p50_ms"] for s in stages]
        p95 = [scenarios[key][s]["p95_ms"] for s in stages]
        xs = [j + (i - 0.5) * width for j in range(len(stages))]
        ax.bar(xs, p50, width=width * 0.9, color=color, label=f"{label} p50")
        ax.errorbar(
            xs,
            p50,
            yerr=[[0] * len(p50), [b - a for a, b in zip(p50, p95, strict=True)]],
            fmt="none",
            ecolor=INK2,
            capsize=3,
            linewidth=1,
        )
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(["decode", "detect", "align", "embed"])
    ax.set_yscale("log")
    _style(ax, "stage (CPU)", "milliseconds (bar = p50, whisker = p95)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Per-stage latency", color=INK, fontsize=11, loc="left")
    return _save(fig, out)


def plot_index(docs: Docs, out: Path) -> Path | None:
    sizes = get(docs["index_scaling"], "result", "sizes")
    if not sizes:
        return None
    largest = sizes[max(sizes, key=int)]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    colors = {"flat": BLUE, "hnsw": ORANGE, "ivf": AQUA}
    for entry in largest:
        pts = [
            (r["latency_single_query_k10"]["p50_ms"], r["recall@1_planted"]) for r in entry["sweep"]
        ]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=colors[entry["backend"]],
            linewidth=2,
            marker="o",
            markersize=5,
            label=entry["backend"],
        )
    ax.set_xscale("log")
    _style(ax, "single-query p50 latency, ms (log)", "recall@1 vs exact (planted queries)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(
        f"Index trade-off at {max(sizes, key=int)} synthetic vectors",
        color=INK,
        fontsize=11,
        loc="left",
    )
    return _save(fig, out)


# --- RESULTS.md ----------------------------------------------------------------------------------


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"


def verification_section(docs: Docs) -> str:
    doc = docs["lfw_verification"]
    if doc is None:
        return _section("LFW verification (1:1)", TBD)
    r = doc["result"]
    lines = [
        f"Run {provenance(doc)}. Model `{r['model']}`. Multi-face policy: {r['detection']['multi_face_policy']}.",
        "",
        f"- Accuracy (10-fold): **{pct(r['accuracy_mean'])} +/- {pct(r['accuracy_std'])}** (std over folds; SEM {pct(r['accuracy_sem'])}). "
        f"Failed pairs count as errors; accuracy on valid pairs only: {pct(r['accuracy_valid_only_mean'])}.",
        f"- Detection failures: {r['n_failed_pairs']} of {r['n_pairs']} pairs ({pct(r['detection_failure_rate_pairs'])}).",
        f"- Pooled EER {pct(r['pooled']['eer'])} (failure-aware {pct(r['pooled']['eer_failure_aware'])}), AUC {r['pooled']['auc']:.4f}.",
        f"- Leakage check: {'passed' if r['leakage_check']['passed'] else 'FAILED'} "
        f"({r['leakage_check']['identities_in_multiple_folds']} identities in several folds, "
        f"{r['leakage_check']['images_in_multiple_folds']} shared images).",
        "",
        "![ROC](img/roc.png)",
        "",
        "| FAR target | Standard protocol TAR | Extended-impostor TAR (FAR measured) |",
        "|---|---|---|",
    ]
    floor = r["tar_at_far_standard_protocol"]["far_resolution_floor"]
    for far, std in r["tar_at_far_standard_protocol"]["targets"].items():
        ext = get(r, "tar_at_far_extended_impostors", "targets", far)
        std_text = (
            pct(std["tar_mean"]) if float(far) >= floor else f"not resolvable (floor {floor:.1e})"
        )
        ext_text = (
            TBD
            if ext is None
            else f"{pct(ext['tar_mean'])} +/- {pct(ext['tar_std'])} ({ext['far_measured_mean']:.1e})"
        )
        lines.append(f"| {float(far):g} | {std_text} | {ext_text} |")
    lines += [
        "",
        "The standard protocol has 3000 impostor pairs, so FAR below about 3.3e-4 cannot be measured there; "
        "the *extended* column uses every cross-identity image pair and is **not** part of the standard protocol.",
        "LFW is near-saturated and demographically skewed; nothing here says anything about fairness.",
    ]
    return _section("LFW verification (1:1)", "\n".join(lines))


def open_set_section(docs: Docs) -> str:
    doc = docs["open_set"]
    if doc is None:
        return _section("Open-set identification (1:N)", TBD)
    lines = [
        f"Run {provenance(doc)}. Mean +/- std over seeded enrolled/non-enrolled splits.",
        "",
        "![DET](img/det.png)",
        "",
    ]
    lines += [
        "| Experiment | Rank-1 (closed-set) | FNIR @ FPIR 1e-1 | FNIR @ FPIR 1e-2 | non-mated probes |",
        "|---|---|---|---|---|",
    ]
    for label, exp in doc["result"]["experiments"].items():
        s = exp["summary"]
        lines.append(
            f"| {label} | {pct(s['rank1_closed_set']['mean'])} +/- {pct(s['rank1_closed_set']['std'])} "
            f"| {pct(s['fnir_at_fpir']['0.1']['mean'])} | {pct(s['fnir_at_fpir']['0.01']['mean'])} | {s['n_nonmated_probes']:.0f} |"
        )
    lines += [
        "",
        "Closed-set rank-1 only asks whether an enrolled person's best match is correct. It ignores people who are "
        "not enrolled, which is the case that matters when a system must say 'nobody here'; FNIR at a fixed FPIR "
        "measures that, and is the number to read.",
    ]
    return _section("Open-set identification (1:N)", "\n".join(lines))


def latency_section(docs: Docs) -> str:
    doc = docs["latency"]
    if doc is None:
        return _section("Latency", TBD)
    r = doc["result"]
    lines = [
        f"Run {provenance(doc)}. Providers: {r['providers']}. Warm-up calls excluded: {r['warmup_calls_excluded']}.",
        "",
        "![latency](img/latency.png)",
        "",
    ]
    lines += ["| Scenario | Stage | p50 | p95 | p99 |", "|---|---|---|---|---|"]
    for name, sc in r["scenarios"].items():
        for stage in ("decode", "detect", "align", "embed_single"):
            s = sc[stage]
            lines.append(
                f"| {name} | {stage} | {ms(s['p50_ms'])} | {ms(s['p95_ms'])} | {ms(s['p99_ms'])} |"
            )
    lines += ["", "| Embed batch | p50 per batch | p50 per image |", "|---|---|---|"]
    for size, s in r["embed_batches"].items():
        lines.append(f"| {size} | {ms(s['p50_ms'])} | {ms(s['per_image_p50_ms'])} |")
    lines += ["", "| Flat search, gallery size | p50 | p95 | p99 |", "|---|---|---|---|"]
    for size, s in r["search_flat"].items():
        lines.append(f"| {size} | {ms(s['p50_ms'])} | {ms(s['p95_ms'])} | {ms(s['p99_ms'])} |")
    cold = r["cold_start"]
    lines += [
        "",
        f"Cold start: detector load {cold['detector_load_s']:.2f} s, embedder load {cold['embedder_load_s']:.2f} s; "
        f"first inference {cold['detector_first_inference_ms']:.0f} ms (detector), {cold['embedder_first_inference_ms']:.0f} ms (embedder).",
        f"GPU: {r['gpu']}",
    ]
    return _section("Latency", "\n".join(lines))


def robustness_section(docs: Docs) -> str:
    doc = docs["robustness"]
    if doc is None:
        return _section("Robustness slices", TBD)
    r = doc["result"]
    lines = [
        f"Run {provenance(doc)}. Fixed threshold {r['threshold']:.4f} ({r['threshold_source']}). "
        f"Samples: {r['samples']}. **Small samples**: differences of a few points are not statistically significant.",
        "",
        "![robustness](img/robustness.png)",
        "",
        "| Setting | Held-out accuracy (95% CI) | Failure rate |",
        "|---|---|---|",
    ]
    for name, entry in r["settings"].items():
        t = entry.get("test")
        if t is None:
            continue
        lo, hi = t["accuracy_ci95"]
        lines.append(
            f"| {name} | {pct(t['accuracy'])} ({pct(lo)} to {pct(hi)}) | {pct(t['failure_rate'])} |"
        )
    return _section("Robustness slices", "\n".join(lines))


def quality_section(docs: Docs) -> str:
    doc = docs["quality_calibration"]
    if doc is None:
        return _section("Quality gates", TBD)
    r = doc["result"]
    enabled = {k: v for k, v in r["thresholds"].items() if isinstance(v, float)}
    held = r["held_out"]
    lines = [
        f"Run {provenance(doc)}. Method: fixed decision threshold {r['tau']:.4f} from clean validation pairs; thresholds "
        "chosen on a validation set (clean pairs of folds 0-1 plus degraded probes) by cross-fitted gain with a Bonferroni-corrected CI; "
        "evaluated once on held-out folds 2-9 plus degraded test probes.",
        "",
        f"Calibrated gates kept: {enabled if enabled else 'none (no gate showed a significant gain)'}.",
        f"Held-out: accuracy of all pairs {pct(held['accuracy_all'])}, reject rate {pct(held['reject_rate'])}, accuracy of kept pairs {pct(held['accuracy_kept'])}.",
        "",
        "![quality curves](img/quality_curves.png)",
    ]
    return _section("Quality gates", "\n".join(lines))


def index_section(docs: Docs) -> str:
    doc = docs["index_scaling"]
    if doc is None:
        return _section("Index scaling", TBD)
    r = doc["result"]
    lines = [
        f"Run {provenance(doc)}. Synthetic random unit vectors, dim {r['dim']}. 'planted' queries are noisy copies of a gallery vector; "
        "'independent' queries have no planted neighbour (a worst case for approximate search, reported for honesty).",
        "",
        "![index](img/index.png)",
        "",
        "| Gallery | Backend | Search params | recall@1 planted | recall@10 planted | recall@1 independent | p50 | p95 | build | delete+rebuild |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for size, entries in r["sizes"].items():
        for entry in entries:
            for row in entry["sweep"]:
                lat = row["latency_single_query_k10"]
                lines.append(
                    f"| {size} | {entry['backend']} | {row['search_params'] or '-'} | {num(row['recall@1_planted'])} | {num(row['recall@10_planted'])} "
                    f"| {num(row['recall@1_independent'])} | {ms(lat['p50_ms'])} | {ms(lat['p95_ms'])} | {entry['build_seconds']:.2f} s "
                    f"| {entry['first_search_after_delete_seconds']:.2f} s |"
                )
    return _section("Index scaling", "\n".join(lines))


def precision_section(docs: Docs) -> str:
    doc = docs["precision"]
    if doc is None:
        return _section("fp16 vs fp32", TBD)
    r = doc["result"]
    a, v = r["embedding_agreement"], r["verification"]
    lines = [
        f"Run {provenance(doc)}. {r['scope']}. Conversion: `{r['fp16_conversion']}`.",
        "",
        f"- Per-image cosine fp32 vs fp16 over {a['n']:.0f} crops: min {a['cosine_min']:.6f}, 1st percentile {a['cosine_p1']:.6f}, mean {a['cosine_mean']:.6f}.",
        f"- LFW accuracy: fp32 {pct(v['fp32']['accuracy_mean'])}, fp16 {pct(v['fp16']['accuracy_mean'])}; "
        f"pairs whose decision flips at the fp32 threshold: {v['decision_flips_at_fp32_mean_threshold']['pairs_flipped']:.0f} "
        f"of {v['decision_flips_at_fp32_mean_threshold']['pairs_compared']:.0f}.",
        f"- Model size: {r['model_bytes']['fp32'] / 1e6:.1f} MB (fp32) vs {r['model_bytes']['fp16'] / 1e6:.1f} MB (fp16).",
        "",
        "| CPU embed time | fp32 p50 per image | fp16 p50 per image |",
        "|---|---|---|",
    ]
    for b in ("1", "16"):
        lines.append(
            f"| batch {b} | {ms(r['cpu_timing']['fp32'][b]['median_ms_per_image'])} | {ms(r['cpu_timing']['fp16'][b]['median_ms_per_image'])} |"
        )
    return _section("fp16 vs fp32", "\n".join(lines))


def results_markdown(docs: Docs) -> str:
    parts = [
        "# Results\n",
        "Generated by `bench/report.py` from `results/*.json`. **Do not edit by hand.** Every number comes from a result "
        "file that records hardware, library versions and the git commit; anything not yet measured is marked "
        f"`{TBD}`.\n",
        "## Key results\n\n" + key_results_table(docs) + "\n",
        verification_section(docs),
        open_set_section(docs),
        quality_section(docs),
        robustness_section(docs),
        latency_section(docs),
        precision_section(docs),
        index_section(docs),
    ]
    return "\n".join(parts)


def update_readme(readme: Path, table: str) -> bool:
    """Replace the block between the RESULTS markers; returns whether the file changed."""
    text = readme.read_text(encoding="utf-8")
    block = f"{README_START}\n{table}\n{README_END}"
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError("README is missing the RESULTS markers")
    new = pattern.sub(lambda _m: block, text)
    readme.write_text(new, encoding="utf-8", newline="\n")
    return new != text


def build(
    docs: Docs, *, docs_dir: Path = ROOT / "docs", readme: Path | None = ROOT / "README.md"
) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    img = docs_dir / "img"
    written: list[Path] = []
    for fn, name in (
        (plot_roc, "roc.png"),
        (plot_det, "det.png"),
        (plot_robustness, "robustness.png"),
        (plot_quality_curves, "quality_curves.png"),
        (plot_latency, "latency.png"),
        (plot_index, "index.png"),
    ):
        path = fn(docs, img / name)
        if path is not None:
            written.append(path)
    results = docs_dir / "RESULTS.md"
    results.write_text(results_markdown(docs), encoding="utf-8", newline="\n")
    written.append(results)
    if readme is not None:
        update_readme(readme, key_results_table(docs))
        written.append(readme)
    return written


def main() -> int:
    for path in build(load_results()):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
