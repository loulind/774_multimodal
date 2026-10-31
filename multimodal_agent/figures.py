"""Report figures (static PNG, matplotlib).

One colour scheme throughout:
  * categorical slots in a fixed order for identity (configurations, modalities);
    colour follows the entity, so a configuration keeps its colour in every figure;
  * a single-hue blue ramp for magnitude (fractions, counts), a blue-grey-red diverging
    scale for values above and below a mean (standardised expression);
  * grey for context, and the status green only as a bar under a tile labelled "true pair".
Scatter plots carry at most three colours. Label classes and clusters are shown as
small multiples (one highlighted, the rest grey) rather than as one many-coloured
scatter. The numbers behind every figure are also in tables/ or the report.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#0d366b", "#1c5cab", "#2a78d6", "#6da7ec", "#b7d3f6", "#f0efec", "#f3c5c0", "#ec9590", "#e34948",
             "#b8302f", "#8c1f1f"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CONTEXT = "#d9d8d2"
GOOD = "#0ca30c"
MAX_POINTS = 1500                 # per modality in embedding maps
MAX_PANELS = 7                    # small multiples per figure
USE_UMAP = True                   # tests switch this off (UMAP compiles on first use)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": 0.6, "grid.linestyle": "-", "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "savefig.dpi": 150, "savefig.bbox": "tight"})
    return plt


def _cmap(colours, name):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, colours)


def _save(fig, path: Path, dpi: int | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):                   # photo grids: JPEG is ~10x smaller
        fig.savefig(path, dpi=dpi or "figure", pil_kwargs={"quality": 85, "optimize": True})
    else:
        fig.savefig(path, dpi=dpi or "figure")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return f"{path.parent.name}/{path.name}"


def config_colours(names: list[str]) -> dict[str, str]:
    """Each configuration keeps one colour in every figure (fixed slot order)."""
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def comparison(panels: list[dict], path: Path) -> str | None:
    """Dot-and-interval plot per metric: the primary configuration highlighted, everything else grey.

    panel = {"title", "rows": [{"name", "value", "lo", "hi", "is_primary"}], "chance", "interval"}"""
    panels = [p for p in panels if p["rows"]]
    if not panels:
        return None
    plt = _plt()
    heights = [max(1.6, 0.28 * len(p["rows"]) + 0.8) for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.6, sum(heights)), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for ax, p in zip(axes[:, 0], panels):
        rows = sorted(p["rows"], key=lambda r: r["value"])
        y = np.arange(len(rows))
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        for yi, r in zip(y, rows):
            c = SERIES[0] if r["is_primary"] else MUTED
            if r.get("lo") is not None and r.get("hi") is not None:
                ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, lw=2, alpha=0.9 if r["is_primary"] else 0.6)
            ax.plot(r["value"], yi, "o", color=c, ms=6, mec=SURFACE, mew=1.5, zorder=3)
            if r["is_primary"] or r is rows[-1]:
                ax.annotate(f"{r['value']:.3f}", (r["hi"] if r.get("hi") is not None else r["value"], yi),
                            xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=8)
        if p.get("chance") is not None:
            ax.axvline(p["chance"], color=MUTED, lw=1)
            ax.annotate("chance", (p["chance"], len(rows) - 0.5), xytext=(3, 0), textcoords="offset points",
                        color=INK2, fontsize=7.5, va="bottom")
        ax.set_yticks(y, [r["name"] for r in rows])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(p["title"] + (f"   ({p['interval']})" if p.get("interval") else ""))
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return _save(fig, path)


def training_curves(curves: dict[str, dict], colours: dict[str, str], selection: str, fold: str,
                    path: Path) -> str | None:
    curves = {n: c for n, c in curves.items() if c.get("val_score")}
    if not curves:
        return None
    plt = _plt()
    from matplotlib.ticker import MaxNLocator

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for ax, key in ((a1, "val_score"), (a2, "train_loss")):
        ends = {}
        for name, c in curves.items():
            v = [np.nan if x is None else x for x in c.get(key) or []]
            if not v:
                continue
            col = colours.get(name, MUTED)
            ep = np.arange(1, len(v) + 1)
            ax.plot(ep, v, color=col, lw=2, label=name)
            ax.plot(ep[-1], v[-1], "o", color=col, ms=5, mec=SURFACE, mew=1.5)
            ends[name] = (ep[-1], v[-1])
        lo, hi = ax.get_ylim()
        ys = sorted(e[1] for e in ends.values() if np.isfinite(e[1]))
        if all(b - a > 0.06 * (hi - lo) for a, b in zip(ys, ys[1:])):      # labels only where they fit
            for name, (x, y) in ends.items():
                ax.annotate(name, (x, y), xytext=(5, 0), textcoords="offset points", va="center",
                            color=INK2, fontsize=7.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax, title in ((a1, f"Validation {selection} score by epoch ({fold})"),
                      (a2, "Training loss by epoch (each configuration's own objective)")):
        ax.set_title(title, fontsize=8.8)
        ax.set_xlabel("epoch")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    a1.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def embedding_map(Za: np.ndarray, Zb: np.ndarray, names: tuple[str, str], labels: np.ndarray | None,
                  classes: list[str], path: Path, seed: int = 0) -> str | None:
    """2-D map (UMAP, cosine) of held-out joint embeddings of both modalities: by modality,
    then one small panel per label class with that class highlighted."""
    rng = np.random.default_rng(seed)
    ia = np.sort(rng.permutation(len(Za))[:MAX_POINTS])
    # Contrastive embeddings of two modalities sit in two offset cones (the "modality gap"),
    # which would dominate any 2-D map. Each modality is centred before mapping so the map
    # shows whether same-label samples meet across modalities; the gap itself is reported in
    # the geometry table.
    centred = []
    for Z in (Za[ia], Zb[ia]):
        Zc = Z - Z.mean(0)
        centred.append(Zc / np.maximum(np.linalg.norm(Zc, axis=1, keepdims=True), 1e-9))
    X = np.vstack(centred)
    if len(X) < 20:
        return None
    try:
        if not USE_UMAP:
            raise RuntimeError("UMAP switched off")
        import umap

        xy = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="cosine", random_state=seed).fit_transform(X)
        how = "UMAP"
    except Exception:                                  # tiny data or missing library: linear map
        Xc = X - X.mean(0)
        xy = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:2].T
        how = "PCA"
    mod = np.repeat([0, 1], len(ia))
    lab = None if labels is None else np.concatenate([labels[ia], labels[ia]])
    shown = []
    if lab is not None and (lab >= 0).any():
        counts = np.bincount(lab[lab >= 0], minlength=len(classes))
        shown = [k for k in np.argsort(-counts) if counts[k] > 0][:MAX_PANELS]
    plt = _plt()
    n = 1 + len(shown)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols + 0.4, 2.3 * rows + 0.3), squeeze=False)
    ax = axes[0, 0]
    for k, col in ((0, SERIES[0]), (1, SERIES[1])):
        sel = mod == k
        ax.scatter(xy[sel, 0], xy[sel, 1], s=3, color=col, alpha=0.55, lw=0, label=names[k])
    ax.set_title("by modality")
    ax.legend(loc="best", markerscale=3, handletextpad=0.2, fontsize=7)
    for j, k in enumerate(shown, start=1):
        ax = axes[j // cols, j % cols]
        sel = lab == k
        ax.scatter(xy[~sel, 0], xy[~sel, 1], s=2, color=CONTEXT, alpha=0.6, lw=0)
        ax.scatter(xy[sel, 0], xy[sel, 1], s=3, color=SERIES[0], alpha=0.8, lw=0)
        ax.set_title(f"{classes[k]} (n={int(sel.sum() // 2)})", fontsize=8.5)
    for ax in axes.ravel():
        _bare(ax)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Held-out joint embeddings of both modalities ({how}; each modality centred, which removes "
                 "the modality gap)", x=0.02, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def retrieval_examples(examples: dict, content, names: tuple[str, str], path: Path, per_direction: int = 3) -> str | None:
    """Queries and their top-5 matches. ``content(position, modality)`` returns ("image", PIL image)
    or ("text", str) for a held-out sample."""
    blocks = [(k, names if k == "a_to_b" else names[::-1], v[:per_direction]) for k, v in examples.items() if v]
    if not blocks:
        return None
    plt = _plt()
    n_rows = sum(len(v) for _, _, v in blocks)
    k = len(blocks[0][2][0]["retrieved"])
    fig, axes = plt.subplots(n_rows, k + 1, figsize=(1.75 * (k + 1), 1.95 * n_rows), squeeze=False)
    r = 0
    for _, (qmod, gmod), rows in blocks:
        for q in rows:
            items = [("query", q["query_position"], qmod, None, q["query_label"])] + \
                    [(f"#{j + 1} · {it['similarity']:.2f}", it["position"], gmod, it["is_partner"], it["label"])
                     for j, it in enumerate(q["retrieved"])]
            for c, (title, pos, m, partner, label) in enumerate(items):
                ax = axes[r, c]
                kind, obj = content(pos, m)
                if kind == "image":
                    obj = obj.copy()
                    obj.thumbnail((160, 160))
                    ax.imshow(obj)
                else:
                    ax.text(0.03, 0.97, textwrap.fill(str(obj)[:220], 26), va="top", ha="left", fontsize=6.2,
                            color=INK, transform=ax.transAxes)
                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f4f3ef", zorder=-1))
                _bare(ax)
                head = f"{qmod} query" if c == 0 else title
                if partner:
                    head += "  (true pair)"
                    ax.plot([0.02, 0.98], [-0.03, -0.03], color=GOOD, lw=3, transform=ax.transAxes, clip_on=False)
                ax.set_title(head, fontsize=7, fontweight="bold" if c == 0 else "normal", color=INK, loc="left")
                if label is not None:
                    ax.set_xlabel(str(label)[:24], fontsize=6.5, color=INK2, labelpad=3)
            r += 1
    fig.suptitle(f"Retrieval examples on held-out data (matches shown as their {names[0]} where available)",
                 x=0.01, ha="left", fontsize=9, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return _save(fig, path, dpi=110)


def prediction_maps(maps: dict, path: Path) -> str | None:
    """Observed vs predicted standardised expression on one held-out slide, best genes first."""
    genes = maps.get("genes") or []
    if not genes:
        return None
    plt = _plt()
    x, y = np.asarray(maps["x"]), np.asarray(maps["y"])
    cmap = _cmap(DIVERGING, "diverging")
    aspect = (np.ptp(y) or 1.0) / (np.ptp(x) or 1.0)
    fig, axes = plt.subplots(len(genes), 2, figsize=(5.6, len(genes) * (2.6 * min(1.4, max(0.5, aspect)) + 0.45)),
                             squeeze=False)
    size = max(2.0, min(14.0, 9000 / max(1, len(x))))
    lim = 2.5

    def z(v):                                   # within-slide z-score: the pattern the within-slide r scores
        v = np.asarray(v, float)
        return (v - v.mean()) / (v.std() or 1.0)

    for row, g in zip(axes, genes):
        for ax, v, title in ((row[0], z(g["observed"]), f"{g['gene']}: observed"),
                             (row[1], z(g["predicted"]), f"predicted (within-slide r = {g['r_within']:.2f})")):
            sc = ax.scatter(x, y, c=v, s=size, cmap=cmap, vmin=-lim, vmax=lim, lw=0)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            _bare(ax)
            ax.set_title(title, fontsize=8)
        cb = fig.colorbar(sc, ax=list(row), shrink=0.8, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, color=AXIS, labelcolor=INK2)
        cb.set_label("z-score within the slide", fontsize=7, color=INK2)
    fig.suptitle(f"Expression predicted from histology, held-out slide {maps.get('group')} "
                 f"({maps.get('n_spots')} spots)", x=0.02, ha="left", fontsize=9, fontweight="bold", color=INK)
    return _save(fig, path)


def confusion(conf: dict, path: Path) -> str | None:
    m = np.asarray(conf.get("matrix") or [])
    if m.size == 0:
        return None
    classes = conf["classes"]
    frac = m / np.maximum(1, m.sum(1, keepdims=True))
    plt = _plt()
    k = len(classes)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * k + 1.6, 0.9 + 0.5 * k))
    im = ax.imshow(frac, cmap=_cmap(BLUES, "blues"), vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if m[i, j]:
                ax.text(j, i, f"{m[i, j]}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if frac[i, j] > 0.55 else INK)
    f1 = conf.get("per_class_f1") or {}
    ax.set_xticks(range(k), classes, rotation=35, ha="right")
    ax.set_yticks(range(k), [f"{c}  (F1 {f1.get(c, 0):.2f})" for c in classes])
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("fraction of the true class", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=7, color=AXIS, labelcolor=INK2)
    ax.set_title(f"Linear probe on the held-out {conf.get('view', '')} embedding: macro-F1 "
                 f"{conf.get('macro_f1', 0):.3f}", fontsize=8.8)
    fig.tight_layout()
    return _save(fig, path)


def cluster_map(cm: dict, path: Path) -> str | None:
    """k-means clusters of held-out joint embeddings: their label composition, and (spatial data)
    where each cluster lies on the slide."""
    cont = cm.get("contingency")
    spatial = cm.get("spatial") and cm.get("x")
    if cont is None and not spatial:
        return None
    plt = _plt()
    k = cm["k"]
    clusters = np.asarray(cm["cluster"])
    n_maps = min(k, MAX_PANELS + 1) if spatial else 0
    cols = 1 + min(4, n_maps)
    rows = max(1, int(np.ceil(n_maps / 4))) if spatial else 1
    fig = plt.figure(figsize=(3.4 + 2.0 * (cols - 1), 2.5 * rows + 0.4))
    gs = fig.add_gridspec(rows, cols, width_ratios=[1.6] + [1] * (cols - 1))
    if cont is not None:
        ax = fig.add_subplot(gs[:, 0])
        counts = np.asarray(cont["counts"])
        frac = counts / np.maximum(1, counts.sum(1, keepdims=True))
        ax.imshow(frac, cmap=_cmap(BLUES, "blues"), vmin=0, vmax=1, aspect="auto")
        for i in range(frac.shape[0]):
            for j in range(frac.shape[1]):
                if counts[i, j]:
                    ax.text(j, i, f"{counts[i, j]}", ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(cont["classes"])), cont["classes"], rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(frac.shape[0]), [f"cluster {c}" for c in cont["clusters"]], fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Label mix per cluster (ARI {cm.get('ari', 0):.3f})", fontsize=8.5)
    if spatial:
        x, y = np.asarray(cm["x"]), np.asarray(cm["y"])
        size = max(1.5, min(10.0, 6000 / max(1, len(x))))
        for c in range(n_maps):
            ax = fig.add_subplot(gs[c // 4, 1 + c % 4])
            sel = clusters == c
            ax.scatter(x[~sel], y[~sel], s=size, color=CONTEXT, lw=0)
            ax.scatter(x[sel], y[sel], s=size, color=SERIES[0], lw=0)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            _bare(ax)
            ax.set_title(f"cluster {c} (n={int(sel.sum())})", fontsize=7.5)
    fig.suptitle(f"k-means clusters (k={k}) of held-out joint embeddings, {cm.get('group')}", x=0.02, ha="left",
                 fontsize=9, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)
