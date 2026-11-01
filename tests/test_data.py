"""Format-generic loaders, column roles and the modality profiler, on synthetic data."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from multimodal_agent.data.loaders import load_path
from multimodal_agent.data.roles import clean_missing
from multimodal_agent.profiler import icc_oneway, morans_i, profile, variance_share


# --- synthetic spatial AnnData: 4 slides, 3 patients, painted spots -------------------

COLOURS = {"tumour": (120, 40, 140), "stroma": (230, 120, 170)}


@pytest.fixture(scope="module")
def spatial_dir(tmp_path_factory):
    import anndata as ad
    from PIL import Image

    d = tmp_path_factory.mktemp("spatial")
    (d / "img").mkdir()
    rng = np.random.default_rng(0)
    obs, xy, counts, uns = [], [], [], {}
    for s, patient in [("s1", "P1"), ("s2", "P1"), ("s3", "P2"), ("s4", "P3")]:
        img = np.full((300, 400, 3), 255, np.uint8)
        for gx in range(6):
            for gy in range(4):
                x, y = 40 + 60 * gx, 40 + 60 * gy
                lab = "tumour" if gx < 3 else "stroma"
                img[y - 15:y + 15, x - 15:x + 15] = COLOURS[lab]
                xy.append((x, y))
                smooth = 5 + 4 * np.sin(gx / 2)
                counts.append(rng.poisson([50 * smooth, 50, *rng.uniform(1, 30, 18)]))
                obs.append({"slide": s, "patient": patient,
                            "label": "undetermined" if (gx, gy) == (0, 0) else lab})
        Image.fromarray(img).save(d / "img" / f"{s}.png")
        uns[s] = {"scalefactors": {"spot_diameter_fullres": 30.0},
                  "metadata": {"source_image_path": f"img/{s}.png"}}
    obs = pd.DataFrame(obs, index=[f"spot{i}" for i in range(len(obs))])
    for c in obs:
        obs[c] = obs[c].astype("category")
    adata = ad.AnnData(X=sp.csr_matrix(np.array(counts, np.int32)), obs=obs,
                       var=pd.DataFrame(index=["smooth", "flat"] + [f"g{i}" for i in range(18)]))
    adata.obsm["spatial"] = np.array(xy, float)
    adata.uns["spatial"] = uns
    adata.write_h5ad(d / "spots.h5ad")
    return d


def test_spatial_anndata_modalities_and_roles(spatial_dir):
    ds = load_path(spatial_dir)
    assert ds.format == "spatial_anndata"
    assert {m.kind for m in ds.modalities.values()} == {"matrix", "spatial", "image"}
    roles = {r.column: r.role for r in ds.roles}
    assert roles == {"slide": "library", "patient": "group", "label": "label"}
    assert ds.obs["label"].isna().sum() == 4          # 'undetermined' -> missing


def test_spot_patches_are_aligned_with_coordinates(spatial_dir):
    ds = load_path(spatial_dir)
    src = ds.modalities["image"].images
    for i in (0, 5, 30, 95):
        patch = np.asarray(src.load(i))
        assert patch.shape == (30, 30, 3)
        lab = "tumour" if ds.obs["label"].iloc[i] == "tumour" or i % 24 == 0 else "stroma"
        assert tuple(patch[15, 15]) == COLOURS[lab]
    wide = np.asarray(src.with_context(2.0).load(0))
    assert wide.shape == (60, 60, 3) and tuple(wide[0, 0]) == (255, 255, 255)   # background ring


def test_spatial_profile(spatial_dir):
    prof = profile(load_path(spatial_dir))
    assert prof.cv_group == "patient"
    assert prof.pairing.n_all_present == 96
    lab = prof.labels[0]
    assert lab.n_classes == 2 and lab.labelled_groups == 3
    img = prof.modality("image").image
    assert img.domain_hint == "stained_tissue" and img.spot_diameter_px_median == 30
    mat = prof.modality("expression").matrix
    assert mat.integer_valued and mat.feature_kind == "genes"
    groups = {g.column: g for g in prof.groups}
    assert groups["slide"].nested_in == "patient"
    sp_prof = prof.modality("spatial").spatial
    assert sp_prof.n_libraries == 4 and sp_prof.spacing_over_diameter == pytest.approx(2.0)


# --- synthetic image + text table ---------------------------------------------------

@pytest.fixture(scope="module")
def table_dir(tmp_path_factory):
    from PIL import Image

    d = tmp_path_factory.mktemp("table")
    (d / "images").mkdir()
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        arr = rng.integers(0, 255, (24, 32), dtype=np.uint8)
        Image.fromarray(arr).save(d / "images" / f"im{i}.png")
        mod = ["CT scan", "MR image"][i % 2]
        rows.append({"id": f"im{i}", "image_path": f"images/im{i}.png",
                     "caption": f"{mod} of patient {i} showing a lesion in region {i % 7}",
                     "split": ["train", "train", "valid", "test"][i % 4], "modality": mod,
                     "cuis": "C001;C002" if i % 3 else "C001;C003", "article": f"PMC{i // 3}"})
    pd.DataFrame(rows).to_csv(d / "pairs.csv", index=False)
    return d


def test_table_roles_and_modalities(table_dir):
    ds = load_path(table_dir)
    assert ds.format == "table"
    roles = {r.column: r.role for r in ds.roles}
    assert roles == {"image_path": "image_path", "caption": "text", "split": "split",
                     "modality": "label", "cuis": "multilabel", "article": "group"}
    assert set(ds.modalities) == {"image", "text"}
    assert ds.modalities["image"].n_present == 60


def test_table_profile(table_dir):
    prof = profile(load_path(table_dir))
    assert prof.split.counts == {"train": 30, "val": 15, "test": 15}
    assert prof.modality("image").image.domain_hint == "greyscale_imaging"
    art = next(g for g in prof.groups if g.column == "article")
    assert art.split_overlap["val_in_train"] > 0   # articles straddle the predefined split
    assert set(art.split_overlap) == {"val_in_train", "test_in_train", "test_in_val"}
    assert prof.modality("text").text.words_median >= 10
    assert prof.pairing.pairs["image|text"] == 60


# --- paired matrices with partial overlap ---------------------------------------------

def test_paired_matrices(tmp_path):
    import anndata as ad

    rng = np.random.default_rng(2)
    rna = ad.AnnData(sp.csr_matrix(rng.poisson(2, (30, 40)).astype(np.float32)),
                     obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
                     var=pd.DataFrame(index=[f"G{i}" for i in range(40)]))
    atac = ad.AnnData(sp.csr_matrix(rng.poisson(1, (25, 50)).astype(np.float32)),
                      obs=pd.DataFrame(index=[f"c{i}" for i in range(10, 35)]),
                      var=pd.DataFrame(index=[f"chr1:{1000 * i}-{1000 * i + 500}" for i in range(50)]))
    rna.write_h5ad(tmp_path / "rna.h5ad")
    atac.write_h5ad(tmp_path / "atac.h5ad")
    ds = load_path(tmp_path)
    assert ds.format == "paired_matrices" and set(ds.modalities) == {"rna", "peaks"}
    prof = profile(ds)
    assert prof.pairing.pairs["peaks|rna"] == 20 and prof.n_samples == 35


# --- statistics with known answers ----------------------------------------------------

def test_icc_and_variance_share_known_answers():
    g = np.repeat(np.arange(5), 40)
    rng = np.random.default_rng(3)
    assert icc_oneway(rng.normal(size=200), g) < 0.1
    assert icc_oneway(g + rng.normal(scale=0.01, size=200), g) > 0.99
    scores = np.column_stack([g + rng.normal(scale=0.01, size=200), rng.normal(size=200)])
    share = variance_share(scores, g)
    assert 0.6 < share < 0.75                     # column 1 is all-between (var 2); column 2 noise (var 1)


def test_morans_i_separates_smooth_from_random():
    rng = np.random.default_rng(4)
    xy = np.array([(x, y) for x in range(20) for y in range(20)], float)
    Z = np.column_stack([np.sin(xy[:, 0] / 3), rng.normal(size=len(xy))])
    smooth, noise = morans_i(Z, xy)
    assert smooth > 0.8 and abs(noise) < 0.15


def test_missing_tokens():
    s = pd.Series(["tumour", "Undetermined", "", "NA", "stroma", None])
    assert clean_missing(s).isna().tolist() == [False, True, True, True, False, True]


def test_colour_share_is_measured_on_the_large_groups_when_most_groups_are_tiny(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(2)
    (tmp_path / "images").mkdir()
    sizes = [30] * 5 + [2] * 25                      # median group size 2, but large groups hold 75%
    rows, i = [], 0
    for g, n in enumerate(sizes):
        tint = rng.integers(40, 215, 3)               # each group has its own colour cast
        for _ in range(n):
            arr = np.clip(tint + rng.normal(0, 8, (16, 16, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(tmp_path / "images" / f"im{i}.png")
            rows.append({"id": f"im{i}", "image_path": f"images/im{i}.png", "centre": f"c{g}"})
            i += 1
    pd.DataFrame(rows).to_csv(tmp_path / "pairs.csv", index=False)
    share = profile(load_path(tmp_path)).modality("image").image.colour_between_group_share
    assert share is not None and share > 0.5
    small = pd.DataFrame(rows[150:])                  # tiny groups only: too small to estimate
    small.to_csv(tmp_path / "pairs.csv", index=False)
    assert profile(load_path(tmp_path)).modality("image").image.colour_between_group_share is None
