"""Synthetic DataProfiles for planner tests (no data files, no downloads)."""

from multimodal_agent.schema import (DataProfile, GroupProfile, ImageProfile, LabelProfile,
                                     MatrixProfile, ModalityProfile, PairingProfile, SpatialProfile,
                                     SplitProfile, TextProfile)


def image(n, source="files", domain="greyscale_imaging", icc=None, dia=None, near_dup=0.0):
    return ModalityProfile(
        name="image", kind="image", n_present=n, missing_fraction=0.0, source="t",
        image=ImageProfile(source=source, n_sampled=100, width_median=256, height_median=256,
                           width_range=[200, 300], spot_diameter_px_median=dia,
                           grayscale_fraction=0.9 if domain == "greyscale_imaging" else 0.0,
                           mean_rgb=[100, 100, 100], saturation_mean=0.1,
                           light_background_fraction=0.0, dark_background_fraction=0.2,
                           stain_hue_fraction=0.7 if domain == "stained_tissue" else 0.02,
                           sharpness_median=100.0,
                           near_duplicate_fraction=None if source == "spot_patches" else near_dup,
                           colour_between_group_share=icc, domain_hint=domain, domain_evidence="t"))


def text(n, words_p95=40.0, near_dup=0.01):
    return ModalityProfile(
        name="text", kind="text", n_present=n, missing_fraction=0.0, source="t",
        text=TextProfile(n_sampled=n, words_median=15, words_p95=words_p95, chars_median=100,
                         vocabulary_size=5000, duplicate_fraction=near_dup / 2,
                         near_duplicate_fraction=near_dup, fraction_over_60_words=0.02))


def matrix(n, name="expression", p=5000, twins=0.05, det=3000, gshare=None, morans=None):
    return ModalityProfile(
        name=name, kind="matrix", n_present=n, missing_fraction=0.0, source="t",
        matrix=MatrixProfile(n_features=p, feature_kind="genes", sparsity=0.8, integer_valued=True,
                             nonnegative=True, library_size_median=3000, library_size_iqr=[1000, 5000],
                             features_per_sample_median=1500, n_features_detected_5pct=det,
                             analysis_space="t", near_twin_fraction=twins, group_variance_share=gshare,
                             morans_i_median=morans))


def spatial(n, dia=145.0):
    return ModalityProfile(name="spatial", kind="spatial", n_present=n, missing_fraction=0.0, source="t",
                           spatial=SpatialProfile(n_libraries=10, samples_per_library=[100, 300, 500],
                                                  nn_distance_px_median=2 * dia,
                                                  spot_diameter_px_median=dia, spacing_over_diameter=2.0))


def label(column, counts, n, groups=None):
    total = sum(counts.values())
    return LabelProfile(column=column, n_classes=len(counts), class_counts=counts,
                        imbalance_ratio=max(counts.values()) / min(counts.values()),
                        minority_class=min(counts, key=counts.get), n_labelled=total,
                        labelled_fraction=total / n, labelled_groups=groups, n_groups=groups)


def profile(mods, n, labels=(), groups=(), cv_group=None, split=None, fmt="table"):
    names = [m.name for m in mods]
    pairs = {f"{a}|{b}": n for i, a in enumerate(names) for b in names[i + 1:]}
    return DataProfile(name="t", path="t", format=fmt, n_samples=n, modalities=list(mods),
                       pairing=PairingProfile(present={m: n for m in names}, pairs=pairs, n_all_present=n),
                       roles=[], labels=list(labels), groups=list(groups), cv_group=cv_group,
                       split=SplitProfile(column="split", counts=split) if split else None)


def spatial_profile(n=13_000, n_groups=8, twins=0.23, icc=0.28, labelled=3000):
    g = GroupProfile(column="patient", role="group", n_levels=n_groups, size_min=n // n_groups - 100,
                     size_median=n / n_groups, size_max=n // n_groups + 100)
    lab = label("label", {"tumour": labelled // 2, "stroma": labelled // 3,
                          "immune": labelled - labelled // 2 - labelled // 3}, n, groups=n_groups)
    return profile([matrix(n, twins=twins, gshare=0.7, morans=0.15),
                    spatial(n), image(n, "spot_patches", "stained_tissue", icc=icc, dia=145.0)],
                   n, labels=[lab], groups=[g], cv_group="patient", fmt="spatial_anndata")


def image_text_profile(n_train=20_000, n_val=2_000, n_test=5_000, labelled=True):
    n = n_train + n_val + n_test
    labs = [label("modality", {"CT": n // 2, "MRI": n // 3, "US": n - n // 2 - n // 3}, n)] if labelled else []
    art = GroupProfile(column="article", role="group", n_levels=int(n * 0.8), size_min=1,
                       size_median=1.0, size_max=10, split_overlap={"test_in_val": 0.19})
    return profile([image(n), text(n)], n, labels=labs, groups=[art], cv_group="article",
                   split={"train": n_train, "val": n_val, "test": n_test})
