#!/usr/bin/env python
"""Download the two primary datasets and convert them to the agent's generic input formats.

This script is the only place in the repository that knows dataset names. It
fetches files (sizes are checked against free disk first, checksums after), and
rewrites the data in formats the agent reads without dataset-specific code:

  spatial AnnData   spots.h5ad: raw counts in X, pixel coordinates in
                    obsm['spatial'], one source image per slide referenced from
                    uns['spatial'][slide], per-spot metadata in obs
  image-text table  pairs.csv: image path, caption, split and per-image metadata,
                    with the images in images/

It makes no modelling choices: no normalisation, gene filtering, patch size or
encoder-specific resizing. Images are only capped in size for storage.

    uv run python scripts/prepare_data.py                      # both datasets, full + quick
    uv run python scripts/prepare_data.py --only quick         # small subsets for smoke tests
    uv run python scripts/prepare_data.py --dataset rocov2 --only full

Outputs (all under data/, which is gitignored):
    data/stimage_her2/spots.h5ad        data/stimage_her2_quick/spots.h5ad
    data/rocov2/pairs.csv + images/     data/rocov2_quick/pairs.csv + images/
    data/raw/...                        downloaded source files
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RAW = DATA / "raw"
SEED = 774

# --- STimage-1K4M (Chen et al. 2024): HER2-positive breast cancer ST cohort -----------
ST_REPO = "jiawennnn/STimage-1K4M"
ST_REVISION = "b2073c6e8bb7e6d8dfd15c0db6134ba4f2a8a84f"      # pinned dataset commit
ST_PREFIX = "Human_Breast_Andersson_10142021_ST_"             # 36 slides, patients A-H
ST_QUICK_SLIDES = ("A1", "B1", "C1", "D1")                    # annotated slides, 4 patients
ST_THUMB_MAX = 600                                            # px, for plotting only

# --- ROCOv2 v2.0.1 (Rückert et al. 2024) -------------------------------------------
ROCO_RECORD = "10821435"
ROCO_N = {"train": 20_000, "valid": 2_000, "test": 5_000}      # stratified by modality
ROCO_QUICK_N = {"train": 600, "valid": 150, "test": 250}       # first members of each zip
ROCO_SHORT_SIDE = 256                                         # storage cap; encoders use 224
# Manually curated imaging-modality concepts (exactly one per image when present).
ROCO_MODALITY_CUIS = {
    "C0040405": "X-Ray Computed Tomography",
    "C1306645": "Plain x-ray",
    "C0024485": "Magnetic Resonance Imaging",
    "C0041618": "Ultrasonography",
    "C0002978": "angiogram",
    "C0032743": "Positron-Emission Tomography",
}


def say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def check_disk(total_bytes: int, dest: Path, what: str) -> None:
    """Refuse to start a download that would leave the disk nearly full."""
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    need = int(total_bytes * 1.2) + 2 * 10**9
    say(f"{what}: {total_bytes / 1e9:.2f} GB to download; {free / 1e9:.1f} GB free")
    if free < need:
        sys.exit(f"not enough free disk for {what}: need ~{need / 1e9:.1f} GB, have {free / 1e9:.1f} GB")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ===================================================================================
# STimage-1K4M -> spatial AnnData
# ===================================================================================

def her2_files() -> dict[str, int]:
    """Repository paths (and sizes) of every file for the cohort at the pinned commit."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = {}
    for folder in ("ST/image", "ST/coord", "ST/gene_exp", "annotation"):
        for item in api.list_repo_tree(ST_REPO, path_in_repo=folder, repo_type="dataset",
                                       revision=ST_REVISION):
            name = item.path.rsplit("/", 1)[-1]
            if name.startswith(ST_PREFIX):
                files[item.path] = int(getattr(item, "size", 0) or 0)
    return files


def download_her2() -> Path:
    from huggingface_hub import hf_hub_download

    raw = RAW / "stimage_her2"
    files = her2_files()
    missing = {p: s for p, s in files.items() if not (raw / p).exists()}
    check_disk(sum(missing.values()), raw, f"STimage HER2 cohort ({len(files)} files)")
    for i, path in enumerate(sorted(missing), 1):
        hf_hub_download(ST_REPO, path, repo_type="dataset", revision=ST_REVISION, local_dir=raw)
        if i % 20 == 0 or i == len(missing):
            say(f"  downloaded {i}/{len(missing)}")
    return raw


def _slide_ids(raw: Path) -> list[str]:
    return sorted(p.name[len(ST_PREFIX):-len("_count.csv")]
                  for p in (raw / "ST" / "gene_exp").glob(f"{ST_PREFIX}*_count.csv"))


def _read_annotation(path: Path) -> pd.Series:
    """Annotation files have two columns (spot id, label) whose header names vary."""
    df = pd.read_csv(path)
    return pd.Series(df.iloc[:, 1].astype(str).to_numpy(), index=df.iloc[:, 0].astype(str))


def build_her2(raw: Path, slides: list[str], out_dir: Path) -> dict:
    """One AnnData for the cohort: counts, coordinates, slide/patient/label, images."""
    import anndata as ad
    import scipy.sparse as sp
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None        # whole-slide PNGs exceed PIL's default guard
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, coords, obs, spatial, notes = [], [], [], {}, []
    genes = None
    for s in slides:
        name = ST_PREFIX + s
        counts = pd.read_csv(raw / "ST" / "gene_exp" / f"{name}_count.csv", index_col=0)
        coord = pd.read_csv(raw / "ST" / "coord" / f"{name}_coord.csv", index_col=0)
        common = counts.index.intersection(coord.index)
        if len(common) < len(counts) or len(common) < len(coord):
            notes.append(f"{s}: {len(counts)} expression rows, {len(coord)} coordinate rows, "
                         f"{len(common)} kept")
        counts, coord = counts.loc[common], coord.loc[common]
        here = set(counts.columns)
        genes = list(counts.columns) if genes is None else [g for g in genes if g in here]
        label = pd.Series(np.nan, index=common, dtype=object)
        anno_path = raw / "annotation" / f"{name}_anno.csv"
        if anno_path.exists():
            anno = _read_annotation(anno_path)
            hit = anno.index.intersection(common)
            label.loc[hit] = anno.loc[hit].to_numpy()
            if len(hit) < len(anno):
                notes.append(f"{s}: {len(anno) - len(hit)} annotated spots have no expression row")
        frames.append(counts)
        coords.append(coord[["xaxis", "yaxis"]].to_numpy(float))
        spot_ids = [f"{s}_{i[len(name) + 1:]}" for i in common]
        obs.append(pd.DataFrame({"slide": s, "patient": s[0], "label": label.to_numpy()},
                                index=spot_ids))
        img_path = raw / "ST" / "image" / f"{name}.png"
        with Image.open(img_path) as im:
            w, h = im.size
            scale = ST_THUMB_MAX / max(w, h)
            thumb = np.asarray(im.convert("RGB").resize((max(1, round(w * scale)),
                                                         max(1, round(h * scale))),
                                                        Image.Resampling.BILINEAR))
        diameter = float(2 * coord["r"].median())
        say(f"    {s}: {len(common)} spots")
        spatial[s] = {
            "images": {"lowres": thumb},
            "scalefactors": {"spot_diameter_fullres": diameter, "tissue_lowres_scalef": scale},
            "metadata": {"source_image_path": str(Path("..") / img_path.relative_to(DATA)),
                         "image_width": w, "image_height": h},
        }
    X = sp.vstack([sp.csr_matrix(f[genes].to_numpy()) for f in frames]).tocsr()
    if np.allclose(X.data, np.round(X.data)):
        X = X.astype(np.int32)
    obs_df = pd.concat(obs)
    for col in ("slide", "patient"):
        obs_df[col] = obs_df[col].astype("category")
    obs_df["label"] = obs_df["label"].astype("category")
    adata = ad.AnnData(X=X, obs=obs_df, var=pd.DataFrame(index=pd.Index(genes, name="gene")))
    adata.obsm["spatial"] = np.vstack(coords)
    adata.uns["spatial"] = spatial
    adata.write_h5ad(out_dir / "spots.h5ad", compression="gzip")
    summary = {"n_spots": adata.n_obs, "n_genes": adata.n_vars, "n_slides": len(slides),
               "n_patients": int(obs_df["patient"].nunique()),
               "labelled_spots": int(obs_df["label"].notna().sum()),
               "label_counts": obs_df["label"].value_counts().to_dict(), "notes": notes}
    say(f"  wrote {out_dir / 'spots.h5ad'}: {adata.n_obs} spots x {adata.n_vars} genes, "
        f"{len(slides)} slides")
    return summary


def prepare_her2(only: str | None) -> None:
    raw = download_her2()
    slides = _slide_ids(raw)
    targets = {"full": (DATA / "stimage_her2", slides),
               "quick": (DATA / "stimage_her2_quick", [s for s in slides if s in ST_QUICK_SLIDES])}
    for size, (out_dir, chosen) in targets.items():
        if only and only != size:
            continue
        say(f"STimage HER2 ({size}): converting {len(chosen)} slides")
        summary = build_her2(raw, chosen, out_dir)
        write_json({"dataset": "STimage-1K4M, HER2-positive breast cancer ST cohort",
                    "source": f"https://huggingface.co/datasets/{ST_REPO}",
                    "revision": ST_REVISION, "slides": chosen, "subset": size,
                    "prepared": datetime.now().isoformat(timespec="seconds"),
                    "conversion": "counts as downloaded; genes = intersection across slides; "
                                  "coordinates are full-resolution pixels (x, y); spot diameter = "
                                  "2 x radius from the coordinate files",
                    **summary}, out_dir / "PREPARED.json")


# ===================================================================================
# ROCOv2 -> image-text table
# ===================================================================================

def zenodo_files(record: str) -> dict[str, dict]:
    r = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60)
    r.raise_for_status()
    return {f["key"]: {"size": int(f["size"]), "md5": f["checksum"].split(":", 1)[1],
                       "url": f"https://zenodo.org/records/{record}/files/{f['key']}?download=1"}
            for f in r.json()["files"]}


def download_file(url: str, dest: Path, size: int, md5: str) -> Path:
    """Stream to disk with resume, then verify the published MD5."""
    if dest.exists() and dest.stat().st_size == size and md5sum(dest) == md5:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        start = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if start and r.status_code != 206:
                    start = 0
                    part.unlink(missing_ok=True)
                r.raise_for_status()
                t, last = time.time(), start
                with open(part, "ab") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
                        last += len(chunk)
                        if size > 2e8 and time.time() - t > 60:
                            say(f"    {dest.name}: {last / 1e9:.2f}/{size / 1e9:.2f} GB")
                            t = time.time()
            break
        except requests.RequestException as err:
            say(f"    {dest.name}: attempt {attempt + 1} failed ({err}); resuming")
            time.sleep(10 * (attempt + 1))
    if not part.exists() or part.stat().st_size != size or md5sum(part) != md5:
        sys.exit(f"download of {dest.name} failed its size/MD5 check")
    part.rename(dest)
    return dest


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP Range requests.

    ``zipfile`` only needs seek/tell/read, so this lets the quick subset pull a
    few hundred members out of a multi-GB archive without downloading all of it.
    Blocks are cached, and members are read in archive order, so contiguous
    members cost one request per block.
    """

    def __init__(self, url: str, size: int, block: int = 8 << 20):
        self.url, self.size, self.block, self.pos = url, size, block, 0
        self.session = requests.Session()
        self.cache: dict[int, bytes] = {}
        self.n_requests = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            lo, hi = i * self.block, min(self.size, (i + 1) * self.block) - 1
            for attempt in range(5):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
                    if r.status_code == 206:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5 * (attempt + 1))
            else:
                raise OSError(f"range request failed for bytes {lo}-{hi}")
            self.n_requests += 1
            if len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = r.content
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            i = self.pos // self.block
            data = self._block(i)
            lo = self.pos - i * self.block
            take = min(len(data) - lo, end - self.pos)
            out += data[lo:lo + take]
            self.pos += take
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def _save_image(raw_bytes: bytes, dest: Path) -> tuple[int, int]:
    """Re-encode one image, capping its shorter side for storage (never upscaling)."""
    from PIL import Image

    with Image.open(io.BytesIO(raw_bytes)) as im:
        im.load()
        if im.mode in ("I", "I;16", "I;16B", "F"):
            arr = np.asarray(im, dtype=np.float64)
            lo, hi = np.percentile(arr, [0.5, 99.5])
            im = Image.fromarray(np.clip(255 * (arr - lo) / max(hi - lo, 1e-9), 0, 255).astype(np.uint8))
        elif im.mode not in ("L", "RGB"):
            im = im.convert("RGB")
        w, h = im.size
        short = min(w, h)
        if short > ROCO_SHORT_SIDE:
            f = ROCO_SHORT_SIDE / short
            im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.BICUBIC)
        im.save(dest, "JPEG", quality=95)
        return w, h


def _roco_tables(raw: Path) -> pd.DataFrame:
    """One row per image across splits: caption, concepts, modality, source article."""
    names = pd.read_csv(raw / "cui_mapping.csv").set_index("CUI")["Canonical name"].to_dict()
    lic = pd.read_csv(raw / "license_information.csv").set_index("ID")
    rows = []
    for split in ("train", "valid", "test"):
        cap = pd.read_csv(raw / f"{split}_captions.csv").set_index("ID")
        con = pd.read_csv(raw / f"{split}_concepts.csv").set_index("ID")["CUIs"]
        man = pd.read_csv(raw / f"{split}_concepts_manual.csv").set_index("ID")["CUIs"]
        df = pd.DataFrame({"caption": cap["Caption"]})
        df["split"] = split
        df["cuis"] = con.reindex(df.index).fillna("")
        manual = man.reindex(df.index).fillna("").astype(str)
        mods = manual.map(lambda s: [c for c in s.split(";") if c in ROCO_MODALITY_CUIS])
        if (mods.map(len) > 1).any():
            sys.exit("an image carries more than one modality concept; check ROCO_MODALITY_CUIS")
        df["modality"] = mods.map(lambda m: ROCO_MODALITY_CUIS[m[0]] if m else np.nan)
        df["concepts"] = df["cuis"].map(lambda s: ";".join(names.get(c, c) for c in s.split(";") if c))
        df["article"] = lic["PMCID"].reindex(df.index)
        df["attribution"] = lic["Attribution"].reindex(df.index)
        rows.append(df)
    out = pd.concat(rows)
    out.index.name = "id"
    return out


def _stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Proportional allocation across modality strata (missing modality is its own stratum)."""
    strata = df["modality"].fillna("<missing>")
    counts = strata.value_counts()
    quota = counts / counts.sum() * n
    alloc = np.floor(quota).astype(int)
    alloc[(quota - alloc).sort_values(ascending=False).index[: n - alloc.sum()]] += 1
    picked = [rng.choice(np.flatnonzero(strata.to_numpy() == s), k, replace=False)
              for s, k in alloc.items() if k > 0]
    return df.iloc[np.sort(np.concatenate(picked))]


def _member_index(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Map image id (file stem) -> archive member."""
    return {Path(m.filename).stem: m for m in zf.infolist()
            if not m.is_dir() and Path(m.filename).suffix.lower() in (".jpg", ".jpeg", ".png")}


def _extract(zf: zipfile.ZipFile, ids: list[str], out_dir: Path) -> dict[str, tuple[int, int]]:
    members = _member_index(zf)
    missing = [i for i in ids if i not in members]
    if missing:
        sys.exit(f"{len(missing)} sampled images are not in the archive (e.g. {missing[:3]})")
    sizes = {}
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(sorted(ids, key=lambda i: members[i].header_offset), 1):
        dest = img_dir / f"{i}.jpg"
        if not dest.exists():
            sizes[i] = _save_image(zf.read(members[i]), dest)
        if k % 2000 == 0:
            say(f"    extracted {k}/{len(ids)}")
    return sizes


def _write_roco(table: pd.DataFrame, out_dir: Path, subset: str, files: dict, extra: dict) -> None:
    table = table.copy()
    table["image_path"] = [f"images/{i}.jpg" for i in table.index]
    cols = ["image_path", "caption", "split", "modality", "cuis", "concepts", "article"]
    table[cols].to_csv(out_dir / "pairs.csv")
    # Licence attributions are for citing any image shown in a figure; they are
    # not data, so they live beside the table rather than in it.
    (out_dir / "meta").mkdir(exist_ok=True)
    table[["article", "attribution"]].to_csv(out_dir / "meta" / "attribution.csv")
    write_json({"dataset": "ROCOv2 v2.0.1", "source": f"https://zenodo.org/records/{ROCO_RECORD}",
                "doi": f"10.5281/zenodo.{ROCO_RECORD}", "license": "CC BY-NC 4.0", "subset": subset,
                "prepared": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
                "n_by_split": table["split"].value_counts().to_dict(),
                "modality_by_split": {s: g["modality"].value_counts(dropna=False).to_dict()
                                      for s, g in table.groupby("split")},
                "image_storage": f"JPEG q95, shorter side capped at {ROCO_SHORT_SIDE} px, no upscaling",
                "source_md5": {k: v["md5"] for k, v in files.items()}, **extra},
               out_dir / "PREPARED.json")
    say(f"  wrote {out_dir / 'pairs.csv'}: {len(table)} pairs")


def prepare_roco(only: str | None, keep_zips: bool) -> None:
    raw = RAW / "rocov2"
    files = zenodo_files(ROCO_RECORD)
    tables = [k for k in files if k.endswith(".csv")]
    check_disk(sum(files[k]["size"] for k in tables if not (raw / k).exists()), raw,
               "ROCOv2 caption/concept tables")
    for k in tables:
        download_file(files[k]["url"], raw / k, files[k]["size"], files[k]["md5"])
    table = _roco_tables(raw)
    say(f"ROCOv2: {len(table)} images in metadata "
        f"({table['split'].value_counts().to_dict()})")

    if only in (None, "quick"):
        out_dir = DATA / "rocov2_quick"
        say(f"ROCOv2 (quick): range-reading the first members of each archive")
        picked, requests_made = [], 0
        for split, n in ROCO_QUICK_N.items():
            key = f"{split}_images.zip"
            remote = HTTPRangeFile(files[key]["url"], files[key]["size"])
            with zipfile.ZipFile(remote) as zf:
                members = sorted(_member_index(zf).items(), key=lambda kv: kv[1].header_offset)
                ids = [i for i, _ in members if i in table.index][:n]
                _extract(zf, ids, out_dir)
            requests_made += remote.n_requests
            picked.append(table.loc[ids])
        _write_roco(pd.concat(picked), out_dir, "quick", files,
                    {"selection": "first members of each split's archive (smoke test only)",
                     "http_range_requests": requests_made})

    if only in (None, "full"):
        out_dir = DATA / "rocov2"
        zips = [f"{s}_images.zip" for s in ROCO_N]
        check_disk(sum(files[k]["size"] for k in zips if not (raw / k).exists()), raw,
                   "ROCOv2 image archives (deleted after extraction)")
        rng = np.random.default_rng(SEED)
        picked = []
        for split, n in ROCO_N.items():
            sample = _stratified_sample(table[table["split"] == split], n, rng)
            if all((out_dir / "images" / f"{i}.jpg").exists() for i in sample.index):
                say(f"ROCOv2 (full): {split}: all {n} sampled images already extracted")
                picked.append(sample)
                continue
            key = f"{split}_images.zip"
            say(f"ROCOv2 (full): {split}: downloading {key} ({files[key]['size'] / 1e9:.2f} GB)")
            path = download_file(files[key]["url"], raw / key, files[key]["size"], files[key]["md5"])
            with zipfile.ZipFile(path) as zf:
                _extract(zf, list(sample.index), out_dir)
            picked.append(sample)
            if not keep_zips:
                path.unlink()
                say(f"  deleted {key} after extraction")
        _write_roco(pd.concat(picked), out_dir, "full", files,
                    {"selection": "proportional stratified random sample by modality "
                                  f"(seed {SEED}) within each official split",
                     "sizes": ROCO_N})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["stimage_her2", "rocov2"], help="default: both")
    ap.add_argument("--only", choices=["quick", "full"], help="default: both sizes")
    ap.add_argument("--keep-zips", action="store_true", help="keep ROCOv2 archives after extraction")
    args = ap.parse_args(argv)
    if args.dataset in (None, "stimage_her2"):
        prepare_her2(args.only)
    if args.dataset in (None, "rocov2"):
        prepare_roco(args.only, args.keep_zips)
    say("done")


if __name__ == "__main__":
    main()
