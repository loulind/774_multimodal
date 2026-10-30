"""What encoders the agent may use, and what each is suited to.

Only models with openly downloadable weights are listed: anything that needs an
account, a token or click-through acceptance of terms is excluded by design.
Each entry records what the model was pretrained on, both so the planner can
match it to the data's domain and so the report can flag possible overlap
between a pretraining corpus and the evaluation data.

Domains are properties of the *encoders*, matched against the profiler's
measured image domain (stained_tissue / greyscale_imaging / colour_other):

  stained_tissue      H&E-type histology
  greyscale_imaging   mostly greyscale medical imaging (radiology and similar)
  colour_other        anything else
  generic             natural-image or web pretraining; always a candidate

Matrices and tables have no pretrained encoder here; they get an MLP trained
from scratch (or a PCA projection as a baseline).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderInfo:
    name: str
    kinds: tuple[str, ...]              # modality kinds it encodes
    source: str                         # loader:identifier (see encoders/pretrained.py)
    domains: tuple[str, ...]
    dim: int                            # output feature dimension
    input_size: int | None = None       # image side in pixels
    max_tokens: int | None = None       # text context length
    partner: str | None = None          # other tower of a pre-aligned dual encoder (CLIP-type)
    finetunable: bool = False           # partial fine-tuning of the last blocks supported
    n_blocks: int | None = None
    size_mb: int = 0
    quick: bool = False                 # small enough for smoke tests
    license: str = ""
    pretraining: str = ""
    trainable: bool = False             # built and trained from scratch here


_ENTRIES = [
    # --- histopathology ---------------------------------------------------------------
    EncoderInfo("phikon", ("image",), "hf_vit:owkin/phikon", ("stained_tissue",), 768, 224,
                finetunable=True, n_blocks=12, size_mb=350, license="Owkin non-commercial",
                pretraining="iBOT self-supervision on 40M TCGA histology tiles"),
    EncoderInfo("plip_image", ("image",), "hf_clip_image:vinid/plip", ("stained_tissue",), 512, 224,
                partner="plip_text", finetunable=True, n_blocks=12, size_mb=600,
                license="not stated (research use)",
                pretraining="CLIP fine-tuned on OpenPath pathology image-text pairs"),
    EncoderInfo("plip_text", ("text",), "hf_clip_text:vinid/plip", ("pathology_text",), 512,
                max_tokens=77, partner="plip_image", size_mb=0, license="not stated (research use)",
                pretraining="CLIP fine-tuned on OpenPath pathology image-text pairs"),
    # --- biomedical image-text --------------------------------------------------------
    EncoderInfo("biomedclip_image", ("image",),
                "open_clip_image:hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                ("greyscale_imaging", "colour_other", "stained_tissue"), 512, 224,
                partner="biomedclip_text", finetunable=True, n_blocks=12, size_mb=784, license="MIT",
                pretraining="PMC-15M: 15M figure-caption pairs from PubMed Central"),
    EncoderInfo("biomedclip_text", ("text",),
                "open_clip_text:hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                ("biomedical_text",), 512, max_tokens=256, partner="biomedclip_image",
                finetunable=True, n_blocks=12, size_mb=0, license="MIT",
                pretraining="PMC-15M: 15M figure-caption pairs from PubMed Central"),
    EncoderInfo("pubmedbert", ("text",),
                "hf_text:microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                ("biomedical_text",), 768, max_tokens=512, size_mb=440, license="MIT",
                pretraining="masked language modelling on PubMed abstracts and PMC full text"),
    # --- general-purpose ---------------------------------------------------------------
    EncoderInfo("clip_b16_image", ("image",), "hf_clip_image:openai/clip-vit-base-patch16", ("generic",),
                512, 224, partner="clip_b16_text", finetunable=True, n_blocks=12, size_mb=600,
                license="MIT", pretraining="CLIP on 400M web image-text pairs"),
    EncoderInfo("clip_b16_text", ("text",), "hf_clip_text:openai/clip-vit-base-patch16", ("generic_text",),
                512, max_tokens=77, partner="clip_b16_image", finetunable=True, n_blocks=12, size_mb=0,
                license="MIT", pretraining="CLIP on 400M web image-text pairs"),
    EncoderInfo("dinov2_base", ("image",), "hf_vit:facebook/dinov2-base", ("generic",), 768, 224,
                finetunable=True, n_blocks=12, size_mb=350, license="Apache-2.0",
                pretraining="DINOv2 self-supervision on LVD-142M curated web images"),
    EncoderInfo("dinov2_small", ("image",), "hf_vit:facebook/dinov2-small", ("generic",), 384, 224,
                finetunable=True, n_blocks=12, size_mb=90, quick=True, license="Apache-2.0",
                pretraining="DINOv2 self-supervision on LVD-142M curated web images"),
    EncoderInfo("resnet50", ("image",), "timm:resnet50.a1_in1k", ("generic",), 2048, 224,
                size_mb=100, license="Apache-2.0", pretraining="supervised ImageNet-1k"),
    EncoderInfo("resnet18", ("image",), "timm:resnet18.a1_in1k", ("generic",), 512, 224,
                size_mb=45, quick=True, license="Apache-2.0", pretraining="supervised ImageNet-1k"),
    EncoderInfo("minilm", ("text",), "sentence:sentence-transformers/all-MiniLM-L6-v2", ("generic_text",),
                384, max_tokens=256, size_mb=90, quick=True, license="Apache-2.0",
                pretraining="contrastive sentence-pair training on 1B web sentence pairs"),
    # --- trained here --------------------------------------------------------------------
    EncoderInfo("mlp", ("matrix", "tabular"), "builtin:mlp", ("any",), 0, trainable=True,
                license="n/a", pretraining="none (trained from scratch)"),
]
REGISTRY: dict[str, EncoderInfo] = {e.name: e for e in _ENTRIES}

DOMAIN_IMAGE = {"stained_tissue": ["phikon", "plip_image", "biomedclip_image"],
                "greyscale_imaging": ["biomedclip_image"],
                "colour_other": ["biomedclip_image"]}
GENERIC_IMAGE = ["dinov2_base", "clip_b16_image", "resnet50"]
DOMAIN_TEXT = {"stained_tissue": ["plip_text", "biomedclip_text", "pubmedbert"],
               "greyscale_imaging": ["biomedclip_text", "pubmedbert"],
               "colour_other": ["biomedclip_text", "pubmedbert"]}
GENERIC_TEXT = ["clip_b16_text", "minilm"]
QUICK_IMAGE = ["dinov2_small", "resnet18"]
QUICK_TEXT = ["minilm"]


def _merge(specific: list[str], generic: list[str], n: int) -> list[str]:
    """Up to n candidates: domain-specific first, but always at least one generic
    model, so that a wrong domain guess can still be corrected by screening."""
    n = max(1, n)
    head = specific[: max(0, n - 1)] if generic else specific[:n]
    return list(dict.fromkeys(head + generic))[:n]


def candidates(kind: str, domain: str | None, n: int, quick: bool = False) -> list[str]:
    """Encoders worth screening for one modality, best prior first."""
    if kind == "image":
        if quick:
            return QUICK_IMAGE[: max(1, n)]
        return _merge(DOMAIN_IMAGE.get(domain or "", []), GENERIC_IMAGE, n)
    if kind == "text":
        if quick:
            return QUICK_TEXT[: max(1, n)]
        return _merge(DOMAIN_TEXT.get(domain or "", []), GENERIC_TEXT, n)
    return ["mlp"]


def is_dual_pair(a: str, b: str) -> bool:
    return REGISTRY.get(a) is not None and REGISTRY[a].partner == b


def summary() -> list[dict]:
    """Compact registry description for the LLM planner and the report."""
    return [{"name": e.name, "kinds": list(e.kinds), "domains": list(e.domains), "dim": e.dim,
             "max_tokens": e.max_tokens, "partner": e.partner, "finetunable": e.finetunable,
             "n_blocks": e.n_blocks, "quick": e.quick, "license": e.license,
             "pretraining": e.pretraining} for e in _ENTRIES]
