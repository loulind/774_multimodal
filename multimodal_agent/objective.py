"""Step 2: turn a free-text objective into a TaskSpec.

Transparent pattern matching: every task the parser assigns is backed by the
phrases that triggered it (``TaskSpec.evidence``), and every modality the
objective names is mapped to a modality of *this* dataset through generic
vocabulary (an "image", "histology" or "radiograph" is an image modality;
"expression", "transcripts" or "peaks" a matrix). When an LLM is available it
may refine the spec (planner/llm.py), and the refinement is validated against
the same schema.

Tasks are kept in order of first mention. The primary task -- whose metric the
run optimises -- is the first *measurable* one: "joint embedding" is the means
by which retrieval or prediction is achieved, not an end with its own metric.
"""

from __future__ import annotations

import re

from .schema import MEASURABLE_TASKS, DataProfile, PredictionSpec, TaskSpec

DASHES = re.compile(r"[‐-―−]")
_SRC = r"(?P<source>[\w&+-]+(?:\s+[\w&+-]+){0,3})"
_TGT = r"(?P<target>[\w&+-]+(?:\s+[\w&+-]+){0,4}?)"

TASK_PATTERNS: dict[str, list[str]] = {
    "retrieval": [r"retriev\w*", r"\bsearch\w*", r"\bmatching\b", r"find (?:similar|matching|related)",
                  r"look[- ]?ups?", r"nearest[- ]neighbou?rs?"],
    "classification": [r"classif\w*", r"categori[sz]\w*", r"diagnos\w*", r"discriminat\w*",
                       r"identify (?:the )?(?:type|class|category|subtype)s?"],
    "clustering": [r"cluster\w*", r"spatial domains?", r"subtype discovery", r"unsupervised",
                   r"discover\w* (?:groups|subtypes|states|populations|domains)",
                   r"group\w* similar"],
    "joint_embedding": [r"joint (?:[\w-]+ )?(?:embedding|representation|space|latent)s?",
                        r"shared (?:[\w-]+ )?(?:embedding|latent|representation|space)s?",
                        r"\balign\w*", r"\bintegrat\w*", r"common (?:embedding )?space",
                        r"multimodal representations?"],
}
PREDICT = [rf"predict\w* (?:of )?(?:the )?{_TGT} from (?:the )?{_SRC}",
           rf"infer\w* (?:of )?(?:the )?{_TGT} from (?:the )?{_SRC}",
           rf"estimat\w* (?:of )?(?:the )?{_TGT} from (?:the )?{_SRC}",
           rf"translat\w* (?:from )?{_SRC} (?:in)?to {_TGT}\b"]

# Generic vocabulary for modality kinds. Matrices get finer families so that,
# with two matrices (say RNA and chromatin), the objective can pick one.
KIND_WORDS: dict[str, list[str]] = {
    "image": [r"images?", r"imaging", r"patch(?:es)?", r"histolog\w*", r"histopatholog\w*",
              r"\bh&e\b", r"whole[- ]slide", r"radiograph\w*", r"radiolog\w*", r"\bscans?\b",
              r"\bct\b", r"\bmri\b", r"x-?rays?", r"ultrasound\w*", r"micrograph\w*", r"microscop\w*",
              r"photo\w*", r"pictures?", r"morpholog\w*"],
    "text": [r"\btexts?\b", r"\btextual\b", r"captions?", r"reports?", r"\bnotes?\b", r"descriptions?",
             r"narratives?", r"sentences?", r"abstracts?"],
    "matrix": [r"gene expression", r"expression", r"transcript\w*", r"\brna\b", r"\bmrna\b", r"\bgenes?\b",
               r"\bcounts?\b", r"proteom\w*", r"protein abundance", r"chromatin", r"\batac\b",
               r"\bpeaks?\b", r"accessibility", r"methylation", r"\bomics?\b"],
    "tabular": [r"clinical (?:variables|data|features|covariates|measurements)", r"tabular",
                r"covariates?", r"demographics?", r"lab(?:oratory)? values"],
    "spatial": [r"spatial(?:ly)?", r"coordinates?", r"locations?"],
}
MATRIX_FAMILIES = {
    "chromatin": ([r"chromatin", r"\batac\b", r"\bpeaks?\b", r"accessibility"], ("peak", "atac", "chrom")),
    "rna": ([r"gene expression", r"expression", r"transcript\w*", r"\brna\b", r"\bmrna\b", r"\bgenes?\b"],
            ("express", "rna", "gene", "transcript")),
    "protein": ([r"proteom\w*", r"protein abundance"], ("prot", "adt")),
}
LABEL_WORDS = (r"\b(?:labels?|class(?:es)?|types?|categor(?:y|ies)|subtypes?|stages?|grades?|"
               r"status|outcomes?|diagnos[ie]s)\b")


def normalise(text: str) -> str:
    return DASHES.sub("-", text).lower()


def _find(patterns: list[str], text: str) -> list[tuple[int, str]]:
    return sorted((m.start(), m.group(0)) for p in patterns for m in re.finditer(p, text))


def _kind_of(phrase: str) -> str | None:
    hits = [(k, pos) for k, pats in KIND_WORDS.items() for pos, _ in _find(pats, phrase)]
    if not hits:
        return None
    return min(hits, key=lambda kv: kv[1])[0]


def _modalities_for(phrase: str, profile: DataProfile) -> list[str]:
    """Dataset modality names that a phrase refers to."""
    kind = _kind_of(phrase)
    if kind is None:
        return []
    names = [m.name for m in profile.modalities if m.kind == kind]
    if kind == "matrix" and len(names) > 1:
        for pats, stems in MATRIX_FAMILIES.values():
            if _find(pats, phrase):
                narrowed = [n for n in names if any(s in n.lower() for s in stems)]
                if narrowed:
                    return narrowed
    return names


def _label_column(text: str, profile: DataProfile, needed: bool) -> tuple[str | None, str]:
    labels = [lab for lab in profile.labels if lab.n_classes >= 2]
    if not labels:
        return None, "no label column in the data"
    tokens = set(re.findall(r"[a-z0-9]+", text))
    named = [lab for lab in labels
             if set(re.findall(r"[a-z0-9]+", lab.column.lower().replace("_", " "))) <= tokens]
    if named:
        lab = max(named, key=lambda lab: lab.n_labelled)
        return lab.column, f"the objective names column '{lab.column}'"
    lab = max(labels, key=lambda lab: lab.n_labelled)
    if len(labels) == 1:
        why = "the only label column"
    else:
        why = f"the most completely labelled of {[x.column for x in labels]}"
    use = "the requested task" if needed else "evaluation only (not named in the objective)"
    return lab.column, f"'{lab.column}' is {why}; used for {use}"


def parse_objective(objective: str, profile: DataProfile) -> TaskSpec:
    text = normalise(objective)
    found: dict[str, list[tuple[int, str]]] = {t: hits for t, pats in TASK_PATTERNS.items()
                                               if (hits := _find(pats, text))}

    prediction, notes = None, []
    for pat in PREDICT:
        for m in re.finditer(pat, text):
            target_mods = _modalities_for(m.group("target"), profile)
            source_mods = _modalities_for(m.group("source"), profile)
            if target_mods and source_mods and target_mods[0] != source_mods[0]:
                prediction = PredictionSpec(source=source_mods[0], target=target_mods[0])
                found.setdefault("cross_modal_prediction", []).append((m.start(), m.group(0)))
            elif re.search(LABEL_WORDS, m.group("target")):
                found.setdefault("classification", []).append((m.start(), m.group(0)))
            else:
                notes.append(f"'{m.group(0)}': could not map both sides to modalities")
        if prediction:
            break

    mentions: dict[str, list[str]] = {}
    order: list[tuple[int, str]] = []
    for kind, pats in KIND_WORDS.items():
        for pos, phrase in _find(pats, text):
            for name in _modalities_for(phrase, profile):
                mentions.setdefault(name, []).append(phrase)
                order.append((pos, name))
    non_spatial = [m.name for m in profile.modalities if m.kind != "spatial"]
    mentioned = [n for _, n in sorted(order) if n in non_spatial]
    mentioned = list(dict.fromkeys(mentioned))

    if prediction:
        pair = [prediction.source, prediction.target]
    elif len(mentioned) >= 2:
        pair = mentioned[:2]
    else:
        ranked = sorted(((count, key) for key, count in profile.pairing.pairs.items()
                         if all(k in non_spatial for k in key.split("|"))), reverse=True)
        pair = ranked[0][1].split("|") if ranked else non_spatial[:1]
        if mentioned and mentioned[0] in pair:
            pair = [mentioned[0]] + [p for p in pair if p != mentioned[0]]

    tasks = [t for _, t in sorted((min(pos for pos, _ in hits), t) for t, hits in found.items())]
    measurable = [t for t in tasks if t in MEASURABLE_TASKS]
    if measurable:
        primary = measurable[0]
    else:
        primary = "retrieval" if len(pair) == 2 else "clustering"
        tasks.append(primary)
        notes.append(f"no measurable task named; '{primary}' assumed as the measure of the representation")
    needs_label = any(t in ("classification", "clustering") for t in tasks)
    label, label_ev = _label_column(text, profile, needs_label)
    if notes:
        found.setdefault("notes", []).extend((0, n) for n in notes)

    return TaskSpec(raw_objective=objective, tasks=tasks, primary_task=primary, pair=pair,
                    prediction=prediction, label_column=label, label_evidence=label_ev,
                    modality_mentions=mentions,
                    evidence={t: [h for _, h in hits] for t, hits in found.items()},
                    source="heuristic")


def task_from_objective(ds, profile: DataProfile, objective: str, seed: int) -> tuple[TaskSpec, DataProfile,
                                                                                      list[str]]:
    """Parse the objective, then take the label it evaluates out of the model inputs.

    A table's categorical columns are both candidate outcomes and covariates. Once the
    objective has fixed the outcome, it is removed from the tabular modality (a label must
    not predict itself) and the data are profiled again. Returns the task, the profile and
    the columns removed.
    """
    from .profiler import profile as profile_data

    task = parse_objective(objective, profile)
    removed = ds.exclude_inputs([task.label_column])
    if removed:
        profile = profile_data(ds, seed)
    return task, profile, removed
