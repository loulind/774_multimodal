"""Command line: one command per dataset and objective.

    mmagent run --data data/<dataset> --objective "..." --out outputs/generated_report_1
    mmagent run ... --quick          # smoke test: small subsample, seconds per fit
    mmagent run ... --no-llm         # rule-based planner, critic and writer (no API key needed)
    mmagent run ... --replay runs/<run>/llm_cache   # reproduce LLM decisions from cache
    mmagent hardware                 # report detected device, disk and budgets
"""

from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mmagent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the agent on one dataset and objective")
    r.add_argument("--data", required=True, help="dataset file or directory")
    r.add_argument("--objective", required=True, help="free-text objective, in quotes")
    r.add_argument("--out", required=True, help="output directory for the report and artefacts")
    r.add_argument("--config", help="YAML with budgets/seed/LLM settings (never model choices)")
    r.add_argument("--quick", action="store_true", help="smoke-test budgets")
    r.add_argument("--no-llm", action="store_true", help="rule-based planner/critic/writer only")
    r.add_argument("--replay", help="directory of cached LLM responses to replay")
    r.add_argument("--label-column", help="override the detected label column")
    r.add_argument("--group-column", help="override the detected group (slide/patient/source) column")
    r.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    r.add_argument("--seed", type=int)
    r.add_argument("--model", help="LLM model id (default from config)")
    sub.add_parser("hardware", help="print detected hardware and budgets")
    pr = sub.add_parser("profile", help="load a dataset and print its modality profile")
    pr.add_argument("--data", required=True, help="dataset file or directory")
    pr.add_argument("--out", help="write the profile as JSON here")
    pr.add_argument("--label-column")
    pr.add_argument("--group-column")
    pl = sub.add_parser("plan", help="dry run: profile, parse the objective and print the rule-based plan")
    pl.add_argument("--data", required=True)
    pl.add_argument("--objective", required=True)
    pl.add_argument("--quick", action="store_true")
    pl.add_argument("--out", help="write plan.json, decision_log.json and screening.json here")
    pl.add_argument("--no-screen", action="store_true", help="skip encoder screening (no downloads)")
    sc = sub.add_parser("schema", help="write the plan's JSON schema")
    sc.add_argument("--out", default="schemas/plan.schema.json")
    args = ap.parse_args(argv)

    if args.cmd == "schema":
        from pathlib import Path

        from .schema import Plan

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(Plan.model_json_schema(), indent=2) + "\n")
        print(f"wrote {args.out}")
        return

    if args.cmd == "plan":
        from pathlib import Path

        from .config import REPO_ROOT, load_config
        from .data.loaders import load_path
        from .decision_log import DecisionLog
        from .encoders.pretrained import FeatureCache
        from .hardware import detect
        from .objective import parse_objective
        from .planner.flow import rule_based
        from .profiler import profile

        cfg = load_config(quick=args.quick)
        hw = detect(cfg.device)
        ds = load_path(args.data)
        prof = profile(ds, cfg.seed)
        budget = cfg.budget(hw.device)
        log = DecisionLog(Path(args.out) / "decision_log.json" if args.out else None)
        task = parse_objective(args.objective, prof)
        out = rule_based(ds, prof, task, budget, hw.device,
                         FeatureCache(REPO_ROOT / cfg.cache_dir / "features"), log, args.quick, cfg.seed,
                         do_screen=not args.no_screen, progress=print)
        if args.out:
            (Path(args.out) / "plan.json").write_text(out.plan.model_dump_json(indent=2))
            (Path(args.out) / "screening.json").write_text(json.dumps(out.screening_table, indent=2,
                                                                      default=str))
        print(prof.brief())
        for f in out.folds:
            print(f"  {f.name}: {f.sizes()} test groups {f.test_groups} val groups {f.val_groups}")
        for s in out.screening:
            print(f"  screening '{s.modality}': " + ", ".join(f"{r.encoder} {r.score:.3f}" for r in s.ranked))
        for d in log.entries:
            print(f"- [{d.step}] {d.decision}: {d.chosen}")
        return

    if args.cmd == "profile":
        from .data.loaders import load_path
        from .profiler import profile

        ds = load_path(args.data, args.label_column, args.group_column)
        prof = profile(ds)
        text = prof.model_dump_json(indent=2)
        if args.out:
            from pathlib import Path

            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text)
        print(text if not args.out else prof.brief())
        return

    if args.cmd == "hardware":
        from .config import load_config
        from .hardware import detect

        hw = detect()
        cfg = load_config()
        print(json.dumps({"hardware": hw.to_record(),
                          "full_budget": cfg.budget(hw.device).model_dump(),
                          "quick_budget": load_config(quick=True).budget(hw.device).model_dump(),
                          "llm": {"model": cfg.llm.model,
                                  "max_output_tokens": cfg.llm.max_output_tokens}},
                         indent=2))
        return

    from .orchestrator import run

    overrides = {"quick": args.quick, "no_llm": args.no_llm, "replay": args.replay,
                 "label_column": args.label_column, "group_column": args.group_column,
                 "device": args.device, "seed": args.seed, "config": args.config}
    if args.model:
        overrides["llm"] = {"model": args.model}
    result = run(args.data, args.objective, args.out,
                 **{k: v for k, v in overrides.items() if v not in (None, False)})
    print(f"\nreport: {result['out_dir']}/report.html")


if __name__ == "__main__":
    main()
