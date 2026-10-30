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
    args = ap.parse_args(argv)

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
