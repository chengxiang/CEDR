"""Command-line interfaces for CEDR training and evaluation."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(
        prog="cedr", description="CEDR continuous diffusion reasoning pipeline"
    )
    sub = p.add_subparsers(dest="command", required=True)

    def base(name, help, required=True):
        q = sub.add_parser(name, help=help)
        q.add_argument("--config", required=required)
        return q

    def teacher(q):
        q.add_argument("--teacher", help="Local snapshot or Hugging Face model ID")
        q.add_argument("--qwen-chunk", type=int)

    def data(q):
        q.add_argument("--data", required=True, help="Prepared dataset directory")

    def output(q):
        q.add_argument("--output", required=True)

    q = sub.add_parser(
        "download", help="Download and verify one model and shared inference assets"
    )
    q.add_argument("--model", required=True)
    q.add_argument(
        "--revision", help="Hugging Face revision (resolved to an immutable commit)"
    )
    output(q)
    q = sub.add_parser(
        "prepare-benchmark", help="Prepare a canonical evaluation cohort"
    )
    q.add_argument(
        "--benchmark",
        required=True,
        choices=["gsm8k", "math500", "humaneval", "mbpp", "coding"],
    )
    q.add_argument("--checkpoint", required=True, help="Downloaded inference package")
    q.add_argument(
        "--input", help="Optional local benchmark JSONL; must match the pinned cohort"
    )
    output(q)
    q = base("prepare", "Tokenize normalized JSONL without truncation")
    q.add_argument("--input", required=True)
    q.add_argument("--tokenizer")
    q.add_argument("--inference", action="store_true")
    output(q)
    q = sub.add_parser("fetch", help="Normalize a Hugging Face dataset to local JSONL")
    for name in ("dataset", "prompt-field", "answer-field", "output"):
        q.add_argument("--" + name, required=True)
    q.add_argument("--subset")
    q.add_argument("--revision", required=True)
    q.add_argument("--split", default="train")
    q.add_argument("--tests-field")
    q.add_argument("--filter-field")
    q.add_argument("--filter-values", nargs="+")
    q.add_argument("--gold-field")
    q.add_argument("--metadata-fields", nargs="*", default=[])
    q.add_argument("--limit", type=int)
    q = sub.add_parser(
        "select", help="Select a weighted stratified representation-fitting panel"
    )
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--count", required=True, type=int)
    q.add_argument("--strata", nargs="+", required=True)
    q.add_argument("--seed", type=int, default=42)
    for command, help in [
        ("covariance", "Estimate FP64 assistant covariance"),
        ("activations", "Prepare optional teacher activations"),
        ("projectors", "Train differentiable-QR teacher-soft-CE projectors"),
        ("cache", "Prepare optional BF16 projected features"),
    ]:
        q = base(command, help)
        data(q)
        teacher(q)
        output(q)
        q.add_argument("--batch-rows", type=int, default=4)
        if command == "projectors":
            q.add_argument("--covariance", required=True)
            q.add_argument("--activation-cache")
            q.add_argument("--resume")
            q.add_argument("--max-updates", type=int)
        if command == "cache":
            q.add_argument("--representation", required=True)
            q.add_argument("--shard-rows", type=int, default=256)
    q = base(
        "train", "Run flow, standalone prompt, joint, or NFT training with torchrun"
    )
    q.add_argument("stage", choices=["flow", "prompt", "joint", "nft"])
    data(q)
    teacher(q)
    output(q)
    q.add_argument("--features", choices=["cache", "live"], default="cache")
    q.add_argument("--cache")
    q.add_argument("--representation")
    q.add_argument("--embedding")
    q.add_argument("--init")
    q.add_argument("--prompt-init")
    q.add_argument("--resume")
    q.add_argument("--elf-selector")
    q.add_argument("--prompt-selector")
    q.add_argument("--micro-batch", type=int)
    q.add_argument("--max-updates", type=int)
    q.add_argument("--no-compile", action="store_true")
    q.add_argument("--pad-id", type=int, default=151643)
    q.add_argument("--reward-validation")
    q.add_argument("--tokenizer")
    q = base(
        "generate",
        "Generate full-canvas answers with independent EMA selectors",
        required=False,
    )
    data(q)
    teacher(q)
    output(q)
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--embedding")
    q.add_argument("--tokenizer")
    q.add_argument("--prompt-source", choices=["learned", "teacher"], default="learned")
    q.add_argument("--elf-selector")
    q.add_argument("--prompt-selector")
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--steps", type=int)
    q.add_argument("--powers", nargs="+", type=float)
    q.add_argument("--sccfg", type=float)
    q.add_argument("--cfg", type=float, default=1.0)
    q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--features", choices=["cache", "live"], default="live")
    q.add_argument("--cache")
    q.add_argument("--representation")
    q = base("score", "Canonical GSM8K, Math-Verify, or EvalPlus scoring")
    data(q)
    output(q)
    q.add_argument("--predictions", required=True)
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--benchmark", choices=["humaneval", "mbpp"])
    q.add_argument("--evalplus-cache")
    q.add_argument(
        "--function-name-repair",
        action="store_true",
        help="Opt in to single-function name aliasing for EvalPlus scoring",
    )
    q = base("validate-rewards", "Validate reference solutions before NFT")
    data(q)
    output(q)
    q = sub.add_parser(
        "export", help="Export checkpoint weights without optimizer/data state"
    )
    q.add_argument("--input", required=True)
    output(q)
    q.add_argument("--elf-selector")
    q.add_argument("--prompt-selector")
    q = sub.add_parser(
        "validate", help="Run short implementation checks (no production training)"
    )
    q.add_argument("--gpu", action="store_true")
    q.add_argument("--teacher")
    q.add_argument("--representation")
    q.add_argument("--output", required=True)
    return p


def dispatch(argv=None):
    args = parser().parse_args(argv)
    if args.command == "download":
        from .packages import download

        return download(args.model, args.output, args.revision)
    if args.command == "prepare-benchmark":
        from .benchmarks import prepare_benchmark

        return prepare_benchmark(args)
    if args.command == "select":
        from .selection import select

        return select(args.input, args.output, args.count, args.strata, args.seed)
    if args.command == "fetch":
        from datasets import load_dataset

        ds = load_dataset(
            args.dataset,
            args.subset,
            split=args.split,
            revision=args.revision,
            streaming=True,
        )
        from .common import atomic_json, sha256

        count = 0
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for i, row in enumerate(ds):
                if args.limit and count >= args.limit:
                    break
                if args.filter_field and str(row[args.filter_field]) not in (
                    args.filter_values or []
                ):
                    continue
                item = dict(
                    id=f"{args.dataset}:{args.split}:{i}",
                    prompt=row[args.prompt_field],
                    answer=row[args.answer_field],
                    metadata={k: row[k] for k in args.metadata_fields},
                )
                if args.tests_field:
                    item["tests"] = row[args.tests_field]
                if args.gold_field:
                    item["metadata"]["gold_answer"] = row[args.gold_field]
                f.write(json.dumps(item) + "\n")
                count += 1
        atomic_json(
            dict(
                dataset=args.dataset,
                revision=args.revision,
                split=args.split,
                subset=args.subset,
                filter_field=args.filter_field,
                filter_values=args.filter_values,
                rows=count,
                sha256=sha256(path),
            ),
            path.with_suffix(".source.json"),
        )
        return
    if args.command == "export":
        from .state import export_checkpoint

        return export_checkpoint(
            args.input, args.output, args.elf_selector, args.prompt_selector
        )
    if args.command == "validate":
        from .validation import run

        return run(args)
    from .config import load_config

    if args.config:
        cfg = load_config(args.config)
    elif args.command == "generate" and Path(args.checkpoint).is_dir():
        from .packages import read_package

        _, cfg = read_package(args.checkpoint, verify=False)
    else:
        raise ValueError("--config is required for checkpoint files")
    if args.command == "prepare":
        from transformers import AutoTokenizer
        from .data import prepare

        tok = AutoTokenizer.from_pretrained(
            args.tokenizer or cfg["teacher"]["model_id"],
            revision=cfg["teacher"]["revision"],
        )
        return prepare(args.input, args.output, tok, cfg, args.inference)
    if args.command in ("covariance", "activations", "projectors", "cache"):
        import torch
        from .teacher import QwenTeacher
        from .data import Rows
        from .representation import estimate, prepare_activations, train_projectors
        from .features import Representation, LiveFeatures, build_cache

        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows = Rows(args.data)
        t = QwenTeacher(
            args.teacher or cfg["teacher"]["model_id"],
            cfg["representation"]["layers"],
            device,
            revision=cfg["teacher"]["revision"],
            chunk_rows=args.qwen_chunk or cfg["teacher"]["chunk_rows"],
        )
        if args.command == "covariance":
            return estimate(
                rows,
                t,
                args.output,
                batch_rows=args.batch_rows,
                length=cfg["data"]["max_tokens"],
                floor=cfg["representation"]["eigenvalue_floor"],
            )
        if args.command == "activations":
            return prepare_activations(
                rows,
                t,
                args.output,
                batch_rows=args.batch_rows,
                length=cfg["data"]["max_tokens"],
            )
        if args.command == "projectors":
            return train_projectors(
                rows,
                t,
                args.covariance,
                args.output,
                cfg,
                batch_rows=args.batch_rows,
                activation_cache=args.activation_cache,
                resume=args.resume,
                max_updates=args.max_updates,
            )
        return build_cache(
            rows,
            LiveFeatures(t, Representation(args.representation, device)),
            args.output,
            batch_rows=args.batch_rows,
            shard_rows=args.shard_rows,
            length=cfg["data"]["max_tokens"],
        )
    if args.command == "train":
        from .train import run

        return run(args, cfg)
    if args.command == "generate":
        from .generation import run

        return run(args, cfg)
    if args.command == "score":
        from .scoring import score

        return score(args, cfg)
    if args.command == "validate-rewards":
        from .scoring import validate_reward_data

        return validate_reward_data(args, cfg)


def main(argv=None):
    # Console entry points call sys.exit(main()). Artifact-returning library
    # functions must not become nonzero process exit statuses.
    dispatch(argv)


if __name__ == "__main__":
    main()
