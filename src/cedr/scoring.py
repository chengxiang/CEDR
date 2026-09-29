"""Canonical scoring adapters and exact coverage checks."""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
from .common import sha256, atomic_json
from .data import Rows
from .rewards import reward, sandbox_prefix
from .code_extraction import extract_code


def _score_one(values):
    task, pred, row = values
    return dict(
        id=row["id"],
        correct=reward(task, pred["prediction"], row),
        seed=pred.get("seed"),
    )


def score(args, cfg):
    repair = getattr(args, "function_name_repair", False)
    if repair and not args.benchmark:
        raise ValueError(
            "--function-name-repair requires --benchmark humaneval or mbpp"
        )
    rows = Rows(args.data)
    gold = [rows[i] for i in range(len(rows))]
    predictions = [
        json.loads(x)
        for x in Path(args.predictions).read_text().splitlines()
        if x.strip()
    ]
    if [p["id"] for p in predictions] != [r["id"] for r in gold]:
        raise ValueError("prediction coverage/order differs from dataset")
    if args.benchmark:
        pairs = [
            (p, r)
            for p, r in zip(predictions, gold)
            if r.get("metadata", {}).get("benchmark", args.benchmark) == args.benchmark
        ]
        if not pairs:
            raise ValueError("requested benchmark has no rows")
        predictions, gold = map(list, zip(*pairs))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(
        task=cfg["task"],
        dataset_sha256=rows.identity,
        predictions_sha256=sha256(args.predictions),
        benchmark=args.benchmark,
    )
    if repair:
        contract["function_name_repair"] = True
    if args.benchmark and args.evalplus_cache:
        cache = Path(args.evalplus_cache)
        if not cache.is_dir():
            raise FileNotFoundError(cache)
        contract["benchmark_files"] = {
            str(p.relative_to(cache)): sha256(p)
            for p in sorted(cache.rglob("*"))
            if p.is_file()
        }
    contract_file = out / "contract.json"
    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise ValueError("scoring inputs changed; use a separate output directory")
    atomic_json(contract, contract_file)
    if args.benchmark:
        # Official EvalPlus runs within the same minimal external sandbox; it
        # requires a pre-downloaded EvalPlus cache because networking is off.
        if not args.evalplus_cache:
            raise ValueError(
                "--evalplus-cache is required for sandboxed benchmark scoring"
            )
        samples = out / "evalplus_samples.jsonl"
        extracted, audits = [], []
        for prediction, row in zip(predictions, gold):
            solution, audit = extract_code(
                prediction["prediction"],
                row["entry_point"],
                function_name_repair=repair,
            )
            extracted.append(dict(task_id=row["id"], solution=solution))
            audits.append(dict(task_id=row["id"], **audit))
        samples.write_text("".join(json.dumps(row) + "\n" for row in extracted))
        audit_path = out / "extraction_audit.jsonl"
        audit_path.write_text("".join(json.dumps(row) + "\n" for row in audits))
        worker = Path(__file__).with_name("evalplus_worker.py")
        command = sandbox_prefix(
            [
                (out, "/results", True),
                (args.evalplus_cache, "/input-cache", False),
                (worker, "/worker.py", False),
            ]
        )
        command += [
            "/worker.py",
            "--dataset",
            args.benchmark,
            "--samples",
            "/results/evalplus_samples.jsonl",
            "--min-time-limit",
            "4",
            "--gt-time-limit-factor",
            "4",
            "--test-details",
            "--parallel",
            str(args.workers),
            "--output-file",
            "/results/evalplus_results.json",
        ]
        subprocess.run(command, check=True)
        result_file = out / "evalplus_results.json"
        result = json.loads(result_file.read_text())
        if set(result["eval"]) != {r["id"] for r in gold} or any(
            len(values) != 1 for values in result["eval"].values()
        ):
            raise ValueError("EvalPlus result coverage differs from predictions")
        base = sum(v[0]["base_status"] == "pass" for v in result["eval"].values())
        plus = sum(
            v[0]["base_status"] == v[0]["plus_status"] == "pass"
            for v in result["eval"].values()
        )
        atomic_json(
            dict(
                status="completed",
                scorer="EvalPlus base/plus",
                code_extraction="single_function_alias" if repair else "standard",
                function_name_repair=repair,
                aliased=sum(a["aliased"] for a in audits),
                extraction_changed=sum(a["code_changed"] for a in audits),
                extraction_audit_sha256=sha256(audit_path),
                samples_sha256=sha256(samples),
                commit="26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
                predictions_sha256=sha256(args.predictions),
                benchmark=args.benchmark,
                dataset_sha256=rows.identity,
                results_sha256=sha256(result_file),
                total=len(gold),
                base_correct=base,
                plus_correct=plus,
                base_accuracy=base / len(gold),
                plus_accuracy=plus / len(gold),
            ),
            out / "scoring.json",
        )
        return
    work = [(cfg["task"], p, r) for p, r in zip(predictions, gold)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            scored = list(pool.map(_score_one, work))
    else:
        scored = list(map(_score_one, work))
    (out / "scored.jsonl").write_text("".join(json.dumps(x) + "\n" for x in scored))
    atomic_json(
        dict(
            status="completed",
            scorer={
                "gsm8k": "canonical numerical",
                "math": "Math-Verify 0.9.0",
                "oci": "reference-validated native tests",
            }[cfg["task"]],
            correct=sum(r["correct"] for r in scored),
            total=len(scored),
            accuracy=sum(r["correct"] for r in scored) / len(scored),
            predictions_sha256=sha256(args.predictions),
            dataset_sha256=rows.identity,
        ),
        out / "scoring.json",
    )


def validate_reward_data(args, cfg):
    """Keep only reference-correct rows; never silently turn failed tests into rewards."""
    data = Rows(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    accepted = []
    rejected = []
    for i in range(len(data)):
        row = data[i]
        if reward(cfg["task"], row["answer"], row):
            accepted.append(row)
        else:
            rejected.append(row["id"])
    (out / "eligible.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "id",
                        "prompt",
                        "answer",
                        "tests",
                        "metadata",
                        "entry_point",
                    )
                }
            )
            + "\n"
            for r in accepted
        )
    )
    atomic_json(
        dict(
            status="completed",
            source_sha256=data.identity,
            retained=len(accepted),
            rejected_ids=rejected,
            eligible_sha256=sha256(out / "eligible.jsonl"),
        ),
        out / "reference_validation.json",
    )
