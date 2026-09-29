"""Pinned benchmark cohorts with native Qwen prompts and stable task ordering."""

from __future__ import annotations
import gzip
import json
from pathlib import Path
from urllib.request import urlopen
from .common import atomic_json, digest
from .data import prepare
from .packages import read_package


def specification():
    return json.loads(Path(__file__).with_name("benchmark_specs.json").read_text())


def normalize(name, rows):
    spec = specification()[name]
    if name == "gsm8k":
        result = [
            dict(
                id=f"gsm8k/{i}",
                prompt=r["question"],
                answer=r.get("answer", r.get("output")),
                metadata=dict(benchmark=name),
            )
            for i, r in enumerate(rows)
        ]
    elif name == "math500":
        result = [
            dict(
                id=r["unique_id"],
                prompt=r["problem"],
                answer=r["solution"],
                metadata=dict(benchmark=name, gold_answer=r["answer"]),
            )
            for r in rows
        ]
    else:
        by_id = {r["task_id"]: r for r in rows}
        if digest([by_id[key] for key in spec["task_ids"]]) != spec["cohort_sha256"]:
            raise ValueError(f"{name} execution tests differ from the pinned cohort")
        result = [
            dict(
                id=key,
                prompt=by_id[key]["prompt"],
                answer=by_id[key]["canonical_solution"],
                entry_point=by_id[key]["entry_point"],
                metadata=dict(benchmark=name),
            )
            for key in spec["task_ids"]
        ]
    if len(result) != spec["rows"] or digest(result) != spec["normalized_sha256"]:
        raise ValueError(
            f"{name} contents or ordering differ from the pinned benchmark"
        )
    return result


def fetch(name):
    spec = specification()[name]
    if name in ("gsm8k", "math500"):
        from datasets import load_dataset

        return list(
            load_dataset(
                spec["dataset"],
                spec.get("subset"),
                revision=spec["revision"],
                split="test",
            )
        )
    with urlopen(spec["url"], timeout=120) as response:
        return [
            json.loads(line)
            for line in gzip.decompress(response.read()).decode().splitlines()
            if line.strip()
        ]


def prepare_benchmark(args):
    from transformers import AutoTokenizer

    _, cfg = read_package(args.checkpoint, verify=False)
    names = ["humaneval", "mbpp"] if args.benchmark == "coding" else [args.benchmark]
    expected_task = (
        "oci"
        if names[0] in ("humaneval", "mbpp")
        else "math" if names[0] == "math500" else "gsm8k"
    )
    if cfg["task"] != expected_task:
        raise ValueError("benchmark and model tasks differ")
    if args.input and len(names) != 1:
        raise ValueError(
            "--input accepts a single benchmark; omit it for the combined coding cohort"
        )
    rows = []
    scoring_rows = {}
    for name in names:
        raw = (
            [
                json.loads(x)
                for x in Path(args.input).read_text().splitlines()
                if x.strip()
            ]
            if args.input
            else fetch(name)
        )
        rows.extend(normalize(name, raw))
        if name in ("humaneval", "mbpp"):
            by_id = {r["task_id"]: r for r in raw}
            scoring_rows[name] = [
                by_id[key] for key in specification()[name]["task_ids"]
            ]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, cohort in scoring_rows.items():
        cache = out / "evalplus-cache"
        cache.mkdir(exist_ok=True)
        (cache / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in cohort)
        )
    source = out / "benchmark.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    tok = AutoTokenizer.from_pretrained(
        Path(args.checkpoint) / "shared/tokenizer", local_files_only=True
    )
    prepare(source, out, tok, cfg, inference=True)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["counts"]["accepted"] != len(rows):
        raise ValueError("benchmark preparation rejected a row")
    atomic_json(
        dict(
            benchmarks={name: specification()[name] for name in names},
            order=[r["id"] for r in rows],
        ),
        out / "benchmark_source.json",
    )
