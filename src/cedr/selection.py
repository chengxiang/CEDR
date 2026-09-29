"""Deterministic stratified panels with explicit inclusion weights."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from .common import atomic_json, sha256


def select(input_path, output, count, strata, seed):
    def key(row):
        return tuple(str(row.get("metadata", {}).get(k, "missing")) for k in strata)

    sizes = Counter()
    with open(input_path) as f:
        for line in f:
            if line.strip():
                sizes[key(json.loads(line))] += 1
    if count < 1 or count > sum(sizes.values()):
        raise ValueError("panel size outside source population")
    ordered = sorted(sizes)
    quota = {k: 0 for k in ordered}
    remaining = count
    # Equal allocation with deterministic redistribution when a stratum exhausts.
    while remaining:
        eligible = [k for k in ordered if quota[k] < sizes[k]]
        share = max(1, remaining // len(eligible))
        for k in eligible:
            add = min(share, sizes[k] - quota[k], remaining)
            quota[k] += add
            remaining -= add
            if not remaining:
                break
    rng = random.Random(seed)
    seen = Counter()
    reservoir = defaultdict(list)
    with open(input_path) as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            k = key(row)
            seen[k] += 1
            q = quota[k]
            if len(reservoir[k]) < q:
                reservoir[k].append((index, row))
            else:
                position = rng.randrange(seen[k])
                if position < q:
                    reservoir[k][position] = (index, row)
    selected = []
    for k, items in reservoir.items():
        for index, row in items:
            row = dict(row)
            row["weight"] = float(row.get("weight", 1)) * sizes[k] / quota[k]
            selected.append((index, row))
    selected.sort()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for _, row in selected))
    atomic_json(
        dict(
            seed=seed,
            count=count,
            strata=strata,
            source_sha256=sha256(input_path),
            output_sha256=sha256(out),
            allocation=[
                dict(
                    stratum=k,
                    population=sizes[k],
                    selected=quota[k],
                    inclusion_probability=quota[k] / sizes[k],
                )
                for k in ordered
            ],
        ),
        out.with_suffix(".selection.json"),
    )
