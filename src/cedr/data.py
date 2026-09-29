"""Native chat tokenization, explicit eligibility, and resumable row order."""

from __future__ import annotations
import json
from pathlib import Path
import re
import numpy as np
import torch
from .common import atomic_json, digest, sha256

MATH_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def format_answer(answer, task):
    if task != "gsm8k":
        return answer
    answer = re.sub(r"<<.*?>>", "", answer)
    if "####" in answer:
        reasoning, final = answer.rsplit("####", 1)
        answer = reasoning.rstrip() + "\n\\boxed{" + final.strip() + "}"
    return answer


def encode_row(row, tokenizer, cfg, *, inference=False):
    task = cfg["task"]
    prompt = row["prompt"]
    if task != "oci":
        prompt = prompt.strip() + MATH_SUFFIX
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if inference:
        ids = prefix
        end = len(ids)
    else:
        answer = format_answer(row["answer"], task)
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=True,
            add_generation_prompt=False,
        )
        if full[: len(prefix)] != prefix:
            raise ValueError("chat template does not preserve prompt prefix")
        # Qwen emits a newline after im_end. Retain exactly one terminal token.
        terminal = tokenizer.convert_tokens_to_ids("<|im_end|>")
        tail = full[len(prefix) :]
        if terminal not in tail:
            raise ValueError("assistant terminal token missing")
        end = len(prefix) + tail.index(terminal)
        ids = full[: end + 1]
    limit = cfg["data"]["max_tokens"]
    prompt_limit = cfg["data"]["max_prompt_tokens"]
    if inference:
        prompt_limit = limit - 1
    if len(ids) > limit or (prompt_limit is not None and len(prefix) > prompt_limit):
        return None
    if not prefix or (not inference and end <= len(prefix)):
        return None
    identity = str(
        row.get("id", digest({"prompt": row["prompt"], "answer": row.get("answer")}))
    )
    return dict(
        id=identity,
        prompt_id=digest(prefix),
        input_ids=ids,
        prompt_length=len(prefix),
        content_end=end,
        prompt=row["prompt"],
        answer=row.get("answer"),
        weight=float(row.get("weight", 1)),
        metadata=row.get("metadata", {}),
        tests=row.get("tests"),
        entry_point=row.get("entry_point"),
    )


def prepare(input_path, output, tokenizer, cfg, inference=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"dataset already published: {out}")
    counts = dict(source=0, accepted=0, rejected=0)
    offsets = []
    prompt_offsets = []
    seen_prompts = set()
    ids = set()
    tmp = out / "rows.jsonl.tmp"
    with open(input_path) as src, tmp.open("wb") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            counts["source"] += 1
            item = encode_row(row, tokenizer, cfg, inference=inference)
            if item is None:
                counts["rejected"] += 1
                continue
            if item["id"] in ids:
                raise ValueError(f'duplicate row id: {item["id"]}')
            ids.add(item["id"])
            offsets.append(dst.tell())
            if item["prompt_id"] not in seen_prompts:
                prompt_offsets.append(len(offsets) - 1)
                seen_prompts.add(item["prompt_id"])
            dst.write((json.dumps(item, ensure_ascii=False) + "\n").encode())
            counts["accepted"] += 1
    if not offsets:
        raise ValueError("no eligible rows")
    tmp.replace(out / "rows.jsonl")
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "prompts.npy", np.asarray(prompt_offsets, dtype=np.int64))
    manifest = dict(
        version=1,
        task=cfg["task"],
        teacher=cfg["teacher"],
        data=cfg["data"],
        inference=inference,
        counts=counts,
        unique_prompts=len(prompt_offsets),
        input_sha256=sha256(input_path),
        files={
            n: sha256(out / n) for n in ("rows.jsonl", "offsets.npy", "prompts.npy")
        },
    )
    atomic_json(manifest, out / "manifest.json")
    return manifest


class Rows:
    def __init__(self, path, *, prompts_only=False):
        self.root = Path(path)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.identity = sha256(self.root / "manifest.json")
        for name, expected in self.manifest["files"].items():
            if sha256(self.root / name) != expected:
                raise ValueError(f"dataset file changed: {name}")
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self.selection = (
            np.load(self.root / "prompts.npy", mmap_mode="r") if prompts_only else None
        )
        self.handle = (self.root / "rows.jsonl").open("rb")
        self.prompts_only = prompts_only

    def __len__(self):
        return len(self.selection) if self.selection is not None else len(self.offsets)

    def __getitem__(self, i):
        physical = int(self.selection[i]) if self.selection is not None else int(i)
        self.handle.seek(int(self.offsets[physical]))
        row = json.loads(self.handle.readline())
        row["physical_index"] = physical
        if self.prompts_only:
            row["input_ids"] = row["input_ids"][: row["prompt_length"]]
            row["content_end"] = row["prompt_length"]
        return row

    def close(self):
        self.handle.close()


def collate(rows, length, pad_id, device="cpu"):
    b = len(rows)
    ids = torch.full((b, length), pad_id, dtype=torch.long, device=device)
    valid = torch.zeros((b, length), dtype=torch.bool, device=device)
    prompt = valid.clone()
    content = valid.clone()
    for i, row in enumerate(rows):
        tokens = row["input_ids"]
        n = len(tokens)
        p = row["prompt_length"]
        e = row["content_end"]
        if n > length or not 0 < p <= e <= n:
            raise ValueError("invalid row geometry; truncation is forbidden")
        ids[i, :n] = torch.tensor(tokens, device=device)
        valid[i, :n] = True
        prompt[i, :p] = True
        content[i, p:e] = True
    return dict(
        ids=ids,
        valid=valid,
        prompt=prompt,
        content=content,
        answer=valid & ~prompt,
        rows=rows,
    )


class Cursor:
    """Global microbatch cursor. Accumulation may cross an epoch boundary.

    Flow/joint training drops the final incomplete global microbatch.
    Prompt imitation retains its partial tail.
    """

    def __init__(
        self,
        size,
        micro_batch,
        world,
        rank,
        seed,
        *,
        first="physical",
        later="global",
        retain_tail=False,
    ):
        self.size = size
        self.micro = micro_batch
        self.world = world
        self.rank = rank
        self.seed = seed
        self.first = first
        self.later = later
        self.retain_tail = retain_tail
        self.epoch = 0
        self.offset = 0
        self._order = None
        self.usable = (
            size
            if retain_tail
            else size // (micro_batch * world) * (micro_batch * world)
        )
        if self.usable < 1:
            raise ValueError("dataset is smaller than one global microbatch")

    def order(self):
        if self._order is None:
            mode = self.first if self.epoch == 0 else self.later
            self._order = (
                np.arange(self.size)
                if mode == "physical"
                else np.random.default_rng(self.seed + self.epoch).permutation(
                    self.size
                )
            )
        return self._order

    def next(self):
        if self.offset >= self.usable:
            self.epoch += 1
            self.offset = 0
            self._order = None
        stop = min(self.offset + self.micro * self.world, self.usable)
        count = stop - self.offset
        lo = self.offset + self.rank * self.micro
        hi = min(lo + self.micro, stop)
        indices = self.order()[lo:hi] if lo < stop else np.empty(0, dtype=np.int64)
        self.offset = stop
        return indices.tolist(), count

    def state_dict(self):
        return {
            k: getattr(self, k)
            for k in (
                "size",
                "micro",
                "world",
                "seed",
                "first",
                "later",
                "retain_tail",
                "epoch",
                "offset",
            )
        }

    def load_state_dict(self, state):
        for k in ("size", "micro", "world", "seed", "first", "later", "retain_tail"):
            if state[k] != getattr(self, k):
                raise ValueError(f"sampler contract changed: {k}")
        self.epoch = state["epoch"]
        self.offset = state["offset"]
        self._order = None
