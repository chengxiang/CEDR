"""Task-specific rewards; infrastructure failures never become incorrect answers."""

from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from .gsm_scoring import extract_gold, extract_pred


def gsm_correct(prediction, gold):
    target = extract_gold(gold)
    if target is None:
        raise ValueError("unparseable GSM8K gold")
    return extract_pred(prediction) == target


def math_correct(prediction, gold, timeout=5):
    from math_verify import parse, verify
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException

    # Boxed-first extraction with strict numerical and symbolic equivalence.
    target = parse(
        f"${gold}$",
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0)],
        fallback_mode="first_match",
        extraction_mode="any_match",
        parsing_timeout=timeout,
        raise_on_error=True,
    )
    if not target:
        raise ValueError("unparseable MATH gold")
    try:
        value = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(boxed_match_priority=0),
                ExprExtractionConfig(),
            ],
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=timeout,
            raise_on_error=True,
        )
        return bool(
            verify(
                target,
                value,
                float_rounding=6,
                numeric_precision=15,
                strict=True,
                allow_set_relation_comp=False,
                timeout_seconds=timeout,
                raise_on_error=True,
            )
        )
    except (TimeoutException, Exception):
        return False


def sandbox_prefix(extra_mounts=()):
    """No host home/work tree or network. Python installation is read-only."""
    command = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    for p in ("/usr", "/lib", "/lib64", "/bin", "/etc/ld.so.cache"):
        if Path(p).exists():
            command += ["--ro-bind", p, p]
    base = str(Path(sys.base_prefix).resolve())
    environment = str(Path(sys.prefix).resolve())
    command += [
        "--ro-bind",
        base,
        "/opt/python",
        "--ro-bind",
        environment,
        "/opt/environment",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--chdir",
        "/tmp",
    ]
    for source, destination, writable in extra_mounts:
        command += [
            "--bind" if writable else "--ro-bind",
            str(Path(source).resolve()),
            destination,
        ]
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    env = dict(
        PATH="/opt/python/bin:/usr/bin:/bin",
        PYTHONPATH=f"/opt/environment/lib/python{version}/site-packages",
        HOME="/tmp",
        LD_LIBRARY_PATH="/opt/python/lib",
        EVALPLUS_MAX_MEMORY_BYTES="4294967296",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="42",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OCI_UNIT_TEST_SANDBOX="1",
    )
    for key, value in env.items():
        command += ["--setenv", key, value]
    return command + [f"/opt/python/bin/python{version}"]


def execute_code(code, tests, timeout=5):
    if (
        not isinstance(tests, list)
        or not tests
        or not all(isinstance(x, str) and x.strip() for x in tests)
    ):
        raise ValueError("native nonempty test strings required")
    worker = Path(__file__).with_name("code_worker.py")
    command = sandbox_prefix([(worker, "/worker.py", False)]) + ["-S", "/worker.py"]
    request = json.dumps(
        dict(code=code, tests=tests, timeout=timeout, stop_on_failure=True)
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(
            request, timeout=len(tests) * (timeout + 1) + 15
        )
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise RuntimeError("sandbox supervisor timed out")
    if process.returncode:
        raise RuntimeError(f"code sandbox failed to start/complete: {stderr[-2000:]}")
    result = json.loads(stdout)
    if not 0 < len(result["tests"]) <= len(tests):
        raise ValueError("invalid sandbox result")
    return result


def sanitize_code(text, entrypoint=None):
    # Load the pinned EvalPlus sanitizer without importing benchmark
    # download modules or model-serving SDKs during training.
    return _load_sanitizer()(text, entrypoint=entrypoint)


from functools import lru_cache


@lru_cache(maxsize=1)
def _load_sanitizer():
    import ast
    import hashlib
    import importlib.util
    import traceback

    spec = importlib.util.find_spec("evalplus")
    if spec is None:
        raise ImportError("Install cedr[code] to use coding rewards")
    package = Path(spec.origin).parent
    expected = {
        "sanitize.py": "0269b69cb27199bfd56d7a00aedc17df5c72f7e60f9d60281a958d9d3dbc50f0",
        "syncheck.py": "3f9309720878e5bbed17ba5c75f8634cd28b0f7ddda69fb515b083fd5a64610e",
    }
    for name, digest in expected.items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != digest:
            raise ValueError(
                "EvalPlus sanitizer revision differs from the pinned scorer"
            )
    namespace = dict(ast=ast, traceback=traceback)
    tree = ast.parse((package / "syncheck.py").read_text())
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "syntax_check"
    ]
    exec(compile(tree, str(package / "syncheck.py"), "exec"), namespace)
    tree = ast.parse((package / "sanitize.py").read_text())
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.ImportFrom)
            and n.module in {"evalplus.data", "evalplus.syncheck"}
        )
        and not (isinstance(n, ast.FunctionDef) and n.name in {"script", "main"})
        and not isinstance(n, ast.If)
    ]
    exec(compile(tree, str(package / "sanitize.py"), "exec"), namespace)
    return namespace["sanitize"]


def final_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += 7
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        if depth == 0:
            return text[start:end]
    return None


def reward(task, text, row):
    if task == "gsm8k":
        return gsm_correct(text, row["answer"])
    if task == "math":
        return math_correct(
            text,
            row["metadata"].get("gold_answer")
            or final_boxed(row["answer"])
            or row["answer"],
        )
    if task == "oci":
        return bool(execute_code(sanitize_code(text), row["tests"])["all_pass"])
    raise ValueError(task)
