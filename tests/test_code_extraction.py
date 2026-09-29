import importlib.util
import json
from types import SimpleNamespace

import pytest

from cedr import code_extraction, scoring
from cedr.cli import parser
from cedr.common import sha256
from cedr.rewards import sanitize_code


@pytest.fixture
def extract():
    if importlib.util.find_spec("evalplus") is None:
        pytest.skip("Install cedr[code] for extraction checks")
    return code_extraction.extract_code


def test_repair_is_opt_in_and_preserves_recursion(extract):
    code = "def wrong(n):\n    return 1 if n == 0 else n * wrong(n - 1)"
    standard, audit = extract(code, "expected")
    assert standard == sanitize_code(code, entrypoint="expected")
    assert not audit["aliased"] and not audit["code_changed"]
    repaired, audit = extract(code, "expected", function_name_repair=True)
    assert audit["aliased"] and audit["code_changed"]
    assert audit["generated_function"] == "wrong"
    namespace = {}
    exec(repaired, namespace)
    assert namespace["expected"] is namespace["wrong"]
    assert namespace["expected"](5) == 120


@pytest.mark.parametrize(
    "code",
    [
        "def expected(x):\n    return x",
        "def wrong(x):\n    return x\nexpected = wrong",
        "def wrong(x):\n    return x\nexpected: int = 1",
        "import math as expected\ndef wrong(x):\n    return x",
        "class expected:\n    pass\ndef wrong(x):\n    return x",
        "def a(x):\n    return x\ndef b(x):\n    return x + 1",
        "x = 1",
    ],
)
def test_existing_bindings_and_ambiguous_code_are_not_repaired(extract, code):
    actual, audit = extract(code, "expected", function_name_repair=True)
    assert actual == sanitize_code(code, entrypoint="expected")
    assert not audit["aliased"] and not audit["code_changed"]


def test_nested_helper_is_not_a_second_top_level_function(extract):
    code = "def outer(x):\n    def inner(y):\n        return y + 1\n    return inner(x)"
    actual, audit = extract(code, "expected", function_name_repair=True)
    assert audit["aliased"] and audit["top_level_functions"] == ["outer"]
    namespace = {}
    exec(actual, namespace)
    assert namespace["expected"](4) == 5


def test_parse_failure_retains_standard_extraction(monkeypatch):
    monkeypatch.setattr(
        code_extraction, "sanitize_code", lambda text, entrypoint=None: text
    )
    code = "def broken("
    actual, audit = code_extraction.extract_code(
        code, "expected", function_name_repair=True
    )
    assert actual == code
    assert audit["reason"] == "unparseable_extracted_code"
    assert not audit["aliased"]


def test_cli_repair_default_and_benchmark_requirement():
    argv = [
        "score",
        "--config",
        "c",
        "--data",
        "d",
        "--predictions",
        "p",
        "--output",
        "o",
    ]
    assert not parser().parse_args(argv).function_name_repair
    assert parser().parse_args(argv + ["--function-name-repair"]).function_name_repair
    with pytest.raises(ValueError, match="requires --benchmark"):
        scoring.score(SimpleNamespace(function_name_repair=True, benchmark=None), {})


@pytest.mark.parametrize("previous_repair", [False, True])
def test_scoring_directory_cannot_mix_extraction_modes(
    tmp_path, monkeypatch, previous_repair
):
    class Rows(list):
        identity = "fixed-dataset"

    monkeypatch.setattr(scoring, "Rows", lambda path: Rows([dict(id="task")]))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(dict(id="task", prediction="pass")) + "\n")
    output = tmp_path / "scores"
    output.mkdir()
    contract = dict(
        task="oci",
        dataset_sha256="fixed-dataset",
        predictions_sha256=sha256(predictions),
        benchmark="humaneval",
    )
    if previous_repair:
        contract["function_name_repair"] = True
    path = output / "contract.json"
    path.write_text(json.dumps(contract))
    before = path.read_bytes()
    args = SimpleNamespace(
        data="unused",
        predictions=predictions,
        output=output,
        benchmark="humaneval",
        evalplus_cache=None,
        function_name_repair=not previous_repair,
    )
    with pytest.raises(ValueError, match="scoring inputs changed"):
        scoring.score(args, dict(task="oci"))
    assert path.read_bytes() == before
