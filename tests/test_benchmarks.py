import pytest
from cedr import benchmarks
from cedr.common import digest


def test_ordered_code_cohort_and_execution_tests(monkeypatch):
    raw = [
        dict(
            task_id=f"Mbpp/{i}",
            prompt=f"question {i}",
            canonical_solution="pass",
            entry_point="solve",
            base_input=[[i]],
            plus_input=[[i + 1]],
        )
        for i in (2, 4)
    ]
    expected = [
        dict(
            id=r["task_id"],
            prompt=r["prompt"],
            answer=r["canonical_solution"],
            entry_point=r["entry_point"],
            metadata=dict(benchmark="mbpp"),
        )
        for r in raw
    ]
    spec = dict(
        task_ids=["Mbpp/2", "Mbpp/4"],
        rows=2,
        normalized_sha256=digest(expected),
        cohort_sha256=digest(raw),
    )
    monkeypatch.setattr(benchmarks, "specification", lambda: dict(mbpp=spec))
    assert benchmarks.normalize("mbpp", raw[::-1]) == expected
    raw[0]["plus_input"] = [[100]]
    with pytest.raises(ValueError, match="execution tests"):
        benchmarks.normalize("mbpp", raw)
