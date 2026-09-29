"""Standard code extraction and optional single-function name repair."""

import ast

from .rewards import sanitize_code


def extract_code(completion, entrypoint, *, function_name_repair=False):
    """Return sanitized code and an audit without executing the completion.

    Repair adds an expected-name alias only when the name is unbound and
    exactly one top-level function is present. It never uses test outcomes.
    """
    standard = sanitize_code(completion, entrypoint=entrypoint)
    audit = dict(
        entrypoint=entrypoint,
        aliased=False,
        generated_function=None,
        code_changed=False,
        reason="standard",
    )
    if not function_name_repair:
        return standard, audit

    code = sanitize_code(completion)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        audit["reason"] = "unparseable_extracted_code"
        return standard, audit

    function_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    functions = [n for n in tree.body if isinstance(n, function_types)]
    bindings = set()
    for node in tree.body:
        if isinstance(node, function_types + (ast.ClassDef,)):
            bindings.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                bindings.update(
                    n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bindings.update(n.asname or n.name.split(".")[0] for n in node.names)
    audit["top_level_functions"] = [n.name for n in functions]
    if entrypoint in bindings:
        audit["reason"] = "expected_name_already_bound"
        return standard, audit
    if len(functions) != 1:
        audit["reason"] = "not_exactly_one_top_level_function"
        return standard, audit

    name = functions[0].name
    repaired = sanitize_code(code + f"\n{entrypoint} = {name}\n", entrypoint=entrypoint)
    audit.update(
        aliased=True,
        generated_function=name,
        code_changed=repaired != standard,
        reason="single_function_alias",
    )
    return repaired, audit
