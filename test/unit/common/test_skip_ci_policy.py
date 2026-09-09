import ast
from pathlib import Path

VALID_SKIP_CI_CATEGORIES = {
    "Accuracy",
    "Architecture",
    "Backend/JIT",
    "Coverage",
    "Matrix",
    "Resource",
    "Third-party",
}


def _skip_ci_reasons(node):
    reasons = []
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        marker = decorator.func
        if not (
            isinstance(marker, ast.Attribute)
            and marker.attr == "skip_ci"
            and isinstance(marker.value, ast.Attribute)
            and marker.value.attr == "mark"
            and isinstance(marker.value.value, ast.Name)
            and marker.value.value.id == "pytest"
        ):
            continue
        reasons.extend(
            keyword.value.value
            for keyword in decorator.keywords
            if keyword.arg == "reason"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        )
    return reasons


def test_skip_ci_markers_have_one_classified_reason():
    test_root = Path(__file__).parents[2]
    violations = []
    for path in test_root.rglob("test_*.py"):
        if path == Path(__file__):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            reasons = _skip_ci_reasons(node)
            if not reasons:
                continue
            if len(reasons) != 1:
                violations.append(f"{path}:{node.lineno} has {len(reasons)} skip_ci markers")
                continue
            category, separator, _ = reasons[0].partition(":")
            if separator != ":" or category not in VALID_SKIP_CI_CATEGORIES:
                violations.append(f"{path}:{node.lineno} has unclassified reason: {reasons[0]}")
    assert not violations, "\n".join(violations)
