"""Hard rule: no dataset-specific branches in the agent's code.

Dataset names may appear in docstrings and comments (to explain a design
choice), never in code or string literals. Only scripts/prepare_data.py, which
downloads and converts the data, may name datasets.
"""

import ast
from pathlib import Path

BANNED = ("stimage", "her2", "andersson", "rocov2", "roco_", "imageclef", "zenodo",
          "jiawennnn", "1k4m")


def test_no_dataset_names_in_package_code():
    offenders = []
    for path in Path("multimodal_agent").rglob("*.py"):
        tree = ast.parse(path.read_text())
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef,
                                           ast.AsyncFunctionDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings:
                text = node.value
            elif isinstance(node, (ast.Name, ast.arg)):
                text = getattr(node, "id", None) or getattr(node, "arg", None)
            elif isinstance(node, ast.Attribute):
                text = node.attr
            if text and any(b in text.lower() for b in BANNED):
                offenders.append(f"{path}:{node.lineno}: {text[:60]}")
    assert not offenders, offenders


def test_configs_and_prompts_name_no_dataset():
    for path in list(Path("configs").glob("*")) + list(Path("prompts").glob("*")):
        text = path.read_text().lower()
        assert not any(b in text for b in BANNED), path
