"""The official challenge task contract, recorded as data rather than executed from upstream.

This used to `exec` `container_validator/tasks.py` out of a vendored copy at import time. The
validator is no longer vendored -- its upstream declares no licence -- so the contract is frozen
into `official_contract.json`: task ids, CLI flags, output filenames and formats. That is a
description of an interface this project must conform to, not a reproduction of upstream code.

Freezing raises one obvious risk, that the record silently diverges from the validator it
describes. `rederive_from_checkout` answers it: given an acquired checkout, it re-derives the
contract from the real `tasks.py` and returns both, so a caller can compare instead of trusting.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

CONTRACT_PATH = Path(__file__).resolve().parent / "official_contract.json"


def _document() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text())


def official_contract() -> dict[str, Any]:
    """The frozen task contract. Offline, and the default everywhere."""
    return _document()["tasks"]


def rederive_from_checkout(root: str | Path) -> dict[str, Any]:
    """Re-derive the contract from an acquired validator checkout, for comparison.

    Not used by the default path: it needs the external validator, which is the reader's to obtain.
    """
    path = Path(root) / "container_validator" / "tasks.py"
    if not path.is_file():
        raise FileNotFoundError(f"No container_validator/tasks.py under {root}.")
    name = "_fomo26_official_validator_tasks"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102 - the pinned, verified upstream
    contract: dict[str, Any] = {}
    for task_id, task in module.TASKS.items():
        inputs = []
        for spec in task.inputs:
            if isinstance(spec, module.InputSpec):
                inputs.append({"kind": "required", "key": spec.key, "flags": [spec.arg]})
            else:
                inputs.append(
                    {
                        "kind": "one_of",
                        "key": spec.group_key,
                        "flags": [option.arg for option in spec.options.values()],
                    }
                )
        contract[task_id] = {
            "inputs": inputs,
            "output_flag": task.output.arg,
            "output_filename": task.output.filename,
            "output_format": task.output.format,
            "suite": task.suite,
        }
    return contract


OFFICIAL_TASKS = tuple(official_contract())
