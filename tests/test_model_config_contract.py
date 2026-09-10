"""Wave 0A: every *reachable* model configuration must resolve to a real network factory.

Audit finding RISK-004 (`docs/audits/FOMO26_CODEBASE_FORENSIC_AUDIT_2026-07-31.md`) showed that
`configs/model/unet_s.yaml` selects `asparagus.modules.networks.unet.unet_s`, which does not
exist. Nothing caught it because no test composes the model configs, and because the factory name
is not written in the config: `configs/model/core/unet.yaml` declares

    _pretrain_net:
      _target_: asparagus.modules.networks.unet.${model.pretrain_net}

so the symbol is only known after Hydra resolves ``model.pretrain_net`` from a *different* file.
No static tool can see that edge.

This module derives the reachable set from the repository instead of hard-coding a list:

* every ``- /model/<name>@model`` entry in any ``configs/**`` defaults list;
* every ``model=<name>`` / ``+model=<name>`` override in any tracked shell or Slurm launcher.

Two known-broken configurations are pinned in ``KNOWN_UNRESOLVABLE`` as a *ratchet*, not an xfail.
The pin asserts the exact broken set and the exact set of files that reference it, so the suite
fails if a new config breaks, if a broken one is fixed (remove it from the pin), or -- the case
that matters -- if an active launcher starts selecting one.

Nothing here modifies production code; Wave 0A is observability only.
"""

from __future__ import annotations

import importlib
import os
import pytest
import re
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

# The repository registers these at runtime; composing outside a live Hydra run needs them here.
for _name, _fn in [("random", lambda a, b: 0), ("version", lambda: "t"), ("eval", eval)]:
    try:
        OmegaConf.register_new_resolver(_name, _fn)
    except Exception:  # noqa: BLE001 - already registered by another test module
        pass

os.environ.setdefault("ASPARAGUS_DATA", "/tmp")
os.environ.setdefault("WANDB_ENTITY", "test")

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"
MODEL_DIR = CONFIG_DIR / "model"

# The four capability slots a model config wires. `configs/model/core/<arch>.yaml` declares a
# `_<slot>` node whose `_target_` interpolates `${model.<slot>}`.
CAPABILITY_SLOTS = ("pretrain_net", "seg_net", "cls_net", "plugin_seg_net")

_MODEL_DEFAULT_RE = re.compile(r"^\s*-\s*/model/([A-Za-z0-9_]+)@model\s*$", re.MULTILINE)
_MODEL_OVERRIDE_RE = re.compile(r"(?<![\w.])\+?model=([A-Za-z0-9_]+)(?![\w.])")
_INTERPOLATION_RE = re.compile(r"\$\{([\w.]+)\}")


# --------------------------------------------------------------------------------------------
# Known-broken registry (the ratchet). Each entry is an audit finding, not an accepted state.
# --------------------------------------------------------------------------------------------

# Both entries this registry originally carried were fixed in the same wave that introduced it:
#
#   unet_s           RISK-004. configs/model/unet_s.yaml named `unet_s` for pretrain_net/seg_net,
#                    but no such factory has ever existed in any branch
#                    (`git log --all -S "def unet_s(" -- asparagus/modules/networks/unet.py` is
#                    empty) and nothing referenced the config. The file was removed rather than
#                    given a speculative architecture.
#   ultradino_vit_s  configs/model/ultradino_vit_s.yaml never existed either, yet
#                    a development debug project selected it, so that project had never
#                    composed. Repointed to /model/dinov2_test, the surviving config in the
#                    same `core/ultradino` group. Those development configs are not part of
#                    the public code release.
#
# The registry stays in place, empty, because the ratchet below is what keeps it empty: an entry
# must never be added without an audit finding justifying it.
KNOWN_UNRESOLVABLE: dict[str, dict[str, object]] = {}

# Configs under these prefixes carry scientific weight. A known-broken model must never be
# selected from here, regardless of what the development/debug tier does.
SCIENTIFIC_CONFIG_PREFIXES = ("configs/projects/fomo26/",)


# --------------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------------


def _tracked(*patterns: str) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend(sorted(REPO.glob(pattern)))
    return [p for p in out if p.is_file()]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def discover_model_references() -> dict[str, set[str]]:
    """Map model-config name -> set of repo-relative files that select it.

    Sources: Hydra defaults lists in any config, and `model=` overrides in shell/Slurm launchers.
    Documentation is deliberately excluded -- a stale `+model=unet_b` in a tutorial is doc drift,
    not a runtime contract (tracked separately in the audit's documentation_drift.tsv).
    """
    refs: dict[str, set[str]] = {}

    for path in _tracked("configs/**/*.yaml", "configs/**/*.yml"):
        rel = path.relative_to(REPO).as_posix()
        for name in _MODEL_DEFAULT_RE.findall(_read(path)):
            refs.setdefault(name, set()).add(rel)

    for path in _tracked("**/*.sh", "**/*.slurm", "**/*.bash"):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith((".venv/", "artifacts/")):
            continue
        for name in _MODEL_OVERRIDE_RE.findall(_read(path)):
            # Only count it when a matching model config exists or is a known-broken name;
            # `model=$SOMETHING` and shell-variable noise must not create phantom entries.
            if (MODEL_DIR / f"{name}.yaml").is_file() or name in KNOWN_UNRESOLVABLE:
                refs.setdefault(name, set()).add(rel)

    return refs


MODEL_REFERENCES = discover_model_references()
ON_DISK_MODELS = sorted(p.stem for p in MODEL_DIR.glob("*.yaml"))
REACHABLE_MODELS = sorted(set(MODEL_REFERENCES) - set(KNOWN_UNRESOLVABLE))


def _compose_model(name: str):
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=f"model/{name}")
    return cfg.model if "model" in cfg else cfg


def _target_nodes(model_cfg) -> dict[str, str]:
    """Return {capability slot -> raw _target_ string} for the model's network nodes."""
    raw = OmegaConf.to_container(model_cfg, resolve=False)
    nodes: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "_target_" in value:
            nodes[str(key).lstrip("_")] = str(value["_target_"])
    return nodes


def _import_network_module(module_path: str):
    """Import a network module, distinguishing "dependency missing" from "factory missing".

    `medvit_3d` imports NATTEN, a CUDA-built extension. Where it is absent or its compiled
    extension will not load, the import fails -- and it does not always fail with `ImportError`:
    a native extension can raise `OSError` or `RuntimeError` instead. That is an environment
    property, not a defect in this repository, so it must never be reported as a missing factory.

    Returns ``(module, None)`` on success and ``(None, reason)`` when the module is unavailable.
    """
    try:
        return importlib.import_module(module_path), None
    except Exception as exc:  # noqa: BLE001 - native extension failures are not all ImportError
        return None, f"{module_path} is unavailable in this environment ({type(exc).__name__}: {exc})"


# --------------------------------------------------------------------------------------------
# Sanity: the discovery itself must not silently collapse
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# The public-release model contract
# --------------------------------------------------------------------------------------------
#
# `len(REACHABLE_MODELS) >= 8` was a breadth *policy*, not a contract: a statement about how much
# historical campaign configuration happened to survive. It could not say *which* eight, so eight
# wrong ones would have satisfied it.
#
# Every expectation below is a literal, written from reviewed release scope rather than read back
# from discovery, so a collapse of the discovery machinery fails the test instead of vacuously
# satisfying it. Together the three sets must account for every model config on disk, and there is
# no minimum count anywhere.

# The FOMO26 public release is candidate D1: a ResEnc-B backbone under an AMAES
# reconstruction-only objective. Exactly one model group is required for that to be usable.
REQUIRED_PUBLIC_MODEL_GROUPS = ("resenc_unet_b",)

# Non-D1 architectures. Their network modules are already dispositioned for removal, so these
# groups are not part of the public release; they are still on disk today, and this pin is the
# ratchet that tracks their removal. Emptying an entry is part of removing the config -- the test
# fails both if a new non-D1 group appears and if one disappears without updating the pin.
NON_D1_MODEL_GROUPS_PENDING_REMOVAL = ("dinov2_test",)

# Groups the public-release scope decision retired outright. Their configs, network modules and
# tests are gone; these names are kept so the contract can assert they stay gone. A retired name
# must fail closed -- never resolve, and never quietly fall back onto a retained model.
RETIRED_MODEL_GROUPS = (
    "medvit3d_b",
    "medvit3d_t",
    "primus_m",
    "primus_s",
)

# Groups the reviewed evidence does not resolve either way: each is held by a retained consumer
# (the inference architecture registry, a retained smoke test, or the ResEnc CLS/REG factory the
# submitted Task-1/3/5 checkpoints deserialize through). The decision is BLOCKED, not "public" and
# not "retired", and they stay retained until it is taken deliberately.
MODEL_GROUPS_PENDING_DECISION = (
    "resenc_unet_b_clsreg",
    "unet_b_lw_dec",
    "unet_m",
    "unet_tiny",
)

# The runtime architecture registry the released CLI advertises. Written out here rather than read
# from `known_architectures()`, so the assertion is against reviewed scope and not against the
# object under test.
EXPECTED_PUBLIC_ARCHITECTURES = ("resenc_b", "unet_m")

# The one alias the runtime declares: Hydra group name -> runtime architecture name.
EXPECTED_ARCHITECTURE_ALIASES = {"resenc_unet_b": "resenc_b"}


# --------------------------------------------------------------------------------------------
# Literal model-to-project selectors
# --------------------------------------------------------------------------------------------
#
# A model config alone is not constructible: its network nodes interpolate keys that exist only in
# a full project config. The previous version of this file found one by taking
# `sorted(project_configs)[0]`, and that is a defect, not a shortcut. Deleting two unrelated
# DinoV2 project configs -- which nothing referenced -- silently moved `dinov2_test` onto
# `DEBUG_CLS.yaml`, which does not define `training.global_crop_size`, and the test began failing
# for a reason that had nothing to do with what it was testing.
#
# So every selector is written out: the exact root config, the exact ordered overrides, and the
# exact class the slot must produce. The root must be a config the release explicitly retains, or
# one this test explicitly owns through overrides. It is never whatever file happens to sort
# first, and there is no fallback: a model with no stable selector is BLOCKED and says so.

RETAINED_PROJECT_CONFIG = "RETAINED_PROJECT_CONFIG"
TEST_OWNED = "TEST_OWNED"

MODEL_SELECTORS: dict[str, dict[str, object]] = {
    "resenc_unet_b": {
        "ownership": RETAINED_PROJECT_CONFIG,
        "classification": "PUBLIC_REQUIRED",
        "root_config": "projects/fomo26/safety/pretrain/resenc_amaes",
        "root_path": "configs/projects/fomo26/safety/pretrain/resenc_amaes.yaml",
        "overrides": (),
        "slot": "_pretrain_net",
        "expected_target_class": "asparagus.modules.networks.resenc_unet.ResidualEncoderUNetSSL",
    },
    "unet_tiny": {
        "ownership": TEST_OWNED,
        "classification": "DECISION_BLOCKED",
        "root_config": "default_finetune_cls",
        "root_path": "configs/default_finetune_cls.yaml",
        "overrides": (
            "+model=unet_tiny",
            "training.target_size=[32,32,32]",
            "training.patch_size=[32,32,32]",
        ),
        "slot": "_cls_net",
        "slot_kwargs": {"input_channels": 1, "output_channels": 2},
        "expected_target_class": "gardening_tools.modules.networks.unet.UNetCLSREG",
    },
}

# Models with no stable selector. Each names the reason; none falls back to another file.
BLOCKED_MODEL_SELECTORS: dict[str, str] = {
    "dinov2_test": (
        "every selecting project config is dispositioned for deletion, and of the three only "
        "DinoV2_2D.yaml instantiates -- DinoV2_3D.yaml and DEBUG_CLS.yaml raise "
        "InterpolationKeyError on model.hidden_size and training.global_crop_size. Choosing "
        "among them by order is exactly the defect this table exists to remove."
    ),
    "resenc_unet_b_clsreg": (
        "selected only by fomo26/finetune task configs dispositioned for deletion. The *factory* "
        "is required -- the submitted Task-1/3/5 checkpoints deserialize through it -- but that "
        "is covered by the checkpoint-role evidence, not by a project-config selector."
    ),
    "unet_b_lw_dec": "selected only by development configs that the public code release does not ship.",
    "unet_m": (
        "no retained project config selects it: all seventeen selectors are dispositioned for "
        "deletion. Its construction is covered where it actually matters -- through the runtime "
        "architecture registry, which advertises it and which "
        "the Task-5 representation-extraction forensics called directly; that campaign code is "
        "not part of the public code release."
    ),
}

# Project configs the release retains. Written out rather than globbed: the point of this list is
# that adding or removing a retained project config is a reviewed change, not a side effect.
RETAINED_PUBLIC_PROJECT_CONFIGS = (
    "configs/projects/fomo26/finetune/task1_lesion.yaml",
    "configs/projects/fomo26/finetune/task1_lesion_amaes_encoder.yaml",
    "configs/projects/fomo26/finetune/task1_lesion_ft.yaml",
    "configs/projects/fomo26/finetune/task1_lesion_scratch.yaml",
    "configs/projects/fomo26/finetune/task1_presence.yaml",
    "configs/projects/fomo26/finetune/task1_presence_ft.yaml",
    "configs/projects/fomo26/finetune/task2_lesion.yaml",
    "configs/projects/fomo26/finetune/task2_lesion_ft.yaml",
    "configs/projects/fomo26/finetune/task3_age.yaml",
    "configs/projects/fomo26/finetune/task3_age_amaes_encoder.yaml",
    "configs/projects/fomo26/finetune/task3_age_ft.yaml",
    "configs/projects/fomo26/finetune/task3_age_scratch.yaml",
    "configs/projects/fomo26/finetune/task4_multiclass.yaml",
    "configs/projects/fomo26/finetune/task4_multiclass_ft.yaml",
    "configs/projects/fomo26/finetune/task5_ppmr.yaml",
    "configs/projects/fomo26/finetune/task5_ppmr_ft.yaml",
    "configs/projects/fomo26/safety/pretrain/base.yaml",
    "configs/projects/fomo26/safety/pretrain/resenc_amaes.yaml",
    "configs/projects/fomo26/safety/pretrain/resenc_amaes_p2_96k.yaml",
)


def test_discovery_machinery_has_not_collapsed():
    """Guard the regexes: if the defaults syntax changes, every other test would vacuously pass."""
    assert MODEL_REFERENCES, "no model references discovered - the defaults-list regex is stale"
    assert ON_DISK_MODELS, "no model configs found on disk - MODEL_DIR is wrong"


def test_required_public_model_groups_are_present_and_reachable():
    """The D1 release contract. Named explicitly, so a collapsed discovery fails here."""
    for name in REQUIRED_PUBLIC_MODEL_GROUPS:
        assert (MODEL_DIR / f"{name}.yaml").is_file(), f"required public model group {name!r} is absent"
        assert name in REACHABLE_MODELS, (
            f"required public model group {name!r} is on disk but no config or launcher selects it, "
            "so the released tree cannot compose it"
        )


def test_model_groups_on_disk_are_exactly_the_reviewed_partition():
    """Ratchet with no minimum count: every model config belongs to exactly one reviewed set."""
    partition = (
        set(REQUIRED_PUBLIC_MODEL_GROUPS) | set(NON_D1_MODEL_GROUPS_PENDING_REMOVAL) | set(MODEL_GROUPS_PENDING_DECISION)
    )
    sizes = len(REQUIRED_PUBLIC_MODEL_GROUPS) + len(NON_D1_MODEL_GROUPS_PENDING_REMOVAL) + len(MODEL_GROUPS_PENDING_DECISION)
    assert sizes == len(partition), "a model group is claimed by more than one reviewed set"
    assert partition == set(ON_DISK_MODELS), (
        "the model configs on disk no longer match the reviewed partition.\n"
        f"  unreviewed on disk : {sorted(set(ON_DISK_MODELS) - partition)}\n"
        f"  reviewed but absent: {sorted(partition - set(ON_DISK_MODELS))}\n"
        "Removing a config means deleting its pin entry in the same change."
    )


def test_registry_advertises_exactly_the_reviewed_public_architectures():
    """Registry contract, asserted against a literal rather than against itself."""
    from finetuning.fomo26_inference.backbones import known_architectures

    assert known_architectures() == sorted(EXPECTED_PUBLIC_ARCHITECTURES), (
        f"the runtime architecture registry is {known_architectures()}, expected "
        f"{sorted(EXPECTED_PUBLIC_ARCHITECTURES)}. Changing it is a public-contract change."
    )


@pytest.mark.parametrize("architecture", sorted(EXPECTED_PUBLIC_ARCHITECTURES))
def test_every_advertised_architecture_actually_constructs(architecture):
    """No dangling factory: an advertised name whose module is gone is a broken CLI choice.

    This is the check that catches a registry entry left behind by a deletion wave:
    `known_architectures()` keeps listing the name, `--architecture` keeps offering it, and the
    failure only appears when a user selects it.
    """
    from finetuning.fomo26_inference.backbones import build_backbone

    net = build_backbone(architecture, input_channels=1, output_channels=2)
    assert sum(p.numel() for p in net.parameters()) > 0, f"{architecture} built a network with no parameters"


def test_architecture_aliases_are_exactly_the_declared_mapping():
    """Duplicate naming is handled explicitly, not by coincidence."""
    from finetuning.fomo26_inference.backbones import (
        ARCHITECTURE_ALIASES,
        canonical_architecture,
        known_architectures,
    )

    assert ARCHITECTURE_ALIASES == EXPECTED_ARCHITECTURE_ALIASES
    for group, runtime_name in EXPECTED_ARCHITECTURE_ALIASES.items():
        assert group in ON_DISK_MODELS, f"alias source {group!r} is not a model config group"
        assert runtime_name in known_architectures(), f"alias target {runtime_name!r} is not registered"
        assert canonical_architecture(group) == runtime_name


def test_required_public_groups_map_into_the_advertised_registry():
    """Discovery and the CLI must meet: every required group must be buildable by name.

    The two namespaces are not equal as sets -- many Hydra model groups, few runtime
    architectures -- so equality is the wrong invariant. What must hold is that the release's
    required groups canonicalize onto architectures the CLI actually offers.
    """
    from finetuning.fomo26_inference.backbones import canonical_architecture, known_architectures

    for group in REQUIRED_PUBLIC_MODEL_GROUPS:
        assert canonical_architecture(group) in known_architectures(), (
            f"required public model group {group!r} canonicalizes to "
            f"{canonical_architecture(group)!r}, which the CLI does not advertise"
        )


def test_cli_advertises_exactly_the_reviewed_architectures():
    """The `--architecture` choices a user actually sees, checked against reviewed scope.

    Read from the entrypoint's own `--help` in a subprocess -- the text a user is shown -- rather
    than by importing the registry, so this is not `known_architectures() == known_architectures()`.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(REPO / "finetuning" / "fomo26_inference" / "pretrained_embedding.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert proc.returncode == 0, proc.stderr
    match = re.search(r"--architecture\s*\{([^}]*)\}", proc.stdout.replace("\n", " "))
    assert match, f"--architecture does not advertise a choice list:\n{proc.stdout}"
    advertised = sorted(x.strip() for x in match.group(1).split(",") if x.strip())
    assert advertised == sorted(EXPECTED_PUBLIC_ARCHITECTURES)


# --------------------------------------------------------------------------------------------
# Selector contract
# --------------------------------------------------------------------------------------------


def test_every_model_group_has_a_selector_or_an_explicit_block():
    """No model may be silently unexercised, and none may be claimed twice."""
    covered = set(MODEL_SELECTORS) | set(BLOCKED_MODEL_SELECTORS)
    assert not (set(MODEL_SELECTORS) & set(BLOCKED_MODEL_SELECTORS)), "a model is both selected and blocked"
    assert covered == set(ON_DISK_MODELS), (
        "the selector table drifted from the model configs on disk.\n"
        f"  on disk without an entry: {sorted(set(ON_DISK_MODELS) - covered)}\n"
        f"  entry without a config  : {sorted(covered - set(ON_DISK_MODELS))}"
    )


def test_required_public_groups_all_have_a_working_selector():
    """A required public group must never be BLOCKED: that would leave it unexercised."""
    for name in REQUIRED_PUBLIC_MODEL_GROUPS:
        assert name in MODEL_SELECTORS, f"{name} is required but has no selector"
        assert MODEL_SELECTORS[name]["ownership"] == RETAINED_PROJECT_CONFIG, (
            f"{name} is required for the public release, so its selector must be a config the "
            "release retains, not one a test owns"
        )


@pytest.mark.parametrize("name", sorted(MODEL_SELECTORS))
def test_selector_root_exists_and_is_explicitly_owned(name):
    """The root is named, present, and its ownership is declared -- never discovered."""
    selector = MODEL_SELECTORS[name]
    root_path = REPO / str(selector["root_path"])
    assert root_path.is_file(), f"{name}: selector root {selector['root_path']} is missing"
    assert selector["ownership"] in {RETAINED_PROJECT_CONFIG, TEST_OWNED}
    assert str(selector["root_path"]) == f"configs/{selector['root_config']}.yaml"
    if selector["ownership"] == TEST_OWNED:
        assert selector["overrides"], f"{name}: a test-owned selector must say how it selects the model"


@pytest.mark.parametrize("name", sorted(MODEL_SELECTORS))
def test_selector_composes_and_constructs_the_expected_class(name):
    """CPU construction through the real production path, from a named root and named overrides."""
    from hydra.utils import instantiate

    selector = MODEL_SELECTORS[name]
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=str(selector["root_config"]), overrides=list(selector["overrides"]))

    slot_cfg = cfg.model.get(str(selector["slot"]))
    assert slot_cfg is not None, f"{name}: {selector['root_config']} has no {selector['slot']} node"

    target = str(OmegaConf.to_container(slot_cfg, resolve=False)["_target_"])
    module, unavailable = _import_network_module(target.split(".${")[0])
    assert module is not None, (
        f"{name}: {unavailable}. A selector in this table must be constructible here; if a model "
        "needs an unavailable native dependency it belongs in BLOCKED_MODEL_SELECTORS."
    )

    net = instantiate(slot_cfg, **dict(selector.get("slot_kwargs") or {}))
    produced = f"{type(net).__module__}.{type(net).__qualname__}"
    assert produced == selector["expected_target_class"], (
        f"{name}: selector produced {produced}, expected {selector['expected_target_class']}"
    )
    assert sum(p.numel() for p in net.parameters()) > 0, f"{name} constructed a network with no parameters"


@pytest.mark.parametrize("blocked", sorted(BLOCKED_MODEL_SELECTORS))
def test_blocked_selectors_state_a_reason_and_are_not_public_required(blocked):
    """A BLOCKED relation is a recorded decision, not an omission."""
    assert len(BLOCKED_MODEL_SELECTORS[blocked]) > 40, f"{blocked}: blocked without a stated reason"
    assert blocked not in REQUIRED_PUBLIC_MODEL_GROUPS


@pytest.mark.parametrize("config_path", RETAINED_PUBLIC_PROJECT_CONFIGS)
def test_every_retained_public_project_config_composes(config_path):
    """Each retained project config is tested on its own; none is a proxy for the others."""
    assert (REPO / config_path).is_file(), f"{config_path} is listed as retained but is absent"
    config_name = config_path[len("configs/") : -len(".yaml")]
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name)
    assert "model" in cfg, f"{config_path} composed without a model node"


def test_this_module_selects_no_config_by_filesystem_order():
    """Anti-regression for the defect that refuted the 30-path deletion set.

    Order-dependent selection is invisible until an unrelated file disappears, so it is banned at
    source level rather than argued about case by case.
    """
    source = _read(Path(__file__))
    body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    for pattern in (r"sorted\([^\n]*\)\s*\[\s*0\s*\]", r"sorted\([^\n]*\)\s*\[\s*-1\s*\]"):
        assert not re.search(pattern, body), f"order-dependent selection found: {pattern}"
    for name in re.findall(r"^\s*(\w+)\s*=\s*sorted\(", body, re.MULTILINE):
        assert not re.search(rf"\b{name}\s*\[\s*-?[01]\s*\]", body), (
            f"{name} is bound to sorted(...) and then indexed: that is order-dependent selection"
        )


# --------------------------------------------------------------------------------------------
# The contract, per reachable model
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", REACHABLE_MODELS)
def test_reachable_model_config_file_exists(name):
    assert (MODEL_DIR / f"{name}.yaml").is_file(), (
        f"'{name}' is selected by {sorted(MODEL_REFERENCES[name])} but "
        f"configs/model/{name}.yaml does not exist, so Hydra composition fails outright"
    )


@pytest.mark.parametrize("name", REACHABLE_MODELS)
def test_reachable_model_config_composes(name):
    model_cfg = _compose_model(name)
    assert model_cfg is not None
    assert _target_nodes(model_cfg), f"model/{name} declares no _target_ network node"


@pytest.mark.parametrize("name", REACHABLE_MODELS)
def test_reachable_model_config_interpolations_resolve(name):
    """Every ${...} inside a network _target_ must be answerable from the composed model config."""
    model_cfg = _compose_model(name)
    for slot, target in _target_nodes(model_cfg).items():
        for key in _INTERPOLATION_RE.findall(target):
            leaf = key.split(".")[-1]
            assert leaf in model_cfg, (
                f"model/{name}: _target_ '{target}' interpolates '{key}', but '{leaf}' is not "
                f"defined in the composed model config"
            )
            assert leaf == slot, (
                f"model/{name}: node '_{slot}' interpolates '{key}'; the slot and the key must "
                "match or the wrong factory is selected"
            )


@pytest.mark.parametrize("name", REACHABLE_MODELS)
def test_reachable_model_config_factories_exist(name):
    """The resolved factory symbol must exist in the network module it names.

    Skips with an explicit reason when the network module needs a native dependency that is not
    installed here (e.g. medvit_3d -> NATTEN). An unavailable dependency is an environment fact;
    only a module that imports cleanly can tell us whether a factory is genuinely missing.
    """
    model_cfg = _compose_model(name)
    missing: list[str] = []
    checked = 0

    for slot, target in _target_nodes(model_cfg).items():
        factory = model_cfg.get(slot)
        if factory in (None, ""):
            continue  # declared-disabled capability; covered by the next test
        module_path = target.split(".${")[0]
        module, unavailable = _import_network_module(module_path)
        if module is None:
            pytest.skip(f"{name}: {unavailable}")
        checked += 1
        if not hasattr(module, str(factory)):
            missing.append(f"{slot}={factory!r} not found in {module_path}")

    assert checked, f"model/{name} enables no capability slot at all"
    assert not missing, f"model/{name} names nonexistent factories: " + "; ".join(missing)


@pytest.mark.parametrize("name", REACHABLE_MODELS)
def test_reachable_model_config_does_not_disable_capabilities_accidentally(name):
    """An empty capability slot must be *declared* empty in the model file, not inherited by
    accident from the `core/<arch>` group.

    `plugin_seg_net:` is intentionally blank for ResEnc, UNet-S/tiny and dinov2_test -- the online
    segmentation plugin is genuinely unavailable there. That is a decision, and the decision has to
    be visible in the model config that owns it.
    """
    model_cfg = _compose_model(name)
    own_text = _read(MODEL_DIR / f"{name}.yaml")

    for slot in CAPABILITY_SLOTS:
        if slot not in model_cfg:
            continue
        if model_cfg.get(slot) not in (None, ""):
            continue
        assert re.search(rf"^\s*{slot}\s*:", own_text, re.MULTILINE), (
            f"model/{name}: capability '{slot}' resolves to empty but is not declared in "
            f"configs/model/{name}.yaml - it was disabled by inheritance, which is unreviewable"
        )

    # A model that enables nothing is a composition failure, not a valid configuration.
    enabled = [s for s in CAPABILITY_SLOTS if model_cfg.get(s) not in (None, "")]
    assert enabled, f"model/{name} disables every capability slot"


# --------------------------------------------------------------------------------------------
# The ratchet: known-broken configs stay inactive, and no new ones appear
# --------------------------------------------------------------------------------------------


def _unresolvable_reason(name: str) -> str | None:
    """Return a reason string if configs/model/<name>.yaml cannot be used, else None."""
    path = MODEL_DIR / f"{name}.yaml"
    if not path.is_file():
        return "MISSING_CONFIG_FILE"
    try:
        model_cfg = _compose_model(name)
    except Exception as exc:  # noqa: BLE001 - any composition failure counts
        return f"COMPOSE_FAILED:{type(exc).__name__}"
    for slot, target in _target_nodes(model_cfg).items():
        factory = model_cfg.get(slot)
        if factory in (None, ""):
            continue
        module, unavailable = _import_network_module(target.split(".${")[0])
        if module is None:
            continue  # dependency unavailable here; not an in-repository defect
        if not hasattr(module, str(factory)):
            return "MISSING_FACTORY"
    return None


def test_unresolvable_model_configs_are_exactly_the_known_set():
    """Ratchet. Fails on a NEW breakage, and also when a pinned one is fixed."""
    candidates = set(ON_DISK_MODELS) | set(KNOWN_UNRESOLVABLE)
    observed = {name: reason for name in sorted(candidates) if (reason := _unresolvable_reason(name))}

    assert set(observed) == set(KNOWN_UNRESOLVABLE), (
        "the set of unresolvable model configs changed.\n"
        f"  observed : {observed}\n"
        f"  pinned   : { {k: v['kind'] for k, v in KNOWN_UNRESOLVABLE.items()} }\n"
        "If a config was fixed, delete its KNOWN_UNRESOLVABLE entry. If a new one broke, fix the "
        "config -- do not extend the pin without an audit finding."
    )
    for name, reason in observed.items():
        assert reason == KNOWN_UNRESOLVABLE[name]["kind"], (
            f"{name}: failure mode changed from {KNOWN_UNRESOLVABLE[name]['kind']} to {reason}"
        )


def test_known_unresolvable_configs_are_not_reachable():
    """The requirement that makes this a ratchet rather than an xfail.

    An xfail on `unet_s` would have kept passing if a launcher started selecting it. This asserts
    the exact referrer set instead, so any new selection -- from a launcher or a project config --
    fails immediately. With the registry empty, it degenerates to "nothing broken is referenced",
    which is the state the wave established.
    """
    for name, entry in KNOWN_UNRESOLVABLE.items():
        expected = set(entry["expected_referrers"])
        actual = set(MODEL_REFERENCES.get(name, set()))
        assert actual == expected, (
            f"'{name}' is a known-broken model config ({entry['finding']}: {entry['detail']}).\n"
            f"  expected referrers: {sorted(expected) or 'none'}\n"
            f"  actual referrers  : {sorted(actual) or 'none'}\n"
            "A broken model configuration must not become reachable. Fix the config or drop the "
            "reference."
        )


def test_no_broken_model_config_reaches_a_scientific_project():
    """Stricter than the referrer pin: no scientific project may select a broken model, ever."""
    for name, entry in KNOWN_UNRESOLVABLE.items():
        offenders = sorted(ref for ref in MODEL_REFERENCES.get(name, set()) if ref.startswith(SCIENTIFIC_CONFIG_PREFIXES))
        assert not offenders, (
            f"'{name}' is unresolvable ({entry['kind']}) but is selected by scientific project config(s): {offenders}"
        )


def test_every_model_reference_in_every_config_resolves():
    """Whole-repo invariant, independent of the reachable-set derivation above.

    Catches the DEBUG_CLS.yaml class of breakage: a project selecting `/model/<name>@model` where
    `configs/model/<name>.yaml` does not exist. The 2026-07-31 audit checked factory *symbols* and
    never checked model-group *membership*, so it could not see this.
    """
    dangling = {
        name: sorted(refs) for name, refs in sorted(MODEL_REFERENCES.items()) if not (MODEL_DIR / f"{name}.yaml").is_file()
    }
    assert not dangling, f"config(s) select a model group that has no config file: {dangling}"


@pytest.mark.parametrize(
    "config_name",
    sorted(
        ref[len("configs/") : -len(".yaml")]
        for refs in MODEL_REFERENCES.values()
        for ref in refs
        if ref.startswith("configs/projects/") and ref.endswith(".yaml")
    ),
)
def test_every_project_config_that_selects_a_model_composes(config_name):
    """Every project config naming a model must actually compose, debug tier included."""
    with initialize_config_dir(version_base="1.2", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name)
    assert "model" in cfg, f"{config_name} composed without a model node"


# --------------------------------------------------------------------------------------------
# Retired groups -- the negative half of the partition
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", RETIRED_MODEL_GROUPS)
def test_retired_model_group_has_no_config_selector_or_block(name):
    """A retired group is absent everywhere, not merely un-selected.

    The positive pins for these groups were removed together with their configs. Without this
    negative assertion the partition would pass just as happily if someone re-added one, or if a
    selector silently resolved it onto a retained model instead.
    """
    assert name not in ON_DISK_MODELS, f"{name} is retired but a model config is still on disk"
    assert name not in MODEL_SELECTORS, f"{name} is retired but still has a selector"
    assert name not in BLOCKED_MODEL_SELECTORS, f"{name} is retired; it needs no block entry"
    assert not (MODEL_DIR / f"{name}.yaml").exists()


@pytest.mark.parametrize("name", RETIRED_MODEL_GROUPS)
def test_retired_model_group_fails_closed_when_composed(name):
    """Composing a retired group raises rather than falling back onto a retained model."""
    with pytest.raises(Exception):
        _compose_model(name)


def test_retired_groups_are_disjoint_from_every_reviewed_set():
    reviewed = (
        set(REQUIRED_PUBLIC_MODEL_GROUPS) | set(NON_D1_MODEL_GROUPS_PENDING_REMOVAL) | set(MODEL_GROUPS_PENDING_DECISION)
    )
    assert not (reviewed & set(RETIRED_MODEL_GROUPS)), "a retired group is still claimed by a reviewed set"
