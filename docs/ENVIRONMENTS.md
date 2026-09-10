# Environments

This release describes two environments, and they are not the same thing. One is the runtime you
install to use this source; the other is a record of what the submitted container was built from.
Conflating them would misstate both, so they are stated separately.

## 1. Public runtime

Declared by `pyproject.toml` and resolved by `uv.lock`. This is the environment the tests and the
entrypoints run in, and the one CI installs.

| Package | Pin | Reason |
|---|---|---|
| `torch` | `==2.7.0` | submitted lineage |
| `lightning` | `==2.4.0` | submitted lineage |
| `torchvision` | `==0.22.0` | submitted lineage |
| `gardening_tools` | `==0.3.2` | submitted lineage, and the version every parity measurement used |
| `natten` | absent | no retained module imports it |

`gardening_tools` is pinned rather than bounded. `asparagus/modules/callbacks/prediction_writer.py`
dispatches on the signature of `save_prediction_from_logits`, so the writer is correct on 0.3.2 and
0.3.5 alike; the pin exists for parity with the submitted lineage, not because a later version
breaks. Note that the parameter naming of the decoder differs between `gardening_tools` releases,
and `tests/amaes_step_reference.json` records the 0.3.2 names — so an unpinned resolution would
fail the numerical contract on structure before it ever reached arithmetic.

`natten` is absent because the model that imported it is not part of this release. The submitted
image did contain it; that fact belongs to the second environment, below, not to this one.

Install it with:

```bash
python -m pip install "uv==0.12.7"
uv sync --locked --all-extras --all-groups
```

## 2. Submitted-container profile

`finetuning/container/historical_sif_profile.json` records what the submitted Apptainer image was
built from — including the NATTEN wheel URLs and SHA-256 digests, and the digest-pinned base image
`pytorch/pytorch@sha256:27c3135420bc184e86977170b6158c6133be3c7cc5c35e9e4fa87bdda629dc2b`.

It is a **record, not an installable environment**. Nothing in this repository installs from it, no
container image is distributed here, and no binary-identical SIF rebuild is claimed or attempted.

## Which environment answers which question

| Question | Environment |
|---|---|
| How do I install and run this code, on CPU or CUDA? | 1 |
| How do I run the tests? | 1 |
| What versions did the released checkpoints and parity measurements use? | 1 — they are the lineage versions |
| What was the submitted container built from? | 2 |

## Regenerating the lock

Locks are generated with **uv 0.12.7**, not with whatever `uv` is on `PATH`. Older uv does not
understand `[tool.uv.extra-build-dependencies]` and silently rewrites the lockfile revision from 3
to 2 — a wrong lock rather than a loud failure.

```bash
uv tool run --from uv==0.12.7 uv lock
uv tool run --from uv==0.12.7 uv lock --check
```
