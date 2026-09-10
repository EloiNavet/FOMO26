"""Contract tests for the FSL rigid rail of ``session_coreg`` (P1 derivative).

FSL is mocked throughout — ``FakeFSL`` records every argv and writes real NIfTI/``.mat`` outputs —
so the whole chain (identity audit, reference choice, single interpolation, shared lattice, rigid
QC, resume, accounting) is exercised without an MRI toolbox. One opt-in test runs the real
``flirt`` when it happens to be on ``PATH``.

These live under ``tests/`` deliberately: ``pyproject.toml`` sets
``testpaths = ["tests", "finetuning/tests"]``, so the older in-package suite at
``asparagus/preprocessing/.../test_session_coreg.py`` is **not** collected by CI. Putting the new
contract here means it actually runs on every push.
"""

from __future__ import annotations

import csv
import json
import nibabel as nib
import numpy as np
import os
import pytest
import shutil
from asparagus_preprocessing import p0_kernel
from asparagus_preprocessing.session_coreg import (
    canonical,
    config as config_mod,
    derivative_manifest,
    fsl,
    metrics,
    pipeline,
    qc as qc_mod,
    run as run_mod,
    session_audit,
)
from asparagus_preprocessing.session_coreg.freesurfer import CommandResult, FreeSurferRunner
from pathlib import Path

# --------------------------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------------------------- #
V2_FIELDS = (
    "sample_id",
    "dataset",
    "participant_id",
    "session_id",
    "cleaned_path",
    "modality_canonical",
    "session_n_scans",
    "shape",
    "pixdim",
    "orientation",
)


def write_nii(path: Path, shape=(24, 22, 20), spacing=(1.0, 1.0, 1.0), seed=0, oblique=False, asymmetric=False) -> Path:
    """A NIfTI with real structure, so NMI and foreground measures are well defined.

    ``asymmetric`` additionally breaks the *rotational* symmetry. The default blob is radially
    symmetric, which is fine for the mocked suite but leaves the three rotation parameters
    mathematically unidentifiable: every rotation about the centre maps the phantom onto itself,
    so a real optimiser minimises the cost equally well at 0 deg and at 90 deg. Only a phantom
    with distinct semi-axes *and* an off-centre lobe gives a 6-DOF fit a unique solution, which
    is what the opt-in real-FLIRT test needs in order to assert anything about the result.
    """
    rng = np.random.default_rng(seed)
    grid = np.indices(shape).astype(np.float32)
    extent = np.array(shape, dtype=np.float32).reshape(3, 1, 1, 1)
    centre = extent / 2.0
    offset = grid - centre
    if asymmetric:
        offset = offset / np.array([1.0, 1.6, 2.4], dtype=np.float32).reshape(3, 1, 1, 1)
    radius = np.sqrt((offset**2).sum(axis=0))
    data = (200.0 * np.exp(-((radius / (0.35 * max(shape))) ** 2))).astype(np.float32)
    if asymmetric:
        lobe = centre + np.array([0.22, -0.16, 0.11], dtype=np.float32).reshape(3, 1, 1, 1) * extent
        lobe_radius = np.sqrt(((grid - lobe) ** 2).sum(axis=0))
        data += (120.0 * np.exp(-((lobe_radius / (0.12 * max(shape))) ** 2))).astype(np.float32)
    data += rng.normal(0, 3.0, shape).astype(np.float32)
    affine = np.diag([*spacing, 1.0]).astype(np.float64)
    if oblique:
        theta = 0.15
        rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        affine[:3, :3] = rot @ affine[:3, :3]
    affine[:3, 3] = [-10.0, -8.0, -6.0]
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def make_corpus(root: Path, sessions: dict, asymmetric: bool = False) -> list:
    """Build a BIDS-ish cleaned tree + canonical rows.

    ``sessions`` maps ``(dataset, sub, ses)`` -> list of ``(modality, shape, spacing)``.
    ``asymmetric`` is forwarded to :func:`write_nii`; only the real-FLIRT test needs it.
    """
    rows = []
    for index, ((dataset, sub, ses), scans) in enumerate(sorted(sessions.items())):
        for order, (modality, shape, spacing) in enumerate(scans):
            path = root / dataset / sub / ses / "anat" / f"{sub}_{ses}_{modality}.nii.gz"
            write_nii(path, shape=shape, spacing=spacing, seed=index * 10 + order, asymmetric=asymmetric)
            rows.append(
                {
                    "dataset": dataset,
                    "participant_id": sub,
                    "session_id": ses,
                    "cleaned_path": str(path),
                    "modality_canonical": modality,
                    "session_n_scans": len(scans),
                    "shape": "x".join(str(s) for s in shape),
                    "pixdim": "x".join(f"{s:g}" for s in spacing),
                    "orientation": "RAS",
                }
            )
    rows.sort(key=lambda r: (r["dataset"], r["participant_id"], r["session_id"], r["cleaned_path"]))
    for i, row in enumerate(rows):
        row["sample_id"] = f"S{i:06d}"
    return rows


def write_manifest(rows: list, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(V2_FIELDS), delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_cleaned_metadata(rows: list, path: Path, overrides: dict = None) -> Path:
    """Cleaned-corpus manifest carrying the scanner/demographic evidence the audit reads."""
    overrides = overrides or {}
    fields = [
        "cleaned_path",
        "Manufacturer",
        "ManufacturersModelName",
        "field_strength_numeric",
        "age_numeric",
        "SoftwareVersions",
        "SeriesDescription",
        "ProtocolName",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = {
                "cleaned_path": row["cleaned_path"],
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "Prisma",
                "field_strength_numeric": "3",
                "age_numeric": "42",
                "SoftwareVersions": "syngo MR E11",
                "SeriesDescription": row["modality_canonical"],
                "ProtocolName": "brain",
            }
            record.update(overrides.get(row["cleaned_path"], {}))
            writer.writerow(record)
    return path


class FakeFSL(fsl.FSLRunner):
    """Simulates ``flirt``: records argv, writes a ``.mat``, resamples onto the ``-ref`` grid.

    ``world_transform`` is the *anatomical* motion the fake registration "found", in scanner
    millimetres; the ``.mat`` written is its FSL-space equivalent for the actual ``-in``/``-ref``
    pair. That is what real FLIRT emits under ``-usesqform``, and it matters: an FSL-space
    identity is **not** a world-space identity when the two images have different fields of view,
    so a fake that wrote ``np.eye(4)`` directly would inject a spurious ~80 mm shift and make the
    rigid QC fire on every session with mismatched FOVs.
    """

    def __post_init__(self):
        super().__post_init__()
        self.calls = []
        self.world_transform = np.eye(4)  # perfectly aligned by default
        self.matrix = None  # set to write a literal FSL-space matrix instead
        self.inverted_matrix = None  # set to make convert_xfm write something other than the inverse
        self.fail_next = False

    def run(self, cmd, expect_output=None):
        cmd = [str(part) for part in cmd]
        forbidden = [flag for flag in fsl.FORBIDDEN_FLAGS if flag in cmd]
        if forbidden:
            raise fsl.FSLError(f"forbidden flags {forbidden}")
        self.calls.append(cmd)
        self.command_log.append(CommandResult(cmd=cmd, returncode=0).to_record())
        if self.fail_next:
            self.fail_next = False
            raise fsl.FSLError("flirt failed (exit 1): simulated failure")

        if cmd[0] == "convert_xfm":
            # Simulate the real tool: read the matrix, invert it, write it back out. Handled
            # before the -omat branch below because convert_xfm carries -omat but no -in/-ref.
            source = metrics.parse_flirt_mat(cmd[cmd.index("-inverse") + 1])
            written = self.inverted_matrix if self.inverted_matrix is not None else np.linalg.inv(source)
            out_mat = Path(cmd[cmd.index("-omat") + 1])
            out_mat.parent.mkdir(parents=True, exist_ok=True)
            out_mat.write_text("\n".join(" ".join(f"{v:.10f}" for v in row) for row in written) + "\n")
            return CommandResult(cmd=cmd, returncode=0)

        if "-omat" in cmd:
            if self.matrix is not None:
                written = np.asarray(self.matrix, dtype=np.float64)
            else:
                moving = nib.load(cmd[cmd.index("-in") + 1])
                reference = nib.load(cmd[cmd.index("-ref") + 1])
                written = metrics.world_to_fsl(reference) @ self.world_transform @ np.linalg.inv(metrics.world_to_fsl(moving))
            out_mat = Path(cmd[cmd.index("-omat") + 1])
            out_mat.parent.mkdir(parents=True, exist_ok=True)
            out_mat.write_text("\n".join(" ".join(f"{v:.10f}" for v in row) for row in written) + "\n")
        if "-out" in cmd:
            moving = nib.load(cmd[cmd.index("-in") + 1])
            reference = nib.load(cmd[cmd.index("-ref") + 1])
            from nibabel.processing import resample_from_to

            resampled = resample_from_to(moving, reference, order=1, cval=float(np.min(moving.dataobj)))
            out = Path(cmd[cmd.index("-out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            nib.save(resampled, out)
        return CommandResult(cmd=cmd, returncode=0)


def fsl_config(**kwargs) -> config_mod.CoregConfig:
    defaults = dict(backend="fsl", do_skull_strip=False, save_brain_mask=False, compute_registration_metrics=True)
    defaults.update(kwargs)
    return config_mod.CoregConfig(**defaults)


def process(rows, root: Path, out: Path, runner=None, config=None, coreg_allowed=True):
    """Run one session end-to-end through the real pipeline with a mocked FSL."""
    config = config or fsl_config()
    sessions = canonical.sessions_from_canonical(rows)
    results = []
    for session in sessions:
        session.coreg_allowed = coreg_allowed
        if not coreg_allowed:
            session.coreg_block_reason = "ambiguous_session:test"
        results.append(
            pipeline.process_session(
                session=session,
                config=config,
                input_root=str(root),
                output_root=str(out),
                runner=runner or FakeFSL(interp=config.image_interp),
            )
        )
    return results


# --------------------------------------------------------------------------------------------- #
# 1. The P0 contract: nothing that does not need a transform may drift from P0
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,spacing,oblique",
    [((24, 22, 20), (1.0, 1.0, 1.0), False), ((32, 30, 8), (0.5, 0.5, 6.0), False), ((20, 18, 16), (1.2, 0.9, 2.5), True)],
)
def test_p0_kernel_matches_the_frozen_p0_implementation(tmp_path, shape, spacing, oblique):
    """The shared kernel must reproduce the P0 corpus exactly, including oblique/thick-slice."""
    from nibabel.processing import resample_from_to

    source = write_nii(tmp_path / "src.nii.gz", shape=shape, spacing=spacing, oblique=oblique)
    img = nib.load(str(source))

    # The verbatim P0 code, as it stood in upstream_pilot.convert_one before extraction.
    src = np.asanyarray(img.dataobj, dtype=np.float32)
    ras = nib.as_closest_canonical(nib.Nifti1Image(src, img.affine, img.header))
    zooms = tuple(float(z) for z in ras.header.get_zooms()[:3])
    out_shape = tuple(int(np.ceil(s * z / 1.0)) for s, z in zip(ras.shape[:3], zooms))
    affine = ras.affine.copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag([1.0 / z for z in zooms])
    expected = resample_from_to(ras, (out_shape, affine), order=3, mode="constant", cval=float(np.min(ras.dataobj)))

    actual = p0_kernel.p0_resample(img)
    assert np.array_equal(np.asanyarray(actual.dataobj, np.float32), np.asanyarray(expected.dataobj, np.float32))
    assert np.array_equal(actual.affine, expected.affine)


# `test_p0_conversion_still_routes_through_the_shared_kernel` drove the Task-5 upstream pilot's
# `convert_one` to prove it consumed p0_kernel rather than a private copy. The pilot is campaign
# code and is not published; the kernel it shared is still covered by the P0 tests above.


def test_single_scan_session_is_bit_identical_to_p0(tmp_path):
    """A single-scan session is passthrough: same bytes as P0, never 'registered'."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0, 1.0, 1.3))]})
    qc = process(rows, root, out)[0]

    assert qc["status"] == "success"
    assert qc["coregistration_applied"] is False
    mod = qc["modalities"][0]
    assert mod["registered"] is False
    produced = np.asanyarray(nib.load(mod["output_path"]).dataobj, dtype=np.float32)
    expected = np.asanyarray(p0_kernel.p0_resample(rows[0]["cleaned_path"]).dataobj, dtype=np.float32)
    assert np.array_equal(produced, expected)


def test_session_reference_is_bit_identical_to_p0(tmp_path):
    """In a multi-scan session the reference is P0 output; only the moving scans are resampled."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0, 1.0, 1.0)), ("T2w", (20, 18, 8), (1.2, 1.2, 4.0))]},
    )
    qc = process(rows, root, out)[0]
    reference = next(m for m in qc["modalities"] if m["is_reference"])
    assert reference["modality_canonical"] == "T1w"

    produced = np.asanyarray(nib.load(reference["output_path"]).dataobj, dtype=np.float32)
    expected = np.asanyarray(p0_kernel.p0_resample(reference["source_path"]).dataobj, dtype=np.float32)
    assert np.array_equal(produced, expected)


# --------------------------------------------------------------------------------------------- #
# 2. Backend command contract
# --------------------------------------------------------------------------------------------- #
def test_estimate_command_is_rigid_6dof_and_always_saves_the_transform(tmp_path):
    cmd = fsl.flirt_estimate_command("mov.nii.gz", "ref.nii.gz", "T.mat")
    assert cmd[0] == "flirt"
    assert cmd[cmd.index("-dof") + 1] == "6", "the production contract is rigid; 12 was the old pilot default"
    assert cmd[cmd.index("-cost") + 1] == "normmi"
    assert "-omat" in cmd, "a registration whose transform was not saved cannot be audited"
    assert "-dof" in cmd, "DOF must be explicit; FSL's default must never decide it"


def test_apply_command_names_its_interpolation_and_reuses_the_estimated_matrix():
    cmd = fsl.flirt_apply_command("mov.nii.gz", "grid.nii.gz", "out.nii.gz", "T.mat")
    assert cmd[cmd.index("-interp") + 1] == "spline", "cubic, matching the P0 kernel's order=3"
    assert "-applyxfm" in cmd and cmd[cmd.index("-init") + 1] == "T.mat"


def test_no_masking_or_skull_stripping_flag_is_ever_emitted():
    for cmd in (
        fsl.flirt_estimate_command("m.nii.gz", "r.nii.gz", "T.mat"),
        fsl.flirt_apply_command("m.nii.gz", "r.nii.gz", "o.nii.gz", "T.mat"),
    ):
        for forbidden in ("-inweight", "-refweight", "-wmseg", "bet", "-fieldmap"):
            assert forbidden not in cmd


def test_runner_refuses_a_command_carrying_non_registration_flags(tmp_path):
    with pytest.raises(fsl.FSLError, match="non-registration flags"):
        fsl.FSLRunner().run(["flirt", "-in", "a", "-ref", "b", "-refweight", "mask.nii.gz"])


def test_twelve_dof_requires_an_explicitly_named_ablation():
    with pytest.raises(ValueError, match="rigid"):
        fsl.FSLRunner(dof=12)
    with pytest.raises(ValueError, match="rigid"):
        fsl_config(dof=12)
    assert fsl.FSLRunner(dof=12, allow_affine_ablation=True).dof == 12


def test_fsl_backend_refuses_skull_stripping_at_both_layers():
    with pytest.raises(ValueError, match="never skull-strips"):
        config_mod.CoregConfig(backend="fsl", do_skull_strip=True)
    with pytest.raises(fsl.FSLError, match="never skull-strips"):
        fsl.FSLRunner().synthseg("in.nii.gz", "seg.nii.gz")


def test_backend_default_interpolation_is_per_toolbox():
    assert config_mod.CoregConfig().image_interp == "trilin"
    assert fsl_config().image_interp == "spline"
    with pytest.raises(ValueError, match="must be one of"):
        fsl_config(image_interp="trilin")  # a FreeSurfer word must not reach FSL


def test_flirt_failure_becomes_a_structured_session_failure(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0, 1.0, 1.0)), ("T2w", (22, 20, 18), (1.0, 1.0, 1.0))]},
    )
    runner = FakeFSL(interp="spline")
    runner.fail_next = True
    qc = process(rows, root, out, runner=runner)[0]

    assert qc["status"] in {"partial", "failed"}
    failed = [m for m in qc["modalities"] if m["status"] == "failed"]
    assert failed and "simulated failure" in failed[0]["reason"]


def test_single_stored_interpolation_per_moving_scan(tmp_path):
    """Exactly one resampling of each moving scan: no native -> iso -> register -> iso chain."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0, 1.0, 1.0)), ("T2w", (20, 18, 8), (1.2, 1.2, 4.0))]},
    )
    runner = FakeFSL(interp="spline")
    config = fsl_config(compute_registration_metrics=False)  # the NMI-before volume is a QC extra
    process(rows, root, out, runner=runner, config=config)

    writes = [c for c in runner.calls if "-out" in c]
    assert len(writes) == 1, f"expected one stored resample, got {len(writes)}"
    estimate = [c for c in runner.calls if "-omat" in c]
    assert len(estimate) == 1
    # Fit and application must share the same -ref, or the matrix would need composing.
    assert estimate[0][estimate[0].index("-ref") + 1] == writes[0][writes[0].index("-ref") + 1]


# --------------------------------------------------------------------------------------------- #
# 3. Session identity
# --------------------------------------------------------------------------------------------- #
def test_same_session_scans_group_and_different_sessions_never_merge(tmp_path):
    root = tmp_path / "cleaned"
    rows = make_corpus(
        root,
        {
            ("PT01", "sub-01", "ses-01"): [("T1w", (16, 16, 16), (1.0,) * 3), ("T2w", (16, 16, 16), (1.0,) * 3)],
            ("PT01", "sub-01", "ses-02"): [("T1w", (16, 16, 16), (1.0,) * 3)],
            ("PT01", "sub-02", "ses-01"): [("T1w", (16, 16, 16), (1.0,) * 3)],
        },
    )
    groups = canonical.group_by_session(rows)
    assert len(groups) == 3
    assert len(groups["PT01/sub-01/ses-01"]) == 2
    assert len(groups["PT01/sub-01/ses-02"]) == 1
    assert {r["cleaned_path"] for r in groups["PT01/sub-01/ses-01"]}.isdisjoint(
        {r["cleaned_path"] for r in groups["PT01/sub-01/ses-02"]}
    )


def test_contradictory_scanner_inside_one_session_blocks_registration(tmp_path):
    """Two scanner models under one session key means two visits, not one."""
    root = tmp_path / "cleaned"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (16,) * 3, (1.0,) * 3), ("T2w", (16,) * 3, (1.0,) * 3)]},
    )
    meta = write_cleaned_metadata(
        rows, tmp_path / "cleaned_manifest.tsv", overrides={rows[1]["cleaned_path"]: {"ManufacturersModelName": "Signa_HDxt"}}
    )
    verdict = session_audit.audit_sessions(rows, cleaned_metadata_path=str(meta))[0]
    assert verdict.status == "ambiguous"
    assert verdict.eligible is False
    assert any("ManufacturersModelName_conflict" in r for r in verdict.hard_reasons)


def test_benign_series_variation_does_not_block_registration(tmp_path):
    """SeriesDescription/SoftwareVersions legitimately differ between series of one visit."""
    root = tmp_path / "cleaned"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (16,) * 3, (1.0,) * 3), ("T2w", (16,) * 3, (1.0,) * 3)]})
    meta = write_cleaned_metadata(
        rows, tmp_path / "m.tsv", overrides={rows[1]["cleaned_path"]: {"SoftwareVersions": "syngo MR E11\\E11"}}
    )
    verdict = session_audit.audit_sessions(rows, cleaned_metadata_path=str(meta))[0]
    assert verdict.eligible is True
    assert any("SoftwareVersions_varies" in f for f in verdict.soft_flags)


def test_manifest_identity_must_match_the_file_path(tmp_path):
    root = tmp_path / "cleaned"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (16,) * 3, (1.0,) * 3)]})
    rows[0]["session_id"] = "ses-09"  # manifest claims a session the path does not carry
    ok, reason = canonical.identity_matches_path(rows[0])
    assert not ok and "session" in reason
    assert session_audit.audit_sessions(rows)[0].status == "ambiguous"


def test_missing_session_id_fails_closed(tmp_path):
    root = tmp_path / "cleaned"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (16,) * 3, (1.0,) * 3)]})
    rows[0]["session_id"] = ""
    verdict = session_audit.audit_sessions(rows)[0]
    assert verdict.status == "ambiguous"


def test_synthetic_session_id_blocks_only_when_explicitly_required(tmp_path):
    root = tmp_path / "cleaned"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (16,) * 3, (1.0,) * 3), ("T2w", (16,) * 3, (1.0,) * 3)]})
    assert session_audit.audit_sessions(rows)[0].eligible is True
    strict = session_audit.audit_sessions(rows, require_explicit_session=True)[0]
    assert strict.status == "ambiguous"
    assert any("synthetic_session_id" in r for r in strict.hard_reasons)


def test_ambiguous_session_is_passthrough_not_dropped(tmp_path):
    """Fail-closed means 'do not register'. The samples still reach the derivative."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root, {("PT01", "sub-01", "ses-01"): [("T1w", (20, 18, 16), (1.0,) * 3), ("T2w", (20, 18, 16), (1.0,) * 3)]}
    )
    qc = process(rows, root, out, coreg_allowed=False)[0]

    assert qc["status"] == "success"
    assert qc["coregistration_applied"] is False
    assert qc["coreg_block_reason"] == "ambiguous_session:test"
    assert len(qc["modalities"]) == 2
    for mod in qc["modalities"]:
        assert mod["registered"] is False
        assert os.path.exists(mod["output_path"])
        produced = np.asanyarray(nib.load(mod["output_path"]).dataobj, dtype=np.float32)
        expected = np.asanyarray(p0_kernel.p0_resample(mod["source_path"]).dataobj, dtype=np.float32)
        assert np.array_equal(produced, expected), "a blocked session must still get exact P0 geometry"


# --------------------------------------------------------------------------------------------- #
# 4. Reference selection
# --------------------------------------------------------------------------------------------- #
def test_isotropic_t1w_beats_thick_slice_flair(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {
            ("PT01", "sub-01", "ses-01"): [
                ("T1w", (24, 24, 24), (1.0, 1.0, 1.0)),
                ("FLAIR", (48, 48, 6), (0.49, 0.49, 6.5)),  # high in-plane, unusable through-plane
            ]
        },
    )
    qc = process(rows, root, out)[0]
    assert qc["reference"]["modality"] == "T1w"
    gated = {c["modality"]: c["gated"] for c in qc["reference"]["candidates"]}
    assert gated["FLAIR"] is True, "a 6.5 mm slice must be gated out of reference selection"


def test_reference_priority_uses_the_curated_label_not_the_filename(tmp_path):
    """The canonical label is authoritative; classify() matches substrings and can be fooled."""
    from asparagus_preprocessing.session_coreg.geometry import ScanGeometry

    misleading = ScanGeometry(path="/x/sub-t1w-cohort_ses-01_T2w.nii.gz", shape=(16,) * 3, voxel_sizes=(1.0,) * 3, ndim=3)
    assert misleading.priority == 0, "filename guess picks T1w"
    labelled = ScanGeometry(
        path="/x/sub-t1w-cohort_ses-01_T2w.nii.gz", shape=(16,) * 3, voxel_sizes=(1.0,) * 3, ndim=3, modality_label="T2w"
    )
    assert labelled.priority == 2 and labelled.modality == "T2w"


def test_canonical_modality_vocabulary_ranks_diffusion_below_structural():
    from asparagus_preprocessing.session_coreg import modalities

    order = [modalities.priority_rank_for(m) for m in ("T1w", "T1c", "T2w", "FLAIR", "PD", "ADC", "DWI_B0")]
    assert order == sorted(order), "structural must outrank diffusion as a session reference"
    assert modalities.priority_rank_for("T1ce") == modalities.priority_rank_for("T1c")
    assert modalities.priority_rank_for("nonsense") == modalities.CANONICAL_OTHER_RANK


def test_reference_selection_is_deterministic(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    spec = {
        ("PT01", "sub-01", "ses-01"): [
            ("T2w", (20, 20, 20), (1.0, 1.0, 1.0)),
            ("FLAIR", (20, 20, 20), (1.0, 1.0, 1.0)),
            ("T1w", (20, 20, 20), (1.0, 1.0, 1.0)),
        ]
    }
    rows = make_corpus(root, spec)
    chosen = {process(rows, root, out / str(i))[0]["reference"]["source_path"] for i in range(3)}
    assert len(chosen) == 1


# --------------------------------------------------------------------------------------------- #
# 5. Shared session geometry
# --------------------------------------------------------------------------------------------- #
def test_registered_modalities_share_one_lattice(tmp_path):
    """The contract cross-modal JEPA relies on: token i is the same box in every modality."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {
            ("PT01", "sub-01", "ses-01"): [
                ("T1w", (24, 22, 20), (1.0, 1.0, 1.0)),
                ("T2w", (18, 16, 9), (1.4, 1.4, 3.0)),
                ("FLAIR", (20, 20, 12), (1.1, 1.1, 2.0)),
            ]
        },
    )
    qc = process(rows, root, out)[0]
    assert qc["status"] == "success"

    images = [nib.load(m["output_path"]) for m in qc["modalities"] if m["status"] == "ok"]
    assert len(images) == 3
    assert len({img.shape for img in images}) == 1
    for img in images[1:]:
        assert np.allclose(img.affine, images[0].affine, atol=1e-4)
        assert "".join(nib.aff2axcodes(img.affine)) == "RAS"
        assert np.allclose(nib.affines.voxel_sizes(img.affine)[:3], 1.0, atol=0.05)


def test_shape_mismatch_across_a_session_is_a_rejection(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (20,) * 3, (1.0,) * 3), ("T2w", (20,) * 3, (1.0,) * 3)]})
    qc = process(rows, root, out)[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    # Corrupt one output so the session no longer shares a lattice, then re-validate.
    write_nii(Path(moving["output_path"]), shape=(9, 9, 9), spacing=(1.0,) * 3)
    fresh = dict(qc, modalities=[dict(m, status="ok") for m in qc["modalities"]])
    summary = qc_mod.validate_session_outputs(fresh, fsl_config())
    assert any("shape_mismatch" in r for r in summary["session_rejects"])


# --------------------------------------------------------------------------------------------- #
# 6. Rigid-transform sanity
# --------------------------------------------------------------------------------------------- #
def _reg(matrix: np.ndarray) -> dict:
    out = metrics.decompose_affine(matrix)
    out.update(parsed=True, world_space=True)
    return out


def test_rigid_transform_passes_and_shear_scale_reflection_are_rejected():
    config = fsl_config()
    theta = np.radians(9.0)
    rigid = np.eye(4)
    rigid[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    rigid[:3, 3] = [3.0, -2.0, 1.0]
    assert qc_mod.rigid_violations(_reg(rigid), config) == []

    shear = np.eye(4)
    shear[0, 1] = 0.05  # determinant is exactly 1 -> a det-only check would pass this
    assert _reg(shear)["determinant"] == pytest.approx(1.0)
    assert any("shear" in v for v in qc_mod.rigid_violations(_reg(shear), config))

    scale = np.diag([1.05, 1.0, 1.0, 1.0])
    assert any("scale" in v or "volume_change" in v for v in qc_mod.rigid_violations(_reg(scale), config))

    reflection = np.diag([-1.0, 1.0, 1.0, 1.0])
    assert any("reflection" in v for v in qc_mod.rigid_violations(_reg(reflection), config))


def test_implausible_motion_is_rejected():
    config = fsl_config()
    far = np.eye(4)
    far[:3, 3] = [200.0, 0.0, 0.0]
    assert any("implausible_translation" in v for v in qc_mod.rigid_violations(_reg(far), config))


def test_unparseable_or_fsl_space_transforms_are_not_judged():
    """A metric we could not compute must not silently become a pass or a fail."""
    config = fsl_config()
    assert qc_mod.rigid_violations({"parsed": False}, config) == []
    assert qc_mod.rigid_violations({"parsed": True, "world_space": False, "determinant": -1.0}, config) == []


def test_non_rigid_transform_fails_the_modality(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (20,) * 3, (1.0,) * 3), ("T2w", (20,) * 3, (1.0,) * 3)]})
    runner = FakeFSL(interp="spline")
    runner.world_transform = np.diag([1.4, 1.0, 1.0, 1.0])  # a scale FLIRT would never produce at -dof 6
    qc = process(rows, root, out, runner=runner)[0]

    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    assert moving["status"] == "failed"
    assert "transform_qc" in moving["reason"]
    assert qc["status"] == "partial"


def test_an_fsl_space_identity_is_not_a_world_space_identity(tmp_path):
    """Why the world-space conversion is mandatory, pinned as a regression.

    Two same-session scans with different fields of view have different FSL frames. Reading a
    FLIRT ``.mat`` as if it were millimetres therefore reports a large bogus translation — which
    is exactly what the rigid QC would reject. Observed on real clinical data: ~78 mm.
    """
    moving = nib.load(str(write_nii(tmp_path / "mov.nii.gz", shape=(60, 60, 20), spacing=(1.0, 1.0, 3.0))))
    reference = nib.load(str(write_nii(tmp_path / "ref.nii.gz", shape=(24,) * 3, spacing=(1.0,) * 3)))

    naive = metrics.decompose_affine(np.eye(4))
    assert naive["translation_norm_mm"] == 0.0

    honest = metrics.decompose_affine(metrics.flirt_world_transform(np.eye(4), moving, reference))
    assert honest["translation_norm_mm"] > 20.0, "an FSL identity hides a real world-space shift"

    # And the converse: a genuine world-space identity round-trips to zero motion.
    fsl_matrix = metrics.world_to_fsl(reference) @ np.eye(4) @ np.linalg.inv(metrics.world_to_fsl(moving))
    aligned = metrics.decompose_affine(metrics.flirt_world_transform(fsl_matrix, moving, reference))
    assert aligned["translation_norm_mm"] == pytest.approx(0.0, abs=1e-9)
    assert aligned["rotation_deg"] == pytest.approx(0.0, abs=1e-9)


def test_flirt_matrix_is_decomposed_in_world_space(tmp_path):
    """A FLIRT matrix is expressed between FSL frames; rigidity must be judged in millimetres."""
    moving = nib.load(str(write_nii(tmp_path / "mov.nii.gz", shape=(30, 28, 10), spacing=(0.6, 0.6, 4.0))))
    reference = nib.load(str(write_nii(tmp_path / "ref.nii.gz", shape=(40,) * 3, spacing=(1.0,) * 3)))

    theta = np.radians(7.0)
    world = np.eye(4)
    world[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    world[:3, 3] = [2.0, -3.0, 1.5]
    in_fsl = metrics.world_to_fsl(reference) @ world @ np.linalg.inv(metrics.world_to_fsl(moving))

    mat_path = tmp_path / "T.mat"
    mat_path.write_text("\n".join(" ".join(f"{v:.10f}" for v in row) for row in in_fsl) + "\n")
    recovered = metrics.flirt_metrics(
        str(mat_path), moving=str(tmp_path / "mov.nii.gz"), reference=str(tmp_path / "ref.nii.gz")
    )

    assert recovered["world_space"] is True
    assert recovered["rotation_deg"] == pytest.approx(7.0, abs=1e-6)
    assert recovered["translation_norm_mm"] == pytest.approx(np.linalg.norm([2.0, -3.0, 1.5]), abs=1e-6)
    assert recovered["determinant"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------------------------- #
# 7. Diffusion is handled conservatively
# --------------------------------------------------------------------------------------------- #
def test_diffusion_gets_its_own_similarity_threshold_not_a_weakened_global_one():
    config = fsl_config()
    structural = {"modality_canonical": "T1w"}
    adc = {"modality_canonical": "ADC"}
    assert qc_mod.nmi_threshold_for(structural, config) == config.qc.nmi_improvement_warn
    assert qc_mod.nmi_threshold_for(adc, config) == config.qc.nmi_improvement_warn_dwi
    assert qc_mod.nmi_threshold_for(adc, config) < qc_mod.nmi_threshold_for(structural, config)
    for label in ("ADC", "DWI_B0", "DWI_B1000", "DWI_TRACE"):
        assert qc_mod.is_diffusion({"modality_canonical": label})
    assert not qc_mod.is_diffusion({"modality_canonical": "FLAIR"})


def _zero_overlap_qc(tmp_path, *, src_count, out_count, status="ok", name="T2w"):
    """A session QC dict carrying the structural foreground evidence and nothing controversial."""
    output = write_nii(tmp_path / f"{name}_out.nii.gz", shape=(20, 18, 16), spacing=(1.0, 1.0, 1.0))
    return {
        "modalities": [
            {
                "name": name,
                "modality_canonical": name,
                "status": status,
                "output_path": str(output),
                "is_reference": False,
                "registration": {
                    "parsed": True,
                    "world_space": True,
                    "determinant": 1.0,
                    "orthogonality_error": 0.0,
                },
                "source_foreground_voxel_count": src_count,
                "output_foreground_voxel_count": out_count,
            }
        ]
    }


def test_interpolation_dust_does_not_count_as_surviving_foreground(tmp_path):
    """The exact S003989 failure: a published volume that is nonzero everywhere but empty.

    A spline resample of a non-overlapping volume leaves values of order 1e-11. Any check phrased
    as "is the image nonzero" passes it, which is why the invariant is decided on the count of
    voxels above the source-derived foreground threshold instead.
    """
    dust = np.full((12, 12, 12), 2.5e-11, dtype=np.float32)
    dust[0, 0, 0] = -7.9e-12
    dust_path = tmp_path / "dust.nii.gz"
    nib.save(nib.Nifti1Image(dust, np.eye(4)), str(dust_path))

    data = np.asanyarray(nib.load(str(dust_path)).dataobj)
    assert np.count_nonzero(data) == data.size, "premise: every voxel is nonzero"

    qc = _zero_overlap_qc(tmp_path, src_count=48_000, out_count=0)
    summary = qc_mod.validate_session_outputs(qc, fsl_config())
    assert any("zero_foreground_overlap" in r for r in summary["session_rejects"])
    assert qc["modalities"][0]["status"] == "failed"


def test_zero_surviving_foreground_is_a_rejection(tmp_path):
    """source foreground > 0 and output foreground == 0 -> rejected, whatever the ratio rounds to."""
    qc = _zero_overlap_qc(tmp_path, src_count=48_000, out_count=0)
    summary = qc_mod.validate_session_outputs(qc, fsl_config())
    assert summary["flags"].get("zero_foreground_overlap") is True
    mod = qc["modalities"][0]
    assert mod["status"] == "failed"
    # Routed through the existing status contract as an output-QC rejection, not a crash.
    assert "output_qc:zero_foreground_overlap" in mod["reason"]


def test_a_low_retention_output_that_kept_real_foreground_stays_publishable(tmp_path):
    """The 0.19-0.25 header-only cases and the ~0.42-0.49 B1000 cases must survive.

    They lost most of their coverage but they still carry anatomy, and the corpus needs them.
    """
    for src_count, out_count in ((48_000, 8_928), (48_000, 22_896), (48_000, 1)):
        qc = _zero_overlap_qc(tmp_path, src_count=src_count, out_count=out_count)
        summary = qc_mod.validate_session_outputs(qc, fsl_config())
        assert not any("zero_foreground_overlap" in r for r in summary["session_rejects"])
        assert qc["modalities"][0]["status"] == "ok"


def test_the_b1000_retention_case_remains_a_warning_and_is_never_rejected(tmp_path):
    """The observed ~0.477 DWI_B1000 samples are warning-level; the new invariant must not touch them."""
    qc = _zero_overlap_qc(tmp_path, src_count=48_000, out_count=22_896, name="DWI_B1000")
    qc["modalities"][0]["foreground_retained_frac"] = 0.477
    summary = qc_mod.validate_session_outputs(qc, fsl_config())
    assert any("clipped_by_reference_fov" in w for w in summary["session_warns"])
    assert not any("zero_foreground_overlap" in r for r in summary["session_rejects"])
    assert qc["modalities"][0]["status"] == "ok"


def test_the_invariant_needs_evidence_and_never_guesses(tmp_path):
    """A modality with no recorded counts -- dry run, or a source too small to threshold -- is
    not rejected. Absence of evidence is not evidence of an empty output."""
    qc = _zero_overlap_qc(tmp_path, src_count=None, out_count=None)
    del qc["modalities"][0]["source_foreground_voxel_count"]
    del qc["modalities"][0]["output_foreground_voxel_count"]
    summary = qc_mod.validate_session_outputs(qc, fsl_config())
    assert not any("zero_foreground_overlap" in r for r in summary["session_rejects"])
    assert qc["modalities"][0]["status"] == "ok"


def test_the_pipeline_records_the_structural_foreground_evidence(tmp_path):
    """The counts must actually be produced by a real run, not only honoured when hand-written."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0,) * 3), ("T2w", (18, 16, 9), (1.4, 1.4, 3.0))]},
    )
    qc = process(rows, root, out)[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    assert moving["source_foreground_voxel_count"] > 0
    assert moving["output_foreground_voxel_count"] > 0
    assert moving["source_foreground_volume_mm3"] > 0
    assert moving["output_foreground_volume_mm3"] > 0
    # The raw ratio is the unrounded evidence behind the rounded reporting value.
    assert moving["foreground_retained_frac_raw"] == pytest.approx(moving["foreground_retained_frac"], abs=5e-5)


def test_a_zero_overlap_candidate_never_survives_at_the_canonical_path(tmp_path):
    """Atomic publication must hold: the rejected candidate is quarantined and nothing is left
    at the path a consumer would read."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0,) * 3), ("T2w", (18, 16, 9), (1.4, 1.4, 3.0))]},
    )

    real_retention = pipeline._foreground_retention

    def empty_output(mod, source_path, resampled_path, runner):
        real_retention(mod, source_path, resampled_path, runner)
        if not mod.get("is_reference"):
            mod["output_foreground_voxel_count"] = 0
            mod["foreground_retained_frac"] = 0.0

    pipeline._foreground_retention = empty_output
    try:
        qc = process(rows, root, out)[0]
    finally:
        pipeline._foreground_retention = real_retention

    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    assert moving["status"] == "failed"
    assert moving["published"] is False
    assert moving["output_path"] == "", "a rejected sample must not advertise a usable path"
    intended = moving["unpublished_output_path"]
    assert intended and not os.path.exists(intended), "nothing may remain at the canonical path"
    assert os.path.exists(moving["quarantined_path"]), "the evidence must be kept for review"


def test_an_optimised_registration_with_real_overlap_is_unaffected(tmp_path):
    """The invariant must be invisible to the ordinary successful path."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0,) * 3), ("T2w", (24, 22, 20), (1.0,) * 3)]},
    )
    qc = process(rows, root, out)[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    assert moving["status"] == "ok"
    assert moving["published"] is True
    assert moving["transform_fitted"] is True
    assert moving["output_foreground_voxel_count"] > 0


def test_foreground_retention_is_a_diagnostic_never_a_rejection(tmp_path):
    """The metric is uncalibrated. It may flag a session for review; it may never fail one."""
    config = fsl_config()
    severe = {"parsed": True, "world_space": True, "determinant": 1.0, "orthogonality_error": 0.0}
    assert qc_mod.rigid_violations(severe, config) == []

    # A genuinely valid output (RAS, 1 mm, finite, non-empty) whose only defect is heavy clipping.
    output = write_nii(tmp_path / "out.nii.gz", shape=(20, 18, 16), spacing=(1.0, 1.0, 1.0))
    qc = {
        "modalities": [
            {
                "name": "T2w",
                "modality_canonical": "T2w",
                "status": "ok",
                "output_path": str(output),
                "is_reference": False,
                "registration": severe,
                "foreground_retained_frac": 0.05,
            }
        ]
    }
    summary = qc_mod.validate_session_outputs(qc, config)
    assert any("clipped_by_reference_fov" in w for w in summary["session_warns"])
    assert summary["flags"].get("fov_clipped") is True
    assert not any("clipped" in r for r in summary["session_rejects"])
    assert qc["modalities"][0]["status"] != "failed", "clipping must not fail a modality"


def test_retained_fraction_is_recorded_continuously_in_the_aggregate_row(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0,) * 3), ("T2w", (18, 16, 9), (1.4, 1.4, 3.0))]},
    )
    qc = process(rows, root, out)[0]
    flat = qc_mod.session_qc_rows(qc)
    moving = next(r for r in flat if not r["is_reference"])
    assert isinstance(moving["foreground_retained_frac"], float)
    assert "modality_canonical" in moving and "orthogonality_error" in moving


def test_smoke_renders_every_session_not_a_strided_sample(tmp_path):
    """The lowest-retention cases cannot be sampled for in advance, so the smoke renders all."""
    root = tmp_path / "cleaned"
    spec = {("PT01", f"sub-{i:03d}", "ses-01"): [("T1w", (16,) * 3, (1.0,) * 3)] for i in range(30)}
    sessions = canonical.sessions_from_canonical(make_corpus(root, spec))

    strided = run_mod._qc_image_keys(sessions, fsl_config(qc_image_stride=20))
    assert len(strided) < len(sessions)
    every = run_mod._qc_image_keys(sessions, fsl_config(qc_image_render_all=True))
    assert every == {s.key for s in sessions}


def test_registration_qc_renders_all_three_planes(tmp_path):
    """Axial alone cannot answer whether clipped foreground was brain or wider coverage."""
    import inspect

    source = inspect.getsource(qc_mod.render_registration_qc)
    assert "sagittal" in source and "coronal" in source and "axial" in source
    assert "planes = [2, 1, 0]" in source

    reference = write_nii(tmp_path / "ref.nii.gz", shape=(24, 22, 20), spacing=(1.0,) * 3)
    moving = write_nii(tmp_path / "mov.nii.gz", shape=(24, 22, 20), spacing=(1.0,) * 3, seed=5)
    png = tmp_path / "qc" / "session_reg.png"
    assert qc_mod.render_registration_qc(str(reference), str(moving), str(moving), None, str(png), label="T2w")
    assert png.exists() and png.stat().st_size > 0


def test_fov_review_ranks_lowest_retention_first_and_points_at_the_montage():
    rows = [
        {"is_reference": True, "foreground_retained_frac": 1.0, "session_key": "a", "dataset": "PT01"},
        {
            "is_reference": False,
            "foreground_retained_frac": 0.91,
            "session_key": "b",
            "dataset": "PT01",
            "subject": "sub-02",
            "session": "ses-01",
            "modality_canonical": "T2w",
        },
        {
            "is_reference": False,
            "foreground_retained_frac": 0.42,
            "session_key": "c",
            "dataset": "PT02",
            "subject": "sub-03",
            "session": "ses-01",
            "modality_canonical": "FLAIR",
        },
    ]
    review = run_mod._fov_review_rows(rows, "/out")
    assert [r["session_key"] for r in review] == ["c", "b"], "lowest retention must come first"
    # The glob must match both the sampled layout (<base>_reg.png) and the per-modality layout
    # written under --qc-images-all (<base>_reg_<MOD>.png).
    import fnmatch

    for row in review:
        pattern = row["qc_montage_glob"]
        base = pattern.split("*")[0]
        assert fnmatch.fnmatch(f"{base}_reg.png", pattern)
        assert fnmatch.fnmatch(f"{base}_reg_{row['modality']}.png", pattern)
    assert not any(r["session_key"] == "a" for r in review), "the reference is not clipped by itself"


def test_distribution_summarises_a_continuous_metric():
    dist = run_mod._distribution([0.5, 0.6, 0.7, 0.8, 0.9, None, "x"])
    assert dist["n"] == 5 and dist["min"] == 0.5 and dist["max"] == 0.9
    assert run_mod._distribution([])["n"] == 0


def test_diffusion_never_escalates_degrees_of_freedom(tmp_path):
    """Poor DWI alignment must not silently buy more DOF."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (24, 22, 20), (1.0,) * 3), ("ADC", (16, 14, 8), (1.8, 1.8, 4.0))]},
    )
    runner = FakeFSL(interp="spline")
    process(rows, root, out, runner=runner)
    for call in (c for c in runner.calls if "-dof" in c):
        assert call[call.index("-dof") + 1] == "6"


# --------------------------------------------------------------------------------------------- #
# 8. Safety
# --------------------------------------------------------------------------------------------- #
def test_output_root_may_not_be_or_contain_the_input_root(tmp_path):
    from asparagus_preprocessing.session_coreg.paths import UnsafeIOError, assert_safe_io

    root = tmp_path / "cleaned"
    root.mkdir()
    with pytest.raises(UnsafeIOError):
        assert_safe_io(str(root), str(root))
    with pytest.raises(UnsafeIOError):
        assert_safe_io(str(root), str(root / "derived"))
    with pytest.raises(UnsafeIOError):
        assert_safe_io(str(root / "inner"), str(root))


def test_source_tree_is_untouched_by_a_run(tmp_path):
    import hashlib

    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (20, 18, 16), (1.0,) * 3), ("T2w", (18, 16, 14), (1.2,) * 3)]},
    )
    before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(str(x) for x in root.rglob("*.nii.gz"))}
    process(rows, root, out)
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(str(x) for x in root.rglob("*.nii.gz"))}
    assert before == after
    assert set(before), "the guard must actually have had files to protect"


def test_nothing_is_written_outside_the_derivative_root(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    other = tmp_path / "bystander"
    other.mkdir()
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (18,) * 3, (1.0,) * 3), ("T2w", (18,) * 3, (1.0,) * 3)]})
    process(rows, root, out)
    assert list(other.iterdir()) == []
    assert out.exists() and any(out.rglob("*.nii.gz"))


def test_transform_is_persisted_outside_the_deleted_work_dir(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (20,) * 3, (1.0,) * 3), ("T2w", (20,) * 3, (1.0,) * 3)]})
    qc = process(rows, root, out)[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    transform = moving["transform_path"]
    assert transform and os.path.exists(transform), "the .mat must survive work-dir cleanup"
    assert "_work" not in transform
    assert metrics.parse_flirt_mat(transform) is not None


# --------------------------------------------------------------------------------------------- #
# 9. Resume
# --------------------------------------------------------------------------------------------- #
def test_validated_success_is_skipped_and_failure_is_retried(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (20,) * 3, (1.0,) * 3), ("T2w", (20,) * 3, (1.0,) * 3)]})
    config = fsl_config()
    process(rows, root, out, config=config)
    assert run_mod.is_session_done(str(out), "PT01/sub-01/ses-01", config) is True

    qc_path = out / "PT01/sub-01/ses-01" / "qc.json"
    payload = json.loads(qc_path.read_text())
    payload["status"] = "partial"
    qc_path.write_text(json.dumps(payload))
    assert run_mod.is_session_done(str(out), "PT01/sub-01/ses-01", config) is False


def test_missing_output_file_makes_a_session_not_done(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (18,) * 3, (1.0,) * 3)]})
    config = fsl_config()
    qc = process(rows, root, out, config=config)[0]
    os.remove(qc["modalities"][0]["output_path"])
    assert run_mod.is_session_done(str(out), "PT01/sub-01/ses-01", config) is False


# --------------------------------------------------------------------------------------------- #
# 10. Corpus definition and accounting
# --------------------------------------------------------------------------------------------- #
def test_canonical_manifest_sha_is_fail_closed(tmp_path):
    root = tmp_path / "cleaned"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (16,) * 3, (1.0,) * 3)]})
    path = write_manifest(rows, tmp_path / "canonical.tsv")
    real = canonical.sha256_of(str(path))
    assert canonical.read_canonical_manifest(str(path), real)
    with pytest.raises(canonical.CanonicalManifestError, match="SHA256 mismatch"):
        canonical.read_canonical_manifest(str(path), "0" * 64)


def test_manifest_missing_a_required_column_is_refused(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("sample_id\tdataset\nS000000\tPT01\n")
    with pytest.raises(canonical.CanonicalManifestError, match="missing required columns"):
        canonical.read_canonical_manifest(str(path))


def test_input_root_is_derived_from_the_dataset_component_not_a_common_ancestor(tmp_path):
    root = tmp_path / "cleaned"
    rows = make_corpus(root, {("PT01", "sub-01", "ses-01"): [("T1w", (16,) * 3, (1.0,) * 3)]})
    # A single-dataset chunk must still resolve the corpus root, not the dataset directory.
    assert canonical.infer_input_root(rows) == str(root)


def test_every_canonical_sample_gets_exactly_one_derivative_row(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {
            ("PT01", "sub-01", "ses-01"): [("T1w", (20, 18, 16), (1.0,) * 3), ("T2w", (18, 16, 14), (1.2,) * 3)],
            ("PT01", "sub-02", "ses-01"): [("T1w", (20, 18, 16), (1.0,) * 3)],
        },
    )
    process(rows, root, out)
    verdicts = {v.session_key: v for v in session_audit.audit_sessions(rows)}
    records = derivative_manifest.build_records(rows, verdicts, str(out))

    assert len(records) == len(rows)
    assert [r["sample_id"] for r in records] == [r["sample_id"] for r in rows]
    summary = derivative_manifest.assert_accounting(records, expected_samples=len(rows))
    assert summary["materialized"] == len(rows)
    assert summary["reference"] == 2 and summary["registered"] == 1
    assert summary["passthrough_single_scan"] == 0  # sub-02's only scan IS its session reference
    assert summary["missing"] == 0


def test_unprocessed_sessions_are_counted_as_missing_never_omitted(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {
            ("PT01", "sub-01", "ses-01"): [("T1w", (18,) * 3, (1.0,) * 3)],
            ("PT01", "sub-02", "ses-01"): [("T1w", (18,) * 3, (1.0,) * 3)],
        },
    )
    process([r for r in rows if r["participant_id"] == "sub-01"], root, out)
    records = derivative_manifest.build_records(rows, {}, str(out))
    summary = derivative_manifest.assert_accounting(records, expected_samples=len(rows))
    assert summary["total_rows"] == 2 and summary["missing"] == 1


def test_accounting_rejects_duplicates_and_a_changed_sample_count(tmp_path):
    base = {name: "" for name in derivative_manifest.DERIVATIVE_FIELDS}
    records = [dict(base, sample_id="S000000", status="reference"), dict(base, sample_id="S000000", status="reference")]
    with pytest.raises(derivative_manifest.AccountingError, match="duplicate sample_id"):
        derivative_manifest.assert_accounting(records)
    with pytest.raises(derivative_manifest.AccountingError, match="added or lost"):
        derivative_manifest.assert_accounting([dict(base, sample_id="S000000", status="reference")], expected_samples=2)
    with pytest.raises(derivative_manifest.AccountingError, match="unknown statuses"):
        derivative_manifest.assert_accounting([dict(base, sample_id="S000000", status="probably_fine")])


def test_stratified_smoke_is_seeded_capped_and_covers_the_required_scenarios(tmp_path):
    root = tmp_path / "cleaned"
    spec = {}
    for i in range(30):
        sub = f"sub-{i:03d}"
        if i % 5 == 0:
            spec[("PT01", sub, "ses-01")] = [("T1w", (16,) * 3, (1.0,) * 3)]
        elif i % 5 == 1:
            spec[("PT01", sub, "ses-01")] = [("T1w", (16,) * 3, (1.0,) * 3), ("T2w", (16,) * 3, (1.0,) * 3)]
        elif i % 5 == 2:
            spec[("PT02", sub, "ses-01")] = [("T1w", (16,) * 3, (1.0,) * 3), ("FLAIR", (16,) * 3, (1.0,) * 3)]
        elif i % 5 == 3:
            spec[("PT02", sub, "ses-01")] = [("T1w", (16,) * 3, (1.0,) * 3), ("ADC", (12, 12, 5), (1.5, 1.5, 5.0))]
        else:
            spec[("PT03", sub, "ses-01")] = [
                ("T1w", (16,) * 3, (1.0,) * 3),
                ("T2w", (16,) * 3, (1.0,) * 3),
                ("DWI_B1000", (12, 12, 5), (1.5, 1.5, 5.0)),
            ]
    rows = make_corpus(root, spec)
    sessions = canonical.sessions_from_canonical(rows)

    first = run_mod.canonical_smoke_keys(sessions, seed=7, max_sessions=20)
    assert first == run_mod.canonical_smoke_keys(sessions, seed=7, max_sessions=20), "must be reproducible"
    assert first != run_mod.canonical_smoke_keys(sessions, seed=8, max_sessions=20)
    assert len(first) <= 20

    picked = {s.key: s for s in sessions if s.key in set(first)}
    labels = {frozenset(str(x.get("modality_canonical")) for x in s.samples) for s in picked.values()}
    assert any(len(s.image_paths) == 1 for s in picked.values()), "single-scan"
    assert any({"T1w", "T2w"} <= label for label in labels), "T1+T2"
    assert any({"T1w", "FLAIR"} <= label for label in labels), "T1+FLAIR"
    assert any({"T1w", "ADC"} <= label for label in labels), "T1+ADC"
    assert any(len(label) >= 3 for label in labels), ">=3 modalities"
    assert len({s.dataset for s in picked.values()}) >= 3, "multiple datasets"


# --------------------------------------------------------------------------------------------- #
# 11. Opt-in: the real binary, when it happens to be installed
# --------------------------------------------------------------------------------------------- #
@pytest.mark.skipif(shutil.which("flirt") is None, reason="FSL not installed; the mocked suite is the CI contract")
def test_real_flirt_produces_a_rigid_transform_on_the_session_lattice(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (32, 30, 28), (1.0,) * 3), ("T2w", (28, 26, 14), (1.2, 1.2, 2.0))]},
        asymmetric=True,
    )
    qc = process(rows, root, out, runner=fsl.FSLRunner(interp="spline"))[0]
    assert qc["status"] == "success", qc.get("reason")
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    registration = moving["registration"]
    assert registration["world_space"] is True
    assert registration["determinant"] == pytest.approx(1.0, abs=1e-3)
    assert registration["orthogonality_error"] < 1e-4
    # The two phantoms are built about the same world centre, so a correct 6-DOF fit is small.
    # This is the assertion the radially symmetric phantom could never support: with rotation
    # unidentifiable, real FLIRT returned 97.6 deg here and the QC rejected it, correctly.
    assert registration["rotation_deg"] < 45.0
    assert registration["translation_norm_mm"] < 60.0


# --------------------------------------------------------------------------------------------- #
# 12. Atomic publication: a canonical output exists only if that sample passed QC
# --------------------------------------------------------------------------------------------- #
def _session_with_two_scans(root: Path):
    return make_corpus(
        root,
        {("PT01", "sub-01", "ses-01"): [("T1w", (32, 30, 28), (1.0,) * 3), ("T2w", (28, 26, 24), (1.2, 1.2, 1.4))]},
    )


def _canonical_outputs(out: Path):
    """Every NIfTI at a canonical derivative path (work dir and quarantine excluded)."""
    return sorted(
        p
        for p in out.rglob("*.nii.gz")
        if "_work" not in p.parts and "_quarantine" not in p.parts and "brainmask" not in p.name
    )


def test_a_transform_rejected_after_flirt_ran_publishes_no_canonical_output(tmp_path):
    """The exact smoke failure: FLIRT succeeds, produces an image, QC rejects the transform."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = FakeFSL(interp="spline")
    # A 150 mm / 120 deg rigid motion: exactly rigid, wildly implausible - a wrong basin.
    angle = np.deg2rad(120.0)
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    runner.world_transform = np.eye(4)
    runner.world_transform[:3, :3] = rot
    runner.world_transform[:3, 3] = [150.0, 0.0, 0.0]

    qc = process(rows, root, out, runner=runner)[0]

    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    reference = next(m for m in qc["modalities"] if m["is_reference"])
    assert moving["status"] == "failed"
    assert "implausible_rotation" in moving["reason"] or "implausible_translation" in moving["reason"]
    # The invariant.
    assert moving["output_path"] == "", "a rejected sample must not advertise a canonical output"
    assert moving.get("published") is False
    assert not os.path.exists(moving["unpublished_output_path"])
    # Evidence is kept.
    assert moving["transform_path"] and os.path.exists(moving["transform_path"])
    assert moving["registration"]["rotation_deg"] > 45.0
    # The reference, which passed, is published normally.
    assert reference["status"] == "ok" and os.path.exists(reference["output_path"])
    assert _canonical_outputs(out) == [Path(reference["output_path"])]


def test_an_empty_candidate_is_rejected_and_never_published(tmp_path):
    """A registration whose output is all zeros must not enter the derivative."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)

    class ZeroingFSL(FakeFSL):
        def apply_transform(self, mov, targ, out_path, lta=None, interp=fsl.FLIRT_INTERP):
            result = super().apply_transform(mov, targ, out_path, lta=lta, interp=interp)
            img = nib.load(out_path)  # blank the moving scan only
            if "T2w" in os.path.basename(mov):
                nib.save(nib.Nifti1Image(np.zeros(img.shape, dtype=np.float32), img.affine), out_path)
            return result

    qc = process(rows, root, out, runner=ZeroingFSL(interp="spline"))[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    assert moving["status"] == "failed"
    assert moving["output_path"] == ""
    assert not os.path.exists(moving["unpublished_output_path"])
    assert qc["status"] == "partial"


def test_a_backend_command_failure_leaves_the_other_modality_publishable(tmp_path):
    """One failed command must cost exactly one modality, not the session's valid outputs."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)

    class FailMovingFSL(FakeFSL):
        def coregister(self, mov, ref, lta):
            if "T2w" in os.path.basename(mov):
                raise fsl.FSLError("flirt exited 1 (simulated)")
            return super().coregister(mov, ref, lta)

    qc = process(rows, root, out, runner=FailMovingFSL(interp="spline"))[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    reference = next(m for m in qc["modalities"] if m["is_reference"])
    assert moving["status"] == "failed"
    assert moving["output_path"] == ""
    assert reference["status"] == "ok" and os.path.exists(reference["output_path"])
    assert _canonical_outputs(out) == [Path(reference["output_path"])]


def test_a_stale_invalid_output_from_an_earlier_attempt_is_cleared(tmp_path):
    """A re-run must not inherit a canonical image an earlier, failing attempt left behind."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)

    # Attempt 1: succeeds, so both scans are published.
    first = process(rows, root, out, runner=FakeFSL(interp="spline"))[0]
    moving_path = next(m["output_path"] for m in first["modalities"] if not m["is_reference"])
    assert os.path.exists(moving_path)
    stale_bytes = Path(moving_path).read_bytes()

    # Attempt 2 (overwrite): the same scan now fails QC. The stale image must not survive.
    runner = FakeFSL(interp="spline")
    runner.world_transform = np.eye(4)
    runner.world_transform[:3, 3] = [200.0, 0.0, 0.0]
    second = process(rows, root, out, runner=runner, config=fsl_config(overwrite=True))[0]

    moving = next(m for m in second["modalities"] if not m["is_reference"])
    assert moving["status"] == "failed"
    assert not os.path.exists(moving_path), "stale output from the previous attempt was left in place"
    quarantined = list((Path(second["session_output_dir"]) / "_quarantine").glob("*"))
    assert quarantined, "the stale image should be quarantined, not silently vanish"
    assert any(p.read_bytes() == stale_bytes for p in quarantined if p.suffix == ".stale" or ".stale" in p.name)


def test_quarantine_can_be_disabled_and_still_never_publishes(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = FakeFSL(interp="spline")
    runner.world_transform = np.eye(4)
    runner.world_transform[:3, 3] = [200.0, 0.0, 0.0]
    qc = process(rows, root, out, runner=runner, config=fsl_config(quarantine_failed_outputs=False))[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    assert moving["status"] == "failed" and moving["output_path"] == ""
    assert not (Path(qc["session_output_dir"]) / "_quarantine").exists()


def test_the_derivative_manifest_carries_no_path_for_a_failed_sample(tmp_path):
    """status=failed must keep transform + metrics but never a canonical NIfTI path."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = FakeFSL(interp="spline")
    runner.world_transform = np.eye(4)
    runner.world_transform[:3, 3] = [200.0, 0.0, 0.0]
    process(rows, root, out, runner=runner)

    records = derivative_manifest.build_records(rows, {}, str(out))
    failed = [r for r in records if r["status"] == "failed"]
    assert len(failed) == 1
    row = failed[0]
    assert row["output_nifti"] == ""
    assert row["transform_path"] and row["reason"]
    assert row["translation_norm_mm"] and float(row["translation_norm_mm"]) > 60.0
    assert derivative_manifest.accounting(records)["materialized"] == 1  # the reference only


# --------------------------------------------------------------------------------------------- #
# 13. Contract/reporting corrections
# --------------------------------------------------------------------------------------------- #
def test_an_unregistered_session_is_not_judged_against_the_shared_grid(tmp_path):
    """coreg_allowed=False means each scan keeps its own P0 grid; that is not a defect."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, coreg_allowed=False)[0]

    assert qc["coregistration_applied"] is False
    rejects = qc["output_qc"]["session_rejects"]
    assert not any("shape_mismatch" in r or "affine_mismatch" in r for r in rejects), rejects
    assert qc["status"] == "success"
    shapes = {tuple(m["output_shape"]) for m in qc["modalities"] if m["status"] == "ok"}
    assert len(shapes) > 1, "the scans really do differ in shape; the check was skipped, not satisfied"


def test_a_registered_session_is_still_judged_against_the_shared_grid(tmp_path):
    """The skip must be conditional: a co-registered session keeps the contract."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, coreg_allowed=True)[0]
    assert qc["coregistration_applied"] is True
    shapes = {tuple(m["output_shape"]) for m in qc["modalities"] if m["status"] == "ok"}
    assert len(shapes) == 1
    assert not qc["output_qc"]["session_rejects"]


def test_an_implausible_but_exactly_rigid_transform_is_flagged_as_such(tmp_path):
    """The flag must describe the evidence: these matrices are rigid, their magnitude is not."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = FakeFSL(interp="spline")
    runner.world_transform = np.eye(4)
    runner.world_transform[:3, 3] = [200.0, 0.0, 0.0]
    qc = process(rows, root, out, runner=runner)[0]

    flags = qc["output_qc"]["flags"]
    assert flags.get("implausible_rigid_transform") is True
    assert "non_rigid_transform" not in flags
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    reg = moving["registration"]
    assert reg["determinant"] == pytest.approx(1.0, abs=1e-6)
    assert reg["orthogonality_error"] < 1e-4


# --------------------------------------------------------------------------------------------- #
# 14. Angular search: an explicit, revertible optimiser constraint
# --------------------------------------------------------------------------------------------- #
def test_angular_search_emits_exactly_the_three_searchr_flags():
    cmd = fsl.flirt_estimate_command("mov.nii.gz", "ref.nii.gz", "t.mat", angular_search_deg=(-30, 30))
    text = " ".join(cmd)
    assert "-searchrx -30 30" in text
    assert "-searchry -30 30" in text
    assert "-searchrz -30 30" in text
    # and nothing else about the registration changed
    assert "-dof 6" in text and "-cost normmi" in text and "-usesqform" in text
    assert "-nosearch" not in text


def test_no_angular_search_leaves_the_command_exactly_as_before():
    assert fsl.flirt_estimate_command("m", "r", "t.mat") == fsl.flirt_estimate_command(
        "m", "r", "t.mat", angular_search_deg=None
    )
    assert not any(part.startswith("-searchr") for part in fsl.flirt_estimate_command("m", "r", "t.mat"))


def test_the_runner_passes_its_angular_search_through_to_flirt(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = FakeFSL(interp="spline", angular_search_deg=(-30, 30))
    process(rows, root, out, runner=runner)
    estimates = [c for c in runner.calls if "-omat" in c]
    assert estimates, "no estimation command was issued"
    for cmd in estimates:
        text = " ".join(cmd)
        assert "-searchrx -30 30" in text and "-searchry -30 30" in text and "-searchrz -30 30" in text


@pytest.mark.parametrize("bad", [(30, -30), (0, 0), (-200, 30), (-30, 200), ("a", "b"), (1,)])
def test_an_invalid_angular_search_fails_closed(bad):
    with pytest.raises(ValueError):
        fsl.FSLRunner(angular_search_deg=bad)


def test_the_angular_search_is_recorded_in_provenance(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, config=fsl_config(angular_search_deg=(-30, 30)))[0]
    assert qc["config"]["angular_search_deg"] == [-30, 30]
    baseline = process(rows, root, tmp_path / "p1b")[0]
    assert baseline["config"]["angular_search_deg"] is None


def test_a_quarantined_image_stays_loadable(tmp_path):
    """The rejected candidate is the evidence a reviewer opens; it must remain a readable NIfTI."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = FakeFSL(interp="spline")
    runner.world_transform = np.eye(4)
    runner.world_transform[:3, 3] = [200.0, 0.0, 0.0]
    qc = process(rows, root, out, runner=runner)[0]

    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    quarantined = moving["quarantined_path"]
    assert quarantined and os.path.exists(quarantined)
    assert quarantined.endswith(".nii.gz"), quarantined  # marker goes before the extension
    assert ".rejected" in os.path.basename(quarantined)
    nib.load(quarantined)  # would raise if the name made the format unparseable


def test_mark_basename_keeps_the_image_extension():
    assert pipeline._mark_basename("a_b.nii.gz", ".rejected") == "a_b.rejected.nii.gz"
    assert pipeline._mark_basename("a_b.nii", ".stale") == "a_b.stale.nii"
    assert pipeline._mark_basename("plain", ".rejected") == "plain.rejected"


# --------------------------------------------------------------------------------------------- #
# 15. The fallback ladder: one retry, only after a genuine QC failure
# --------------------------------------------------------------------------------------------- #
class LadderFSL(FakeFSL):
    """Applies a different world transform per attempt, keyed by the search flags in the command.

    ``constrained_transform`` is used when the command carries ``-searchrx`` (attempt 1);
    ``unconstrained_transform`` when it does not (attempt 2).
    """

    def __post_init__(self):
        super().__post_init__()
        self.constrained_transform = np.eye(4)
        self.unconstrained_transform = np.eye(4)
        self.estimates = []

    def run(self, cmd, expect_output=None):
        cmd = [str(p) for p in cmd]
        if "-omat" in cmd:
            constrained = any(p.startswith("-searchr") for p in cmd)
            self.estimates.append("constrained" if constrained else "unconstrained")
            self.world_transform = self.constrained_transform if constrained else self.unconstrained_transform
        return super().run(cmd, expect_output=expect_output)


def _bad_transform(mm=200.0):
    t = np.eye(4)
    t[:3, 3] = [mm, 0.0, 0.0]
    return t


def _fallback_config(**kw):
    return fsl_config(angular_search_deg=(-30, 30), fallback_unconstrained_retry=True, **kw)


def test_a_passing_first_attempt_is_final_and_never_retried(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
    runner.constrained_transform = np.eye(4)
    runner.unconstrained_transform = _bad_transform()  # would fail, must never be reached

    qc = process(rows, root, out, runner=runner, config=_fallback_config())[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])

    assert runner.estimates == ["constrained"], runner.estimates
    assert len(moving["registration_attempts"]) == 1
    assert moving["selected_attempt"] == 1
    assert moving["retried"] is False
    assert moving["status"] == "ok"
    assert os.path.exists(moving["output_path"])


def test_a_failing_first_attempt_is_recovered_by_the_unconstrained_retry(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
    runner.constrained_transform = _bad_transform()  # rejected by rigid QC
    runner.unconstrained_transform = np.eye(4)  # passes

    qc = process(rows, root, out, runner=runner, config=_fallback_config())[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])

    assert runner.estimates == ["constrained", "unconstrained"]
    attempts = moving["registration_attempts"]
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert attempts[0]["qc_outcome"] == "fail" and "implausible_translation" in attempts[0]["reason"]
    assert attempts[1]["qc_outcome"] == "pass"
    assert attempts[0]["angular_search_deg"] == [-30, 30]
    assert attempts[1]["angular_search_deg"] is None
    assert moving["selected_attempt"] == 2
    assert moving["retried"] is True
    assert moving["status"] == "ok"
    assert os.path.exists(moving["output_path"])
    # The accepted metrics are attempt 2's, not attempt 1's.
    assert moving["registration"]["translation_norm_mm"] < 60.0
    assert qc["status"] == "success"


def test_when_both_attempts_fail_nothing_is_published(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
    runner.constrained_transform = _bad_transform(200.0)
    runner.unconstrained_transform = _bad_transform(250.0)

    qc = process(rows, root, out, runner=runner, config=_fallback_config())[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])

    assert runner.estimates == ["constrained", "unconstrained"]
    assert moving["selected_attempt"] is None
    assert moving["status"] == "failed"
    assert moving["output_path"] == ""
    assert not os.path.exists(moving["unpublished_output_path"])
    assert len(moving["registration_attempts"]) == 2
    assert all(a["qc_outcome"] == "fail" for a in moving["registration_attempts"])
    assert _canonical_outputs(out) == [Path(next(m["output_path"] for m in qc["modalities"] if m["is_reference"]))]


def test_the_failed_first_candidate_is_never_published_when_the_retry_wins(tmp_path):
    """The retry must publish its own image, not attempt 1's rejected one."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
    runner.constrained_transform = _bad_transform()
    runner.unconstrained_transform = np.eye(4)

    qc = process(rows, root, out, runner=runner, config=_fallback_config())[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    published = nib.load(moving["output_path"])
    attempt1_img = moving["registration_attempts"][0]["image_path"]
    attempt2_img = moving["registration_attempts"][1]["image_path"]
    assert attempt1_img != attempt2_img, "attempts must not share a path"
    # The published image is the retry's: it retains foreground, the rejected one did not.
    assert float(np.count_nonzero(np.asanyarray(published.dataobj))) > 0
    assert moving["foreground_retained_frac"] > 0.5


def test_a_retry_never_overwrites_a_valid_first_attempt_output(tmp_path):
    """Attempt paths are distinct, so a second attempt cannot clobber a good first one."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
    runner.constrained_transform = np.eye(4)
    runner.unconstrained_transform = _bad_transform()
    qc = process(rows, root, out, runner=runner, config=_fallback_config())[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    assert moving["selected_attempt"] == 1
    assert len(runner.estimates) == 1
    assert (
        nib.load(moving["output_path"]).shape
        == nib.load(next(m["output_path"] for m in qc["modalities"] if m["is_reference"])).shape
    )


def test_provenance_records_both_attempts_in_full(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
    runner.constrained_transform = _bad_transform()
    runner.unconstrained_transform = np.eye(4)
    qc = process(rows, root, out, runner=runner, config=_fallback_config())[0]

    stored = json.load(open(Path(qc["session_output_dir"]) / "qc.json"))
    moving = next(m for m in stored["modalities"] if not m["is_reference"])
    required = {
        "attempt",
        "angular_search_deg",
        "transform_path",
        "image_path",
        "command",
        "translation_norm_mm",
        "rotation_deg",
        "determinant",
        "orthogonality_error",
        "nmi_before",
        "nmi_after",
        "nmi_improvement",
        "foreground_retained_frac",
        "qc_outcome",
        "reason",
    }
    for attempt in moving["registration_attempts"]:
        missing = required - set(attempt)
        assert not missing, f"attempt {attempt['attempt']} missing {sorted(missing)}"
        assert os.path.exists(attempt["transform_path"])
    assert moving["selected_attempt"] == 2
    assert stored["config"]["fallback_unconstrained_retry"] is True
    assert stored["config"]["angular_search_deg"] == [-30, 30]


def test_the_ladder_is_inert_without_the_flag(tmp_path):
    """A constrained run without the retry flag behaves exactly as before: one attempt, no retry."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
    runner.constrained_transform = _bad_transform()
    runner.unconstrained_transform = np.eye(4)
    qc = process(rows, root, out, runner=runner, config=fsl_config(angular_search_deg=(-30, 30)))[0]
    moving = next(m for m in qc["modalities"] if not m["is_reference"])
    assert runner.estimates == ["constrained"]
    assert len(moving["registration_attempts"]) == 1
    assert moving["status"] == "failed"
    assert moving["output_path"] == ""


def test_the_ladder_needs_a_constrained_first_attempt_to_have_anything_to_fall_back_from():
    """Without angular_search_deg the two attempts would be identical, so there is no ladder."""
    forward = pipeline.FORWARD
    cfg = fsl_config(fallback_unconstrained_retry=True)  # no angular_search_deg
    assert pipeline._attempt_ladder(cfg) == [{"type": forward, "search": pipeline._RUNNER_DEFAULT}]
    cfg2 = fsl_config(angular_search_deg=(-30, 30), fallback_unconstrained_retry=True)
    assert pipeline._attempt_ladder(cfg2) == [
        {"type": forward, "search": pipeline._RUNNER_DEFAULT},
        {"type": forward, "search": None},
    ]
    # and with no retry configured at all, still exactly one attempt
    assert pipeline._attempt_ladder(fsl_config(angular_search_deg=(-30, 30))) == [
        {"type": forward, "search": pipeline._RUNNER_DEFAULT}
    ]


@pytest.mark.parametrize("first_ok", [True, False])
def test_resume_is_safe_after_either_ladder_outcome(tmp_path, first_ok):
    """Re-running a session must reach the same terminal state and leave no invalid output."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)

    def run_once():
        runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
        runner.constrained_transform = np.eye(4) if first_ok else _bad_transform()
        runner.unconstrained_transform = np.eye(4) if first_ok else _bad_transform(250.0)
        return process(rows, root, out, runner=runner, config=_fallback_config(overwrite=True))[0]

    first = run_once()
    second = run_once()
    for qc in (first, second):
        moving = next(m for m in qc["modalities"] if not m["is_reference"])
        if first_ok:
            assert moving["status"] == "ok" and os.path.exists(moving["output_path"])
        else:
            assert moving["status"] == "failed" and moving["output_path"] == ""
            assert not os.path.exists(moving["unpublished_output_path"])
    assert first["status"] == second["status"]


def test_a_larger_stratified_request_deepens_strata_and_reaches_its_target():
    """The pilot must be a scaled-up smoke, and must actually reach the requested size."""
    sessions = []
    for i in range(400):
        mods = [("T1w", 1.0), ("T2w", 1.0)] if i % 2 else [("T1w", 1.0), ("FLAIR", 4.0), ("ADC", 1.0)]
        sessions.append(
            run_mod.Session(
                key=f"DS{i % 25:02d}/sub-{i:03d}/ses-01",
                session_dir=f"/x/DS{i % 25:02d}/sub-{i:03d}/ses-01",
                dataset=f"DS{i % 25:02d}",
                subject=f"sub-{i:03d}",
                session="ses-01",
                image_paths=[f"/x/{i}_{m}.nii.gz" for m, _ in mods],
                samples=[{"modality_canonical": m, "pixdim": f"1x1x{z:g}"} for m, z in mods],
            )
        )
    small = run_mod.canonical_smoke_keys(sessions, seed=0, max_sessions=40)
    big = run_mod.canonical_smoke_keys(sessions, seed=0, max_sessions=300)
    assert len(small) == 40
    assert len(big) == 300
    assert len(set(big)) == 300, "no duplicates"
    # deterministic
    assert big == run_mod.canonical_smoke_keys(sessions, seed=0, max_sessions=300)
    assert len({k.split("/")[0] for k in big}) == 25, "all datasets represented"


def test_a_changed_registration_policy_invalidates_resume(tmp_path):
    """Resuming under a different policy must reprocess, not inherit the old run's outputs.

    ``is_session_done`` used to look only at the files on disk, so resuming a run with a different
    angular search (or dof, cost, interpolation, backend) silently kept sessions produced under
    the previous policy -- one derivative, two registration policies, no record of the split.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = FakeFSL(interp="spline")
    qc = process(rows, root, out, runner=runner, config=_fallback_config())[0]
    assert qc["status"] == "success"
    key = qc["session_key"]

    # Same policy -> reuse.
    assert run_mod.is_session_done(str(out), key, _fallback_config()) is True

    # Each registration-defining field, changed alone, must invalidate reuse.
    for changed in (
        fsl_config(angular_search_deg=None, fallback_unconstrained_retry=True),
        fsl_config(angular_search_deg=(-30, 30), fallback_unconstrained_retry=False),
        fsl_config(angular_search_deg=(-90, 90), fallback_unconstrained_retry=True),
        fsl_config(angular_search_deg=(-30, 30), fallback_unconstrained_retry=True, cost="corratio"),
        fsl_config(angular_search_deg=(-30, 30), fallback_unconstrained_retry=True, image_interp="trilinear"),
    ):
        assert run_mod.is_session_done(str(out), key, changed) is False, changed


def test_an_unregistered_session_stays_done_across_resumes(tmp_path):
    """A passthrough session keeps per-scan grids; that must not read as 'never finished'.

    Only a *registered* session is contracted to share one lattice. Requiring one shape for
    audit-blocked and single-scan sessions marked them unfinished forever, so every resume
    reprocessed them.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)  # two scans, different shapes and spacings
    # Blocked by the identity audit, exactly as an ambiguous session is.
    qc = process(rows, root, out, runner=FakeFSL(interp="spline"), config=_fallback_config(), coreg_allowed=False)[0]

    assert qc["coregistration_applied"] is False
    assert qc["status"] == "success"
    shapes = {tuple(nib.load(m["output_path"]).shape) for m in qc["modalities"]}
    assert len(shapes) > 1, "this test is only meaningful when the grids genuinely differ"
    assert run_mod.is_session_done(str(out), qc["session_key"], _fallback_config()) is True


def test_a_snapshot_predating_a_policy_field_invalidates_resume(tmp_path):
    """An older run must not be silently absorbed just because it lacks the newer fields.

    Real incident: a production output root held 40 sessions from a revision whose snapshot had
    no ``angular_search_deg`` and no ``fallback_unconstrained_retry``. Skipping absent fields made
    32 of them count as "done", so a production run would have inherited scans registered with no
    bounded search and no fallback -- two registration policies in one derivative, unrecorded.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, runner=FakeFSL(interp="spline"), config=_fallback_config())[0]
    key, qc_path = qc["session_key"], out / qc["session_key"] / "qc.json"
    assert run_mod.is_session_done(str(out), key, _fallback_config()) is True

    stored = json.loads(qc_path.read_text())
    for field in ("angular_search_deg", "fallback_unconstrained_retry"):
        older = json.loads(json.dumps(stored))
        older["config"].pop(field)
        qc_path.write_text(json.dumps(older))
        assert run_mod.is_session_done(str(out), key, _fallback_config()) is False, field

    # A snapshot with no config at all is the oldest case of the same thing.
    oldest = json.loads(json.dumps(stored))
    oldest["config"] = {}
    qc_path.write_text(json.dumps(oldest))
    assert run_mod.is_session_done(str(out), key, _fallback_config()) is False


# --------------------------------------------------------------------------------------------- #
# 17. Attempt 3: the inverse-direction rescue
#
# The paired reference-policy gate showed the two estimation directions are not equally
# conditioned. Registering a thick slab *onto* a full-coverage volume let the optimiser slide it
# along its own thick axis (8 definitive failures); fomo50k_legacy avoided that only by electing
# the slab as the session reference, which costs ~half the field of view. Attempt 3 estimates in
# the better-conditioned direction and inverts, so the output lattice never changes.
#
# What these tests must protect, in order of importance:
#   - the existing success path is untouched (A, B, N, O);
#   - the rescue cannot publish something the existing QC would reject (D, M);
#   - the inversion means what the pipeline claims it means (E, F);
#   - the output is still one resampling of the original moving scan onto the P0 grid (G, H, I).
# --------------------------------------------------------------------------------------------- #
class RescueFSL(FakeFSL):
    """Keyed on the attempt, so each rung of the ladder can be given its own outcome.

    Attempts are told apart by the matrix path the pipeline chose (``_attempt1``, ``_attempt2``,
    ``_attempt3_reverse``) rather than by the search flags, because attempt 3 deliberately carries
    the *same* bound as attempt 1 -- direction is the only variable that changes.

    ``reverse_transform`` is the anatomical motion the reverse fit "finds", expressed as usual in
    scanner millimetres and in the estimation's own direction (reference -> moving). The pipeline
    then inverts it, so a rescue that should succeed is configured with the *inverse* of the
    motion that would make the moving scan land correctly.
    """

    def __post_init__(self):
        super().__post_init__()
        self.attempt1_transform = np.eye(4)
        self.attempt2_transform = np.eye(4)
        self.reverse_transform = np.eye(4)
        self.estimates = []  # (attempt, direction) in the order the optimiser was invoked
        self.applies = []  # every command that wrote an image

    def run(self, cmd, expect_output=None):
        cmd = [str(p) for p in cmd]
        if cmd and cmd[0] == "flirt" and "-omat" in cmd:
            omat = cmd[cmd.index("-omat") + 1]
            if "_attempt3_reverse" in omat:
                self.world_transform = self.reverse_transform
                self.estimates.append((3, "reference_to_moving"))
            elif "_attempt2" in omat:
                self.world_transform = self.attempt2_transform
                self.estimates.append((2, "moving_to_reference"))
            else:
                self.world_transform = self.attempt1_transform
                self.estimates.append((1, "moving_to_reference"))
        if cmd and cmd[0] == "flirt" and "-out" in cmd and "-applyxfm" in cmd:
            self.applies.append(cmd)
        return super().run(cmd, expect_output=expect_output)


def _rotation(theta: float, axis: int) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    rot = np.eye(3)
    rot[i, i] = c
    rot[i, j] = -s
    rot[j, i] = s
    rot[j, j] = c
    return rot


def _image_with_affine(shape, spacing, neurological: bool) -> "nib.Nifti1Image":
    """An image whose affine has the given handedness -- FSL flips the first axis for one only."""
    affine = np.diag([*spacing, 1.0]).astype(float)
    if not neurological:
        affine[0, 0] = -affine[0, 0]
    affine[:3, :3] = _rotation(np.deg2rad(4.0), axis=0) @ affine[:3, :3]  # slightly oblique, as real data is
    affine[:3, 3] = [-90.0, -110.0, -70.0]
    return nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)


def _rescue_config(**kw):
    return fsl_config(angular_search_deg=(-30, 30), fallback_unconstrained_retry=True, inverse_direction_rescue=True, **kw)


def _rescue_runner():
    return RescueFSL(interp="spline", angular_search_deg=(-30, 30))


def _moving_of(qc):
    return next(m for m in qc["modalities"] if not m["is_reference"])


# --- A/B/C/D: the ladder runs exactly as far as it must, and no further ----------------------- #
def test_a_passing_first_attempt_runs_neither_the_retry_nor_the_rescue(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt2_transform = _bad_transform()  # must never be reached
    runner.reverse_transform = _bad_transform()

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    moving = _moving_of(qc)

    assert runner.estimates == [(1, "moving_to_reference")]
    assert [a["attempt"] for a in moving["registration_attempts"]] == [1]
    assert moving["selected_attempt"] == 1
    assert moving["status"] == "ok"
    assert os.path.exists(moving["output_path"])
    assert not any(c[0] == "convert_xfm" for c in runner.calls), "no inversion may run when attempt 1 passes"


def test_a_recovered_second_attempt_still_never_reaches_the_rescue(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = np.eye(4)  # the existing retry rescues it
    runner.reverse_transform = _bad_transform()  # must never be reached

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    moving = _moving_of(qc)

    assert runner.estimates == [(1, "moving_to_reference"), (2, "moving_to_reference")]
    assert [a["attempt"] for a in moving["registration_attempts"]] == [1, 2]
    assert moving["selected_attempt"] == 2
    assert moving["status"] == "ok"
    assert not any(c[0] == "convert_xfm" for c in runner.calls)


def test_the_rescue_runs_only_after_both_forward_attempts_fail_and_is_published(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = np.eye(4)  # inverts to an identity: a clean fit

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    moving = _moving_of(qc)
    attempts = moving["registration_attempts"]

    assert runner.estimates == [(1, "moving_to_reference"), (2, "moving_to_reference"), (3, "reference_to_moving")]
    assert [a["attempt"] for a in attempts] == [1, 2, 3]
    assert [a["attempt_type"] for a in attempts] == ["forward", "forward", "inverse_direction_rescue"]
    assert [a["qc_outcome"] for a in attempts] == ["fail", "fail", "pass"]
    assert moving["selected_attempt"] == 3
    assert moving["status"] == "ok"
    assert qc["status"] == "success"
    assert os.path.exists(moving["output_path"])
    # Attempt 3 keeps attempt 1's bound: direction is the only variable that changed.
    assert attempts[0]["angular_search_deg"] == [-30, 30]
    assert attempts[1]["angular_search_deg"] is None
    assert attempts[2]["angular_search_deg"] == [-30, 30]
    # The published metrics are the rescue's.
    assert moving["registration"]["translation_norm_mm"] < 60.0


def test_when_all_three_attempts_fail_nothing_reaches_a_canonical_path(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = _bad_transform(mm=300.0)

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    moving = _moving_of(qc)

    assert len(moving["registration_attempts"]) == 3
    assert all(a["qc_outcome"] == "fail" for a in moving["registration_attempts"])
    assert moving["selected_attempt"] is None
    assert moving["status"] != "ok"
    assert qc["status"] != "success"
    assert not os.path.exists(moving["output_path"]), "a definitive failure must not be published"


# --- E/F: the inversion is real, and means what the pipeline says it means -------------------- #
def test_the_rescue_publishes_the_inverse_of_the_matrix_it_estimated(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    rot = np.eye(4)
    rot[:3, :3] = _rotation(np.deg2rad(7.0), axis=1)
    rot[:3, 3] = [3.0, -2.0, 1.5]
    runner.reverse_transform = rot

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    rescue = _moving_of(qc)["registration_attempts"][2]

    forward = metrics.parse_flirt_mat(rescue["forward_matrix_path"])
    inverted = metrics.parse_flirt_mat(rescue["transform_path"])
    assert forward is not None and inverted is not None
    assert np.allclose(forward @ inverted, np.eye(4), atol=1e-6), "the applied matrix is not the inverse"
    # Both survive the work dir being deleted: the fit cannot be re-checked without them.
    assert os.path.exists(rescue["forward_matrix_path"]) and os.path.exists(rescue["transform_path"])
    assert rescue["forward_matrix_path"] != rescue["transform_path"]


def test_inverting_the_fsl_matrix_is_exactly_inverting_the_world_transform(tmp_path):
    """The mathematical claim the rescue rests on, checked against the repo's own frame algebra.

    A FLIRT matrix lives in the ``-in``/``-ref`` pair's scaled-mm spaces, so a matrix fitted with
    (in=grid, ref=moving) and then inverted is decoded with (moving=moving, reference=grid) --
    the pair swapped, exactly as ``convert_xfm -inverse`` documents. If that were wrong, no
    composition would rescue it and the published transform would be silently in the wrong frame.

    Both handedness combinations are exercised because FSL flips the first axis for one and not
    the other, which is precisely where a frame error would hide.
    """
    for flip_grid, flip_moving in ((False, False), (True, False), (False, True), (True, True)):
        grid = _image_with_affine((40, 38, 36), (1.0, 1.0, 1.0), flip_grid)
        moving = _image_with_affine((44, 42, 8), (0.9, 0.9, 6.0), flip_moving)
        known = np.eye(4)  # the anatomical motion moving -> grid we want back at the end
        known[:3, :3] = _rotation(np.deg2rad(11.0), axis=2)
        known[:3, 3] = [4.0, -6.0, 2.0]

        # What the reverse fit produces: an FSL matrix for (in=grid, ref=moving), i.e. grid->moving.
        reverse_fsl = metrics.world_to_fsl(moving) @ np.linalg.inv(known) @ np.linalg.inv(metrics.world_to_fsl(grid))
        recovered = metrics.flirt_world_transform(np.linalg.inv(reverse_fsl), moving=moving, reference=grid)

        assert np.allclose(recovered, known, atol=1e-9), f"frame error at flip={flip_grid},{flip_moving}"
        # And a physical point: the moving scan's world coordinates land where they should.
        point = np.array([12.0, -25.0, 7.0, 1.0])
        assert np.allclose(recovered @ point, known @ point, atol=1e-9)


# --- G/H/I: the output is still one resampling of the original scan onto the P0 grid ---------- #
def test_the_rescue_resamples_the_original_moving_scan_onto_the_exact_p0_grid(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = np.eye(4)

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    moving = _moving_of(qc)
    rescue = moving["registration_attempts"][2]

    apply_cmd = rescue["apply_command"]
    estimate_cmd = rescue["estimate_command"]
    # The estimation swapped the roles ...
    assert estimate_cmd[estimate_cmd.index("-in") + 1] == rescue["estimation_in"]
    assert estimate_cmd[estimate_cmd.index("-ref") + 1] == rescue["estimation_ref"]
    assert rescue["estimation_direction"] == "reference_to_moving"
    # ... and the application did not: original moving in, P0 target grid out.
    assert apply_cmd[apply_cmd.index("-in") + 1] == rescue["apply_in"] == estimate_cmd[estimate_cmd.index("-ref") + 1]
    assert apply_cmd[apply_cmd.index("-ref") + 1] == rescue["apply_ref"] == estimate_cmd[estimate_cmd.index("-in") + 1]
    # The applied matrix is the inverted one; ``transform_path`` is where it was kept afterwards,
    # since the work dir is deleted on success.
    assert apply_cmd[apply_cmd.index("-init") + 1].endswith("_attempt3.mat")
    assert os.path.exists(rescue["transform_path"])
    assert "-usesqform" not in apply_cmd, "-init and -usesqform are mutually exclusive in FLIRT"
    # The moving scan is never the lattice: the -ref of the application is the session's grid.
    assert apply_cmd[apply_cmd.index("-ref") + 1] != moving["source_path"]


def test_the_rescue_performs_exactly_one_final_resampling(tmp_path):
    """Three attempts, three candidate images -- but only one of them is ever published.

    The rescue must not add a second interpolation of its own (estimate on one grid, resample
    onto another): the published image is a single spline pass from the original NIfTI.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = np.eye(4)
    # No NMI-before resample to count, and the work dir kept so the applied input can be inspected.
    config = _rescue_config(compute_registration_metrics=False, clean_work_on_success=False)

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=config)[0]
    moving = _moving_of(qc)
    rescue = moving["registration_attempts"][2]

    applied = rescue["apply_command"][rescue["apply_command"].index("-init") + 1]
    for_rescue = [c for c in runner.applies if c[c.index("-init") + 1] == applied]
    assert len(for_rescue) == 1, f"the rescue resampled {len(for_rescue)} times"
    # ``-in`` is the RAS passthrough, which this backend materialises as a hard link, so it is the
    # original NIfTI byte for byte -- never an intermediate that would add an interpolation.
    applied_in = for_rescue[0][for_rescue[0].index("-in") + 1]
    assert open(applied_in, "rb").read() == open(moving["source_path"], "rb").read()
    # convert_xfm touches matrices only; it can never be an interpolation.
    inversions = [c for c in runner.calls if c[0] == "convert_xfm"]
    assert len(inversions) == 1 and "-out" not in inversions[0]


def test_a_rescued_scan_shares_the_exact_session_lattice(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = np.eye(4)

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    assert qc["status"] == "success"
    images = [nib.load(m["output_path"]) for m in qc["modalities"]]
    assert len({img.shape for img in images}) == 1
    for img in images[1:]:
        assert np.allclose(img.affine, images[0].affine, atol=1e-4)
    reference = next(m for m in qc["modalities"] if m["is_reference"])
    assert nib.load(reference["output_path"]).shape == images[0].shape


# --- J: provenance ---------------------------------------------------------------------------- #
def test_rescue_provenance_records_both_transforms_and_every_command(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = np.eye(4)

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    attempts = _moving_of(qc)["registration_attempts"]
    required = {
        "attempt",
        "attempt_type",
        "angular_search_deg",
        "estimation_direction",
        "estimation_in",
        "estimation_ref",
        "forward_matrix_path",
        "estimate_command",
        "inversion_command",
        "apply_command",
        "apply_in",
        "apply_ref",
        "transform_path",
        "image_path",
        "qc_outcome",
        "qc_rejects",
        "reason",
        "translation_norm_mm",
        "rotation_deg",
        "determinant",
        "orthogonality_error",
        "nmi_before",
        "nmi_after",
        "nmi_improvement",
        "foreground_retained_frac",
    }
    for attempt in attempts:
        assert not required - set(attempt), f"attempt {attempt['attempt']} missing {sorted(required - set(attempt))}"

    forward, rescue = attempts[:2], attempts[2]
    # Only the rescue inverts anything, and its inversion command is the documented FSL one.
    assert all(a["inversion_command"] is None and a["forward_matrix_path"] == "" for a in forward)
    inversion = rescue["inversion_command"]
    assert inversion[0] == "convert_xfm" and "-inverse" in inversion
    # It inverts the matrix the reverse fit produced, into the one the application then uses.
    assert inversion[inversion.index("-inverse") + 1].endswith("_attempt3_reverse.mat")
    assert inversion[inversion.index("-omat") + 1] == rescue["apply_command"][rescue["apply_command"].index("-init") + 1]
    assert all(a["estimation_direction"] == "moving_to_reference" for a in forward)
    # Earlier attempts keep their own record: a rescue never rewrites the evidence it followed.
    assert forward[0]["translation_norm_mm"] > 60.0
    assert _moving_of(qc)["registration"]["translation_norm_mm"] < 60.0
    stored = json.loads((out / qc["session_key"] / "qc.json").read_text())
    assert stored["config"]["inverse_direction_rescue"] is True


# --- K/L: the resume contract ------------------------------------------------------------------ #
def test_changing_the_rescue_policy_invalidates_resume(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    process(rows, root, out, runner=_rescue_runner(), config=_rescue_config())[0]
    key = canonical.sessions_from_canonical(rows)[0].key

    assert run_mod.is_session_done(str(out), key, _rescue_config()) is True
    # The same corpus asked for under the forward-only policy is a different derivative.
    assert run_mod.is_session_done(str(out), key, _fallback_config()) is False


def test_a_snapshot_predating_the_rescue_field_invalidates_resume(tmp_path):
    """Sessions produced before this feature existed must never count as done under it.

    Same failure mode as the 40-session incident that motivated this check: an older snapshot has
    fewer fields, and reading that silence as agreement mixes two registration policies in one
    derivative with nothing recording the split.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, runner=_rescue_runner(), config=_rescue_config())[0]
    qc_path = out / qc["session_key"] / "qc.json"

    stored = json.loads(qc_path.read_text())
    older = json.loads(json.dumps(stored))
    older["config"].pop("inverse_direction_rescue")
    qc_path.write_text(json.dumps(older))
    assert run_mod.is_session_done(str(out), qc["session_key"], _rescue_config()) is False
    assert "inverse_direction_rescue" in run_mod.RESUME_INVALIDATING_FIELDS


# --- M: a failed rescue candidate cannot survive at a canonical path -------------------------- #
def test_a_failed_rescue_candidate_is_quarantined_not_published(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = _bad_transform(mm=300.0)

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    moving = _moving_of(qc)

    assert not os.path.exists(moving["output_path"])
    quarantine = out / qc["session_key"] / "_quarantine"
    assert quarantine.exists() and any(quarantine.iterdir()), "the evidence must survive for review"
    assert all(str(p).startswith(str(out)) for p in quarantine.iterdir())


def test_an_inversion_that_is_not_an_inverse_is_refused(tmp_path):
    """The whole rescue rests on the inverted matrix meaning "moving -> grid".

    A silently wrong matrix would still produce a plausible-looking image, so the tool's output is
    checked against a numerical inverse rather than trusted.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = np.eye(4)
    runner.inverted_matrix = np.eye(4)  # convert_xfm "returns" something that is not the inverse

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_rescue_config())[0]
    moving = _moving_of(qc)

    rescue = moving["registration_attempts"][2]
    assert rescue["qc_outcome"] == "fail"
    assert "convert_xfm" in rescue["reason"] and "inverse" in rescue["reason"]
    assert moving["selected_attempt"] is None
    assert not os.path.exists(moving["output_path"])


def test_the_rescue_is_refused_on_a_backend_without_an_inversion_contract():
    with pytest.raises(ValueError, match="FSL backend only"):
        config_mod.CoregConfig(backend="freesurfer", inverse_direction_rescue=True)
    with pytest.raises(ValueError, match="do_coregister"):
        fsl_config(inverse_direction_rescue=True, do_coregister=False)
    # And a runner that does not advertise the contract fails closed rather than silently
    # degrading to the forward-only ladder.
    assert fsl.FSLRunner.supports_inverse_direction is True
    assert getattr(FreeSurferRunner, "supports_inverse_direction", False) is False


def test_convert_xfm_is_reported_separately_from_the_always_required_tools():
    assert fsl.REQUIRED_TOOLS == ("flirt",)
    assert fsl.INVERSE_RESCUE_TOOLS == ("convert_xfm",)
    report = fsl.check_tools()
    assert set(report["inverse_rescue_tools"]) == {"convert_xfm"}
    assert "inverse_rescue_ok" in report


# --- N/O: nothing that already worked has moved ------------------------------------------------ #
def test_the_forward_ladder_is_unchanged_when_the_rescue_is_off(tmp_path):
    """The default and the two-attempt policy must issue exactly the commands they did before."""
    assert [s["type"] for s in pipeline._attempt_ladder(fsl_config())] == ["forward"]
    assert [s["type"] for s in pipeline._attempt_ladder(_fallback_config())] == ["forward", "forward"]
    assert [s["type"] for s in pipeline._attempt_ladder(_rescue_config())] == [
        "forward",
        "forward",
        "inverse_direction_rescue",
    ]
    # A rescue configured without a bounded first attempt still has both forward rungs collapse to
    # one, exactly as before: the retry needs something to fall back *from*.
    assert [s["type"] for s in pipeline._attempt_ladder(fsl_config(inverse_direction_rescue=True))] == [
        "forward",
        "inverse_direction_rescue",
    ]

    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = LadderFSL(interp="spline", angular_search_deg=(-30, 30))
    runner.constrained_transform = _bad_transform()
    runner.unconstrained_transform = np.eye(4)
    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_fallback_config())[0]
    moving = _moving_of(qc)
    assert [a["attempt_type"] for a in moving["registration_attempts"]] == ["forward", "forward"]
    assert all(a["forward_matrix_path"] == "" for a in moving["registration_attempts"])
    assert not any(c[0] == "convert_xfm" for c in runner.calls)


def test_a_rescued_session_still_satisfies_the_p0_geometry_contract(tmp_path):
    """The reference and every passthrough stay bit-identical to P0 even when a scan was rescued."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _rescue_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    runner.reverse_transform = np.eye(4)

    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, runner=runner, config=_rescue_config())[0]
    reference = next(m for m in qc["modalities"] if m["is_reference"])

    expected = tmp_path / "p0_reference.nii.gz"
    p0_kernel.write_p0_resampled(reference["source_path"], str(expected), 1.0)
    got, want = nib.load(reference["output_path"]), nib.load(str(expected))
    assert got.shape == want.shape
    assert np.allclose(got.affine, want.affine, atol=1e-6)
    assert np.array_equal(np.asanyarray(got.dataobj), np.asanyarray(want.dataobj))
    for image in (got,):
        assert nib.aff2axcodes(image.affine) == ("R", "A", "S")
        assert np.allclose(nib.affines.voxel_sizes(image.affine)[:3], 1.0, atol=0.05)


# --------------------------------------------------------------------------------------------- #
# 18. The header-only fallback
#
# Measured on the 8 scans the optimiser definitively lost: an independent comparator
# (fomo50k_legacy fitted the same pair with the slab as ITS reference) puts the true relationship
# 0.30-1.35 mm and 0.16-2.56 deg from the scanner header, and tri-planar QC confirms the header
# places ventricles, corpus callosum, brainstem and skull inside the reference contours in all 8.
# FSL's own translation-only schedule and simple3D.sch drove the brain out of the field of view
# (165-394 mm, retention 0.0-0.28), so neither is usable.
#
# It is not a registration, and the tests below exist mostly to make sure it can never be mistaken
# for one, or reached while an optimised attempt is still viable.
# --------------------------------------------------------------------------------------------- #
class HeaderFallbackFSL(RescueFSL):
    """Adds the header-alignment command on top of the attempt-keyed fake.

    ``-applyxfm -usesqform`` writes the transform the *headers* imply, which for two images in one
    scanner frame is the world-space identity -- verified against real FSL 6.0.4, where this
    command decomposed to 0.00 mm / 0.00 deg. The attempt-keyed transforms must not leak into it,
    so it is intercepted before ``RescueFSL`` picks one.
    """

    def __post_init__(self):
        super().__post_init__()
        self.header_calls = []

    def run(self, cmd, expect_output=None):
        cmd = [str(p) for p in cmd]
        if cmd and cmd[0] == "flirt" and "-usesqform" in cmd and "-applyxfm" in cmd and "-omat" in cmd:
            self.header_calls.append(cmd)
            self.world_transform = np.eye(4)
            return FakeFSL.run(self, cmd, expect_output=expect_output)
        return super().run(cmd, expect_output=expect_output)


def _header_config(**kw):
    return fsl_config(angular_search_deg=(-30, 30), fallback_unconstrained_retry=True, header_only_fallback=True, **kw)


def _header_runner():
    return HeaderFallbackFSL(interp="spline", angular_search_deg=(-30, 30))


def test_the_header_fallback_runs_only_after_every_optimised_attempt_has_failed(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = np.eye(4)  # passes, so nothing else may run

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_header_config())[0]
    moving = _moving_of(qc)

    assert [a["attempt_type"] for a in moving["registration_attempts"]] == ["forward"]
    assert moving["selected_attempt"] == 1
    assert moving["registration_method"] == "optimised"
    assert not runner.header_calls or all("_attempt" not in c[c.index("-omat") + 1] for c in runner.header_calls)


def test_a_lost_scan_is_published_where_its_scanner_header_places_it(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_header_config())[0]
    moving = _moving_of(qc)
    attempts = moving["registration_attempts"]

    assert [a["attempt_type"] for a in attempts] == ["forward", "forward", "header_only"]
    assert [a["qc_outcome"] for a in attempts] == ["fail", "fail", "pass"]
    assert moving["selected_attempt"] == 3
    assert moving["status"] == "ok"
    assert os.path.exists(moving["output_path"])
    # It optimised nothing, so it has no estimation and no search.
    fb = attempts[2]
    assert fb["estimation_direction"] == "none:scanner_header"
    assert fb["estimation_in"] == "" and fb["estimation_ref"] == ""
    assert fb["angular_search_deg"] is None
    assert fb["estimate_command"] is None and fb["inversion_command"] is None
    # One command, carrying -usesqform (never -init), writing both the image and the transform.
    cmd = fb["apply_command"]
    assert "-usesqform" in cmd and "-init" not in cmd and "-applyxfm" in cmd
    assert cmd[cmd.index("-in") + 1] == fb["apply_in"] and cmd[cmd.index("-ref") + 1] == fb["apply_ref"]
    assert cmd[cmd.index("-interp") + 1] == "spline"


def test_a_header_aligned_scan_is_never_recorded_as_registered(tmp_path):
    """It shares the session output lattice, but nothing was fitted, and every field must say so."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    rows = _session_with_two_scans(root)

    qc = process(rows, root, out, runner=runner, config=_header_config())[0]
    moving = _moving_of(qc)
    # "registered" means an optimised transform was fitted and accepted -- not "reached the
    # registration path". The ladder did run, and that is recorded separately.
    assert moving["registration_method"] == "header_only"
    assert moving["registered"] is False
    assert moving["transform_fitted"] is False
    assert moving["coregistration_attempted"] is True

    records = derivative_manifest.build_records(rows, {}, str(out))
    row = next(r for r in records if r["sample_id"] == moving["sample_id"])
    assert row["status"] == "header_only_aligned"
    assert row["registration_method"] == "header_only"
    assert row["registered"] is False
    assert row["transform_fitted"] is False
    assert row["strict_pairing_eligible"] is False
    assert "no transform was fitted" in row["reason"]
    # An auditable header-derived matrix still exists; its presence must prove nothing.
    assert moving["transform_path"] and row["transform_path"]
    # Still part of the corpus, and still countable separately from a fitted registration.
    summary = derivative_manifest.accounting(records)
    assert summary["header_only_aligned"] == 1
    assert summary["registered"] == 0
    assert summary["transform_fitted"] == 0
    assert summary["materialized"] == len(records)
    assert "header_only_aligned" in derivative_manifest.MATERIALIZED


def test_an_optimised_registration_carries_the_positive_half_of_the_contract(tmp_path):
    """The counterpart: a scan attempt 1 fitted must assert every field the header path denies."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, runner=_header_runner(), config=_header_config())[0]
    moving = _moving_of(qc)
    assert moving["registration_method"] == "optimised"
    assert moving["registered"] is True
    assert moving["transform_fitted"] is True

    records = derivative_manifest.build_records(rows, {}, str(out))
    row = next(r for r in records if r["sample_id"] == moving["sample_id"])
    assert (row["status"], row["registered"], row["transform_fitted"]) == ("registered", True, True)
    assert row["strict_pairing_eligible"] is True

    # The reference is the anchor that fit was measured against, so it stays pairable.
    reference = next(r for r in records if r["status"] == "reference")
    assert reference["strict_pairing_eligible"] is True
    assert reference["transform_fitted"] is False and reference["registration_method"] == "none"

    summary = derivative_manifest.accounting(records)
    assert summary["transform_fitted"] == 1
    assert summary["strict_pairing_eligible"] == 2  # the fitted moving scan and its reference


def test_a_reference_whose_session_fitted_nothing_is_not_strictly_pairable(tmp_path):
    """Otherwise a header-only session would still offer its reference as a strict pair partner."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    rows = _session_with_two_scans(root)
    process(rows, root, out, runner=runner, config=_header_config())

    records = derivative_manifest.build_records(rows, {}, str(out))
    assert {r["status"] for r in records} == {"reference", "header_only_aligned"}
    assert not any(r["strict_pairing_eligible"] for r in records)
    assert derivative_manifest.accounting(records)["strict_pairing_eligible"] == 0


def test_the_manifest_reclassifies_a_snapshot_written_before_registered_was_narrowed(tmp_path):
    """Existing qc.json files must classify correctly without re-running any registration.

    Older snapshots stored ``registered=true`` for anything that entered the ladder, header-only
    included. The method is authoritative, so the old value is read as "attempted" only.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, runner=runner, config=_header_config())[0]

    qc_path = out / qc["session_key"] / "qc.json"
    stored = json.loads(qc_path.read_text())
    for mod in stored["modalities"]:
        if mod.get("is_reference"):
            continue
        mod["registered"] = True  # the old meaning
        mod.pop("transform_fitted")
        mod.pop("coregistration_attempted")
    qc_path.write_text(json.dumps(stored))

    row = next(r for r in derivative_manifest.build_records(rows, {}, str(out)) if r["status"] != "reference")
    assert row["status"] == "header_only_aligned"
    assert row["registered"] is False
    assert row["transform_fitted"] is False
    assert row["strict_pairing_eligible"] is False


def test_the_method_vocabulary_cannot_drift_between_pipeline_and_manifest(tmp_path):
    assert pipeline.HEADER_ONLY == derivative_manifest.METHOD_HEADER_ONLY
    assert derivative_manifest.METHOD_OPTIMISED == "optimised"
    assert derivative_manifest.METHOD_NONE == "none"


def test_an_attempted_registration_that_fitted_nothing_is_never_a_passthrough(tmp_path):
    """Fail closed: 'ok' with no fitted transform and no header alignment has no valid meaning."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    qc = process(rows, root, out, runner=_header_runner(), config=_header_config())[0]

    qc_path = out / qc["session_key"] / "qc.json"
    stored = json.loads(qc_path.read_text())
    for mod in stored["modalities"]:
        if mod.get("is_reference"):
            continue
        mod["transform_fitted"] = False
        mod["registered"] = False
        mod["registration_method"] = "none"
        mod["coregistration_attempted"] = True
        mod["status"] = "ok"
    qc_path.write_text(json.dumps(stored))

    row = next(r for r in derivative_manifest.build_records(rows, {}, str(out)) if r["status"] != "reference")
    assert row["status"] == "failed"
    assert "no transform was fitted" in row["reason"]
    assert row["strict_pairing_eligible"] is False


def test_a_blocked_session_is_never_relabelled_as_a_single_scan(tmp_path):
    """The session's own qc.json records why it was blocked, so the roll-up must not need the audit.

    Passing no verdicts used to silently turn a multi-modality session blocked for a scanner
    conflict into ``passthrough_single_scan`` with reason ``single_scan_session`` -- a statement
    contradicted by the very record it was derived from.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = make_corpus(
        root,
        {
            ("PT01", "sub-01", "ses-01"): [
                ("T1w", (20, 18, 16), (1.0,) * 3),
                ("T2w", (20, 18, 16), (1.0,) * 3),
                ("FLAIR", (20, 18, 16), (1.0,) * 3),
            ]
        },
    )
    process(rows, root, out, coreg_allowed=False)

    # No verdicts supplied: the qc.json alone has to carry the answer.
    records = derivative_manifest.build_records(rows, {}, str(out))
    summary = derivative_manifest.assert_accounting(records, expected_samples=len(rows))
    assert summary["passthrough_ambiguous_session"] == 2  # the two non-reference scans
    assert summary["passthrough_single_scan"] == 0
    assert summary["reference"] == 1
    blocked = [r for r in records if r["status"] == "passthrough_ambiguous_session"]
    assert all("ambiguous_session:test" in r["reason"] for r in blocked)
    assert not any(r["strict_pairing_eligible"] for r in records)
    # The reference of a session that fitted nothing is not a strict pair partner either.
    assert next(r for r in records if r["status"] == "reference")["strict_pairing_eligible"] is False


def test_the_tensorisation_rows_carry_the_distinction_to_the_tensor_stage(tmp_path):
    """Once the NIfTIs are gone, the .pt manifest is the only place this can still be read."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    rows = _session_with_two_scans(root)
    process(rows, root, out, runner=runner, config=_header_config())

    records = derivative_manifest.build_records(rows, {}, str(out))
    tensor_rows = derivative_manifest.tensorisation_rows(records)
    assert len(tensor_rows) == len(records)  # header-only stays in the corpus
    header = next(r for r in tensor_rows if r["status"] == "header_only_aligned")
    assert header["transform_fitted"] is False and header["strict_pairing_eligible"] is False
    assert header["registration_method"] == "header_only"


def test_the_header_fallback_still_shares_the_session_lattice(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)

    qc = process(_session_with_two_scans(root), root, out, runner=runner, config=_header_config())[0]
    assert qc["status"] == "success"
    images = [nib.load(m["output_path"]) for m in qc["modalities"]]
    assert len({img.shape for img in images}) == 1
    for img in images[1:]:
        assert np.allclose(img.affine, images[0].affine, atol=1e-4)


def test_a_header_aligned_candidate_still_has_to_pass_the_same_qc(tmp_path):
    """It is a last resort, not an exemption: the unchanged checks still decide."""
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    rows = make_corpus(
        root, {("PT01", "sub-01", "ses-01"): [("T1w", (32, 30, 28), (1.0,) * 3), ("T2w", (28, 26, 24), (1.2, 1.2, 1.4))]}
    )
    qc = process(rows, root, out, runner=runner, config=_header_config())[0]
    fb = _moving_of(qc)["registration_attempts"][2]
    assert "qc_rejects" in fb and fb["qc_outcome"] in ("pass", "fail")
    # And it is judged by the very same function the session validator re-applies.
    assert fb["qc_rejects"] == [] or fb["reason"]


def test_a_header_aligned_scan_with_genuine_overlap_is_still_published(tmp_path):
    """The zero-overlap invariant targets empty outputs, not the fallback itself.

    178 samples reached the corpus this way and all but one carry anatomy; rejecting the rung
    wholesale would discard them.
    """
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    rows = make_corpus(
        root, {("PT01", "sub-01", "ses-01"): [("T1w", (32, 30, 28), (1.0,) * 3), ("T2w", (28, 26, 24), (1.2, 1.2, 1.4))]}
    )
    qc = process(rows, root, out, runner=runner, config=_header_config())[0]
    moving = _moving_of(qc)
    assert moving["registration_method"] == "header_only"
    assert moving["transform_fitted"] is False
    assert moving["status"] == "ok"
    assert moving["published"] is True
    assert moving["output_foreground_voxel_count"] > 0


def test_the_header_fallback_is_last_in_the_ladder_and_configurable(tmp_path):
    assert [s["type"] for s in pipeline._attempt_ladder(_header_config())] == ["forward", "forward", "header_only"]
    both = _header_config(inverse_direction_rescue=True)
    assert [s["type"] for s in pipeline._attempt_ladder(both)] == [
        "forward",
        "forward",
        "inverse_direction_rescue",
        "header_only",
    ]
    # Off by default, and never reachable on a backend or a run that cannot mean it.
    assert [s["type"] for s in pipeline._attempt_ladder(_fallback_config())] == ["forward", "forward"]
    with pytest.raises(ValueError, match="FSL backend only"):
        config_mod.CoregConfig(backend="freesurfer", header_only_fallback=True)
    with pytest.raises(ValueError, match="do_coregister"):
        fsl_config(header_only_fallback=True, do_coregister=False)
    with pytest.raises(ValueError, match="init_from_header"):
        fsl_config(header_only_fallback=True, init_from_header=False)


def test_changing_the_header_fallback_policy_invalidates_resume(tmp_path):
    root, out = tmp_path / "cleaned", tmp_path / "p1"
    rows = _session_with_two_scans(root)
    runner = _header_runner()
    runner.attempt1_transform = _bad_transform()
    runner.attempt2_transform = _bad_transform(mm=250.0)
    qc = process(rows, root, out, runner=runner, config=_header_config())[0]
    key, qc_path = qc["session_key"], out / qc["session_key"] / "qc.json"

    assert run_mod.is_session_done(str(out), key, _header_config()) is True
    assert run_mod.is_session_done(str(out), key, _fallback_config()) is False
    assert "header_only_fallback" in run_mod.RESUME_INVALIDATING_FIELDS

    stored = json.loads(qc_path.read_text())
    stored["config"].pop("header_only_fallback")
    qc_path.write_text(json.dumps(stored))
    assert run_mod.is_session_done(str(out), key, _header_config()) is False


# Four tests here read `session_coreg/slurm/submit_coreg_fsl.sh` and `coreg_fsl_aggregate.slurm`
# to prove the launcher and the array stage agreed on one registration policy. Those scripts
# carried a Slurm account and cluster absolute paths and are not part of the public code release,
# so the tests went with them. The coregistration pipeline they launched is still covered here.
