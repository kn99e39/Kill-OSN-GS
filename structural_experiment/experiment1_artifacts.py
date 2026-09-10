"""Content-addressed persistence for Experiment 1 control planes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .canonical import read_json, write_json
from .experiment1_contracts import AssociationDeclaration, ContractError, Experiment1ControlPlane, fingerprint


MANIFEST_SCHEMA = "structural-experiment/experiment1-manifest/v1"


def control_plane_payload(controls: Experiment1ControlPlane) -> dict[str, Any]:
    """Serialize every frozen input needed to audit a later A-only variation."""
    return {
        "observed_partition": controls.observed_partition.to_dict(),
        "generated_partition": controls.generated_partition.to_dict(),
        "reference_partition": controls.reference_partition.to_dict(),
        "preprocessing_fingerprint": controls.preprocessing_fingerprint,
        "sampling_fingerprint": controls.sampling_fingerprint,
        "renderer_fingerprint": controls.renderer_fingerprint,
        "evaluation": controls.evaluation.to_dict(),
        "reference_model_fingerprint": controls.reference_model_fingerprint,
        "seed": controls.seed,
        "frozen_fingerprints": controls.frozen_fingerprints(),
    }


def write_experiment1_manifest(
    path: str | Path,
    *,
    controls: Experiment1ControlPlane,
    association: AssociationDeclaration,
    artifact_locations: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Persist an audit manifest without materialising, modifying, or inferring A."""
    payload = {
        "schema": MANIFEST_SCHEMA,
        "control_plane": control_plane_payload(controls),
        "association": association.to_dict(),
        "association_fingerprint": association.fingerprint,
        "artifact_locations": dict(artifact_locations or {}),
    }
    payload["manifest_fingerprint"] = fingerprint(payload)
    write_json(path, payload)
    return payload


def verify_experiment1_manifest(path: str | Path, *, controls: Experiment1ControlPlane) -> dict[str, Any]:
    """Verify serialized frozen controls before allowing a later controlled run."""
    payload = read_json(path)
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ContractError(f"unsupported Experiment 1 manifest schema: {payload.get('schema')!r}")
    expected_fingerprint = payload.get("manifest_fingerprint")
    unsigned = dict(payload)
    unsigned.pop("manifest_fingerprint", None)
    if expected_fingerprint != fingerprint(unsigned):
        raise ContractError("Experiment 1 manifest has a mismatched content fingerprint")
    recorded = payload.get("control_plane", {}).get("frozen_fingerprints")
    if recorded != controls.frozen_fingerprints():
        raise ContractError("Experiment 1 manifest controls do not match the supplied frozen control plane")
    return payload
