"""Content-addressed persistence for Experiment 1 control planes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .canonical import read_json, write_json
from .experiment1_contracts import AssociationDeclaration, ContractError, Experiment1ControlPlane, fingerprint


MANIFEST_SCHEMA = "structural-experiment/experiment1-manifest/v1"
GENPC_MANIFEST_SCHEMA = "structural-experiment/genpc-experiment1-manifest/v1"


def control_plane_payload(controls: Any) -> dict[str, Any]:
    """Serialize every frozen input needed to audit a later A-only variation."""
    # GenPC adds R by wrapping the historical Experiment1ControlPlane.  Keep
    # the v1 payload for historical controls and add only the explicit R
    # block for the wrapper.
    base = getattr(controls, "base", controls)
    return {
        "observed_partition": base.observed_partition.to_dict(),
        "generated_partition": base.generated_partition.to_dict(),
        "reference_partition": base.reference_partition.to_dict(),
        "preprocessing_fingerprint": base.preprocessing_fingerprint,
        "sampling_fingerprint": base.sampling_fingerprint,
        "renderer_fingerprint": base.renderer_fingerprint,
        "evaluation": base.evaluation.to_dict(),
        "reference_model_fingerprint": base.reference_model_fingerprint,
        "seed": base.seed,
        "frozen_fingerprints": controls.frozen_fingerprints(),
        **({"reconciliation": controls.reconciliation.to_dict()} if hasattr(controls, "reconciliation") else {}),
    }


def write_experiment1_manifest(
    path: str | Path,
    *,
    controls: Any,
    association: AssociationDeclaration,
    artifact_locations: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Persist an audit manifest without materialising, modifying, or inferring A."""
    payload = {
        "schema": GENPC_MANIFEST_SCHEMA if hasattr(controls, "reconciliation") else MANIFEST_SCHEMA,
        "control_plane": control_plane_payload(controls),
        "association": association.to_dict(),
        "association_fingerprint": association.fingerprint,
        "artifact_locations": dict(artifact_locations or {}),
    }
    payload["manifest_fingerprint"] = fingerprint(payload)
    write_json(path, payload)
    return payload


def verify_experiment1_manifest(path: str | Path, *, controls: Any) -> dict[str, Any]:
    """Verify serialized frozen controls before allowing a later controlled run."""
    payload = read_json(path)
    expected_schema = GENPC_MANIFEST_SCHEMA if hasattr(controls, "reconciliation") else MANIFEST_SCHEMA
    if payload.get("schema") != expected_schema:
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
