"""Read-only geometry identity helpers for Experiment 1.

Only raw file bytes are authoritative for lineage.  Geometry decoding is used
for index-bound validation and never rewrites the input or derives regions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any, BinaryIO

from .experiment1_contracts import ContractError, RawArtifactLineage, sha256_file


@dataclass(frozen=True)
class DecodedGeometry:
    vertex_count: int
    face_count: int
    positions: tuple[tuple[float, float, float], ...] = ()
    faces: tuple[tuple[int, ...], ...] = ()


_PLY_SCALARS: dict[str, tuple[str, int]] = {
    "char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8),
}


def _read_ply_header(handle: BinaryIO) -> tuple[str, list[tuple[str, int, list[tuple[str, ...]]]]]:
    if handle.readline().rstrip(b"\r\n") != b"ply":
        raise ContractError("not a PLY file")
    fmt = ""
    elements: list[tuple[str, int, list[tuple[str, ...]]]] = []
    current: tuple[str, int, list[tuple[str, ...]]] | None = None
    while True:
        raw = handle.readline()
        if not raw:
            raise ContractError("PLY header ended before end_header")
        line = raw.decode("ascii", errors="strict").strip()
        if line == "end_header":
            break
        if not line or line.startswith("comment") or line.startswith("obj_info"):
            continue
        parts = line.split()
        if parts[:1] == ["format"] and len(parts) == 3:
            fmt = parts[1]
        elif parts[:1] == ["element"] and len(parts) == 3:
            current = (parts[1], int(parts[2]), [])
            elements.append(current)
        elif parts[:1] == ["property"] and current is not None:
            if len(parts) == 3:
                current[2].append(("scalar", parts[1], parts[2]))
            elif len(parts) == 5 and parts[1] == "list":
                current[2].append(("list", parts[2], parts[3], parts[4]))
            else:
                raise ContractError("unsupported PLY property declaration")
    if fmt not in {"ascii", "binary_little_endian"}:
        raise ContractError(f"unsupported PLY format {fmt!r}; only ascii and binary_little_endian are read")
    return fmt, elements


def _scalar_ascii(token: str, type_name: str) -> int | float:
    if type_name not in _PLY_SCALARS:
        raise ContractError(f"unsupported PLY scalar type {type_name}")
    return float(token) if _PLY_SCALARS[type_name][0] in {"f", "d"} else int(token)


def _scalar_binary(handle: BinaryIO, type_name: str) -> int | float:
    try:
        code, size = _PLY_SCALARS[type_name]
    except KeyError as exc:
        raise ContractError(f"unsupported PLY scalar type {type_name}") from exc
    payload = handle.read(size)
    if len(payload) != size:
        raise ContractError("unexpected EOF in PLY body")
    return struct.unpack("<" + code, payload)[0]


def decode_ply(path: str | Path) -> DecodedGeometry:
    """Decode vertex/face counts and positions from ASCII or binary-LE PLY."""
    with Path(path).open("rb") as handle:
        fmt, elements = _read_ply_header(handle)
        positions: list[tuple[float, float, float]] = []
        faces: list[tuple[int, ...]] = []
        for name, count, properties in elements:
            for _ in range(count):
                values: dict[str, Any] = {}
                for declaration in properties:
                    if declaration[0] == "scalar":
                        _, type_name, property_name = declaration
                        if fmt == "ascii":
                            # ASCII is consumed per record below; this branch is unreachable.
                            raise AssertionError("ASCII property should be read as a record")
                        values[property_name] = _scalar_binary(handle, type_name)
                    else:
                        _, count_type, value_type, property_name = declaration
                        n = int(_scalar_binary(handle, count_type))
                        values[property_name] = tuple(int(_scalar_binary(handle, value_type)) for _ in range(n))
                if name == "vertex" and fmt != "ascii":
                    positions.append((float(values["x"]), float(values["y"]), float(values["z"])))
                if name == "face" and fmt != "ascii":
                    candidates = next((value for value in values.values() if isinstance(value, tuple)), ())
                    faces.append(tuple(int(index) for index in candidates))
            if fmt == "ascii":
                # ASCII records must be read here, not property-by-property.
                # Re-open after parsing header is deliberately avoided; consume element records now.
                raise ContractError("internal ASCII PLY parser error")
        vertex_count = next((count for name, count, _ in elements if name == "vertex"), 0)
        face_count = next((count for name, count, _ in elements if name == "face"), 0)
        return DecodedGeometry(vertex_count=vertex_count, face_count=face_count, positions=tuple(positions), faces=tuple(faces))


def _decode_ascii_ply(path: Path) -> DecodedGeometry:
    with path.open("rb") as handle:
        fmt, elements = _read_ply_header(handle)
        if fmt != "ascii":
            raise ContractError("expected ASCII PLY")
        positions: list[tuple[float, float, float]] = []
        faces: list[tuple[int, ...]] = []
        for name, count, properties in elements:
            for _ in range(count):
                tokens = handle.readline().decode("ascii", errors="strict").split()
                if not tokens:
                    raise ContractError("unexpected EOF in ASCII PLY body")
                offset = 0
                values: dict[str, Any] = {}
                for declaration in properties:
                    if declaration[0] == "scalar":
                        _, type_name, property_name = declaration
                        if offset >= len(tokens):
                            raise ContractError("truncated ASCII PLY record")
                        values[property_name] = _scalar_ascii(tokens[offset], type_name)
                        offset += 1
                    else:
                        _, count_type, value_type, property_name = declaration
                        if offset >= len(tokens):
                            raise ContractError("truncated ASCII PLY list")
                        n = int(_scalar_ascii(tokens[offset], count_type))
                        offset += 1
                        if len(tokens) < offset + n:
                            raise ContractError("truncated ASCII PLY list values")
                        values[property_name] = tuple(int(_scalar_ascii(token, value_type)) for token in tokens[offset:offset + n])
                        offset += n
                if name == "vertex":
                    try:
                        positions.append((float(values["x"]), float(values["y"]), float(values["z"])))
                    except KeyError as exc:
                        raise ContractError("vertex PLY requires x/y/z for Experiment 1") from exc
                elif name == "face":
                    candidates = next((value for value in values.values() if isinstance(value, tuple)), ())
                    faces.append(tuple(int(index) for index in candidates))
        vertex_count = next((count for name, count, _ in elements if name == "vertex"), 0)
        face_count = next((count for name, count, _ in elements if name == "face"), 0)
        return DecodedGeometry(vertex_count=vertex_count, face_count=face_count, positions=tuple(positions), faces=tuple(faces))


def _glb_chunks(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ContractError("GLB is too short")
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or length != len(payload):
        raise ContractError("not a valid glTF 2.0 GLB")
    offset = 12
    json_chunk: bytes | None = None
    bin_chunk = b""
    while offset < len(payload):
        chunk_length, chunk_type = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        chunk = payload[offset:offset + chunk_length]
        offset += chunk_length
        if len(chunk) != chunk_length:
            raise ContractError("truncated GLB chunk")
        if chunk_type == b"JSON":
            json_chunk = chunk
        elif chunk_type == b"BIN\x00":
            bin_chunk = chunk
    if json_chunk is None:
        raise ContractError("GLB lacks a JSON chunk")
    return json.loads(json_chunk.decode("utf-8").rstrip(" \t\r\n\x00")), bin_chunk


def _gltf_accessor(document: dict[str, Any], blob: bytes, index: int) -> tuple[int, tuple[tuple[float, ...], ...]]:
    accessor = document["accessors"][index]
    view_index = accessor.get("bufferView")
    if view_index is None or accessor.get("sparse"):
        raise ContractError("sparse or implicit GLB accessors are not supported")
    view = document["bufferViews"][view_index]
    component = accessor["componentType"]
    component_codes = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
    if component not in component_codes:
        raise ContractError("unsupported GLB component type")
    code, scalar_size = component_codes[component]
    component_count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}.get(accessor["type"])
    if component_count is None:
        raise ContractError("unsupported GLB accessor type")
    count = int(accessor["count"])
    stride = int(view.get("byteStride", scalar_size * component_count))
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    records: list[tuple[float, ...]] = []
    for item in range(count):
        at = start + item * stride
        end = at + scalar_size * component_count
        if end > len(blob):
            raise ContractError("GLB accessor exceeds BIN chunk")
        records.append(tuple(struct.unpack_from("<" + code * component_count, blob, at)))
    return component, tuple(records)


def decode_glb(path: str | Path) -> DecodedGeometry:
    """Decode the first TRIANGLES primitive from a glTF 2.0 binary file."""
    document, blob = _glb_chunks(Path(path))
    meshes = document.get("meshes", [])
    if not meshes or not meshes[0].get("primitives"):
        raise ContractError("GLB has no mesh primitive")
    primitive = meshes[0]["primitives"][0]
    if primitive.get("mode", 4) != 4 or "POSITION" not in primitive.get("attributes", {}):
        raise ContractError("Experiment 1 expects a TRIANGLES GLB primitive with POSITION")
    component, positions = _gltf_accessor(document, blob, primitive["attributes"]["POSITION"])
    if component != 5126:
        raise ContractError("GLB POSITION must use float32")
    if "indices" in primitive:
        component, indices = _gltf_accessor(document, blob, primitive["indices"])
        if component not in {5121, 5123, 5125}:
            raise ContractError("GLB indices must be unsigned integers")
        flat = [int(item[0]) for item in indices]
    else:
        flat = list(range(len(positions)))
    if len(flat) % 3:
        raise ContractError("TRIANGLES index count must be divisible by three")
    faces = tuple(tuple(flat[offset:offset + 3]) for offset in range(0, len(flat), 3))
    return DecodedGeometry(
        vertex_count=len(positions),
        face_count=len(faces),
        positions=tuple(tuple(float(value) for value in row[:3]) for row in positions),
        faces=faces,
    )


def decode_geometry(path: str | Path) -> DecodedGeometry:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".ply":
        with source.open("rb") as handle:
            _format, _elements = _read_ply_header(handle)
        return _decode_ascii_ply(source) if _format == "ascii" else decode_ply(source)
    if suffix == ".glb":
        return decode_glb(source)
    raise ContractError(f"unsupported raw geometry extension {source.suffix!r}")


def lineage_from_file(
    path: str | Path,
    *,
    artifact_id: str,
    source_locator: str = "",
    source_revision: str = "",
) -> RawArtifactLineage:
    source = Path(path)
    geometry = decode_geometry(source)
    representation = "glb" if source.suffix.lower() == ".glb" else "ply"
    return RawArtifactLineage(
        artifact_id=artifact_id,
        original_file_sha256=sha256_file(source),
        representation=representation,
        vertex_count=geometry.vertex_count,
        face_count=geometry.face_count,
        source_locator=source_locator or str(source),
        source_revision=source_revision,
    )
