from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .models import AtomRecord, PhaseModel, Reflection


_CRS_RE = re.compile(r"^CRS(?P<index>\d+)\s+(?P<body>.*)$")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def _floats(text: str) -> List[float]:
    return [float(match.group(0)) for match in _FLOAT_RE.finditer(text)]


def _normalize_space_group(space_group: str) -> str:
    return " ".join(space_group.upper().split())


def parse_gsas_exp(path: str | Path) -> List[PhaseModel]:
    phase_data: Dict[int, Dict[str, object]] = {}
    for line in Path(path).read_text(encoding="latin-1", errors="replace").splitlines():
        match = _CRS_RE.match(line.rstrip())
        if match is None:
            continue
        index = int(match.group("index"))
        body = match.group("body")
        data = phase_data.setdefault(index, {"atoms": []})

        if "PNAM" in body:
            data["name"] = body.split("PNAM", 1)[1].strip()
        elif "ABC" in body:
            values = _floats(body.split("ABC", 1)[1])
            if len(values) >= 3:
                data["abc"] = tuple(values[:3])
        elif "ANGLES" in body:
            values = _floats(body.split("ANGLES", 1)[1])
            if len(values) >= 3:
                data["angles"] = tuple(values[:3])
        elif "SG SYM" in body:
            data["space_group"] = body.split("SG SYM", 1)[1].strip()
        elif re.search(r"\bAT\s+\d+A\b", body):
            atom_match = re.search(
                r"AT\s+(?P<label>\d+A)\s+(?P<element>[A-Za-z0-9_]+)\s+"
                r"(?P<x>[-+.\dEe]+)\s+(?P<y>[-+.\dEe]+)\s+"
                r"(?P<z>[-+.\dEe]+)\s+(?P<occ>[-+.\dEe]+)",
                body,
            )
            if atom_match is not None:
                atoms = data.setdefault("atoms", [])
                atoms.append(
                    AtomRecord(
                        label=atom_match.group("label"),
                        element=atom_match.group("element").strip(),
                        x=float(atom_match.group("x")),
                        y=float(atom_match.group("y")),
                        z=float(atom_match.group("z")),
                        occupancy=float(atom_match.group("occ")),
                    )
                )

    phases: List[PhaseModel] = []
    for index, data in sorted(phase_data.items()):
        abc = data.get("abc")
        angles = data.get("angles")
        if abc is None or angles is None:
            continue
        phases.append(
            PhaseModel(
                index=index,
                name=str(data.get("name", f"Phase {index}")).strip(),
                a=float(abc[0]),
                b=float(abc[1]),
                c=float(abc[2]),
                alpha=float(angles[0]),
                beta=float(angles[1]),
                gamma=float(angles[2]),
                space_group=str(data.get("space_group", "")).strip(),
                atoms=tuple(data.get("atoms", ())),
            )
        )
    if not phases:
        raise ValueError(f"No usable CRS phase definitions were found in {path}.")
    return phases


def validate_ni_cubic_phase(phase: PhaseModel) -> None:
    validate_cubic_fm3m_phase(phase)
    if not any(atom.element.upper().startswith("NI") for atom in phase.atoms):
        raise ValueError("The selected phase does not contain a Ni atom record.")


def validate_cubic_fm3m_phase(phase: PhaseModel) -> None:
    if not (
        abs(phase.a - phase.b) < 1e-6
        and abs(phase.a - phase.c) < 1e-6
        and abs(phase.alpha - 90.0) < 1e-6
        and abs(phase.beta - 90.0) < 1e-6
        and abs(phase.gamma - 90.0) < 1e-6
    ):
        raise ValueError("Only cubic phase models are supported by this reflection generator.")
    if _normalize_space_group(phase.space_group) != "F M 3 M":
        raise ValueError("Only F m 3 m phase models are supported by this reflection generator.")


def _fcc_allowed(h: int, k: int, l: int) -> bool:
    parities = {h % 2, k % 2, l % 2}
    return len(parities) == 1


def _multiplicity(h: int, k: int, l: int) -> int:
    values = (h, k, l)
    permutations = len(set(itertools.permutations(values, 3)))
    sign_count = 1
    for value in values:
        if value != 0:
            sign_count *= 2
    return permutations * sign_count


def generate_fcc_reflections(phase: PhaseModel, d_min: float, d_max: float) -> List[Reflection]:
    validate_cubic_fm3m_phase(phase)
    low = min(float(d_min), float(d_max))
    high = max(float(d_min), float(d_max))
    if low <= 0.0:
        raise ValueError("d_min must be positive.")
    max_index = max(1, int(math.ceil(phase.a / low)))
    reflections: Dict[Tuple[int, int, int], Reflection] = {}
    for h in range(0, max_index + 1):
        for k in range(0, h + 1):
            for l in range(0, k + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if not _fcc_allowed(h, k, l):
                    continue
                denominator = math.sqrt(h * h + k * k + l * l)
                if denominator <= 0.0:
                    continue
                d_spacing = phase.a / denominator
                if low <= d_spacing <= high:
                    key = (h, k, l)
                    reflections[key] = Reflection(
                        h=h,
                        k=k,
                        l=l,
                        d_spacing=d_spacing,
                        multiplicity=_multiplicity(h, k, l),
                    )
    return sorted(reflections.values(), key=lambda reflection: reflection.d_spacing)
