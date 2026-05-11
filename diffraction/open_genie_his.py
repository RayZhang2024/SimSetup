from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .models import AXIS_TOF, FocusedSpectrum, InstrumentCalibration


_FILENAME_RE = re.compile(r"ENGINX(?P<run>\d+)_(?P<bank>\d+)\.his$", re.IGNORECASE)
_TYPE_NAMES = ("GXRealarray", "Float", "Integer", "String", "UndefinedObject")


def _exact_tag_offset(data: bytes, tag: str) -> Optional[int]:
    encoded = tag.encode("ascii")
    needle = struct.pack(">I", len(encoded)) + encoded
    offset = data.find(needle)
    if offset < 0:
        return None
    return offset + 4


def _record_type(data: bytes, tag_offset: int) -> Optional[tuple[str, int]]:
    best: Optional[tuple[str, int]] = None
    search_end = min(len(data), tag_offset + 128)
    for type_name in _TYPE_NAMES:
        offset = data.find(type_name.encode("ascii"), tag_offset, search_end)
        if offset >= 0 and (best is None or offset < best[1]):
            best = (type_name, offset)
    return best


def _scalar_payload_start(type_name: str, type_offset: int) -> int:
    return type_offset + len(type_name)


def _read_float(data: bytes, tag: str) -> Optional[float]:
    tag_offset = _exact_tag_offset(data, tag)
    if tag_offset is None:
        return None
    record_type = _record_type(data, tag_offset)
    if record_type is None or record_type[0] != "Float":
        return None
    payload = _scalar_payload_start(record_type[0], record_type[1])
    value_offset = payload + 11
    if value_offset + 8 > len(data):
        return None
    return float(struct.unpack_from(">d", data, value_offset)[0])


def _read_integer(data: bytes, tag: str) -> Optional[int]:
    tag_offset = _exact_tag_offset(data, tag)
    if tag_offset is None:
        return None
    record_type = _record_type(data, tag_offset)
    if record_type is None or record_type[0] != "Integer":
        return None
    payload = _scalar_payload_start(record_type[0], record_type[1])
    value_offset = payload + 13
    if value_offset + 4 > len(data):
        return None
    return int(struct.unpack_from(">I", data, value_offset)[0])


def _read_string(data: bytes, tag: str) -> Optional[str]:
    tag_offset = _exact_tag_offset(data, tag)
    if tag_offset is None:
        return None
    record_type = _record_type(data, tag_offset)
    if record_type is None or record_type[0] != "String":
        return None
    payload = _scalar_payload_start(record_type[0], record_type[1])
    length_offset = payload + 10
    if length_offset + 4 > len(data):
        return None
    length = int(struct.unpack_from(">I", data, length_offset)[0])
    value_offset = length_offset + 4
    if length < 0 or value_offset + length > len(data):
        return None
    return data[value_offset : value_offset + length].decode("latin-1", errors="replace").rstrip("\x00")


def _read_real_array(data: bytes, tag: str) -> Optional[np.ndarray]:
    tag_offset = _exact_tag_offset(data, tag)
    if tag_offset is None:
        return None
    record_type = _record_type(data, tag_offset)
    if record_type is None or record_type[0] != "GXRealarray":
        return None
    payload = _scalar_payload_start(record_type[0], record_type[1])
    header = data[payload : min(len(data), payload + 96)]
    for offset in range(0, max(0, len(header) - 8)):
        length = int.from_bytes(header[offset : offset + 4], "big", signed=False)
        duplicate = int.from_bytes(header[offset + 4 : offset + 8], "big", signed=False)
        if length <= 0 or length != duplicate:
            continue
        data_offset = payload + offset + 8
        byte_count = length * 8
        if data_offset + byte_count > len(data):
            continue
        values = np.frombuffer(data, dtype=">f8", count=length, offset=data_offset)
        return values.astype(float, copy=True)
    return None


def _first_present_float(data: bytes, tags: Iterable[str]) -> Optional[float]:
    for tag in tags:
        value = _read_float(data, tag)
        if value is not None:
            return value
    return None


def _parse_filename(path: Path) -> tuple[Optional[str], Optional[int]]:
    match = _FILENAME_RE.search(path.name)
    if match is None:
        return None, None
    return match.group("run"), int(match.group("bank"))


def read_his_spectrum(path: Path) -> FocusedSpectrum:
    source_path = Path(path)
    data = source_path.read_bytes()
    x = _read_real_array(data, "gXx")
    y = _read_real_array(data, "gXy")
    e = _read_real_array(data, "gXe")
    if x is None or y is None:
        raise ValueError(f"{source_path.name} does not contain gXx/gXy histogram arrays.")
    if e is not None and e.size != y.size:
        e = None

    filename_run, filename_bank = _parse_filename(source_path)
    run_number = _read_string(data, "gXrunUno") or filename_run
    bank_number = _read_integer(data, "gXbank") or filename_bank
    bank_name = _read_string(data, "gXbankUname")
    difc = _first_present_float(data, ("gXpeaksUdifc", "gXdifc"))
    tzero = _first_present_float(data, ("gXpeaksUzero", "gXzero")) or 0.0
    difa = _first_present_float(data, ("gXpeaksUdifa", "gXdifa")) or 0.0
    calibration = None
    if difc is not None and abs(difc) > 1e-12:
        calibration = InstrumentCalibration(difc=float(difc), tzero=float(tzero), difa=float(difa))

    metadata: Dict[str, object] = {
        "raw_file": _read_string(data, "gXfile"),
        "parameter_file": _read_string(data, "gXparameterUfile"),
        "title": _read_string(data, "gXtitle"),
        "time": _read_string(data, "gXtime"),
        "user_name": _read_string(data, "gXuserUname"),
        "instrument": _read_string(data, "gXinstUname"),
        "difc": difc,
        "difa": difa,
        "tzero": tzero,
    }

    return FocusedSpectrum(
        source_path=source_path,
        x=x,
        y=y,
        e=e,
        native_axis=AXIS_TOF,
        x_is_edges=(x.size == y.size + 1),
        calibration=calibration,
        metadata={key: value for key, value in metadata.items() if value is not None},
        run_number=run_number,
        bank_number=bank_number,
        bank_name=bank_name,
        x_label=_read_string(data, "gXxlabel") or "Time-of-Flight",
        y_label=_read_string(data, "gXylabel") or "Neutron counts",
    )
