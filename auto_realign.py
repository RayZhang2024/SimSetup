"""Geometry-based suggestions for the Pick Point local coordinate frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class AutoRealignError(ValueError):
    """Raised when a mesh does not contain enough shape information to align."""


@dataclass(frozen=True)
class AutoRealignSuggestion:
    origin: np.ndarray
    plane_normal: np.ndarray
    in_plane_axis: np.ndarray
    profile: str
    warnings: Tuple[str, ...]
    eigenvalues: np.ndarray

    @property
    def summary(self) -> str:
        text = f"Estimated {self.profile} frame"
        if self.warnings:
            text += ". " + " ".join(self.warnings)
        else:
            text += ". Principal directions are well separated."
        return text


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise AutoRealignError("Could not determine a non-zero frame direction.")
    return np.asarray(vector, dtype=float) / length


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    result = _unit(vector)
    dominant = int(np.argmax(np.abs(result)))
    if result[dominant] < 0.0:
        result = -result
    return result


def _project_reference(reference: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = _unit(normal)
    reference = np.asarray(reference, dtype=float)
    candidate = reference - normal * float(np.dot(reference, normal))
    if np.linalg.norm(candidate) <= 1e-9:
        candidates = np.eye(3, dtype=float)
        fallback = candidates[int(np.argmin(np.abs(candidates @ normal)))]
        candidate = fallback - normal * float(np.dot(fallback, normal))
    candidate = _unit(candidate)
    if float(np.dot(candidate, reference)) < 0.0:
        candidate = -candidate
    return candidate


def _surface_covariance(points: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise AutoRealignError("The mesh must contain at least three 3D points.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        raise AutoRealignError("The mesh must contain triangle faces.")
    if not np.isfinite(points).all():
        raise AutoRealignError("The mesh contains non-finite coordinates.")
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(points):
        raise AutoRealignError("The mesh has invalid triangle indices.")

    triangle_points = points[triangles]
    cross_products = np.cross(
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross_products, axis=1)
    valid = np.isfinite(areas) & (areas > max(float(areas.max(initial=0.0)) * 1e-12, 1e-15))
    if not np.any(valid):
        raise AutoRealignError("The mesh has no non-degenerate surface triangles.")

    triangle_points = triangle_points[valid]
    areas = areas[valid]
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise AutoRealignError("The mesh has no usable surface area.")

    # Exact first and second moments for a uniform point over each triangle,
    # weighted by triangle area. This makes the estimate insensitive to how
    # densely a planar STL surface happened to be tessellated.
    reference = np.average(triangle_points.mean(axis=1), axis=0, weights=areas)
    relative = triangle_points - reference
    vertex_sum = relative.sum(axis=1)
    vertex_second = np.einsum("nki,nkj->nij", relative, relative)
    triangle_second = (vertex_second + np.einsum("ni,nj->nij", vertex_sum, vertex_sum)) / 12.0
    second_moment = np.einsum("n,nij->ij", areas, triangle_second) / total_area
    triangle_means = relative.mean(axis=1)
    mean = np.average(triangle_means, axis=0, weights=areas)
    covariance = second_moment - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return reference + mean, covariance


def estimate_auto_realign(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    preferred_normal: np.ndarray,
    preferred_in_plane: np.ndarray,
) -> AutoRealignSuggestion:
    """Suggest a frame from a triangulated surface without changing the mesh.

    ``preferred_normal`` and ``preferred_in_plane`` are current-frame directions
    used only when the mesh is symmetric around one of its principal axes.
    """
    points = np.asarray(points, dtype=float)
    surface_center, covariance = _surface_covariance(points, triangles)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0.0)
    if not np.isfinite(values).all() or float(values[-1]) <= 1e-15:
        raise AutoRealignError("The mesh is too small or flat to estimate a 3D frame.")

    minor = _canonical_sign(vectors[:, 0])
    middle = _canonical_sign(vectors[:, 1])
    major = _canonical_sign(vectors[:, 2])
    epsilon = max(float(values[-1]) * 1e-12, 1e-15)
    middle_gap = float(values[1] / max(float(values[0]), epsilon))
    major_gap = float(values[2] / max(float(values[1]), epsilon))
    warnings = []

    # A thin plate has one clearly small variance. Its normal is stable even
    # when its in-plane outline is square and therefore has no unique roll.
    if float(values[0] / max(float(values[1]), epsilon)) <= 0.18:
        profile = "plate"
        plane_normal = minor
        if major_gap < 1.15:
            in_plane_axis = _project_reference(preferred_in_plane, plane_normal)
            warnings.append("The plate outline is symmetric; in-plane roll uses the current frame as a reference.")
        else:
            in_plane_axis = major

    # A short, axisymmetric tube can have its longitudinal direction as the
    # isolated minor component (the two radial components are similar).
    elif major_gap < 1.15 and middle_gap >= 1.15:
        profile = "axisymmetric tube"
        in_plane_axis = minor
        plane_normal = _project_reference(preferred_normal, in_plane_axis)
        warnings.append("The radial directions are symmetric; roll uses the current frame as a reference.")

    # End caps can make the tube axis the major component while its two radial
    # components remain nearly equal, even when length is only about its diameter.
    elif middle_gap < 1.15 and major_gap >= 1.15:
        profile = "axisymmetric tube"
        in_plane_axis = major
        plane_normal = _project_reference(preferred_normal, in_plane_axis)
        warnings.append("The radial directions are symmetric; roll uses the current frame as a reference.")

    # A long body has a stable major axis. If the cross-section is circular or
    # square, choose a repeatable roll from the current frame and report it.
    elif major_gap >= 2.0:
        profile = "elongated tube or pipe"
        in_plane_axis = major
        if middle_gap < 1.15:
            plane_normal = _project_reference(preferred_normal, major)
            warnings.append("The cross-section is symmetric; roll uses the current frame as a reference.")
        else:
            plane_normal = middle

    else:
        profile = "principal-axis"
        if major_gap < 1.15 and middle_gap < 1.15:
            raise AutoRealignError(
                "The mesh is too symmetric to infer a unique frame. Use manual realignment."
            )
        if major_gap >= 1.15:
            in_plane_axis = major
            if middle_gap < 1.15:
                plane_normal = _project_reference(preferred_normal, major)
                warnings.append("Two cross-axis directions are similar; roll uses the current frame as a reference.")
            else:
                plane_normal = minor
        else:
            plane_normal = minor
            in_plane_axis = _project_reference(preferred_in_plane, minor)
            warnings.append("Two in-plane directions are similar; the current frame sets the in-plane direction.")
        warnings.append("The mesh does not clearly match a plate or tube; review this principal-axis suggestion carefully.")

    plane_normal = _unit(plane_normal)
    in_plane_axis = _project_reference(in_plane_axis, plane_normal)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    origin = 0.5 * (bounds_min + bounds_max)
    if not np.isfinite(origin).all() or not np.isfinite(surface_center).all():
        raise AutoRealignError("Could not determine a finite origin for this mesh.")
    return AutoRealignSuggestion(
        origin=origin,
        plane_normal=plane_normal,
        in_plane_axis=in_plane_axis,
        profile=profile,
        warnings=tuple(warnings),
        eigenvalues=values,
    )


def triangulated_faces(polydata) -> np.ndarray:
    """Return triangle indices for a PyVista/VTK PolyData surface."""
    triangulated = polydata.triangulate()
    packed = np.asarray(triangulated.faces, dtype=np.int64)
    if packed.size == 0:
        raise AutoRealignError("The mesh has no polygon faces to analyze.")
    faces = []
    cursor = 0
    while cursor < len(packed):
        count = int(packed[cursor])
        cursor += 1
        indices = packed[cursor : cursor + count]
        cursor += count
        if count != 3:
            raise AutoRealignError("The mesh could not be triangulated for analysis.")
        faces.append(indices)
    return np.asarray(faces, dtype=np.int64)
