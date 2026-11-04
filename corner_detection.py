"""Find sharp corner candidates on triangulated surface meshes."""

from __future__ import annotations

from typing import Optional, Set

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import ConvexHull, QhullError, cKDTree

from auto_realign import AutoRealignError, estimate_auto_realign, triangulated_faces


class CornerDetectionError(ValueError):
    """Raised when a mesh cannot be analyzed for corner candidates."""


def _rdp_indices(points: np.ndarray, tolerance: float) -> list[int]:
    """Return indices retained by Ramer-Douglas-Peucker for an open polyline."""
    if len(points) <= 2:
        return list(range(len(points)))

    retained = np.zeros(len(points), dtype=bool)
    retained[[0, -1]] = True
    pending = [(0, len(points) - 1)]
    while pending:
        start, end = pending.pop()
        segment = points[end] - points[start]
        length = float(np.linalg.norm(segment))
        interior = points[start + 1 : end]
        if len(interior) == 0:
            continue
        if length <= 1e-12:
            distances = np.linalg.norm(interior - points[start], axis=1)
        else:
            relative = interior - points[start]
            distances = np.abs(segment[0] * relative[:, 1] - segment[1] * relative[:, 0]) / length
        farthest_offset = int(np.argmax(distances))
        if float(distances[farthest_offset]) > tolerance:
            farthest = start + 1 + farthest_offset
            retained[farthest] = True
            pending.append((start, farthest))
            pending.append((farthest, end))
    return np.flatnonzero(retained).tolist()


def _closed_rdp_indices(points: np.ndarray, tolerance: float) -> list[int]:
    """Simplify a closed convex polyline while preserving its cyclic order."""
    if len(points) <= 3:
        return list(range(len(points)))

    # Two farthest-point sweeps provide stable, near-diameter split points in
    # linear time, avoiding a quadratic distance matrix for large STL hulls.
    first = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    second = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))

    def arc_indices(start: int, end: int) -> list[int]:
        result = [start]
        while result[-1] != end:
            result.append((result[-1] + 1) % len(points))
        return result

    first_arc = arc_indices(first, second)
    second_arc = arc_indices(second, first)
    first_kept = _rdp_indices(points[first_arc], tolerance)[:-1]
    second_kept = _rdp_indices(points[second_arc], tolerance)[:-1]
    result = [first_arc[index] for index in first_kept]
    result.extend(second_arc[index] for index in second_kept)
    return result


def _plate_outline_corners(
    surface: pv.PolyData,
    suggestion,
    *,
    turn_angle_degrees: float,
    corner_scale_fraction: float,
) -> Optional[np.ndarray]:
    """Return plate silhouette corners, or None when generic detection applies."""
    if suggestion is None or suggestion.profile != "plate":
        return None

    normal = suggestion.plane_normal
    in_plane = suggestion.in_plane_axis
    second_in_plane = np.cross(normal, in_plane)
    projected = np.column_stack((surface.points @ in_plane, surface.points @ second_in_plane))
    if not np.isfinite(projected).all():
        return None
    try:
        hull_ids = ConvexHull(projected).vertices
    except QhullError:
        return None
    hull_points = projected[hull_ids]
    if len(hull_ids) < 3:
        return None

    projected_span = np.ptp(hull_points, axis=0)
    narrow_span = float(projected_span.min())
    if narrow_span <= 1e-12:
        return None
    # Keep the scale control useful on noisy borders while making sure a large
    # tolerance cannot collapse the short side of an otherwise sharp plate.
    tolerance = min(
        float(np.linalg.norm(np.ptp(surface.points, axis=0))) * corner_scale_fraction,
        narrow_span * 0.1,
    )
    simplified = _closed_rdp_indices(hull_points, tolerance)
    if len(simplified) < 3:
        return None

    candidate_hull_indices = []
    for index, hull_index in enumerate(simplified):
        previous = hull_points[simplified[index - 1]] - hull_points[hull_index]
        following = hull_points[simplified[(index + 1) % len(simplified)]] - hull_points[hull_index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator <= 1e-12:
            continue
        interior_angle = float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(previous, following)) / denominator, -1.0, 1.0)
                )
            )
        )
        turning_angle = 180.0 - interior_angle
        if turning_angle >= turn_angle_degrees:
            candidate_hull_indices.append(int(hull_ids[hull_index]))
    if not candidate_hull_indices:
        return np.empty((0, 3), dtype=float)
    return np.asarray(surface.points[candidate_hull_indices], dtype=float).copy()


def _slice_perimeter_corners(
    surface: pv.PolyData,
    suggestion,
    *,
    turn_angle_degrees: float,
    corner_scale_fraction: float,
    diagonal: float,
    has_three_surface_directions,
) -> Optional[np.ndarray]:
    """Find persistent corners on perimeter sections near the main-axis ends."""
    if suggestion is None or int(surface.n_open_edges) > 0:
        return None

    if suggestion.profile == "plate":
        axis = np.asarray(suggestion.plane_normal, dtype=float)
        require_three_surface_directions = False
    else:
        # For tubes and elongated bodies the main axis gives their end profile;
        # for a principal-axis shape it gives the largest stable section axis.
        axis = np.asarray(suggestion.in_plane_axis, dtype=float)
        require_three_surface_directions = True
    axis_length = float(np.linalg.norm(axis))
    if not np.isfinite(axis_length) or axis_length <= 1e-12:
        return None
    axis = axis / axis_length

    reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(np.eye(3) @ axis)))]
    in_plane_u = reference - axis * float(np.dot(reference, axis))
    in_plane_length = float(np.linalg.norm(in_plane_u))
    if in_plane_length <= 1e-12:
        return None
    in_plane_u /= in_plane_length
    in_plane_v = np.cross(axis, in_plane_u)

    coordinate = surface.points @ axis
    lower = float(coordinate.min())
    upper = float(coordinate.max())
    axis_span = upper - lower
    if axis_span <= max(diagonal * 1e-10, 1e-12):
        return None

    center = np.asarray(surface.center, dtype=float)
    locator = cKDTree(surface.points)
    offsets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4)
    simplify_tolerance = diagonal * corner_scale_fraction
    cluster_tolerance = min(
        diagonal * 0.02,
        max(diagonal * corner_scale_fraction * 2.0, axis_span * 0.01),
    )
    accepted_points = []
    usable_sections = 0

    for boundary in (lower, upper):
        groups = []
        for offset_index, offset_fraction in enumerate(offsets):
            direction = 1.0 if boundary == lower else -1.0
            section_position = boundary + direction * axis_span * offset_fraction
            origin = center + axis * (section_position - float(np.dot(center, axis)))
            section = surface.slice(normal=axis, origin=origin)
            if section.n_cells == 0 or section.n_points == 0 or section.lines.size == 0:
                continue
            section = section.clean(tolerance=max(diagonal * 1e-8, 1e-10))
            stripper = vtk.vtkStripper()
            stripper.SetInputData(section)
            stripper.JoinContiguousSegmentsOn()
            stripper.Update()
            contours = pv.wrap(stripper.GetOutput())
            packed_lines = np.asarray(contours.lines, dtype=np.int64)
            cursor = 0
            loops = []
            while cursor < packed_lines.size:
                point_count = int(packed_lines[cursor])
                cursor += 1
                ids = packed_lines[cursor : cursor + point_count]
                cursor += point_count
                if point_count < 4:
                    continue
                coordinates = np.asarray(contours.points[ids], dtype=float)
                if float(np.linalg.norm(coordinates[0] - coordinates[-1])) > diagonal * 1e-5:
                    continue
                coordinates = coordinates[:-1]
                if len(coordinates) < 3:
                    continue
                projected = np.column_stack((coordinates @ in_plane_u, coordinates @ in_plane_v))
                area = 0.5 * abs(
                    float(
                        np.dot(projected[:, 0], np.roll(projected[:, 1], -1))
                        - np.dot(projected[:, 1], np.roll(projected[:, 0], -1))
                    )
                )
                if area > diagonal * diagonal * 1e-10:
                    loops.append((area, coordinates, projected))

            if not loops:
                continue
            usable_sections += 1
            # Keep the outer perimeter. Tiny loops often come from holes, mesh
            # seams, or isolated surface fragments near an extremum.
            _area, coordinates, projected = max(loops, key=lambda loop: loop[0])
            simplified = _closed_rdp_indices(projected, simplify_tolerance)
            if len(simplified) < 3:
                continue

            for corner_index, point_index in enumerate(simplified):
                previous = projected[simplified[corner_index - 1]] - projected[point_index]
                following = projected[simplified[(corner_index + 1) % len(simplified)]] - projected[point_index]
                denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
                if denominator <= 1e-12:
                    continue
                angle = float(
                    np.degrees(
                        np.arccos(
                            np.clip(float(np.dot(previous, following)) / denominator, -1.0, 1.0)
                        )
                    )
                )
                if 180.0 - angle < turn_angle_degrees:
                    continue

                slice_point = coordinates[point_index]
                extended = slice_point + axis * (boundary - float(np.dot(slice_point, axis)))
                projected_corner = np.array(
                    [float(np.dot(extended, in_plane_u)), float(np.dot(extended, in_plane_v))]
                )
                group = next(
                    (
                        item
                        for item in groups
                        if float(np.linalg.norm(item["projected"] - projected_corner)) < cluster_tolerance
                    ),
                    None,
                )
                if group is None:
                    groups.append(
                        {
                            "projected": projected_corner,
                            "projected_samples": [projected_corner],
                            "levels": {offset_index},
                            "extended": [extended],
                        }
                    )
                else:
                    group["levels"].add(offset_index)
                    group["extended"].append(extended)
                    group["projected_samples"].append(projected_corner)
                    group["projected"] = np.mean(group["projected_samples"], axis=0)

        for group in groups:
            if len(group["levels"]) < 2:
                continue
            target = np.mean(np.asarray(group["extended"], dtype=float), axis=0)
            nearest_distance, nearest_id = locator.query(target)
            if float(nearest_distance) > cluster_tolerance:
                continue
            candidate = surface.points[int(nearest_id)].copy()
            if require_three_surface_directions and not has_three_surface_directions(candidate):
                continue
            accepted_points.append(candidate)

    if usable_sections == 0:
        return None
    if not accepted_points:
        return np.empty((0, 3), dtype=float)

    unique_points = []
    for point in accepted_points:
        if all(float(np.linalg.norm(point - existing)) > cluster_tolerance for existing in unique_points):
            unique_points.append(point)
    return np.asarray(unique_points, dtype=float)


def detect_surface_corners(
    mesh: pv.PolyData,
    *,
    feature_angle_degrees: float = 60.0,
    turn_angle_degrees: float = 60.0,
    corner_scale_fraction: float = 0.005,
) -> np.ndarray:
    """Return persistent sharp corners on model-end section perimeters.

    Watertight models are sliced near both ends of a shape-specific axis. The
    largest closed perimeter is simplified, and only turns that persist across
    multiple nearby slices are kept. Candidates are snapped back to vertices
    on the original surface. Open or incomplete meshes use the plate-outline
    or feature-edge fallback; generic feature junctions also need three
    distinct local surface directions to avoid marking a point along a ridge.
    """
    if mesh is None or mesh.n_points < 3 or mesh.n_cells == 0:
        raise CornerDetectionError("Load a surface mesh with faces before detecting corners.")
    if not np.isfinite(feature_angle_degrees) or not 0.0 < feature_angle_degrees < 180.0:
        raise CornerDetectionError("The surface feature angle must be between 0 and 180 degrees.")
    if not np.isfinite(turn_angle_degrees) or not 0.0 < turn_angle_degrees < 180.0:
        raise CornerDetectionError("The corner turn angle must be between 0 and 180 degrees.")
    if not np.isfinite(corner_scale_fraction) or not 0.0 < corner_scale_fraction < 0.5:
        raise CornerDetectionError("The corner scale must be between 0 and 50 percent of the mesh size.")

    surface = mesh.triangulate().clean()
    if surface.n_cells == 0:
        raise CornerDetectionError("The mesh does not contain usable surface faces.")
    spans = np.ptp(surface.points, axis=0)
    diagonal = float(np.linalg.norm(spans))
    if not np.isfinite(diagonal) or diagonal <= 1e-12:
        raise CornerDetectionError("The mesh is too small or degenerate to detect corners.")

    suggestion = None
    try:
        suggestion = estimate_auto_realign(
            surface.points,
            triangulated_faces(surface),
            preferred_normal=np.array([1.0, 0.0, 0.0]),
            preferred_in_plane=np.array([0.0, 1.0, 0.0]),
        )
    except (AutoRealignError, ValueError, RuntimeError):
        pass

    surface_vertex_locator = None
    vertex_normal_offsets = None
    ordered_vertex_normals = None

    def has_three_surface_directions(feature_point: np.ndarray) -> bool:
        """Reject edge-network junctions that do not meet three face planes."""
        nonlocal surface_vertex_locator, vertex_normal_offsets, ordered_vertex_normals
        if surface_vertex_locator is None:
            triangles = triangulated_faces(surface)
            triangle_points = surface.points[triangles]
            triangle_normals = np.cross(
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 0],
            )
            lengths = np.linalg.norm(triangle_normals, axis=1)
            valid = np.isfinite(lengths) & (lengths > 1e-12)
            if not np.any(valid):
                return False
            triangles = triangles[valid]
            triangle_normals = triangle_normals[valid] / lengths[valid, None]
            vertex_ids = triangles.ravel()
            face_ids = np.repeat(np.arange(len(triangles), dtype=np.int64), 3)
            order = np.argsort(vertex_ids, kind="stable")
            counts = np.bincount(vertex_ids, minlength=surface.n_points)
            vertex_normal_offsets = np.concatenate(([0], np.cumsum(counts)))
            ordered_vertex_normals = triangle_normals[face_ids[order]]
            surface_vertex_locator = cKDTree(surface.points)

        vertex_id = int(surface_vertex_locator.query(feature_point)[1])
        start = int(vertex_normal_offsets[vertex_id])
        end = int(vertex_normal_offsets[vertex_id + 1])
        local_normals = ordered_vertex_normals[start:end]
        normal_alignment_limit = float(np.cos(np.deg2rad(45.0)))
        distinct_normals = []
        for normal in local_normals:
            if all(
                abs(float(np.dot(normal, existing))) < normal_alignment_limit
                for existing in distinct_normals
            ):
                distinct_normals.append(normal)
                if len(distinct_normals) >= 3:
                    return True
        return False

    slice_corners = _slice_perimeter_corners(
        surface,
        suggestion,
        turn_angle_degrees=turn_angle_degrees,
        corner_scale_fraction=corner_scale_fraction,
        diagonal=diagonal,
        has_three_surface_directions=has_three_surface_directions,
    )
    if slice_corners is not None:
        if len(slice_corners) == 0:
            return slice_corners
        order = np.lexsort((slice_corners[:, 2], slice_corners[:, 1], slice_corners[:, 0]))
        return slice_corners[order]

    plate_corners = _plate_outline_corners(
        surface,
        suggestion,
        turn_angle_degrees=turn_angle_degrees,
        corner_scale_fraction=corner_scale_fraction,
    )
    if plate_corners is not None:
        order = np.lexsort((plate_corners[:, 2], plate_corners[:, 1], plate_corners[:, 0]))
        return plate_corners[order]

    feature_edges = surface.extract_feature_edges(
        feature_angle=float(feature_angle_degrees),
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=True,
    )
    feature_edges = feature_edges.clean(tolerance=max(diagonal * 1e-7, 1e-9))
    packed_lines = np.asarray(feature_edges.lines, dtype=np.int64)
    if feature_edges.n_points == 0 or packed_lines.size == 0:
        return np.empty((0, 3), dtype=float)

    positive_spans = spans[spans > max(diagonal * 1e-12, 1e-12)]
    smallest_span = float(positive_spans.min()) if positive_spans.size else diagonal
    support_distance = min(diagonal * corner_scale_fraction, smallest_span * 0.25)

    neighbors: list[Set[int]] = [set() for _ in range(feature_edges.n_points)]
    cursor = 0
    while cursor < packed_lines.size:
        point_count = int(packed_lines[cursor])
        cursor += 1
        point_ids = packed_lines[cursor : cursor + point_count]
        cursor += point_count
        for first, second in zip(point_ids[:-1], point_ids[1:]):
            first_id = int(first)
            second_id = int(second)
            if first_id == second_id:
                continue
            neighbors[first_id].add(second_id)
            neighbors[second_id].add(first_id)

    candidate_ids = []
    points = np.asarray(feature_edges.points, dtype=float)
    direction_alignment_limit = float(np.cos(np.deg2rad(30.0)))

    def trace_feature_arm(start_id: int, origin_id: int) -> tuple[np.ndarray, float]:
        previous_id = origin_id
        current_id = start_id
        path_length = float(np.linalg.norm(points[current_id] - points[origin_id]))
        visited = {origin_id}
        while path_length < support_distance and current_id not in visited:
            visited.add(current_id)
            if len(neighbors[current_id]) != 2:
                break
            next_ids = neighbors[current_id] - {previous_id}
            if len(next_ids) != 1:
                break
            next_id = next(iter(next_ids))
            segment_length = float(np.linalg.norm(points[next_id] - points[current_id]))
            if segment_length <= 1e-12:
                break
            previous_id, current_id = current_id, next_id
            path_length += segment_length

        direction = points[current_id] - points[origin_id]
        direction_length = float(np.linalg.norm(direction))
        if direction_length <= 1e-12:
            return np.zeros(3, dtype=float), path_length
        return direction / direction_length, path_length

    has_open_boundary = int(surface.n_open_edges) > 0
    for point_id, adjacent in enumerate(neighbors):
        degree = len(adjacent)
        if degree >= 3:
            directions = []
            for neighbor_id in sorted(adjacent):
                direction, arm_length = trace_feature_arm(neighbor_id, point_id)
                if arm_length >= support_distance:
                    directions.append((arm_length, neighbor_id, direction))
            directions.sort(key=lambda item: (-item[0], item[1]))
            distinct_axes = []
            for _length, _neighbor_id, direction in directions:
                if all(
                    abs(float(np.dot(direction, existing_axis))) < direction_alignment_limit
                    for existing_axis in distinct_axes
                ):
                    distinct_axes.append(direction)
            if len(distinct_axes) >= 3 and has_three_surface_directions(points[point_id]):
                candidate_ids.append(point_id)
            continue
        if degree != 2 or not has_open_boundary:
            continue

        first_id, second_id = sorted(adjacent)
        first_direction, first_length = trace_feature_arm(first_id, point_id)
        second_direction, second_length = trace_feature_arm(second_id, point_id)
        if min(first_length, second_length) < support_distance:
            continue
        cosine = float(
            np.clip(
                np.dot(first_direction, second_direction),
                -1.0,
                1.0,
            )
        )
        direction_angle = float(np.degrees(np.arccos(cosine)))
        turn_angle = 180.0 - direction_angle
        if turn_angle >= turn_angle_degrees:
            candidate_ids.append(point_id)

    if not candidate_ids:
        return np.empty((0, 3), dtype=float)

    candidates = points[np.asarray(candidate_ids, dtype=np.int64)].copy()
    order = np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0]))
    return candidates[order]
