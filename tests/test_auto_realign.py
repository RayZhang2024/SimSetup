from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import trimesh

from auto_realign import AutoRealignError, estimate_auto_realign


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"


def rotation_matrix() -> np.ndarray:
    ax, ay, az = np.deg2rad([23.0, -31.0, 47.0])
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]]
    )
    ry = np.array(
        [[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]]
    )
    rz = np.array(
        [[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]]
    )
    return rz @ ry @ rx


def estimate(mesh: trimesh.Trimesh, preferred_normal=None, preferred_in_plane=None):
    return estimate_auto_realign(
        mesh.vertices,
        mesh.faces,
        preferred_normal=np.eye(3)[0] if preferred_normal is None else preferred_normal,
        preferred_in_plane=np.eye(3)[1] if preferred_in_plane is None else preferred_in_plane,
    )


class AutoRealignTests(unittest.TestCase):
    def test_rotated_plate_recovers_thin_normal_and_long_in_plane_axis(self) -> None:
        mesh = trimesh.creation.box(extents=[16.0, 8.0, 0.4])
        rotation = rotation_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [12.0, -7.0, 4.5]
        mesh.apply_transform(transform)

        suggestion = estimate(mesh)

        np.testing.assert_allclose(suggestion.origin, [12.0, -7.0, 4.5], atol=1e-10)
        self.assertEqual(suggestion.profile, "plate")
        self.assertGreater(abs(float(np.dot(suggestion.plane_normal, rotation[:, 2]))), 0.995)
        self.assertGreater(abs(float(np.dot(suggestion.in_plane_axis, rotation[:, 0]))), 0.995)
        self.assertAlmostEqual(float(np.dot(suggestion.plane_normal, suggestion.in_plane_axis)), 0.0, places=10)

    def test_rotated_elongated_rectangular_sample_recovers_long_axis(self) -> None:
        mesh = trimesh.creation.box(extents=[4.0, 2.0, 24.0])
        rotation = rotation_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [-3.0, 8.0, 1.0]
        mesh.apply_transform(transform)

        suggestion = estimate(mesh)

        self.assertEqual(suggestion.profile, "elongated tube or pipe")
        self.assertGreater(abs(float(np.dot(suggestion.in_plane_axis, rotation[:, 2]))), 0.995)
        self.assertAlmostEqual(float(np.dot(suggestion.plane_normal, suggestion.in_plane_axis)), 0.0, places=10)

    def test_symmetric_plate_uses_reference_for_ambiguous_roll(self) -> None:
        mesh = trimesh.creation.box(extents=[10.0, 10.0, 0.4])
        rotation = rotation_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        mesh.apply_transform(transform)
        preferred = rotation @ np.array([0.0, 1.0, 0.0])

        suggestion = estimate(mesh, preferred_in_plane=preferred)

        self.assertTrue(any("in-plane roll" in note for note in suggestion.warnings))
        self.assertGreater(abs(float(np.dot(suggestion.in_plane_axis, preferred))), 0.999)

    def test_symmetric_mesh_is_rejected_as_ambiguous(self) -> None:
        mesh = trimesh.creation.box(extents=[8.0, 8.0, 8.0])

        with self.assertRaisesRegex(AutoRealignError, "too symmetric"):
            estimate(mesh)

    def test_round_pipe_recovers_axis_and_reports_ambiguous_roll(self) -> None:
        mesh = trimesh.creation.cylinder(radius=20.0, height=40.0, sections=48)
        rotation = rotation_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        mesh.apply_transform(transform)

        suggestion = estimate(mesh)

        self.assertEqual(suggestion.profile, "axisymmetric tube")
        self.assertGreater(abs(float(np.dot(suggestion.in_plane_axis, rotation[:, 2]))), 0.99)
        self.assertTrue(any("roll uses the current frame" in note for note in suggestion.warnings))

    def test_area_weighted_result_is_unchanged_by_triangle_subdivision(self) -> None:
        mesh = trimesh.creation.box(extents=[11.0, 5.0, 0.5])
        rotation = rotation_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        mesh.apply_transform(transform)
        original = estimate(mesh)

        points = mesh.vertices.tolist()
        faces = mesh.faces.tolist()
        a, b, c = faces.pop(0)
        center_index = len(points)
        points.append(np.mean(np.asarray(points)[[a, b, c]], axis=0).tolist())
        faces.extend([[a, b, center_index], [b, c, center_index], [c, a, center_index]])
        refined = estimate_auto_realign(
            np.asarray(points),
            np.asarray(faces),
            preferred_normal=np.eye(3)[0],
            preferred_in_plane=np.eye(3)[1],
        )

        np.testing.assert_allclose(refined.eigenvalues, original.eigenvalues, rtol=1e-12, atol=1e-12)
        self.assertGreater(abs(float(np.dot(refined.plane_normal, original.plane_normal))), 0.999999)
        self.assertGreater(abs(float(np.dot(refined.in_plane_axis, original.in_plane_axis))), 0.999999)

    def test_axis_signs_are_stable_when_triangle_winding_is_reversed(self) -> None:
        mesh = trimesh.creation.box(extents=[14.0, 7.0, 0.6])
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix()
        mesh.apply_transform(transform)
        original = estimate(mesh)
        reversed_faces = estimate_auto_realign(
            mesh.vertices,
            mesh.faces[:, ::-1],
            preferred_normal=np.eye(3)[0],
            preferred_in_plane=np.eye(3)[1],
        )

        np.testing.assert_allclose(reversed_faces.plane_normal, original.plane_normal, atol=1e-12)
        np.testing.assert_allclose(reversed_faces.in_plane_axis, original.in_plane_axis, atol=1e-12)

    def test_repository_pipe_stl_has_a_stable_longitudinal_axis(self) -> None:
        mesh = trimesh.load(MODEL / "2_thin_pipe.stl", force="mesh")
        suggestion = estimate(mesh)

        self.assertEqual(suggestion.profile, "axisymmetric tube")
        self.assertGreater(abs(float(np.dot(suggestion.in_plane_axis, np.array([0.0, 1.0, 0.0])))), 0.99)
        self.assertTrue(np.isfinite(suggestion.origin).all())

    def test_repository_plate_stl_is_recognized(self) -> None:
        mesh = trimesh.load(MODEL / "big5.stl", force="mesh")
        suggestion = estimate(mesh)

        self.assertEqual(suggestion.profile, "plate")
        self.assertEqual(suggestion.warnings, ())

    def test_repository_t_weld_stl_returns_valid_orthogonal_suggestion(self) -> None:
        mesh = trimesh.load(MODEL / "T_weld.stl", force="mesh")
        suggestion = estimate(mesh)

        self.assertTrue(np.isfinite(suggestion.origin).all())
        self.assertAlmostEqual(float(np.linalg.norm(suggestion.plane_normal)), 1.0, places=10)
        self.assertAlmostEqual(float(np.linalg.norm(suggestion.in_plane_axis)), 1.0, places=10)
        self.assertAlmostEqual(float(np.dot(suggestion.plane_normal, suggestion.in_plane_axis)), 0.0, places=10)
        self.assertTrue(any("does not clearly match" in note for note in suggestion.warnings))


if __name__ == "__main__":
    unittest.main()
