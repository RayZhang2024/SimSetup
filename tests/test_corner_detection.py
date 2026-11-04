from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pyvista as pv

from corner_detection import CornerDetectionError, detect_surface_corners


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"


class SurfaceCornerDetectionTests(unittest.TestCase):
    def test_closed_box_returns_its_eight_surface_corners(self) -> None:
        mesh = pv.Cube().triangulate().clean()

        corners = detect_surface_corners(mesh)

        expected = {
            (x, y, z)
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
        }
        self.assertEqual(len(corners), 8)
        self.assertEqual({tuple(np.round(point, 6)) for point in corners}, expected)

    def test_rectangular_solid_returns_only_vertices_at_axis_extremes(self) -> None:
        mesh = pv.Cube(x_length=10.0, y_length=6.0, z_length=2.0).triangulate().clean()

        corners = detect_surface_corners(mesh)

        expected = {
            (x, y, z)
            for x in (-5.0, 5.0)
            for y in (-3.0, 3.0)
            for z in (-1.0, 1.0)
        }
        self.assertEqual({tuple(np.round(point, 6)) for point in corners}, expected)

    def test_open_rectangular_surface_finds_bent_boundary_corners(self) -> None:
        mesh = pv.Plane(i_size=12.0, j_size=6.0, i_resolution=6, j_resolution=4).triangulate().clean()

        corners = detect_surface_corners(mesh)

        self.assertEqual(len(corners), 4)
        self.assertEqual(
            {tuple(np.round(point, 6)) for point in corners},
            {(-6.0, -3.0, 0.0), (-6.0, 3.0, 0.0), (6.0, -3.0, 0.0), (6.0, 3.0, 0.0)},
        )

    def test_smooth_pipe_rims_are_not_misreported_as_corners(self) -> None:
        mesh = pv.Cylinder(radius=1.0, height=5.0, resolution=48, capping=True).triangulate().clean()

        corners = detect_surface_corners(mesh)

        self.assertEqual(corners.shape, (0, 3))

    def test_rotated_cube_keeps_eight_corners_on_the_surface(self) -> None:
        mesh = pv.Cube().triangulate().clean()
        mesh.rotate_x(19.0, inplace=True)
        mesh.rotate_y(-33.0, inplace=True)
        mesh.rotate_z(52.0, inplace=True)

        corners = detect_surface_corners(mesh)

        self.assertEqual(len(corners), 8)
        for point in corners:
            closest = mesh.points[mesh.find_closest_point(point)]
            np.testing.assert_allclose(closest, point, atol=1e-9)

    def test_repository_t_weld_stl_returns_surface_corner_candidates(self) -> None:
        mesh = pv.read(MODEL / "T_weld.stl").triangulate().clean()

        corners = detect_surface_corners(mesh)

        self.assertEqual(len(corners), 16)
        self.assertTrue(np.isin(np.round(corners[:, 2], 6), (0.0, 100.0)).all())

    def test_repository_pipe_stl_has_no_artificial_polygonal_corners(self) -> None:
        mesh = pv.read(MODEL / "2_thin_pipe.stl").triangulate().clean()

        corners = detect_surface_corners(mesh)

        self.assertEqual(corners.shape, (0, 3))

    def test_repository_plate_models_return_four_outline_corners(self) -> None:
        for name in (
            "big2.stl",
            "big2_local.stl",
            "big5.stl",
            "big5_local_frame.stl",
            "big5_local_frame_2.stl",
        ):
            with self.subTest(model=name):
                mesh = pv.read(MODEL / name).triangulate().clean()

                corners = detect_surface_corners(mesh)

                self.assertEqual(len(corners), 4)
                self.assertTrue(np.isfinite(corners).all())
                for point in corners:
                    closest = mesh.points[mesh.find_closest_point(point)]
                    np.testing.assert_allclose(closest, point, atol=1e-8)

    def test_repository_bent_body_keeps_only_stable_end_corners(self) -> None:
        mesh = pv.read(MODEL / "bent2_1_local_frame.stl").triangulate().clean()

        corners = detect_surface_corners(mesh)

        self.assertEqual(len(corners), 1)
        closest = mesh.points[mesh.find_closest_point(corners[0])]
        np.testing.assert_allclose(closest, corners[0], atol=1e-8)

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaises(CornerDetectionError):
            detect_surface_corners(pv.Cube(), feature_angle_degrees=180.0)


if __name__ == "__main__":
    unittest.main()
