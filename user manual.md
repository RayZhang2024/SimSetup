# SimSetup Placement Tab User Manual

This manual covers only the Placement tab in SimSetup. The Placement tab is used to load a sample model, fit the model position from measured feature points, inspect the sample in the instrument view, estimate path lengths and exposure values, and create placement output files.

## Start the App

From the project folder, start the GUI with:

```powershell
.\.venv\Scripts\python .\gui_app.py
```

You can also load files at startup:

```powershell
.\.venv\Scripts\python .\gui_app.py --csv .\example_measurements.csv --mesh .\model\2_thin_pipe.stl
```

## Placement Tab Layout

The Placement tab contains:

- A left control panel for files, setup values, manual placement, and live stage pose.
- A 3D instrument view for the sample, stage, beam, detectors, points, and path overlays.
- A measurement table for fitted feature-point correspondences.
- A prediction table for target model points, calculated stage readouts, path lengths, and exposure estimates.

## Basic Workflow

1. Load a sample mesh or supported SAT model.
2. Enter the setup geometry values.
3. Load or enter measured feature-point correspondences.
4. Run Fit placement.
5. Check the residuals and 3D view.
6. Load or enter prediction points.
7. Generate stage readouts, path lengths, and exposure estimates.
8. Save the project or export output files.

## Load a Sample Model

Use File > Load mesh/SAT or the Load mesh/SAT toolbar button.

Supported inputs include STL and common mesh formats such as PLY, OBJ, OFF, GLB, and GLTF. Simple polyhedral ACIS SAT files can also load. Curved or complex SAT files must be converted to STL first because the Placement tab needs triangle mesh geometry for ray tracing, clipping, and path-length calculations.

After loading a model, the 3D view shows the sample in the current placement state. If no fit or manual placement is active, the model is shown in its model-coordinate position.

## Setup Geometry

Open the setup/instrument controls from the Placement tab settings if you need to edit full geometry values.

Important values:

- Pivot X/Y/Z: the fixed measurement point in world coordinates. This is the point where the beam, gauge volume, and measurement target are centered.
- Theodolite X/Y/Z: the theodolite or sight-line origin used by the theodolite view and sight-line overlay.
- Slit X/Y/Z, slit width, slit height: the imaging beam slit position and aperture.
- Beam length: the visual and ray-tracing extent for the incoming beam.
- Detector dimensions: the imaging detector display and map extents.
- Collimator: the gauge-volume depth used for exposure and gauge-volume calculations.
- Material: the material model used for exposure estimates.
- Target uncertainty: the target uncertainty used when a calibrated custom material profile is selected.

Changing setup values updates the 3D view and invalidates detector/path calculations that depend on the old geometry.

## Measurement Table

The measurement table is used to fit the model-to-stage transform. Each row is one measured feature point.

Columns:

- Label: point name.
- Model X/Y/Z: point coordinates in the CAD or mesh model coordinate system.
- Stage X/Y/Z: stage readout when that physical point was moved to the pivot.

You can load a CSV with:

```csv
label,model_x,model_y,model_z,stage_x,stage_y,stage_z
```

Use the measurement toolbar to load, save, add, delete, clear, fit, move selected points to the pivot, and control automatic point-to-pivot behavior.

## Fit Placement

After entering at least enough measured point correspondences, click Fit placement.

The app computes the rigid transform from model coordinates to the stage frame. The fit report shows translation, rotation, and residuals. Use the residuals to check whether the entered model points and stage readouts are consistent.

If residuals are unexpectedly large, check:

- Model coordinates are in the same units as the stage readouts.
- Each row pairs the correct model feature with the corresponding measured stage readout.
- Stage X/Y/Z values were recorded when the feature point was at the pivot.
- The mesh is loaded in the expected model coordinate system.

## Manual Sample Placement

Use Manual Sample Placement when you want to bypass the fitted transform or inspect a known transform manually.

Controls:

- Enable manual placement: uses manual values instead of the fitted transform.
- Model->Stage X/Y/Z: manual translation from model coordinates to stage coordinates.
- Local Rot X/Y/Z: manual local rotations.
- Load transform: copies the current fitted transform into the manual controls.

The Live Stage Pose controls still apply on top of either fitted placement or manual placement.

## Live Stage Pose

The Live Stage Pose controls simulate the current stage readout.

- Stage X/Y/Z: current stage translations in the stage-local coordinate frame.
- Omega: current stage rotation about the pivot.
- Reset pose: returns the live stage pose to zero.

Selecting a measurement or prediction row can move that point to the pivot when Auto move on select is enabled. This updates the live stage pose so the selected point is centered at the measurement position.

## 3D View Controls

Use the view toolbar or View menu to inspect the setup.

Common controls:

- Isometric and axis views: jump to standard camera directions.
- Theodolite view: look from the theodolite position toward the pivot.
- Parallel projection: switch between perspective and orthographic view.
- Show Stage, Beam, Gauge Volume, Detectors, Feature Points, Prediction Points, Sample Triad, Sight Line, Diffraction Vectors, and Path Length Overlays: turn scene elements on or off.
- Compute detector map: calculate the imaging detector path-length map for the current setup.
- Export detector map: save the current detector map as FITS or TIFF.

## Prediction Table

The prediction table is used for planned measurement points.

Columns:

- Label: target point name.
- Model X/Y/Z: target point in model coordinates.
- Stage X/Y/Z: calculated stage readout that moves the target point to the pivot.
- Path 1 and Path 2: calculated diffraction path lengths for the two banks.
- UAmp1 and UAmp2: estimated exposure values for the two banks.

Typical prediction workflow:

1. Load or enter prediction model points.
2. Generate stage readouts.
3. Generate path lengths.
4. Estimate UAmp.
5. Create scan file.

Generating stage readouts requires a fitted transform or enabled manual placement. Generating path lengths also requires a loaded mesh and a valid placed sample.

## Exposure Estimates

The Estimate UAmp command uses the current material, collimator, slit width, slit height, and path lengths.

For built-in reference materials, the estimate uses the built-in Rietveld count-time law. For calibrated custom materials, it uses the material acquisition profile and Target uncertainty.

If Estimate UAmp fails, check that Path 1 and Path 2 have already been generated and that the material setup is valid.

## Save and Export

Available Placement outputs:

- Save Project: saves the current setup, mesh, placement, tables, view settings, and profiles to a `.simsetup` project.
- Load Project: restores a saved `.simsetup` project.
- Export Fit JSON: saves the fitted transform and residual report.
- Save measurement CSV: saves measured feature points.
- Save prediction CSV: saves prediction points and generated values.
- Create scan file: creates a scan command file from prediction rows.
- Export detector map: saves the detector path-length map as FITS or TIFF.

Project files embed the current mesh as STL, even if the original model was loaded from another supported format.

## Common Errors

Failed to load mesh:

- The file is not a supported mesh.
- A SAT file contains curved ACIS geometry. Export it as STL from the CAD/source tool.

Fit failed:

- Not enough valid point correspondences.
- Non-numeric table values.
- Mismatched model and stage points.

Move to pivot failed:

- No fitted transform is available and manual placement is disabled.
- The selected row does not contain valid model coordinates.

Path or detector map failed:

- No mesh is loaded.
- Placement has not been fitted or manually enabled.
- The selected point is outside the model or the ray path does not intersect the mesh as expected.

Create scan file failed:

- Prediction stage readouts are missing.
- UAmp values are missing. Run Estimate UAmp first.
