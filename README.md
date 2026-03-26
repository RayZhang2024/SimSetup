# Sample Placement Fit Prototype

## Environment setup

Create and use the repo-local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

All commands below should be run from that virtual environment.

The GUI implementation targets this environment and uses:

- `PyQt5` for the desktop application shell
- `pyvista` and `pyvistaqt` for embedded 3D rendering
- `trimesh` for STL and mesh loading

This workspace now contains a standalone tool that reconstructs how a physical sample is mounted on the stage from point correspondences.

## GUI application

Launch the desktop application from the repo-local virtual environment:

```powershell
python .\gui_app.py
```

Optional startup inputs:

```powershell
python .\gui_app.py --csv .\example_measurements.csv --mesh .\sample.stl
```

The GUI supports:

- loading or editing measurement rows
- loading an STL or other mesh format supported by `trimesh`
- fitting the model-to-stage transform
- inspecting the beam, theodolite view, stage, and fitted sample in 3D
- jumping the camera to `Iso`, axis-aligned, and theodolite views
- switching between perspective and parallel projection
- moving the live stage pose so the selected fitted point lands at the pivot
- computing a detector path-length map for a collimated beam over the slit area
- exporting the detector map as `.fits` or `.tiff`
- exporting the fit result to JSON

## What the tool solves

You measure several physical feature points on the mounted sample.
For each point you record:

- the point coordinates in the CAD or STL model
- the stage readout `x, y, z` when that point is brought to the fixed measurement point

From those measurements, the tool computes the rigid transform that places the digital sample model into the stage frame of the digital twin.

## Geometry assumption

The fitter now uses the corrected stage model:

- the beam and theodolite intersection is a fixed measurement point
- `x, y, z` readouts are translations in the stage's own local frame
- `omega` rotates the whole stage frame about the fixed measurement point around the vertical `z` axis
- the sample is rigidly fixed to the stage

During point measurement, when a feature point is moved onto the pivot, the stage-local readout is the negative of that feature point's coordinate in the stage frame:

```text
point_in_stage_frame = -stage_readout
```

So the fit only needs paired `model xyz` and `stage xyz` values. `Omega` is not used in the measurement table.

## CSV format

Use a CSV file with this header:

```csv
label,model_x,model_y,model_z,stage_x,stage_y,stage_z
```

Each row is one measured feature point.

An example file is included at [example_measurements.csv](/c:/Users/fov76845/GitProjects/SimSetup/example_measurements.csv).

## Usage

Run the synthetic verification case:

```powershell
python .\placement_fit.py --demo
```

Fit a real measurement file, assuming the fixed measurement point is the world origin:

```powershell
python .\placement_fit.py .\measurements.csv
```

Run the included example dataset:

```powershell
python .\placement_fit.py .\example_measurements.csv
```

## Outputs

The tool prints:

- the fitted translation
- the fitted rotation as a quaternion and Euler ZYX angles
- the 4x4 homogeneous transform matrix
- RMS and max residual error
- per-point residuals

The optional JSON file contains the same transform plus the reconstructed stage-frame position for each measured point.

## How to use the result in a digital twin

Let `T_model_to_stage` be the fitted transform from this tool.

Then for any live stage readout `(x, y, z, omega)`, the sample model position in setup coordinates is:

```text
point_world = pivot_world + Rz(omega) * (point_stage + stage_readout_local)
```

So the live `x/y/z` translation is applied inside the rotated stage frame, not in the fixed world frame.
