# Raw ENGIN-X NXS calibration and focusing

## Scope and convention

The raw workflow supports the supplied ENGIN-X histogram NeXus layout at
`/raw_data_1/detector_1/counts` with dimensions period x spectrum x TOF bin.
Event-mode NeXus and native ISIS `.raw` files are not supported. Mantid is not
used.

The bank calibration convention is:

```text
TOF (microsecond) = DIFC (microsecond/angstrom) * d (angstrom) + TZERO (microsecond)
```

The accepted raw-workflow calibration is linear. A quadratic whole-pattern fit
may be retained as a diagnostic, but `DIFA` is not applied without independent
model-selection evidence.

## Spectrum and bank mapping

Rows are mapped through `spectrum_index`, never row position:

| Bank | ENGIN-X name | Spectrum IDs | NXS polar angle |
| --- | --- | --- | --- |
| 1 | North / +90 degree | 1-1200 | +75 to +105 degrees |
| 2 | South / -90 degree | 1201-2400 | -105 to -75 degrees |
| excluded | transmission | 2405-2513 | 180 degrees |

The supplied files omit spectrum IDs 2401-2404. Both diffraction banks contain
1,200 spectra and the 109 transmission spectra contribute no counts to focusing,
calibration, or reference normalization.

## Pixel focusing

Each diffraction pixel is focused before summation. The relative geometry
coefficient is

```text
geometry_DIFC = 505.5568 * (L1 + L2) * sin(abs(2theta) / 2)
```

where `L2` and `2theta` come from the NXS detector geometry. `L1=50 m` is an
explicit nominal ENGIN-X instrument-profile value. The NXS
`source_detector_distance=0` value is recorded but is not treated as a physical
flight path. The bank-equivalent map is:

```text
TOF_bank = TOF_pixel * median(bank geometry_DIFC) / geometry_DIFC_pixel
```

Mapped histogram edges are overlap-rebinned onto the shared bank edge grid and
then summed. This conserves counts over matching support. Poisson counting
variance is split in proportion to overlap (Poisson thinning); covariance between
pieces of a split source bin is not represented.

Geometry therefore provides relative alignment, not an independent absolute
calibration. The unlabeled `user_table07`-`user_table09` datasets are ignored.
CeO2 peak fitting corrects the bank-level DIFC/TZERO once after focusing.

## Exposure and reference correction

The period-specific `/raw_data_1/periods/proton_charge` value is required in
`uAh` and must be finite and positive. The unsmoothed default correction is:

```text
(C_sample / Q_sample) / (C_reference / Q_reference)
```

Reference bins that are nonfinite, zero, or below the configured weak-signal
threshold are masked. V-Nb can contain structure, so response smoothing is
explicit and disabled by default. Charge normalization does not by itself prove
detector-efficiency or reference-structure correction.

## Calibration persistence

Saved JSON records include file size and SHA-256 identity, run and charge data,
instrument-profile provenance, grouping and masks, focusing convention,
uncertainty approximation, reference-response settings, CeO2 standard, fitted
coefficients and uncertainties, accepted/rejected peaks, residuals, usable TOF
range, processing version, and scientific validation gaps. Reloaded calibration
is applied explicitly and an application counter prevents applying it twice to
the same focused spectrum.

## Current scientific status

The three supplied runs have been processed locally and produce internally
consistent bank-level CeO2 fits. This does not establish independent absolute
accuracy. Remaining gaps are an independently trusted focused reference,
confirmation of the nominal geometry provenance, and separation of V-Nb
structure from detector response.
