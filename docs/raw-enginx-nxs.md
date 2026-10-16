# Raw ENGIN-X NXS calibration and focusing

## Scope and convention

The raw workflow supports the supplied ENGIN-X histogram NeXus layout at
`/raw_data_1/detector_1/counts` with dimensions period x spectrum x TOF bin.
Event-mode NeXus and native ISIS `.raw` files are not supported. Mantid is not
used.

There are two separate workflows:

1. **Create calibration** reads one open-beam/V-Nb NXS and one CeO2 NXS, processes
   both banks, displays diagnostics, and saves one JSON. A sample is not read.
2. **Analyze samples** reads an active calibration JSON and one or more sample NXS
   files. The original reference and CeO2 files are not required or reopened.

After both banks finish calibration, **Create Calibration** offers three read-only
popups. The open-beam preview overlays the bank-summed charge-normalized response
and a smoothed comparison curve; only the selected smoothing option affects the
stored response. The CeO2 review shows the corrected pattern, matched reflections,
and selectable local Exp-Voigt fits with linear backgrounds and residuals. The
quadratic-fit review plots fitted centres with their TOF uncertainties, the accepted
weighted DIFC/DIFA/TZERO curve, residuals, and peak acceptance status. Opening these
views does not activate or rewrite a calibration.

The bank calibration convention is:

```text
TOF (microsecond) = DIFC (microsecond/angstrom) * d (angstrom)
                  + DIFA (microsecond/angstrom^2) * d^2 (angstrom^2) + TZERO (microsecond)
```

The accepted raw-workflow calibration is a weighted quadratic least-squares fit
of individual CeO2 peak centres using their TOF uncertainties. Peak residuals
and outlier handling use this fitted model; the whole-pattern quadratic fit is
retained only as a diagnostic and its DIFA is not substituted. At least four
accepted peaks with three distinct d-spacings and a well-conditioned fit are
required. Coefficient uncertainties come from the weighted fit covariance,
inflated when the reduced chi-square exceeds one. DIFA=0 is only a peak-matching
seed, not a constraint on the accepted calibration.

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

The calibration JSON stores the target geometry DIFC and every included pixel's
geometry DIFC. Sample focusing reuses that basis and rejects changed geometry;
it does not independently choose a new median. A per-bin detector coverage
fraction is retained. Reference correction is valid only where both the sample
and saved reference have full detector coverage, preventing different pixel
support from biasing the ratio.

Geometry therefore provides relative alignment, not an independent absolute
calibration. The unlabeled `user_table07`-`user_table09` datasets are ignored.
CeO2 peak fitting corrects the bank-level DIFC/DIFA/TZERO once after focusing.

## Exposure and reference correction

The period-specific `/raw_data_1/periods/proton_charge` value is required in
`uAh` and must be finite and positive. Reference response values are stored as
charge-normalized histogram bin integrals in `counts/uAh`. The unsmoothed
correction is:

```text
(C_sample / Q_sample) / (C_reference / Q_reference)
```

When sample and calibration bin edges differ, saved response integrals and
variances are overlap-rebinned. This preserves integrated intensity and avoids
bin-width bias; center interpolation is not used. Nonfinite, nonpositive, or
incompletely covered reference bins are masked. V-Nb can contain structure, so
response smoothing is enabled by default in Create Calibration and can be turned
off. Charge normalization does not by itself prove detector-efficiency or
reference-structure correction.

## Calibration persistence

Saved JSON records include the actual per-bank reference response, bin edges,
units, statistical uncertainties, validity and detector-coverage masks, file
identity, run and charge data, instrument-profile provenance, grouping and pixel
masks, the fixed focusing basis, CeO2 standard, fitted coefficients and
uncertainties, individual-peak acceptance and residuals, usable TOF range,
processing version, and scientific validation gaps. New bundles store fitted
DIFA and its uncertainty; previously saved linear DIFA=0 bundles remain loadable.

Loading another JSON invalidates sample spectra processed with a different
bundle fingerprint. Raw-NXS spectra retain the accepted JSON calibration during
plotting and fitting; the legacy focused-histogram calibration cannot replace it.
An application counter also prevents applying calibration twice.

## Default smoothing: real-run comparison

The default-smoothed calibration was compared with an explicitly unsmoothed
calibration using reference run 377438, CeO2 run 377898, sample run 378089,
the same CeO2 phase (`Hist/CEO2.EXP`), and no pixel masks in either run. Both
banks used the same focusing settings. The quadratic coefficients below are
from accepted individual peak centres; each uncertainty is one standard error
from the weighted fit. DIFA is in `us/angstrom^2`, DIFC in `us/angstrom`, and
TZERO and RMS in `us`.

| Response | Bank | DIFC +/- uncertainty | DIFA +/- uncertainty | TZERO +/- uncertainty | RMS | Valid reference bins | Valid corrected sample bins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoothed (default) | 1 | 18414.6862 +/- 7.5180 | -4.1477 +/- 2.5792 | -14.3816 +/- 5.1562 | 0.6057 | 10186/10186 | 8936/10186 |
| Smoothed (default) | 2 | 18405.1995 +/- 8.9330 | -8.8240 +/- 3.0787 | -25.8746 +/- 6.1014 | 1.1715 | 10186/10186 | 8932/10186 |
| Unsmoothed (comparison) | 1 | 18416.9626 +/- 8.5801 | -4.8453 +/- 2.9417 | -16.3017 +/- 5.8857 | 0.8564 | 10181/10186 | 8936/10186 |
| Unsmoothed (comparison) | 2 | 18405.8610 +/- 9.0400 | -9.1679 +/- 3.1134 | -26.0497 +/- 6.1802 | 1.2102 | 10185/10186 | 8931/10186 |

Each bank and both settings accepted the same 11 reflections: 442, 531, 440,
333, 422, 420, 331, 400, 222, 311, and 220. Reflections 600 and 511 were
rejected as already matched. The full-pattern fit is diagnostic only and was
not substituted for the accepted individual-peak calibration:

| Response | Bank | Full-pattern DIFC | Full-pattern DIFA | Full-pattern TZERO |
| --- | ---: | ---: | ---: | ---: |
| Smoothed (default) | 1 | 18434.5551 | -12.0310 | -21.3430 |
| Smoothed (default) | 2 | 18418.9724 | -14.5393 | -28.9790 |
| Unsmoothed (comparison) | 1 | 18436.3205 | -12.5323 | -22.8131 |
| Unsmoothed (comparison) | 2 | 18419.6349 | -14.8425 | -29.2375 |

Across bins valid in both reference responses, the median absolute smoothed
versus unsmoothed change was 1.74% (bank 1) and 1.78% (bank 2); the 95th
percentiles were 40.17% and 39.37%. The very large maxima (30442% and 9221%)
occur where the unsmoothed response is close to zero, so percentage changes
there are unstable. On matching TOF bins with finite, nonzero corrected sample
intensity in both runs (7655 and 7798 bins), median absolute changes were
1.35% and 1.43%; 95th percentiles were 6.41% and 7.43%, and 99th percentiles
were 17.57% and 20.00%. The medians of corrected sample intensity changed
from 0.55651 to 0.55573 (bank 1) and from 0.80145 to 0.80184 (bank 2).
Thus the bulk sample response changes modestly, but localized differences
are material and should not be interpreted as a detector-efficiency correction
without independent validation. Bank 2 gained one valid corrected sample bin
with smoothing; bank 1's valid-bin count was unchanged.

Each mode's bundle was saved and reloaded before sample analysis. The reloaded
JSON preserved the exact response arrays and bin edges used in calibration,
and recorded `reference_response_settings.smoothed` as `true` or `false`
respectively. These comparison JSON files were temporary and are not committed.

## Current scientific status

The three supplied runs have been processed locally and produce internally
consistent bank-level CeO2 fits. This does not establish independent absolute
accuracy. Remaining gaps are an independently trusted focused reference,
confirmation of the nominal geometry provenance, and separation of V-Nb
structure from detector response. The fitted DIFA is empirical and has not
been independently separated from focusing or geometry error; the different
whole-pattern estimate is retained for comparison, not substituted.
