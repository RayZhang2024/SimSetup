from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .batch_fitting import SpectrumFitRequest, fit_spectrum


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated diffraction fit.")
    parser.add_argument("request_path", type=Path)
    parser.add_argument("result_path", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    args = parser.parse_args()
    try:
        with args.request_path.open("rb") as handle:
            request: SpectrumFitRequest = pickle.load(handle)
        def save_checkpoint(payload: dict[str, object]) -> None:
            temporary_path = args.checkpoint_path.with_suffix(".tmp")
            with temporary_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_path.replace(args.checkpoint_path)

        result = fit_spectrum(
            request,
            use_gsas_convergence=True,
            checkpoint_callback=save_checkpoint,
        )
        payload = ("result", result)
    except BaseException as exc:
        payload = ("error", f"{type(exc).__name__}: {exc}")
    with args.result_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
