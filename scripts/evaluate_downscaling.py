#!/usr/bin/env python3
"""Quantify the learned super-resolution step against plain interpolation.

    python scripts/evaluate_downscaling.py --factor 8 --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from twr.config import OUTPUT_DIR  # noqa: E402
from twr.downscale import (  # noqa: E402
    ResidualSuperResolver,
    bilinear_upsample,
    evaluate_downscaling,
    make_paired_fields,
)


def plot_example(factor: int, fine_size: int, seed: int, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coarse, fine, terrain = make_paired_fields(
        n_samples=40, fine_size=fine_size, factor=factor, seed=seed
    )
    model = ResidualSuperResolver(factor).fit(coarse[:32], fine[:32], terrain)
    index = int(np.argmax(fine[32:].max(axis=(1, 2)))) + 32
    interpolated = bilinear_upsample(coarse[index], factor)
    predicted = model.predict(coarse[index : index + 1])[0]

    panels = [
        (coarse[index], f"coarse input ({coarse.shape[1]}x{coarse.shape[2]})"),
        (interpolated, "bilinear interpolation"),
        (predicted, "learned super-resolution"),
        (fine[index], f"target ({fine_size}x{fine_size})"),
    ]
    vmax = float(max(fine[index].max(), predicted.max()))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1))
    for axis, (field, title) in zip(axes, panels, strict=True):
        image = axis.imshow(field, vmin=0, vmax=vmax, cmap="YlGnBu", interpolation="nearest")
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.8, label="precipitation (arbitrary units)")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "figures" / "downscaling_example.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--factor", type=int, default=8, help="upsampling factor")
    parser.add_argument("--fine-size", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=48)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    metrics = evaluate_downscaling(
        n_train=args.n_train,
        n_test=args.n_test,
        fine_size=args.fine_size,
        factor=args.factor,
        seed=args.seed,
    )

    coarse_size = args.fine_size // args.factor
    print(f"Downscaling {coarse_size}x -> {args.fine_size}x (factor {args.factor})")
    print(f"  bilinear interpolation RMSE : {metrics.rmse_interpolated:.4f}")
    print(f"  learned residual model RMSE : {metrics.rmse_model:.4f}")
    print(f"  skill score vs interpolation: {metrics.skill_score:+.1%}")
    print(f"  test fields                 : {metrics.n_test}")

    payload = {
        **metrics.as_dict(),
        "factor": args.factor,
        "fine_size": args.fine_size,
        "note": (
            "Synthetic paired fields. Stands in for NLDAS-3 class forcing downscaled "
            "toward CONUS404 class reanalysis. A ridge regression on patch features "
            "replaces the CNN so the demo runs without a GPU; the interface is the same."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "downscaling_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")

    if args.plot:
        figure = plot_example(args.factor, args.fine_size, args.seed, args.output_dir)
        print(f"wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
