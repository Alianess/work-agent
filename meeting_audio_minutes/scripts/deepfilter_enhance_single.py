from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from df.enhance import enhance, init_df
from df.io import load_audio, resample, save_audio
from df.model import ModelParams


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DeepFilterNet enhancement for one audio file without DataLoader workers."
    )
    parser.add_argument("audio", help="Input noisy wav file.")
    parser.add_argument("--model-base-dir", required=True, help="Local DeepFilterNet model directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--pf", action="store_true", help="Enable DeepFilterNet post-filter.")
    parser.add_argument("--atten-lim", type=int, default=None, help="Attenuation limit in dB.")
    parser.add_argument("--epoch", default="best", help="Checkpoint epoch.")
    parser.add_argument("--log-level", default="info", help="DeepFilterNet log level.")
    parser.add_argument("--no-suffix", action="store_true", help="Keep original filename.")
    parser.add_argument(
        "--no-delay-compensation",
        dest="compensate_delay",
        action="store_false",
        help="Disable delay compensation.",
    )
    parser.add_argument("--no-df-stage", action="store_true", help="Disable deep filtering stage.")
    parser.set_defaults(compensate_delay=True)
    args = parser.parse_args()

    # Avoid OpenMP over-spawning on local Mac CPUs and avoid sandbox SHM-sensitive workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    audio_path = Path(args.audio).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, df_state, suffix = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        log_level=args.log_level,
        config_allow_defaults=True,
        epoch=args.epoch,
        mask_only=args.no_df_stage,
    )
    suffix = None if args.no_suffix else suffix

    df_sr = ModelParams().sr
    noisy_audio, meta = load_audio(str(audio_path), df_sr, "cpu")
    with torch.no_grad():
        enhanced = enhance(
            model,
            df_state,
            noisy_audio,
            pad=args.compensate_delay,
            atten_lim_db=args.atten_lim,
        )
    enhanced = resample(enhanced.to("cpu"), df_sr, meta.sample_rate)
    save_audio(str(audio_path), enhanced, sr=meta.sample_rate, output_dir=str(output_dir), suffix=suffix, log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
