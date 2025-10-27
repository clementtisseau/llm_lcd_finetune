from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend for SSH/servers
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def load_metrics(jsonl_path: Path) -> pd.DataFrame:
    """Read metrics.jsonl into a pandas DataFrame."""
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {jsonl_path}")

    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                # Skip malformed lines but warn
                print(f"Warning: could not parse line: {line[:120]}... ({e})")

    if not rows:
        raise ValueError("No rows parsed from metrics file.")

    df = pd.DataFrame(rows)

    # Keep expected columns if present, but don't fail if some are missing
    wanted = [
        "ts", "epoch", "global_step", "batch", "split", "tokens_in_batch",
        "loss_token_avg", "ppl", "grad_norm", "lr"
    ]
    cols = [c for c in wanted if c in df.columns] + [c for c in df.columns if c not in wanted]
    df = df[cols]

    # Types
    num_cols = ["global_step", "epoch", "batch", "tokens_in_batch",
                "loss_token_avg", "ppl", "grad_norm", "lr"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")

    # Sort by global_step (fallback to ts if needed)
    if "global_step" in df.columns and df["global_step"].notna().any():
        df = df.sort_values(["global_step", "ts"] if "ts" in df.columns else ["global_step"]).reset_index(drop=True)
    elif "ts" in df.columns:
        df = df.sort_values(["ts"]).reset_index(drop=True)

    return df


def maybe_smooth(series: pd.Series, smooth: Optional[int]) -> pd.Series:
    """Optional rolling-mean smoothing (window=N steps)."""
    if smooth and smooth > 1:
        return series.rolling(window=int(smooth), min_periods=1).mean()
    return series


def plot_series(x, y, title: str, ylabel: str, out_path: Path, dpi: int = 150):
    """
    Single-metric, single-figure plot using matplotlib (no seaborn, one chart per figure).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(x, y)  # do not set explicit colors/styles (keeps defaults)
    plt.title(title)
    plt.xlabel("Optimizer step (global_step)")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def make_plots(df: pd.DataFrame, outdir: Path, smooth: Optional[int] = None, dpi: int = 150):
    """
    Create all requested plots:
      Train: loss, ppl, grad_norm, lr
      Eval:  loss, ppl, lr
      X-axis: global_step
    """
    if "global_step" not in df.columns:
        raise ValueError("Missing 'global_step' in metrics. Make sure you log it.")

    outdir = Path(outdir)

    # TRAIN
    train = df[df.get("split", "") == "train"]
    if not train.empty:
        x = train["global_step"]
        if "loss_token_avg" in train.columns:
            y = maybe_smooth(train["loss_token_avg"], smooth)
            plot_series(x, y, "Train loss/token vs steps", "Loss/token", outdir / "train_loss.png", dpi=dpi)
        if "ppl" in train.columns:
            y = maybe_smooth(train["ppl"], smooth)
            plot_series(x, y, "Train perplexity vs steps", "Perplexity", outdir / "train_ppl.png", dpi=dpi)
        if "grad_norm" in train.columns:
            y = maybe_smooth(train["grad_norm"], smooth)
            plot_series(x, y, "Train grad norm vs steps", "Gradient L2 norm", outdir / "train_grad_norm.png", dpi=dpi)
        if "lr" in train.columns:
            y = maybe_smooth(train["lr"], smooth)
            plot_series(x, y, "Train learning rate vs steps", "LR", outdir / "train_lr.png", dpi=dpi)
    else:
        print("No train rows found.")

    # EVAL
    eval_df = df[df.get("split", "") == "eval"]
    if not eval_df.empty:
        x = eval_df["global_step"]
        if "loss_token_avg" in eval_df.columns:
            y = maybe_smooth(eval_df["loss_token_avg"], smooth)
            plot_series(x, y, "Eval loss/token vs steps", "Loss/token", outdir / "eval_loss.png", dpi=dpi)
        if "ppl" in eval_df.columns:
            y = maybe_smooth(eval_df["ppl"], smooth)
            plot_series(x, y, "Eval perplexity vs steps", "Perplexity", outdir / "eval_ppl.png", dpi=dpi)
        if "lr" in eval_df.columns:
            y = maybe_smooth(eval_df["lr"], smooth)
            plot_series(x, y, "Eval learning rate vs steps", "LR", outdir / "eval_lr.png", dpi=dpi)
    else:
        print("No eval rows found.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot training/eval metrics from JSONL (saves PNGs).")
    parser.add_argument("--metrics", type=Path, default="/scratch/ctisseau/finetuned-models/Qwen-Qwen2.5-0.5B_classical_sft_c5c0c8e1a308a4b6f52a/metrics.jsonl", help="Path to metrics.jsonl")
    parser.add_argument("--outdir", type=Path, default="/home/ctisseau/llm_lcd_finetune/RPN_generation/metrics", help="Directory to save PNGs")
    parser.add_argument("--smooth", type=int, default=0, help="Rolling window (in steps) for smoothing; 0/1 disables")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI (default 150)")
    args = parser.parse_args()

    df = load_metrics(args.metrics)
    make_plots(df, outdir=args.outdir, smooth=args.smooth, dpi=args.dpi)
    print(f"Saved PNGs to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()