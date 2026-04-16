"""
Peak Hour Analytics
====================
Reads all CSV logs produced by detect_cpu.py / detect_gpu.py and generates:
  1. Hourly footfall bar chart (outputs/peak_hours.png)
  2. Per-minute timeline (outputs/timeline.png)
  3. Summary printed to console

Usage:
    python analytics.py                        # auto-detect latest log
    python analytics.py --log data/logs/footfall_20240101_120000.csv
    python analytics.py --all                  # merge all logs
"""

import argparse
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_DIR = Path("data/logs")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_logs(log_path: str | None = None, merge_all: bool = False) -> pd.DataFrame:
    if merge_all:
        files = sorted(glob.glob(str(LOG_DIR / "footfall*.csv")))
        if not files:
            raise SystemExit("No log files found in data/logs/")
        print(f"[Analytics] Merging {len(files)} log file(s) …")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    elif log_path:
        df = pd.read_csv(log_path)
    else:
        files = sorted(glob.glob(str(LOG_DIR / "footfall*.csv")))
        if not files:
            raise SystemExit("No log files found in data/logs/  —  run detect_cpu.py first.")
        df = pd.read_csv(files[-1])
        print(f"[Analytics] Using latest log: {files[-1]}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def plot_hourly(df: pd.DataFrame) -> None:
    hourly = df.groupby("hour")["entries_this_minute"].sum().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(hourly.index, hourly.values, color="#4C9BE8", edgecolor="white", linewidth=0.5)

    peak_hour = int(hourly.idxmax())
    bars[peak_hour].set_color("#E84C4C")
    ax.annotate(
        f"Peak: {peak_hour:02d}:00–{peak_hour+1:02d}:00\n({int(hourly[peak_hour])} entries)",
        xy=(peak_hour, hourly[peak_hour]),
        xytext=(peak_hour + 0.5, hourly[peak_hour] + max(hourly.values) * 0.05),
        fontsize=9, color="#E84C4C",
        arrowprops=dict(arrowstyle="->", color="#E84C4C"),
    )

    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Entries", fontsize=11)
    ax.set_title("Cafe Footfall — Hourly Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUTPUT_DIR / "peak_hours.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Analytics] Saved → {out}")


def plot_timeline(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_sorted["timestamp"], df_sorted["entries_this_minute"],
            color="#4C9BE8", linewidth=1.2, alpha=0.8)
    ax.fill_between(df_sorted["timestamp"], df_sorted["entries_this_minute"],
                    alpha=0.15, color="#4C9BE8")

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Entries / min", fontsize=11)
    ax.set_title("Cafe Footfall — Per-Minute Timeline", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    out = OUTPUT_DIR / "timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Analytics] Saved → {out}")


def print_summary(df: pd.DataFrame) -> None:
    total = df["entries_this_minute"].sum()
    hourly = df.groupby("hour")["entries_this_minute"].sum()
    peak_h = int(hourly.idxmax())

    print("\n" + "=" * 45)
    print("  CAFE FOOTFALL SUMMARY")
    print("=" * 45)
    print(f"  Total entries logged : {total}")
    print(f"  Peak hour            : {peak_h:02d}:00 – {peak_h+1:02d}:00  ({int(hourly[peak_h])} entries)")
    print(f"  Busiest 3 hours      :")
    for h, v in hourly.nlargest(3).items():
        print(f"      {h:02d}:00 – {h+1:02d}:00  →  {int(v)} entries")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peak Hour Analytics for Cafe Footfall")
    parser.add_argument("--log", default=None, help="Path to a specific CSV log file")
    parser.add_argument("--all", action="store_true", help="Merge all logs in data/logs/")
    args = parser.parse_args()

    df = load_logs(args.log, args.all)
    print_summary(df)
    plot_hourly(df)
    plot_timeline(df)
    print("[Analytics] Done. Charts saved to outputs/")
