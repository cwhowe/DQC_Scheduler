from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot summarized scheduler experiment results.")
    ap.add_argument("suite_dir")
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    rows = load_csv(suite_dir / "summary_aggregated.csv")
    figs_dir = suite_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    # Group by suite/workload.
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["suite"], r["workload"])].append(r)

    for (suite, workload), grp in grouped.items():
        grp = sorted(grp, key=lambda r: r["method"])
        methods = [r["method"] for r in grp]

        # Makespan
        vals = [float(r["makespan_s_mean"]) for r in grp]
        plt.figure(figsize=(9, 4.5))
        plt.bar(methods, vals)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Makespan (s)")
        plt.title(f"{suite} / {workload}: makespan")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{suite}__{workload}__makespan.png", dpi=160)
        plt.close()

        # Mean latency
        vals = [float(r["lat_mean_s_mean"]) for r in grp]
        plt.figure(figsize=(9, 4.5))
        plt.bar(methods, vals)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Mean job latency (s)")
        plt.title(f"{suite} / {workload}: mean latency")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{suite}__{workload}__latency.png", dpi=160)
        plt.close()

        # Overhead stack: schedule + partition + recon
        schedule = [float(r["schedule_mean_s_mean"]) for r in grp]
        partition = [float(r["partition_mean_s_mean"]) for r in grp]
        recon = [float(r["recon_mean_s_mean"]) for r in grp]
        x = range(len(methods))
        plt.figure(figsize=(9, 4.5))
        plt.bar(x, schedule, label="schedule")
        plt.bar(x, partition, bottom=schedule, label="partition")
        plt.bar(x, recon, bottom=[a + b for a, b in zip(schedule, partition)], label="reconstruction")
        plt.xticks(list(x), methods, rotation=25, ha="right")
        plt.ylabel("Mean overhead (s)")
        plt.title(f"{suite} / {workload}: classical overhead")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / f"{suite}__{workload}__overhead_stack.png", dpi=160)
        plt.close()

        # Utilization / fairness
        util = [float(r["utilization_mean"]) for r in grp]
        fair = [float(r["fairness_jain_mean"]) for r in grp]
        plt.figure(figsize=(9, 4.5))
        plt.plot(methods, util, marker="o", label="utilization")
        plt.plot(methods, fair, marker="o", label="fairness_jain")
        plt.xticks(rotation=25, ha="right")
        plt.ylim(bottom=0)
        plt.title(f"{suite} / {workload}: utilization and fairness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / f"{suite}__{workload}__util_fairness.png", dpi=160)
        plt.close()

    print(f"Wrote figures under {figs_dir}")


if __name__ == "__main__":
    main()
