"""run_ci_seeds.py — Run E11, E19, and E20 across multiple seeds for CI analysis.

Produces repeated-seed CSVs in the same seed_N/ layout used by E1:
    results/paper_ci/seed_N/e11_slo_constrained.csv
    results/paper_ci/seed_N/e19_stream_composition.csv
    results/paper_ci/seed_N/e20_batch_stream.csv

Once this completes, update analyze_ci.py to point the single-run analyses
at results/paper_ci/ instead of results/experiments/ and they will
automatically switch to the stronger paired-seed bootstrap + Wilcoxon path.

Usage
-----
    python run_ci_seeds.py
    python run_ci_seeds.py --seeds 2026,2027,2028,2029,2030 --n-jobs 40
    python run_ci_seeds.py --experiments E11,E19   # subset
    python run_ci_seeds.py --fast                  # n_jobs=20 for quick check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Default seed list — matches the seeds already used for E1
# ---------------------------------------------------------------------------
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034]

# ---------------------------------------------------------------------------
# Experiment definitions
#   key       → (runner_script, --experiments flag or None, output_filename)
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "E11": (
        "run_experiments.py",
        "E11",
        "e11_slo_constrained.csv",
    ),
    "E19": (
        "run_experiments_new.py",
        "E19",
        "e19_stream_composition.csv",
    ),
    "E20": (
        "run_e20_batch_stream.py",
        None,                          # E20 runner has no --experiments flag
        "e20_batch_stream.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--outdir", default="results/paper_ci",
        help="Root output directory; seed_N/ subdirs are created here",
    )
    p.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seed list (default: %(default)s)",
    )
    p.add_argument(
        "--n-jobs", type=int, default=40,
        help="Jobs per condition per seed (default: %(default)s)",
    )
    p.add_argument(
        "--experiments", default="E11,E19,E20",
        help="Comma-separated experiments to run (default: %(default)s)",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="Use n_jobs=20 for a quick sanity check",
    )
    p.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter to use (default: same as this script)",
    )
    return p.parse_args()


def run_seed(
    seed: int,
    experiment_key: str,
    runner: str,
    experiments_flag: str | None,
    outfile: str,
    seed_dir: str,
    n_jobs: int,
    python: str,
) -> bool:
    """Run one experiment for one seed. Returns True on success."""
    dest = os.path.join(seed_dir, outfile)

    # Skip if already done
    if os.path.exists(dest):
        print(f"    [skip] {dest} already exists")
        return True

    cmd = [
        python, runner,
        "--outdir", seed_dir,
        "--n-jobs", str(n_jobs),
        "--seed", str(seed),
    ]
    if experiments_flag is not None:
        cmd += ["--experiments", experiments_flag]

    print(f"    running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    [ERROR] seed={seed} {experiment_key} failed "
              f"(returncode={result.returncode})")
        return False

    print(f"    done in {elapsed:.1f}s  → {dest}")
    return True


def main() -> None:
    args = parse_args()
    if args.fast:
        args.n_jobs = 20

    seeds   = [int(s.strip()) for s in args.seeds.split(",")]
    wanted  = {e.strip().upper() for e in args.experiments.split(",")}
    unknown = wanted - set(EXPERIMENTS)
    if unknown:
        print(f"[ERROR] Unknown experiments: {unknown}. Valid: {set(EXPERIMENTS)}")
        sys.exit(1)

    print(f"[CI SEEDS]  outdir  = {args.outdir}")
    print(f"[CI SEEDS]  seeds   = {seeds}")
    print(f"[CI SEEDS]  n_jobs  = {args.n_jobs}")
    print(f"[CI SEEDS]  running = {sorted(wanted)}")

    t_total = time.time()
    failures = []

    for seed in seeds:
        seed_dir = os.path.join(args.outdir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n--- seed {seed}  ({seed_dir}) ---")

        for key in ["E11", "E19", "E20"]:
            if key not in wanted:
                continue
            runner, exp_flag, outfile = EXPERIMENTS[key]
            print(f"  [{key}]")
            ok = run_seed(
                seed=seed,
                experiment_key=key,
                runner=runner,
                experiments_flag=exp_flag,
                outfile=outfile,
                seed_dir=seed_dir,
                n_jobs=args.n_jobs,
                python=args.python,
            )
            if not ok:
                failures.append((seed, key))

    elapsed = time.time() - t_total
    print(f"\n[CI SEEDS]  finished in {elapsed:.1f}s")

    if failures:
        print(f"[CI SEEDS]  FAILURES: {failures}")
        sys.exit(1)
    else:
        print(f"[CI SEEDS]  all seeds completed successfully")
        print(f"\nNext step: update analyze_ci.py paths for E11/E19/E20:")
        print(f"  E11_SINGLE → Path('{args.outdir}')  +  use _load_seeds")
        print(f"  E19_SINGLE → Path('{args.outdir}')  +  use _load_seeds")
        print(f"  E20_SINGLE → Path('{args.outdir}')  +  use _load_seeds")


if __name__ == "__main__":
    main()
