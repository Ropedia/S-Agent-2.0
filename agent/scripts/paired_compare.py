#!/usr/bin/env python3
"""Paired (McNemar) comparison of two sagent configurations on their MATCHED questions.

    .venv/bin/python scripts/paired_compare.py <job_a> <job_b> [--by-type]

Why the intersection is not optional
------------------------------------
Comparing two configurations' overall percentages compares two different subsets of
questions whenever either run is partial -- and SAgent finishes short episodes
first, so a partial run's subset is systematically the EASY end. This project
reported a "-3.4pp regression" (44/100 vs 46/97) that vanished to exactly zero
once the questions were matched, and a "+3.1pp" at n=107 that finished at
-0.5pp. Both were this error.

So this script refuses to print a difference for anything but the intersection,
and prints the two subset sizes next to it so a reader can see how much was
dropped.

McNemar rather than a two-proportion test: the same questions are scored twice,
so the pairing carries information. Only the discordant pairs (b+/a- and a+/b-)
enter the statistic; concordant pairs are uninformative about which configuration is
better. Exact binomial two-sided, which is correct for the small discordant
counts we get at n<=200 (the chi-square approximation needs b+c >= 25).
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

# Trial dirs are `MMSI_<question-id>__<random-hash>`; the hash differs between
# runs of the same question, so it must be stripped to match across configurations.
TRIAL_RE = re.compile(r"^MMSI_(\d+)__")


def outcomes(job: Path, keep_infra_failures: bool = False) -> dict[str, int]:
    """question id -> 1 correct / 0 wrong. Missing reward = not scored, omitted.

    A trial that timed out or died before producing `agent/trajectory.json` is
    scored `reward=0`, i.e. it enters a comparison as a WRONG ANSWER. That is not
    evidence about the thing under test, and the count is not symmetric between
    jobs -- measured on the three 200-question runs: 5 / 2 / 8 such trials, so
    arm2 carried six more forced zeros than arm1 and read ~1pp worse for it.
    (arm1 additionally had 2 trials with no reward at all, which drop out here
    anyway -- the same failure class landing in two different buckets.)

    So these are excluded by default and the count is reported, which makes the
    asymmetry visible instead of silently costing the configuration that crashed more.
    Pass keep_infra_failures=True to reproduce an as-scored number.
    """
    scored: dict[str, int] = {}
    for trial in sorted(job.glob("MMSI_*")):
        m = TRIAL_RE.match(trial.name)
        if not m or not trial.is_dir():
            continue
        reward = trial / "verifier" / "reward.txt"
        if not reward.is_file():
            continue
        if (
            not keep_infra_failures
            and not (trial / "agent" / "trajectory.json").is_file()
        ):
            continue
        raw = reward.read_text().strip()
        try:
            scored[m.group(1)] = 1 if float(raw) >= 0.5 else 0
        except ValueError:
            continue
    return scored


def infra_failures(job: Path) -> int:
    """Trials scored but with no trajectory -- crashed/timed out before running."""
    n = 0
    for trial in sorted(job.glob("MMSI_*")):
        if not TRIAL_RE.match(trial.name) or not trial.is_dir():
            continue
        if (trial / "verifier" / "reward.txt").is_file() and not (
            trial / "agent" / "trajectory.json"
        ).is_file():
            n += 1
    return n


MMSI_PARQUET = Path(
    "/scratch/ywxzml3j/wry/action/repro/data/MMSI-Bench/MMSI_Bench.parquet"
)


def question_types(job: Path) -> dict[str, str]:
    """question id -> question_type.

    Joined from the source parquet, not the trial config: the trials carry no
    question_type (checked), and the parquet's `id` is the same integer that
    the trial dir name encodes as `MMSI_<id>__<hash>`.
    """
    if not MMSI_PARQUET.is_file():
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}
    frame = pd.read_parquet(MMSI_PARQUET, columns=["id", "question_type"])
    return {str(row.id): str(row.question_type) for row in frame.itertuples()}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs.

    b = A right / B wrong, c = A wrong / B right. Under the null each
    discordant pair is a fair coin, so the count is Binomial(b+c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval, which behaves at the small n we run."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def report(
    a_name: str,
    b_name: str,
    a: dict[str, int],
    b: dict[str, int],
    infra: tuple[int, int] | None = None,
) -> None:
    shared = sorted(set(a) & set(b))
    n = len(shared)
    a_hit = sum(a[q] for q in shared)
    b_hit = sum(b[q] for q in shared)
    both = sum(1 for q in shared if a[q] and b[q])
    a_only = sum(1 for q in shared if a[q] and not b[q])
    b_only = sum(1 for q in shared if b[q] and not a[q])
    neither = n - both - a_only - b_only

    print(f"\n{a_name}  vs  {b_name}")
    print(
        f"  scored: {len(a)} / {len(b)}   matched: {n}"
        f"   (dropped {len(a) - n} / {len(b) - n})"
    )
    if infra is not None and (infra[0] or infra[1]):
        print(
            f"  excluded as infra failures (no trajectory): {infra[0]} / {infra[1]}"
            + ("   <-- ASYMMETRIC" if abs(infra[0] - infra[1]) >= 3 else "")
        )
    if n == 0:
        print("  no overlap -- nothing comparable")
        return

    for name, hit in ((a_name, a_hit), (b_name, b_hit)):
        lo, hi = wilson(hit, n)
        print(
            f"  {name:<34} {hit:>3}/{n} = {100 * hit / n:5.1f}%  CI[{lo:.1f},{hi:.1f}]"
        )

    delta = 100 * (b_hit - a_hit) / n
    p = mcnemar_exact(a_only, b_only)
    print(
        f"  contingency: both {both}  only-A {a_only}  only-B {b_only}  neither {neither}"
    )
    print(
        f"  delta (B - A): {delta:+.1f}pp   discordant {a_only + b_only}"
        f"   McNemar exact two-sided p = {p:.3f}"
    )
    # A pp threshold alone is not a verdict. Four times now a mid-run delta has
    # cleared 3pp on a handful of discordant pairs and been written into a doc as
    # a finding, then vanished at full n (v20 -3.4pp, v24 +3.1pp, arm1 51.8%,
    # arm2 -8.4pp -> all ~0). Below ~30 discordant pairs the sign itself is not
    # stable, so say UNDECIDABLE rather than naming a direction: the printed
    # word is what ends up quoted, and it has to refuse to mislead on its own.
    discordant = a_only + b_only
    if discordant < 30:
        verdict = (
            f"UNDECIDABLE -- only {discordant} discordant pairs (need ~30). "
            "Do not quote this delta or design against it."
        )
    elif p > 0.05:
        verdict = f"not distinguishable (p={p:.3f}); |delta| {abs(delta):.1f}pp"
    elif abs(delta) > 3.0:
        verdict = f"actionable ({delta:+.1f}pp, p={p:.3f})"
    else:
        verdict = f"significant but small ({delta:+.1f}pp, p={p:.3f})"
    print(f"  -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_a")
    ap.add_argument("job_b")
    ap.add_argument(
        "--by-type",
        action="store_true",
        help="also break the matched set down by question_type",
    )
    ap.add_argument(
        "--keep-infra-failures",
        action="store_true",
        help="count crashed/timed-out trials as wrong answers (as-scored). Off by "
        "default: their count is not symmetric between jobs, so including them "
        "penalises whichever configuration crashed more, which is not the thing under test.",
    )
    args = ap.parse_args()

    a_path, b_path = Path(args.job_a), Path(args.job_b)
    a = outcomes(a_path, args.keep_infra_failures)
    b = outcomes(b_path, args.keep_infra_failures)
    infra = (
        None
        if args.keep_infra_failures
        else (
            infra_failures(a_path),
            infra_failures(b_path),
        )
    )
    report(a_path.name, b_path.name, a, b, infra)

    if not args.by_type:
        return
    types = question_types(b_path) or question_types(a_path)
    if not types:
        print(
            "\n  [no question_type available (MMSI parquet missing) -- skipping breakdown]"
        )
        return
    shared = sorted(set(a) & set(b))
    buckets: dict[str, list[str]] = {}
    for q in shared:
        buckets.setdefault(types.get(q, "unknown"), []).append(q)
    print("\n  by question_type (matched questions only):")
    for qtype, qs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        ah = sum(a[q] for q in qs)
        bh = sum(b[q] for q in qs)
        b_only = sum(1 for q in qs if b[q] and not a[q])
        a_only = sum(1 for q in qs if a[q] and not b[q])
        print(
            f"    {qtype:<28} n={len(qs):<4} A {ah:>3} ({100 * ah / len(qs):4.1f}%)"
            f"   B {bh:>3} ({100 * bh / len(qs):4.1f}%)"
            f"   {100 * (bh - ah) / len(qs):+5.1f}pp  disc {a_only}/{b_only}"
        )


if __name__ == "__main__":
    main()
