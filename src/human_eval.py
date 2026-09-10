"""Human evaluation report (Section 3.2 / Table 4).

Reads ``results/human_eval.csv`` after the two raters have filled the
``m1_*`` and ``m2_*`` columns with 1 (yes) / 0 (no), and prints the
percentage of "yes" per rater per criterion and Cohen's kappa between the
two raters.

    python -m src.human_eval
"""

from __future__ import annotations

import csv

from src.config import RESULTS_DIR

CRITERIA = ["fluency", "relevance", "answerability"]


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two raters on a binary label."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(v) / n) * (b.count(v) / n) for v in (0, 1))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    path = RESULTS_DIR / "human_eval.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rated = [r for r in rows if r["m1_fluency"] != "" and r["m2_fluency"] != ""]
    if not rated:
        print(f"{path}: 0 / {len(rows)} rows rated. "
              "Fill the m1_* and m2_* columns with 1/0 first.")
        return

    print(f"{len(rated)} / {len(rows)} rows rated\n")
    print(f"{'criterion':<15}{'M1 % yes':>10}{'M2 % yes':>10}{'kappa':>9}")
    print("-" * 44)
    for c in CRITERIA:
        m1 = [int(r[f"m1_{c}"]) for r in rated]
        m2 = [int(r[f"m2_{c}"]) for r in rated]
        print(f"{c:<15}{100 * sum(m1) / len(m1):>9.1f}%"
              f"{100 * sum(m2) / len(m2):>9.1f}%{cohen_kappa(m1, m2):>9.3f}")


if __name__ == "__main__":
    main()
