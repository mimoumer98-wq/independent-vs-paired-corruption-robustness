"""Analysis-only script (no model inference) for the blur x DDTI boundary:
tests whether the train-matched Paired (rho=-1) model's accuracy on DDTI is
non-monotonic in blur severity. Reads the per-seed arrays already saved by
evaluate_matched.py --corruption blur (DDTI branch) -- no re-evaluation.

Three paired t-tests (scipy.stats.ttest_rel, n=5 seeds), Holm-corrected as one
family: rise sigma=0.5 vs 0.0, rise sigma=1.0 vs 0.0, fall sigma=1.5 vs 1.0.

Example:
    python analyze_blur_nonmonotonicity.py \\
        --matched-json results/blur_matched_analysis.json --out results/blur_nonmonotonicity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def holm(pdict):
    items = sorted(pdict.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, rm = {}, 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        rm = max(rm, a)
        adj[k] = rm
    return adj


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matched-json", type=Path, required=True,
                    help="Output of 'evaluate_matched.py --corruption blur' for the DDTI run")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    data = json.loads(args.matched_json.read_text())
    per_seed = data["per_seed"]
    A = {sev: np.array(per_seed[sev]["paired"]) for sev in per_seed}

    tests = {}

    def paired(label, hi, lo):
        hi_k, lo_k = str(hi), str(lo)
        d = A[hi_k] - A[lo_k]
        t, pv = stats.ttest_rel(A[hi_k], A[lo_k])
        tests[label] = {"hi_sev": hi, "lo_sev": lo, "mean_diff": float(d.mean()),
                         "per_seed_diff": [round(float(x), 4) for x in d], "n_seeds": len(d),
                         "t": float(t), "p_raw": float(pv), "wins": int((d > 0).sum())}

    paired("rise_0.5_vs_0.0", 0.5, 0.0)
    paired("rise_1.0_vs_0.0", 1.0, 0.0)
    paired("fall_1.5_vs_1.0", 1.5, 1.0)

    hp = holm({k: v["p_raw"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = hp[k]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "test": "scipy.stats.ttest_rel paired t-test, n seeds; Holm over the 3-test family",
        "paired_model_nonmonotonicity_tests": tests,
    }, indent=2), encoding="utf-8")

    print("=== PAIRED MODEL NON-MONOTONICITY TESTS (DDTI, blur) ===")
    for k, v in tests.items():
        print(f"  {k}: mean_diff={v['mean_diff']:+.4f} wins={v['wins']}/{v['n_seeds']} "
              f"t={v['t']:+.2f} p_raw={v['p_raw']:.4f} p_holm={v['p_holm']:.4f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
