"""Generate the single-view-baseline comparison table (speckle) as
elsarticle/booktabs LaTeX, from evaluate_singleview.py's output.

Run: python gen_table3.py --singleview-json ... --out-dir .
"""
import argparse
import json
from pathlib import Path


def pf(p):
    s = f"{p:.3f}"
    return "\\textbf{" + s + "}" if p < 0.05 else s


HEAD = (
    "\\begin{table*}[t]\n\\centering\\small\n"
    "\\caption{Single-view baseline vs.\\ the two-view conditions under speckle: mean "
    "per-image Dice for the single-view baseline (SV), Independent ($\\rho=0$) and Paired "
    "($\\rho=-1$), with the paired differences Independent$-$SV and SV$-$Paired and their "
    "Holm-corrected $p$ (bold if $p<0.05$), on TN3K and DDTI.}\n"
    "\\label{tab:singleview}\n"
    "\\begin{tabular}{l rrr rr rr}\n\\toprule\n"
    "$\\sigma$ & SV & Ind. & Paired & $\\Delta_{\\mathrm{I}-\\mathrm{SV}}$ & $p$ & "
    "$\\Delta_{\\mathrm{SV}-\\mathrm{P}}$ & $p$ \\\\\n\\midrule\n"
)
FOOT = "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--singleview-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    cmp = json.loads(args.singleview_json.read_text())["comparison"]

    t = HEAD
    for ds, nm in [("tn3k", "TN3K (in-distribution)"), ("ddti", "DDTI (out-of-distribution)")]:
        t += "\\multicolumn{8}{l}{\\textit{" + nm + "}} \\\\\n"
        for s in sorted(cmp[ds]["speckle"], key=float):
            r = cmp[ds]["speckle"][s]
            t += (f"{s} & {r['singleview']:.3f} & {r['independent']:.3f} & {r['paired']:.3f} & "
                  f"+{r['diff_independent_minus_singleview']:.3f} & {pf(r['p_holm_independent_vs_singleview'])} & "
                  f"+{r['diff_singleview_minus_paired']:.3f} & {pf(r['p_holm_singleview_vs_paired'])} \\\\\n")
        if ds == "tn3k":
            t += "\\midrule\n"
    t += FOOT

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "table3_singleview.tex").write_text(t, encoding="utf-8")
    print(f"wrote table3_singleview.tex in {args.out_dir}")


if __name__ == "__main__":
    main()
