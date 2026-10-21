"""Generate the zero-shot and train-matched comparison tables as
elsarticle/booktabs LaTeX (no vertical rules), directly from the evaluation
JSONs.

Run: python gen_tables.py --zero-shot-json ... --gaussian-matched-json ... \\
        --blur-matched-json ... --out-dir .
"""
import argparse
from pathlib import Path


def pfmt(p):
    s = f"{p:.3f}"
    return f"\\textbf{{{s}}}" if p < 0.05 else s


def rows_zeroshot(p0, corr):
    out = []
    for sev in sorted(p0["tn3k"]["independent_vs_paired"][corr], key=float):
        tm = p0["tn3k"]["means"][corr][sev]; ta = p0["tn3k"]["independent_vs_paired"][corr][sev]
        dm = p0["ddti"]["means"][corr][sev]; da = p0["ddti"]["independent_vs_paired"][corr][sev]
        out.append(f"{sev} & {tm['independent']:.3f} & {tm['paired']:.3f} & {ta['cohens_d']:.2f} & {pfmt(ta['p_holm'])} "
                   f"& {dm['independent']:.3f} & {dm['paired']:.3f} & {da['cohens_d']:.2f} & {pfmt(da['p_holm'])} \\\\")
    return out


def rows_matched(data, corr):
    out = []
    sevs = [s for s in sorted(data["tn3k"]["means"], key=float) if float(s) > 0]
    for sev in sevs:
        tm = data["tn3k"]["means"][sev]; ta = data["tn3k"]["independent_vs_paired"][sev]
        dm = data["ddti"]["means"][sev]; da = data["ddti"]["independent_vs_paired"][sev]
        out.append(f"{sev} & {tm['independent']:.3f} & {tm['paired']:.3f} & {ta['cohens_d']:.2f} & {pfmt(ta['p_holm'])} "
                   f"& {dm['independent']:.3f} & {dm['paired']:.3f} & {da['cohens_d']:.2f} & {pfmt(da['p_holm'])} \\\\")
    return out


HEAD = (r"""\begin{table*}[t]
\centering\small
\caption{%s}
\label{%s}
\begin{tabular}{l rrrr rrrr}
\toprule
 & \multicolumn{4}{c}{TN3K (in-distribution)} & \multicolumn{4}{c}{DDTI (out-of-distribution)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
$\sigma$ & Ind. & Paired & $d_z$ & $p$ & Ind. & Paired & $d_z$ & $p$ \\
\midrule
""")
FOOT = "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"


def group(name, rows):
    return f"\\multicolumn{{9}}{{l}}{{\\textit{{{name}}}}} \\\\\n" + "\n".join(rows) + "\n"


def main():
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--zero-shot-json", type=Path, required=True)
    p.add_argument("--gaussian-matched-json", type=Path, required=True)
    p.add_argument("--blur-matched-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    p0 = json.loads(args.zero_shot_json.read_text())
    g = json.loads(args.gaussian_matched_json.read_text())
    b = json.loads(args.blur_matched_json.read_text())

    cap1 = ("Zero-shot comparison (speckle-trained models, no re-training): mean per-image Dice for "
            "Independent ($\\rho=0$) and Paired ($\\rho=-1$), their effect size Cohen's $d_z$ "
            "(Independent$-$Paired), and the Holm-corrected $p$ (bold if $p<0.05$), by corruption "
            "family and severity, on TN3K and DDTI.")
    t1 = HEAD % (cap1, "tab:zeroshot")
    for corr, nm in [("speckle", "Speckle"), ("gaussian", "Additive Gaussian"),
                      ("blur", "Gaussian blur"), ("gamma", "Gamma-speckle")]:
        t1 += group(nm, rows_zeroshot(p0, corr))
        if corr != "gamma":
            t1 += "\\midrule\n"
    t1 += FOOT
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "table1_zeroshot.tex").write_text(t1, encoding="utf-8")

    cap2 = ("Train-matched comparison (each condition re-trained under the evaluation corruption): "
            "mean per-image Dice for Independent ($\\rho=0$) and Paired ($\\rho=-1$), Cohen's $d_z$ "
            "(Independent$-$Paired), and Holm-corrected $p$ (bold if $p<0.05$), on TN3K and DDTI.")
    t2 = HEAD % (cap2, "tab:matched")
    t2 += group("Additive Gaussian", rows_matched(g, "gaussian")) + "\\midrule\n"
    t2 += group("Gaussian blur", rows_matched(b, "blur"))
    t2 += FOOT
    (args.out_dir / "table2_matched.tex").write_text(t2, encoding="utf-8")
    print(f"wrote table1_zeroshot.tex and table2_matched.tex in {args.out_dir}")


if __name__ == "__main__":
    main()
