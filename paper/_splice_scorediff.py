"""Splice tab_scorediff_new.tex (built by compute_score_diff.py) into MAIN.tex,
replacing the Tab.4 (score_diff_merged) tabular body. Backs up MAIN.tex first.
Pipeline: sanity_gate.py -> compute_score_diff.py -> this. See PIPELINE.md.
"""
import shutil, datetime

stamp = datetime.date.today().strftime("%Y%m%d")
shutil.copy("MAIN.tex", f"MAIN.tex.bak_scorediff_{stamp}")
src = open("MAIN.tex", encoding="utf-8").read().split("\n")
body = open("tab_scorediff_new.tex", encoding="utf-8").read().rstrip("\n").split("\n")
MARK = "\\multirow{8}{*}{\\rotatebox{90}{ADULT}}"
start = end = None
for i, l in enumerate(src):
    if start is None and MARK in l and i > 600:
        start = i
    if start is not None and "\\end{tabular}" in l and i > start:
        end = i
        break
assert start and end, (start, end)
new = src[:start] + body + src[end:]
open("MAIN.tex", "w", encoding="utf-8").write("\n".join(new))
print(f"spliced: lines {start+1}..{end} ({end-start} old) -> {len(body)} new; backup MAIN.tex.bak_scorediff_{stamp}")
