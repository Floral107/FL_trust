"""Splice tab_scorefluct_new.tex (built by compute_score_fluct.py) into MAIN.tex,
replacing the Tab.3 (score_fluct) tabular body. Backs up MAIN.tex first.
Pipeline: sanity_gate.py -> compute_score_fluct.py -> this. See PIPELINE.md.
"""
import shutil, datetime

stamp = datetime.date.today().strftime("%Y%m%d")
shutil.copy("MAIN.tex", f"MAIN.tex.bak_fluct_{stamp}")
src = open("MAIN.tex", encoding="utf-8").read().split("\n")
body = open("tab_scorefluct_new.tex", encoding="utf-8").read().rstrip("\n").split("\n")
MARK = "\\multirow{4}{*}{\\rotatebox{90}{ADULT}}"
start = end = None
for i, l in enumerate(src):
    if start is None and MARK in l and i > 500:
        start = i
    if start is not None and "\\end{tabular}" in l and i > start:
        end = i
        break
assert start and end, (start, end)
new = src[:start] + body + src[end:]
open("MAIN.tex", "w", encoding="utf-8").write("\n".join(new))
print(f"fluct spliced: lines {start+1}..{end} ({end-start} old) -> {len(body)} new; backup MAIN.tex.bak_fluct_{stamp}")
