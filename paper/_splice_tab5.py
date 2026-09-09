"""Splice tab5_new.tex (built by build_tab5.py) into MAIN.tex, replacing the
Tab.5 (tab:util) tabular body. Backs up MAIN.tex first.
Pipeline: sanity_gate.py -> build_tab5.py -> this. See PIPELINE.md.
"""
import shutil, datetime

stamp = datetime.date.today().strftime("%Y%m%d")
shutil.copy("MAIN.tex", f"MAIN.tex.bak_tab5_{stamp}")
src = open("MAIN.tex", encoding="utf-8").read().split("\n")
body = open("tab5_new.tex", encoding="utf-8").read().rstrip("\n").split("\n")
MARK = "\\multirow{3}{*}{$\\mathtt{loss}$}"
start = end = None
for i, l in enumerate(src):
    if start is None and MARK in l:
        start = i
    if start is not None and "\\end{tabular}" in l and i > start:
        end = i
        break
assert start and end, (start, end)
new = src[:start] + body + src[end:]
open("MAIN.tex", "w", encoding="utf-8").write("\n".join(new))
print(f"tab5 spliced: lines {start+1}..{end} ({end-start} old) -> {len(body)} new; backup MAIN.tex.bak_tab5_{stamp}")
