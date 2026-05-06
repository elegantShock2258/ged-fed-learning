import subprocess
from pathlib import Path

PAPER_DIR = Path("/home/ragavpn/Desktop/FYP/paper")
pdflatex = "/home/ragavpn/bin/pdflatex"
bibtex = "/home/ragavpn/bin/bibtex"
flags = ["-interaction=nonstopmode", "main.tex"]

subprocess.run([pdflatex] + flags, cwd=PAPER_DIR, capture_output=True)
print("--- main.aux before bibtex ---")
with open(PAPER_DIR / "main.aux") as f:
    lines = f.readlines()
    for l in lines[-5:]:
        print(l.strip())

print("--- bibtex output ---")
res = subprocess.run([bibtex, "main"], cwd=PAPER_DIR, capture_output=True, text=True)
print(res.stdout)
