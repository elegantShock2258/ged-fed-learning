import re

with open("paper.tex", "r") as f:
    content = f.read()

# 1. Fix Abstract
content = content.replace(
    "CelebA and FLamby pipelines have been fully architected and constitute immediate future experimental targets.",
    "We discuss the theoretical extension of this framework to higher-dimensional domains such as CelebA and FLamby."
)

# 2. Rename Limitations Section
content = content.replace(
    "\\subsection{Known Limitations (Self-Critique)}\\label{ssec:limitations}",
    "\\section{Limitations and Future Work}\\label{sec:limitations}"
)

# Fix references to the section
content = content.replace("Section~\\ref{ssec:limitations}", "Section~\\ref{sec:limitations}")

# 3. Clean up "Gap X" naming conventions to look like formal subsections
def repl_gap(match):
    # match.group(0) is the full match, e.g., \textbf{Gap 1 — Frozen Graph and Topological Drift:}
    text = match.group(0)
    
    # Extract the title part
    # Look for "—" or "\textrightarrow"
    title = ""
    if "—" in text:
        title = text.split("—")[1].strip()
    elif "\\textrightarrow" in text:
        title = text.split("\\textrightarrow")[1].strip()
    else:
        title = text
        
    # Clean up trailing colon or \\
    title = title.replace(":\\\\", "").replace(":", "").replace("}\\n", "}").replace("}", "").strip()
    
    return f"\\subsection{{{title}}}"

content = re.sub(r"\\textbf{Gap \d+\s*(—|\\textrightarrow).*?}", repl_gap, content)

# 4. Explicitly address BFT bound (Gap 7)
content = content.replace(
    "The bound $f_{\\max} < n(1-\\tau)/(2\\tau)$ is empirically observed",
    "\\textbf{Conjecture 1 (Byzantine Fault Tolerance Bound):} The system tolerates a maximum fraction of adversarial clients $f_{\\max} < n(1-\\tau)/(2\\tau)$. While empirically supported by our experiments ($f=0.25$ resilient at $\\tau=0.07$), formal mathematical proof remains an open problem."
)

with open("paper.tex", "w") as f:
    f.write(content)

print("Paper refined for IEEE submission.")
