#!/usr/bin/env python3
"""
build_paper.py
--------------
Finds pdflatex (wherever it is), copies real figures if available,
then runs the full 3-pass pdflatex + bibtex build.
Run from anywhere:  python3 paper/build_paper.py
"""

import os
import subprocess
import shutil
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
PAPER_DIR    = SCRIPT_DIR          # main.tex lives next to this script
PROJECT_ROOT = PAPER_DIR.parent    # FYP/
MAIN_TEX     = PAPER_DIR / "main.tex"
FIGURES_DIR  = PAPER_DIR / "figures"
RESULTS_FIGS = PROJECT_ROOT / "results" / "figures"

# Map figure stem → source files to try (in preference order)
FIGURE_MAP = {
    "g01_acceptance_bars": ["G01_per_round_acceptance_bars.png"],
    "g02_ged_distribution": ["G02_ged_score_distribution.png"],
    "g03_roc_curve":        ["G03_roc_curve.png"],
    "latent_shift":         ["latent_shift_asia.png"],
    "por_architecture":     ["por_architecture.png"],
}

# ── 1. Find pdflatex ──────────────────────────────────────────────────────────
def find_pdflatex() -> str:
    # 1a. On PATH
    exe = shutil.which("pdflatex")
    if exe:
        return exe

    # 1b. Common install locations
    import glob
    home = Path.home()
    explicit = [
        home / ".TinyTeX/bin/x86_64-linux/pdflatex",
        home / ".TinyTeX/bin/aarch64-linux/pdflatex",
        home / ".tinytex/bin/x86_64-linux/pdflatex",
    ]
    for p in explicit:
        if p.is_file():
            return str(p)
    patterns = [
        "/usr/local/texlive/*/bin/x86_64-linux/pdflatex",
        "/usr/local/texlive/*/bin/aarch64-linux/pdflatex",
        "/opt/texlive/*/bin/x86_64-linux/pdflatex",
        "/snap/texlive/*/bin/pdflatex",
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]

    return None


# ── 2. Copy real figures if available ────────────────────────────────────────
def copy_figures():
    FIGURES_DIR.mkdir(exist_ok=True)
    if not RESULTS_FIGS.exists():
        print("  ⚠  results/figures/ not found — using placeholders")
        return
    for stem, candidates in FIGURE_MAP.items():
        dest_png = FIGURES_DIR / f"{stem}.png"
        for cand in candidates:
            src = RESULTS_FIGS / cand
            if src.exists():
                shutil.copy2(src, dest_png)
                print(f"  ✓  Copied {cand} → figures/{stem}.png")
                break
        else:
            print(f"  ⚠  No real figure for {stem} — placeholder will be used")


# ── 3. Generate figures ───────────────────────────────────────────────────────
def generate_figures():
    print("  Generating fresh visualization plots...")
    vis_dir = PROJECT_ROOT / "visualizations"
    scripts = [
        "g00_por_architecture.py",
        "g00_latent_shift.py",
        "g01_acceptance_bars.py",
        "g02_ged_distribution.py",
        "g03_roc_curve.py"
    ]
    for script in scripts:
        cmd = [sys.executable, str(vis_dir / script)]
        try:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, capture_output=True)
            print(f"  ✓  Executed {script}")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠  Failed to execute {script}:\n{e.stderr.decode()}")

# ── 4. Run a shell command, stream output ─────────────────────────────────────
def run(cmd: list, cwd: Path):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    # Show only warnings/errors from stdout (suppress the wall of OK lines)
    for line in result.stdout.splitlines():
        low = line.lower()
        if any(kw in low for kw in ["error", "warning", "undefined", "missing",
                                     "overfull", "underfull", "!",]):
            print(f"   {line}")
    if result.stderr.strip():
        print(result.stderr[:800])
    return result.returncode


# ── 4. Main build sequence ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  IEEE T-AI Paper Builder")
    print("=" * 60)

    # Locate pdflatex
    pdflatex = find_pdflatex()
    if not pdflatex:
        print("\n❌  pdflatex not found.")
        print("   Install with:  sudo apt install texlive-full")
        print("   Or TinyTeX:    wget -qO- https://yihui.org/tinytex/install-unx.sh | sh")
        sys.exit(1)
    print(f"\n✅  pdflatex: {pdflatex}")

    bibtex = shutil.which("bibtex") or str(Path(pdflatex).parent / "bibtex")
    print(f"✅  bibtex  : {bibtex}")

    # Generate and Copy figures
    print("\n── Generating & Copying figures ──────────────────────────")
    generate_figures()
    copy_figures()

    flags = [
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "main.tex",
    ]

    # Pass 1
    print("\n── Pass 1: pdflatex ─────────────────────────────────────")
    rc = run([pdflatex] + flags, PAPER_DIR)

    # BibTeX
    print("\n── BibTeX ───────────────────────────────────────────────")
    run([bibtex, "main"], PAPER_DIR)

    # Pass 2
    print("\n── Pass 2: pdflatex ─────────────────────────────────────")
    run([pdflatex] + flags, PAPER_DIR)

    # Pass 3
    print("\n── Pass 3: pdflatex ─────────────────────────────────────")
    rc = run([pdflatex] + flags, PAPER_DIR)

    out_pdf = PAPER_DIR / "main.pdf"
    print("\n" + "=" * 60)
    if out_pdf.exists() and out_pdf.stat().st_size > 10_000:
        print(f"✅  BUILD SUCCESSFUL")
        print(f"   PDF: {out_pdf}")
        print(f"   Size: {out_pdf.stat().st_size / 1024:.1f} KB")
    else:
        print("❌  Build may have failed — check main.log for details")
        log = PAPER_DIR / "main.log"
        if log.exists():
            # Print last 30 lines of log
            lines = log.read_text(errors="replace").splitlines()
            print("\n── Last 30 lines of main.log ──")
            for l in lines[-30:]:
                print(f"   {l}")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
