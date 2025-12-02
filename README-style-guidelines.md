# README Style Guidelines (Design Language)

Purpose: provide a concise, reproducible, and cross-platform design language for README files and short documentation across this repository. It draws from the project’s `README.md` header style (badges + short table), the attached scientific writer ruleset, and project metadata (e.g., `.gitignore`). These rules aim to standardize tone, layout, commands, and cross-platform setup instructions and to emphasise that this repository provides a snapshot of an analysis at the time of publishing rather than an actively-maintained package.

Important constraints

- No emojis anywhere in repository READMEs or other README-like documentation (policy rule: absolute).
- Badges are allowed (and encouraged) in the top-level README header; keep badge tables minimal and aligned, similar to the existing `README.md`.
- Be explicit about OS-specific commands and differences for Windows, Linux, and macOS; show clear, separate command blocks for each.
- Keep READMEs and folder-level README files small, focused, and actionable.
-- Avoid making claims that sound promotional ("state-of-the-art", "best", "revolutionary"). Instead, state facts and reference the supporting sources.
-- This repository should be treated as a publication snapshot: do not document CI, PR workflows, or release automation as if they are required or expected unless they are actually present and relevant.

Core principles

- Precision: choose precise language and include concrete commands, versions, and paths.
- Objectivity: no self-promoting claims; present facts, limitations, and where to find evidence (links or references). Use neutral phrasing.
- Concision: prefer short sentences and clear prose; avoid redundancy.
-- Reproducible instructions: provide steps that allow a reader on the target OS to reproduce the setup, run the analysis, and regenerate figures and outputs.
- No hallucinations: do not invent version numbers, commands or data locations. Use placeholder tags when needed (see below).

Structure conventions

-- Top-level README: should follow the layout used by `README.md` already in this repository: badges table at top; short paragraph describing the repository; a `Repository layout` section; `Reproduce the analysis`, `Reproducibility notes`, and `Citation` sections.

- Subfolder README: one or two paragraphs of purpose followed by limited headings: `Contents`, `Quick usage`, `Inputs/outputs`, `Reproducibility`, `Examples`, and `Troubleshooting`.
- Headings: use H2 for high-level sections (##), H3 for subsections, and so on.
- Short Table-of-Contents: include when the README is long (> 600 words).
- Badge placement: header area only. Keep the badge table consistent with the main README's style.

Tone and wording

- Use active voice for instructions, passive voice when describing experiments or results.
- Replace marketing words: "utilize", "leverage", "cutting-edge", "robust", etc. Prefer "use", "apply", "improve" with numeric or measurable qualifiers.
- Reserve "significant/significantly" for statistical claims backed by a test. Otherwise, use "notable", "substantial", or simply provide numbers and let readers decide.
- Avoid adverbs unless they add technical precision. E.g., instead of "quickly converge", use "converges in N iterations or T seconds on these inputs".

Cross-platform instructions (Windows / macOS / Linux)

- Always show commands for all three major platforms if available: use separate sections:
    - Linux / macOS: `bash` or `zsh` commands
    - Windows (PowerShell): PowerShell commands
    - Windows (cmd.exe): `cmd.exe` commands where appropriate
- Examples
    - Create a Python venv

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```cmd
REM Windows (cmd)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- If different commands are required for Windows vs Linux/macOS, state the reason (different path styles, script names, shell behavior).
- Use environment variable examples for each OS; e.g., setting a variable:

```bash
# Linux / macOS
export DATA_DIR="/path/to/data"
```

```powershell
# Windows PowerShell
$env:DATA_DIR = "C:\path\to\data"
```

```cmd
REM Windows cmd
set DATA_DIR=C:\path\to\data
```

Code blocks and language hints

-- Provide language hint for code blocks: e.g., ```bash```, ```python```, ```r```, ```powershell```.

- Use `source` vs `.` consistently for shell files.
- Use full, unambiguous commands (prefer `python3` on Linux/macOS). For cross-platform, show both `python3` and `python` with a note.

Placeholders and no-hallucination tags

- Use placeholders when information is missing or requires user input:
    - `[SPECIFY USER OR PATH]`
    - `[INSERT VERSION OR TAG]`
    - `[INSERT DATA LINK OR PATH]`
    - `[SOFTWARE VERSION]` or `[NOTE: specify OS]`
- If a command or an option is not available on a platform, say so explicitly instead of guessing.

Gitignore and visibility guidance

- Check `.gitignore` before mentioning paths. If a path is ignored (e.g., `**/data/`, `**/figures/`, `renv/`, `.venv`), the README must instruct users how to obtain or generate these files, not assume they are in git.
- Recommended wording for large files: "Data/figures are not committed. Place raw data under `PATH`, then run `download` or `prepare` steps (provide commands)."
- When listing example results that are not committed, include a link to rendered outputs under `renders/` where appropriate.

Badges and release information

- Badges are allowed and encouraged; keep them to the header region and make them meaningful:
    - `Status` (e.g., Under development), `License`, `Language`, `Release version`, `Zenodo/DOI` — similar to the existing top-of-file style.
- Keep badges up to date and link them to their source (e.g., DOI badge points to Zenodo). Avoid using badges that don't link to a valid source.

Citation and acknowledgement

- Provide a `Citation` block and include a DOI or a recommended citation in BibTeX format.
- Avoid promotional language.

Subfolder README template (short):

- Purpose (one or two sentences)
- Contents (what to expect in the folder)
- Quick usage (commands for all OSes a user is likely to run)
- Inputs and outputs (where to put input files; what outputs will be generated; paths to results)
- Troubleshooting / common gotchas
- Reproducibility: point to `renv/`, `requirements.txt`, or `setup_env.R` and show the commands to restore envs.

Example subfolder README

````markdown
## Purpose
Short description of the analysis or script purpose.

## Contents
- `scriptA.py`: script that does X
- `scriptB.R`: script that runs Y

## Quick usage
Linux / macOS:
```bash
python3 scriptA.py --input /path/to/inputs --out results/
```

Windows (PowerShell):
```powershell
python scriptA.py --input C:\path\to\inputs --out results/ 
```

## Inputs
Place raw data files under `data/input/` (not committed).

## Outputs
Results will appear in `data/results/`. These folders are ignored by `.gitignore`.
````

Accessibility and readability

- Use descriptive alt text for images and keep file names readable and consistent.

Style checklist (quick)

- No emojis (policy)
- Badges in header only
- Include OS-specific commands
- Use placeholders where needed
- Avoid marketing language
- Use statistics language correctly (reserve "significant")
- Check `.gitignore` before referencing any file paths
- Short subfolder READMEs use the provided template
