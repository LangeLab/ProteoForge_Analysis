# Small bash script to convert all notebooks within
# a folder and its subfolders to html while preserving
# folder structure in the output directory.
# Uses jupyter nbconvert. Template argument is required; default is
# pretty-jupyter's 'pj' template.

# Usage: nb_to_html.sh [options]
# Options:
#   -o, --outdir DIR       Output directory for generated HTML (default: renders)
#   -i, --ignore PATTERN   Comma-separated list of path patterns to ignore (default: sandbox,.venv)
#   -t, --template NAME    nbconvert template name to use (default: none)
#   -n, --dry-run          Show what would be done without running conversions
#   -h, --help             Show this help

set -o nounset
set -o pipefail

OUTDIR="renders"
IGNORE_DEFAULT="sandbox,.venv"
IGNORE="$IGNORE_DEFAULT"
# Template is mandatory; default to pretty-jupyter's 'pj'
TEMPLATE="pj"
DRY_RUN=0
ROOT='.'

print_help() {
    sed -n '1,160p' "$0" | sed -n '1,120p'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--outdir)
            OUTDIR="$2"; shift 2;;
        -i|--ignore)
            IGNORE="$2"; shift 2;;
        -t|--template)
            TEMPLATE="$2"; shift 2;;
        -r|--root)
            ROOT="$2"; shift 2;;
        -n|--dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            echo "Usage: $0 [options]";
            echo "  -o, --outdir DIR       Output directory for generated HTML (default: renders)";
            echo "  -i, --ignore PATTERN   Comma-separated list of path patterns to ignore (default: $IGNORE_DEFAULT)";
            echo "  -t, --template NAME    nbconvert template name to use (default: pj)";
            echo "  -r, --root DIR         Root folder to search for notebooks (default: .)";
            echo "  -n, --dry-run          Show what would be done without running conversions";
            exit 0;;
        *)
            echo "Unknown option: $1"; exit 1;;
    esac
done

# Start timer
start=$(date +%s.%N)

# Build find exclude predicates from IGNORE
IFS=',' read -ra IGN_ARR <<< "$IGNORE"
FIND_EXCLUDES=""
for p in "${IGN_ARR[@]}"; do
    p_trim=$(echo "$p" | xargs)
    [[ -z "$p_trim" ]] && continue
    FIND_EXCLUDES+=" -not -path '*/$p_trim/*'"
done

# Also exclude the output dir (so renders/ isn't processed)
# Note: make sure to exclude both relative and absolute patterns
FIND_EXCLUDES+=" -not -path '$OUTDIR/*' -not -path './$OUTDIR/*'"

# Also exclude the output dir under the ROOT if user points to a subfolder
FIND_EXCLUDES+=" -not -path '$ROOT/$OUTDIR/*' -not -path './$ROOT/$OUTDIR/*'"

echo "Out dir: $OUTDIR"
echo "Ignore patterns: $IGNORE"
echo "Using template: $TEMPLATE"
[[ $DRY_RUN -eq 1 ]] && echo "Dry run: no files will be converted or moved"

# Create output dir if not dry-run
if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$OUTDIR"
    touch "$OUTDIR/.init" || true
fi

# Find notebooks and convert under ROOT
# We use eval to expand the generated FIND_EXCLUDES string
CMD_FIND="find '$ROOT' -name '*.ipynb' $FIND_EXCLUDES"

echo "Finding notebooks under: $ROOT"
echo "Using find: $CMD_FIND"

# Use while read to handle spaces in filenames
eval "$CMD_FIND" | while IFS= read -r nb; do
    # Skip if inside the OUTDIR (safety)
    case "$nb" in
        ./$OUTDIR/*|$OUTDIR/*) continue;;
    esac

    # Build nbconvert command
    nbcmd=(jupyter nbconvert --to html)
    # TEMPLATE is always set (default 'pj' if not provided)
    nbcmd+=(--template "$TEMPLATE")
    nbcmd+=("$nb")

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "Would run: ${nbcmd[*]}"
        # Also show where output would go
        html_rel=${nb%.ipynb}.html
        outpath="$OUTDIR/${html_rel#./}"
        echo "Would write: $outpath"
        continue
    fi

    echo "Converting: $nb"
    "${nbcmd[@]}"

    # Move generated html to OUTDIR while preserving directory structure
    # Compute path relative to ROOT
    # Strip initial ./ if present
    nb_no_prefix=${nb#./}
    rel_path=${nb_no_prefix#${ROOT#/}}
    # If ROOT is '.', then rel_path becomes the same as nb_no_prefix
    if [[ "$ROOT" == "." ]]; then
        rel_path=$nb_no_prefix
    else
        # Remove any leading slash left from parameter substitution
        rel_path=${rel_path#/}
    fi

    html_rel=${rel_path%.ipynb}.html
    dest_dir="$OUTDIR/$(dirname "$html_rel")"
    mkdir -p "$dest_dir"
    if [[ -f "$html_rel" ]]; then
        mv -f "$html_rel" "$dest_dir/"
    else
        echo "Warning: expected output $html_rel not found for notebook $nb"
    fi
done

# End timer
end=$(date +%s.%N)

# Print time
echo "Time elapsed: " $(echo "$end - $start" | bc) "seconds"
