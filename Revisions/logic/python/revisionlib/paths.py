from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
REVISIONS_ROOT = REPO_ROOT / 'Revisions'
REVISION_PYTHON_ROOT = REVISIONS_ROOT / 'logic' / 'python'
SIMULATION_ROOT = REPO_ROOT / 'Simulation'
SIMULATION_DATA_ROOT = SIMULATION_ROOT / 'data'
SIMULATION_FIGURES_ROOT = SIMULATION_ROOT / 'figures'
BENCHMARK_ROOT = REPO_ROOT / 'Benchmark'
BENCHMARK_DATA_ROOT = BENCHMARK_ROOT / 'data'
BENCHMARK_RESULTS_ROOT = BENCHMARK_DATA_ROOT / 'results'
REVISION_OUTPUTS_ROOT = REVISIONS_ROOT / 'outputs' / 'simulation'
REVISION_BENCHMARK_OUTPUTS_ROOT = REVISIONS_ROOT / 'outputs' / 'benchmark'


def ensure_directory(path: Path | str) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def revision_output_dir(*parts: str) -> Path:
    return ensure_directory(REVISION_OUTPUTS_ROOT.joinpath(*parts))


def revision_benchmark_output_dir(*parts: str) -> Path:
    return ensure_directory(REVISION_BENCHMARK_OUTPUTS_ROOT.joinpath(*parts))