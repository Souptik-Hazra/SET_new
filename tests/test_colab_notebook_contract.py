import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "SET_Pharma_Colab_Pipeline.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_cells(notebook):
    return [
        (index, "".join(cell.get("source", [])))
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    ]


def literal_tuple_from_module(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def test_colab_notebook_is_valid_and_every_code_cell_compiles():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for index, source in code_cells(notebook):
        compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")


def test_notebook_imports_the_runtime_and_uses_all_stage_entrypoints():
    notebook_source = "\n".join(source for _, source in code_cells(load_notebook()))
    assert "from scripts.colab_runtime import" in notebook_source
    assert "run_ingestion(" in notebook_source
    assert "run_models(" in notebook_source
    assert "run_mathematical_validation(" in notebook_source
    assert "sync_artifacts(DRIVE_ARTIFACT_ROOT, REPO_DIR)" in notebook_source
    assert "sync_artifacts(REPO_DIR, DRIVE_ARTIFACT_ROOT)" in notebook_source


def test_notebook_branch_and_dataset_paths_match_repository_contract():
    notebook_source = "\n".join(source for _, source in code_cells(load_notebook()))
    assert 'BRANCH = "feature/colab-pipeline-execution"' in notebook_source
    assert 'REPO_DIR = Path("/content/SET_new")' in notebook_source
    assert 'DRIVE_RAW_CSV = Path("/content/drive/MyDrive/SET_Pharma/data/raw/combined_secondary_care_medicines.csv")' in notebook_source
    assert 'LOCAL_RAW_CSV = REPO_DIR / "data" / "raw" / "combined_secondary_care_medicines.csv"' in notebook_source
    assert (ROOT / "scripts" / "colab_runtime.py").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "config" / "config.yaml").is_file()
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/data/raw/*" in gitignore


def test_runtime_artifact_contracts_match_python_pipeline_outputs():
    helper = ROOT / "scripts" / "colab_runtime.py"
    ingestion = (ROOT / "src" / "set_forecasting" / "data" / "ingestion.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "src" / "set_forecasting" / "pipeline.py").read_text(encoding="utf-8")
    mathematical = (ROOT / "src" / "set_forecasting" / "validation" / "mathematical.py").read_text(encoding="utf-8")

    for artifact in literal_tuple_from_module(helper, "INGESTION_ARTIFACTS"):
        assert Path(artifact).name in ingestion, f"Ingestion output not found: {artifact}"
    for artifact in literal_tuple_from_module(helper, "MODEL_ARTIFACTS"):
        assert Path(artifact).name in pipeline, f"Model output not found: {artifact}"
    for artifact in literal_tuple_from_module(helper, "MATH_ARTIFACTS"):
        assert Path(artifact).name in mathematical, f"Validation output not found: {artifact}"


def test_notebook_and_helper_use_same_stage_parameters():
    notebook_source = "\n".join(source for _, source in code_cells(load_notebook()))
    helper = (ROOT / "scripts" / "colab_runtime.py").read_text(encoding="utf-8")
    assert "min_periods=MIN_PERIODS" in notebook_source
    assert "min_nonzero=MIN_NONZERO" in notebook_source
    assert "bootstrap=BOOTSTRAP_RESAMPLES" in notebook_source
    assert "seed=RANDOM_SEED" in notebook_source
    assert "def run_models(" in helper
    assert "def run_mathematical_validation(" in helper
    assert "require_stage(\"ingestion\"" in helper
    assert "require_stage(\"model_pipeline\"" in helper
