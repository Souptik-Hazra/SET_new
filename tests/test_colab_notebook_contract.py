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
    assert "importlib.util.find_spec(name)" in notebook_source
    assert "src.visualization.generate_figures" in notebook_source
    assert (ROOT / "src" / "visualization" / "__init__.py").is_file()
    assert "run_ingestion(" in notebook_source
    assert "run_models(" in notebook_source
    assert "run_mathematical_validation(" in notebook_source
    assert "sync_artifacts(DRIVE_ARTIFACT_ROOT, REPO_DIR)" in notebook_source
    assert "sync_artifacts(REPO_DIR, DRIVE_ARTIFACT_ROOT)" in notebook_source


def test_notebook_branch_and_dataset_paths_match_repository_contract():
    notebook_source = "\n".join(source for _, source in code_cells(load_notebook()))
    assert 'BRANCH = "feature/colab-pipeline-execution"' in notebook_source
    assert 'git", "checkout", "--detach", f"origin/{BRANCH}"' in notebook_source
    assert "DEFAULT_NOTEBOOK_COMMIT" not in notebook_source
    assert "NOTEBOOK_COMMIT" not in notebook_source
    assert 'REPO_DIR = Path("/content/SET_new")' in notebook_source
    assert 'DRIVE_RAW_CSV = Path("/content/drive/MyDrive/Dataset/combined_secondary_care_medicines.csv")' in notebook_source
    assert 'LOCAL_RAW_CSV = REPO_DIR / "data" / "raw" / "combined_secondary_care_medicines.csv"' in notebook_source
    assert (ROOT / "scripts" / "colab_runtime.py").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "src" / "set_forecasting" / "config.py").is_file()
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/data/raw/*" in gitignore


def test_runtime_artifact_contracts_match_python_pipeline_outputs():
    helper = ROOT / "scripts" / "colab_runtime.py"
    ingestion = (ROOT / "src" / "set_forecasting" / "data" / "ingestion.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "src" / "set_forecasting" / "pipeline.py").read_text(encoding="utf-8")
    mathematical = (ROOT / "src" / "set_forecasting" / "validation" / "mathematical.py").read_text(encoding="utf-8")

    for artifact in literal_tuple_from_module(helper, "INGESTION_ARTIFACTS"):
        assert Path(artifact).name in ingestion, f"Ingestion output not found: {artifact}"
    ingestion_output_names = {Path(item).name for item in literal_tuple_from_module(helper, "INGESTION_ARTIFACTS")}
    for artifact in literal_tuple_from_module(helper, "MODEL_ARTIFACTS"):
        source = ingestion if Path(artifact).name in ingestion_output_names else pipeline
        assert Path(artifact).name in source, f"Model output not found: {artifact}"
    for artifact in literal_tuple_from_module(helper, "MATH_ARTIFACTS"):
        assert Path(artifact).name in mathematical, f"Validation output not found: {artifact}"


def test_notebook_and_helper_use_same_stage_parameters():
    notebook_source = "\n".join(source for _, source in code_cells(load_notebook()))
    helper = (ROOT / "scripts" / "colab_runtime.py").read_text(encoding="utf-8")
    assert "min_periods=MIN_PERIODS" in notebook_source
    assert "min_nonzero=MIN_NONZERO" in notebook_source
    assert "train_frac=TRAIN_FRAC" in notebook_source
    assert "val_frac=VALIDATION_FRAC" in notebook_source
    assert "test_frac=TEST_FRAC" in notebook_source
    assert "bootstrap=BOOTSTRAP_RESAMPLES" in notebook_source
    assert "seed=RANDOM_SEED" in notebook_source
    assert "def run_models(" in helper
    assert "def run_mathematical_validation(" in helper
    assert "require_stage(\"ingestion\"" in helper
    assert "require_stage(\"model_pipeline\"" in helper



def test_notebook_generates_displays_and_syncs_figures():
    notebook_source = "\n".join(source for _, source in code_cells(load_notebook()))
    assert "from src.visualization.generate_figures import generate_figures" in notebook_source
    assert "generated_figures = generate_figures(REPO_DIR)" in notebook_source
    assert "display(Image(filename=str(figure_path)" in notebook_source
    assert "sync_artifacts(REPO_DIR, DRIVE_ARTIFACT_ROOT)" in notebook_source
    generator = ROOT / "src" / "visualization" / "generate_figures.py"
    assert generator.is_file()
    generator_source = generator.read_text(encoding="utf-8")
    assert "figure_generation_report.json" in generator_source
    assert "radcsr_bins =" not in generator_source
    assert "static_bins =" not in generator_source
    assert "phase9_evaluation_summary.csv" not in generator_source


def test_python_config_is_valid_and_notebook_uses_its_defaults():
    from src.set_forecasting.config import get_config, validate_config

    config = validate_config(get_config())
    notebook_source = "\\n".join(source for _, source in code_cells(load_notebook()))
    assert 'from src.set_forecasting.config import get_config, validate_config' in notebook_source
    assert 'MIN_PERIODS = CONFIG["data"]["min_periods"]' in notebook_source
    assert 'MIN_NONZERO = CONFIG["data"]["min_nonzero"]' in notebook_source
    assert config["radcsr"]["kappa_mu"] >= config["radcsr"]["kappa_sigma"]
    assert config["evaluation"]["bootstrap_resamples"] >= 100


def test_active_python_modules_do_not_read_yaml_configuration():
    paths = [
        ROOT / "scripts" / "run_pipeline.py",
        ROOT / "scripts" / "run_inventory_sensitivity.py",
        ROOT / "scripts" / "colab_runtime.py",
        ROOT / "src" / "set_forecasting" / "pipeline.py",
        ROOT / "src" / "scmd_reliability.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "config.yaml" not in source, f"YAML config reference remains in {path}"
        assert "import yaml" not in source, f"YAML dependency remains in {path}"


def test_notebook_defaults_to_fresh_artifacts_and_makes_restore_opt_in():
    notebook_source = "\n".join(source for _, source in code_cells(load_notebook()))
    assert "RESTORE_CHECKPOINTS = False" in notebook_source
    assert "if RESTORE_CHECKPOINTS:" in notebook_source
    assert 'for relative in ("data/processed", "output"):' in notebook_source
    assert "shutil.rmtree(target)" in notebook_source



def test_notebook_contains_and_installs_forecasting_implementation():
    notebook = load_notebook()
    notebook_source = "\n".join(source for _, source in code_cells(notebook))
    markdown_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "def create_features(data):" in notebook_source
    assert "def train_and_predict(" in notebook_source
    assert 'objective="quantile"' in notebook_source
    assert "for quantile in (.1,.5,.9):" in notebook_source
    assert "PreviousObservationNaive" in notebook_source
    assert "SeasonalNaive12" in notebook_source
    assert "pipeline_module.train_and_predict = train_and_predict" in notebook_source
    assert "Notebook-owned forecasting implementation" in markdown_source
    assert "RADCSR formulas and coefficients" in markdown_source
    assert "Inventory outcomes and stockout labels remain simulated" in markdown_source

def test_model_fingerprint_includes_notebook_forecasting_source():
    helper = (ROOT / "scripts" / "colab_runtime.py").read_text(encoding="utf-8")
    assert 'model_code.append("notebooks/SET_Pharma_Colab_Pipeline.ipynb")' in helper
