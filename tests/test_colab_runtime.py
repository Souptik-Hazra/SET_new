from pathlib import Path

from scripts import colab_runtime


def test_stage_checkpoint_skips_only_when_manifest_and_outputs_match(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(colab_runtime, "MANIFEST_PATH", tmp_path / "output" / "colab_run_manifest.json")
    artifact = "data/processed/result.parquet"
    calls = []

    def action():
        calls.append("ran")
        target = tmp_path / artifact
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"parquet-placeholder")
        return "ok"

    first = colab_runtime.run_stage("demo", "fingerprint-a", (artifact,), action)
    second = colab_runtime.run_stage("demo", "fingerprint-a", (artifact,), action)

    assert first["status"] == "COMPLETED"
    assert second["status"] == "SKIPPED_VALID_CHECKPOINT"
    assert calls == ["ran"]


def test_stage_reruns_when_fingerprint_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(colab_runtime, "MANIFEST_PATH", tmp_path / "output" / "colab_run_manifest.json")
    artifact = "output/result.json"
    calls = []

    def action():
        calls.append(1)
        target = tmp_path / artifact
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(len(calls)), encoding="utf-8")

    colab_runtime.run_stage("demo", "fingerprint-a", (artifact,), action)
    colab_runtime.run_stage("demo", "fingerprint-b", (artifact,), action)

    assert len(calls) == 2


def test_stage_reruns_when_artifact_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(colab_runtime, "MANIFEST_PATH", tmp_path / "output" / "colab_run_manifest.json")
    artifact = "data/processed/result.parquet"
    calls = []

    def action():
        calls.append(1)
        target = tmp_path / artifact
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"complete")

    colab_runtime.run_stage("demo", "fingerprint-a", (artifact,), action)
    (tmp_path / artifact).unlink()
    colab_runtime.run_stage("demo", "fingerprint-a", (artifact,), action)

    assert len(calls) == 2


def test_sync_artifacts_copies_processed_and_output_but_not_raw(tmp_path):
    source = tmp_path / "runtime"
    destination = tmp_path / "drive"
    (source / "data" / "processed").mkdir(parents=True)
    (source / "data" / "raw").mkdir(parents=True)
    (source / "output").mkdir(parents=True)
    (source / "data" / "processed" / "x.parquet").write_bytes(b"processed")
    (source / "data" / "raw" / "private.csv").write_text("raw", encoding="utf-8")
    (source / "output" / "report.json").write_text("{}", encoding="utf-8")

    copied = colab_runtime.sync_artifacts(source, destination)

    assert "data/processed/x.parquet" in copied
    assert "output/report.json" in copied
    assert not (destination / "data" / "raw" / "private.csv").exists()


def test_failed_forced_stage_invalidates_previous_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(colab_runtime, "MANIFEST_PATH", tmp_path / "output" / "colab_run_manifest.json")
    artifact = "output/result.json"
    target = tmp_path / artifact
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("previous-success", encoding="utf-8")
    colab_runtime._write_manifest({
        "version": 1,
        "stages": {"demo": {
            "fingerprint": "fingerprint-a",
            "artifacts": [artifact],
            "status": "completed",
        }},
    })

    def failing_action():
        raise RuntimeError("simulated failure")

    import pytest
    with pytest.raises(RuntimeError, match="simulated failure"):
        colab_runtime.run_stage(
            "demo", "fingerprint-a", (artifact,), failing_action, force=True
        )

    assert not colab_runtime.stage_is_complete("demo", "fingerprint-a", (artifact,))


def test_upstream_rerun_invalidates_downstream_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(colab_runtime, "MANIFEST_PATH", tmp_path / "output" / "colab_run_manifest.json")
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    artifact = "output/ingestion.json"
    (output / "ingestion.json").write_text("ok", encoding="utf-8")
    (output / "math.json").write_text("old", encoding="utf-8")
    colab_runtime._write_manifest({
        "version": 1,
        "stages": {
            "ingestion": {"fingerprint": "old", "artifacts": [artifact], "status": "completed"},
            "mathematical_validation": {"fingerprint": "math-old", "artifacts": ["output/math.json"], "status": "completed"},
        },
    })

    colab_runtime.run_stage(
        "ingestion", "new", (artifact,), lambda: None,
        force=True, invalidate_stages=("mathematical_validation",),
    )

    assert not colab_runtime.stage_is_complete(
        "mathematical_validation", "math-old", ("output/math.json",)
    )


def test_fingerprint_changes_when_stage_code_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runtime, "ROOT", tmp_path)
    source = tmp_path / "input.csv"
    config = tmp_path / "config" / "config.yaml"
    code = tmp_path / "src" / "module.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    code.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("source", encoding="utf-8")
    config.write_text("seed: 42", encoding="utf-8")
    code.write_text("VALUE = 1", encoding="utf-8")

    before = colab_runtime.run_fingerprint(source, {"stage": "test"}, ("src/module.py",))
    code.write_text("VALUE = 2", encoding="utf-8")
    after = colab_runtime.run_fingerprint(source, {"stage": "test"}, ("src/module.py",))

    assert before != after


def test_require_stage_rejects_missing_or_stale_upstream_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(colab_runtime, "MANIFEST_PATH", tmp_path / "output" / "colab_run_manifest.json")

    import pytest
    with pytest.raises(RuntimeError, match="Required stage 'ingestion'"):
        colab_runtime.require_stage("ingestion", "expected", ("data/processed/series.parquet",))

    artifact = tmp_path / "data" / "processed" / "series.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"series")
    colab_runtime._write_manifest({
        "version": 1,
        "stages": {"ingestion": {
            "fingerprint": "expected",
            "artifacts": ["data/processed/series.parquet"],
            "status": "completed",
        }},
    })
    colab_runtime.require_stage(
        "ingestion", "expected", ("data/processed/series.parquet",)
    )

    with pytest.raises(RuntimeError, match="Required stage 'ingestion'"):
        colab_runtime.require_stage(
            "ingestion", "stale", ("data/processed/series.parquet",)
        )



def test_python_313_is_required(monkeypatch):
    from types import SimpleNamespace
    import pytest

    monkeypatch.setattr(
        colab_runtime.sys, "version_info",
        SimpleNamespace(major=3, minor=12),
    )
    monkeypatch.delenv("SET_ALLOW_UNSUPPORTED_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="requires Python 3.13.18"):
        colab_runtime.assert_supported_python()

    monkeypatch.setattr(
        colab_runtime.sys, "version_info",
        SimpleNamespace(major=3, minor=13),
    )
    result = colab_runtime.assert_supported_python()
    assert result["python_supported"] is True
    assert result["python_override"] is False



def test_model_checkpoint_requires_every_declared_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_runtime, "ROOT", tmp_path)
    monkeypatch.setattr(colab_runtime, "MANIFEST_PATH", tmp_path / "output" / "colab_run_manifest.json")
    artifacts = colab_runtime.MODEL_ARTIFACTS
    for relative in artifacts:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"verified")
    colab_runtime._write_manifest({
        "version": 1,
        "stages": {"model_pipeline": {
            "fingerprint": "complete",
            "artifacts": list(artifacts),
            "status": "completed",
        }},
    })
    assert colab_runtime.stage_is_complete("model_pipeline", "complete", artifacts)
    for relative in artifacts:
        target = tmp_path / relative
        target.unlink()
        assert not colab_runtime.stage_is_complete("model_pipeline", "complete", artifacts), relative
        target.write_bytes(b"restored")



def test_canonical_pipeline_rejects_python_other_than_313(monkeypatch):
    import pytest
    from src.set_forecasting import pipeline

    monkeypatch.setattr(
        pipeline.platform,
        "python_version_tuple",
        lambda: ("3", "12", "0"),
    )
    with pytest.raises(RuntimeError, match="requires Python 3.13.x"):
        pipeline.run_model_pipeline()
