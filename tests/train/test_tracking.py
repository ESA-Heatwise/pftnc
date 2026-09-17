from pftnc.train.tracking import ArtifactStore, MlflowTracker


def test_artifact_store_writes_json(tmp_path):
    path = ArtifactStore.write_json(tmp_path / "nested" / "metrics.json", {"mse": 1.5})

    assert path.read_text() == '{\n  "mse": 1.5\n}'


def test_tracker_logs_metric_without_step(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "pftnc.train.tracking.mlflow.log_metric",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    MlflowTracker("test").log_metric("mse", 1.5)

    assert calls == [(("mse", 1.5), {})]


def test_tracker_logs_metric_with_step(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "pftnc.train.tracking.mlflow.log_metric",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    MlflowTracker("test").log_metric("mse", 1.5, step=3)

    assert calls == [(("mse", 1.5), {"step": 3})]
