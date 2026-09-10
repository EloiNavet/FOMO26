from asparagus.pipeline.auto_configuration import logging as logging_module


def test_local_csv_history_is_opt_in_and_coexists_with_base_logger(monkeypatch, tmp_path) -> None:
    created = []

    class FakeBase:
        def __init__(self, **kwargs):
            created.append(("base", kwargs))

    class FakeCsv:
        def __init__(self, **kwargs):
            created.append(("csv", kwargs))

    monkeypatch.setattr(logging_module, "BaseLogger", FakeBase)
    monkeypatch.setattr(logging_module, "CSVLogger", FakeCsv)
    loggers = logging_module.logging(
        ckpt_wandb_id=None,
        ckpt_mlflow_id=None,
        log_file_name="train.log",
        run_dir=str(tmp_path),
        version="v3",
        wandb_experiment="task5-v3",
        wandb_logging=False,
        csv_logging=True,
        mlflow_logging=False,
    )

    assert len(loggers) == 2
    assert created[0][0] == "base"
    assert created[1] == ("csv", {"save_dir": str(tmp_path), "name": "local_history", "version": ""})
