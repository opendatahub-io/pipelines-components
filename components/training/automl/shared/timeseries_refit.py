"""Refit selected time series dependencies once and export their original ensembles."""

from __future__ import annotations

import copy
import logging
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TimeSeriesRefitPlan:
    """Preserve selected model identities, configurations and ensemble dependencies.

    AutoGluon's native refit_full has no time limit and copies some model families
    instead of retraining them. Fit each unique base configuration explicitly on
    the full data, then copy the learned ensemble and register its dependencies.
    The private trainer/model APIs used here match AutoGluon's own refit_full.
    """

    def __init__(self, predictor: Any, selected_models: list[str], time_limit: int):
        """Resolve the dependency graph before allocating the preset refit budget."""
        from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel

        self.models: dict[str, Any] = {}
        self.dependencies: dict[str, list[str]] = {}
        self.base_models: dict[str, Any] = {}
        self.fitted: dict[str, Any] = {}
        self.failures: dict[str, Exception] = {}
        self.fit_time_seconds = 0.0
        self.refit_records: list[dict[str, Any]] = []
        visiting = set()

        def visit(name):
            if name in visiting:
                raise ValueError(f"Cyclic time series ensemble dependency: {name}")
            if name in self.models:
                return
            visiting.add(name)
            model = predictor._trainer.load_model(name)
            if isinstance(model, AbstractTimeSeriesEnsembleModel):
                dependencies = list(model.model_names)
                if not dependencies:
                    raise ValueError(f"Ensemble {name!r} has no fitted dependencies")
                for dependency in dependencies:
                    visit(dependency)
                self.dependencies[name] = dependencies
            else:
                self.base_models[name] = model._get_model_base()
            self.models[name] = model
            visiting.remove(name)

        for name in selected_models:
            visit(name)
        self.time_limit_per_model = max(1, time_limit // len(self.base_models))

    def required_models(self, name: str) -> list[str]:
        """Return dependencies followed by the requested model, without duplicates."""
        required = {}

        def visit(model_name):
            if model_name in required:
                return
            for dependency in self.dependencies.get(model_name, []):
                visit(dependency)
            required[model_name] = None

        visit(name)
        return list(required)

    def fit_base_models(self, name: str, train_data: Any, path: Path, predictor_kwargs: dict) -> None:
        """Train missing base dependencies, remembering successes and failures."""
        from autogluon.timeseries import TimeSeriesPredictor

        for base_name in self.required_models(name):
            if base_name not in self.base_models:
                continue
            if base_name in self.failures:
                raise RuntimeError(f"Dependency {base_name!r} failed refit") from self.failures[base_name]
            if base_name in self.fitted:
                continue
            base_model = self.base_models[base_name]
            hyperparameters = copy.deepcopy(base_model.get_params()["hyperparameters"])
            hyperparameters["ag_args"] = {"name": base_name}
            model_path = path / base_name
            model_path.mkdir(parents=True, exist_ok=True)
            refit = TimeSeriesPredictor(path=str(model_path), **predictor_kwargs)
            started = time.perf_counter()
            record = {"model": base_name, "status": "failed"}
            try:
                try:
                    refit.fit(
                        train_data=train_data,
                        hyperparameters={type(base_model): hyperparameters},
                        time_limit=self.time_limit_per_model,
                        enable_ensemble=False,
                        # Exactly one fixed configuration: use all rows without reserving
                        # another validation horizon or fitting backtest windows.
                        skip_model_selection=True,
                    )
                finally:
                    # AutoGluon persists the full refit frame. Release it before
                    # caching or exporting predictors, also when fitting fails.
                    # Evaluation, backtesting and inference receive data in memory.
                    (model_path / "utils" / "data" / "train.pkl").unlink(missing_ok=True)
                if refit.model_names() != [base_name]:
                    raise RuntimeError(f"Unexpected refit model set for {base_name!r}: {refit.model_names()}")
                self.fitted[base_name] = refit
                record["status"] = "completed"
            except Exception as exc:
                self.failures[base_name] = exc
                raise
            finally:
                elapsed = time.perf_counter() - started
                self.fit_time_seconds += elapsed
                record["elapsed_seconds"] = elapsed
                self.refit_records.append(record)
                logger.info("Time series base refit: %s", record)

    def export(self, name: str, output_path: Path) -> Any:
        """Export a self-contained predictor whose default is the selected model."""
        from autogluon.timeseries import TimeSeriesPredictor

        required = self.required_models(name)
        base_names = [model for model in required if model in self.base_models]
        template = self.fitted[base_names[0]]
        shutil.copytree(template.path, output_path)
        exported = TimeSeriesPredictor.load(str(output_path))
        trainer = exported._trainer
        for base_name in base_names[1:]:
            model = copy.deepcopy(self.fitted[base_name]._trainer.load_model(base_name))
            model_path = Path(trainer.path) / base_name
            shutil.copytree(model.path, model_path)
            model.set_contexts(str(model_path))
            trainer._add_model(model)
        for ensemble_name in required:
            if ensemble_name not in self.dependencies:
                continue
            ensemble = copy.deepcopy(self.models[ensemble_name])
            ensemble.set_contexts(str(Path(trainer.path) / ensemble_name))
            # Refit uses the original names, so learned weights and dependency
            # identity remain intact, including distinct configs of one family.
            trainer._add_model(ensemble, base_models=self.dependencies[ensemble_name])
            trainer.save_model(ensemble)
        trainer.model_best = name
        trainer.save()
        exported.save()
        return exported

    def metadata(self, name: str) -> dict[str, Any]:
        """Describe the actual model served behind a catalog output label."""
        return {
            "prediction_model": name,
            "refit_base_models": [model for model in self.required_models(name) if model in self.base_models],
            "ensemble_dependencies": {
                model: self.dependencies[model] for model in self.required_models(name) if model in self.dependencies
            },
        }
