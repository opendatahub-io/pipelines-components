"""Regression tests for dependency-preserving time series refits."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ..timeseries_refit import TimeSeriesRefitPlan


class BaseModel:
    """A model configuration, with a separately stored trained asset."""

    def __init__(self, name, **hyperparameters):
        """Keep identity and all fit options for assertions."""
        self.name = name
        self.hyperparameters = hyperparameters

    def _get_model_base(self):
        return self

    def get_params(self):
        """Return AutoGluon's constructor configuration."""
        return {"hyperparameters": self.hyperparameters}


class Ensemble:
    """A fitted ensemble whose weights must survive export unchanged."""

    def __init__(self, name, weights):
        """Store the selected ensemble's learned weights."""
        self.name = name
        self.model_to_weight = weights

    @property
    def model_names(self):
        """Return selected dependencies."""
        return list(self.model_to_weight)

    def set_contexts(self, path):
        """Relocate saved model assets."""
        self.path = path


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    """Expose only the AutoGluon APIs used by the refit plan."""
    predictor_cls = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "autogluon.timeseries", SimpleNamespace(TimeSeriesPredictor=predictor_cls))
    monkeypatch.setitem(
        sys.modules,
        "autogluon.timeseries.models.ensemble",
        SimpleNamespace(AbstractTimeSeriesEnsembleModel=Ensemble),
    )
    created = []

    def construct(path, **kwargs):
        refit = mock.MagicMock(path=path)
        refit.path = path
        name = Path(path).name
        refit.model_names.return_value = [name]
        # A backend asset beyond the model pickle must be copied too.
        model_path = Path(path) / "models" / name
        model_path.mkdir(parents=True)
        (model_path / "weights.bin").write_bytes(name.encode())
        fitted_model = mock.MagicMock(name=name)
        fitted_model.path = str(model_path)
        refit._trainer.load_model.return_value = fitted_model
        created.append(refit)
        return refit

    predictor_cls.side_effect = construct
    exported = mock.MagicMock()
    exported._trainer.path = str(tmp_path / "export" / "models")
    predictor_cls.load.return_value = exported
    return SimpleNamespace(cls=predictor_cls, created=created, exported=exported)


def selection(*models):
    """Create a source trainer with explicit model lookup."""
    predictor = mock.MagicMock()
    predictor._trainer.load_model.side_effect = {model.name: model for model in models}.__getitem__
    return predictor


def test_single_member_ensemble_and_standalone_train_only_once(sdk, tmp_path):
    """Weight-one TFT and its standalone selection share one restricted fit."""
    tft = BaseModel("TemporalFusionTransformer", hidden_dim=32, ag_args_fit={"max_time_limit_ratio": 0.8})
    ensemble = Ensemble("WeightedEnsemble", {tft.name: 1.0})
    plan = TimeSeriesRefitPlan(selection(tft, ensemble), [ensemble.name, tft.name], 3600)
    data = object()
    for name in [ensemble.name, tft.name, ensemble.name]:
        plan.fit_base_models(name, data, tmp_path / "cache", {"target": "sales"})

    sdk.cls.assert_called_once()
    sdk.created[0].fit.assert_called_once_with(
        train_data=data,
        hyperparameters={
            BaseModel: {
                "hidden_dim": 32,
                "ag_args_fit": {"max_time_limit_ratio": 0.8},
                "ag_args": {"name": tft.name},
            }
        },
        time_limit=3600,
        enable_ensemble=False,
        skip_model_selection=True,
    )
    assert "ag_args" not in tft.hyperparameters
    assert plan.refit_records[0]["model"] == tft.name
    assert plan.refit_records[0]["status"] == "completed"
    assert plan.refit_records[0]["elapsed_seconds"] >= 0

    plan.export(ensemble.name, tmp_path / "export")
    trainer = sdk.exported._trainer
    saved_ensemble = trainer.save_model.call_args.args[0]
    assert saved_ensemble.model_to_weight == {tft.name: 1.0}
    assert saved_ensemble is not ensemble
    trainer._add_model.assert_called_once_with(saved_ensemble, base_models=[tft.name])
    assert trainer.model_best == ensemble.name
    trainer.save.assert_called_once()
    sdk.exported.save.assert_called_once()
    assert (tmp_path / "export" / "models" / tft.name / "weights.bin").exists()
    assert plan.metadata(ensemble.name) == {
        "prediction_model": ensemble.name,
        "refit_base_models": [tft.name],
        "ensemble_dependencies": {ensemble.name: [tft.name]},
    }


def test_multi_member_same_family_preserves_configs_weights_and_assets(sdk, tmp_path):
    """Distinct configurations of the same family retain their own names and weights."""
    first = BaseModel("TFT_small", hidden_dim=16)
    second = BaseModel("TFT_large", hidden_dim=64)
    wrapper = mock.MagicMock(name=second.name)
    wrapper.name = second.name
    wrapper._get_model_base.return_value = second
    ensemble = Ensemble("WeightedEnsemble", {first.name: 0.25, second.name: 0.75})
    plan = TimeSeriesRefitPlan(selection(first, wrapper, ensemble), [ensemble.name, first.name, second.name], 600)
    original_weights = copy.deepcopy(ensemble.model_to_weight)
    for name in [ensemble.name, second.name, first.name]:
        plan.fit_base_models(name, object(), tmp_path / "cache", {})
    assert len(sdk.created) == 2
    assert plan.time_limit_per_model == 300
    for refit, model in zip(sdk.created, [first, second], strict=True):
        assert refit.fit.call_args.kwargs["hyperparameters"] == {
            BaseModel: {**model.hyperparameters, "ag_args": {"name": model.name}}
        }
    plan.export(ensemble.name, tmp_path / "export")
    trainer = sdk.exported._trainer
    saved = trainer.save_model.call_args.args[0]
    assert saved.model_to_weight == original_weights == ensemble.model_to_weight
    assert trainer._add_model.call_args.kwargs["base_models"] == [first.name, second.name]
    assert (tmp_path / "export" / "models" / second.name / "weights.bin").read_bytes() == second.name.encode()
    assert plan.metadata(ensemble.name)["refit_base_models"] == [first.name, second.name]


def test_failed_shared_dependency_is_not_retrained(sdk, tmp_path):
    """A failed dependency fails its ensemble and standalone output without retrying."""
    tft = BaseModel("TFT")
    ensemble = Ensemble("WeightedEnsemble", {tft.name: 1.0})
    plan = TimeSeriesRefitPlan(selection(tft, ensemble), [ensemble.name, tft.name], 600)
    original_construct = sdk.cls.side_effect

    def fail_fit(**kwargs):
        refit = original_construct(**kwargs)
        refit.fit.side_effect = RuntimeError("OOM")
        return refit

    sdk.cls.side_effect = fail_fit
    with pytest.raises(RuntimeError, match="OOM"):
        plan.fit_base_models(ensemble.name, object(), tmp_path / "cache", {})
    with pytest.raises(RuntimeError, match="Dependency 'TFT' failed refit"):
        plan.fit_base_models(tft.name, object(), tmp_path / "cache", {})
    assert len(sdk.created) == 1
    assert plan.refit_records[0]["status"] == "failed"
    assert plan.refit_records[0]["elapsed_seconds"] >= 0


def test_nested_ensembles_share_dependencies_and_export_in_order(sdk, tmp_path):
    """Nested selected ensembles preserve the entire dependency graph."""
    base = BaseModel("TFT")
    inner = Ensemble("InnerEnsemble", {base.name: 1.0})
    outer = Ensemble("OuterEnsemble", {inner.name: 0.5, base.name: 0.5})
    plan = TimeSeriesRefitPlan(selection(base, inner, outer), [outer.name, inner.name], 600)
    plan.fit_base_models(outer.name, object(), tmp_path / "cache", {})
    assert plan.required_models(outer.name) == [base.name, inner.name, outer.name]
    plan.export(outer.name, tmp_path / "export")
    saved_names = [call.args[0].name for call in sdk.exported._trainer.save_model.call_args_list]
    assert saved_names == [inner.name, outer.name]
    assert sdk.exported._trainer.model_best == outer.name


@pytest.mark.parametrize("weights", [{}, {"WeightedEnsemble": 1.0}])
def test_invalid_ensemble_fails_without_default_fit(sdk, weights):
    """Missing or cyclic dependencies cannot silently launch default training."""
    ensemble = Ensemble("WeightedEnsemble", weights)
    with pytest.raises(ValueError):
        TimeSeriesRefitPlan(selection(ensemble), [ensemble.name], 600)
    sdk.cls.assert_not_called()
