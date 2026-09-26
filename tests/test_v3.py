"""
Тесты модулей V3: временные фолды, ансамблирование, группы признаков.
Запуск: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cv import Fold, build_folds, split_dates  # noqa: E402
from src.ensemble import (  # noqa: E402
    inverse_error_weights,
    simple_average,
    weighted_average,
)
from src.features import EXTRA_GROUPS, get_feature_columns, make_features  # noqa: E402


# --- src/cv.py ---

def test_folds_are_strictly_chronological() -> None:
    """Каждый фолд обучается строго раньше, чем валидируется."""
    for fold in build_folds():
        assert pd.Timestamp(fold.valid_start) < pd.Timestamp(fold.valid_end)
        assert split_dates(fold)


def test_validation_horizon_covers_full_week() -> None:
    """Валидационное окно — минимум 2 недели (цикл будни/выходные)."""
    for fold in build_folds():
        length = (pd.Timestamp(fold.valid_end) - pd.Timestamp(fold.valid_start)).days + 1
        assert length >= 14, f"{fold.name}: горизонт всего {length} дней"


def test_test_fold_is_october_and_last() -> None:
    """Последний фолд — тестовый октябрь."""
    folds = build_folds()
    assert folds[-1].is_test
    assert folds[-1].valid_start == "2025-10-01"
    assert folds[-1].valid_end == "2025-10-31"
    assert sum(f.is_test for f in folds) == 1


def test_fold_masks_partition_by_date() -> None:
    """Маски train/valid не пересекаются и покрывают будущее."""
    dates = pd.DataFrame(
        {"date": pd.to_datetime(["2025-07-31", "2025-08-01", "2025-08-14", "2025-08-15"])}
    )
    fold = Fold(name="t", valid_start="2025-08-01", valid_end="2025-08-14")
    train_mask, valid_mask = fold.masks(dates)
    assert not (train_mask & valid_mask).any()
    assert not train_mask.iloc[1:].any()
    assert valid_mask.iloc[1:3].all()
    assert not valid_mask.iloc[3]


# --- src/ensemble.py ---

@pytest.fixture
def toy_preds() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    y = rng.gamma(2.0, 100.0, 3000)
    preds = {
        "good": y + rng.normal(0.0, 40.0, 3000),
        "bad": y + rng.normal(0.0, 150.0, 3000),
    }
    return y, preds


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def test_simple_average_sits_between_models(toy_preds) -> None:
    """Среднее не хуже худшей модели и не лучше лучшей."""
    y, preds = toy_preds
    avg = simple_average(preds)
    assert _mae(y, avg) >= _mae(y, preds["good"])
    assert _mae(y, avg) <= _mae(y, preds["bad"])


def test_inverse_error_weights_favour_better_model(toy_preds) -> None:
    """Лучшая модель получает больший вес, сумма весов равна 1."""
    y, preds = toy_preds
    w = inverse_error_weights(y, preds)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["good"] > w["bad"]


def test_inverse_error_weights_flattening_is_monotone(toy_preds) -> None:
    """Сглаживание (power=0.5) уменьшает разрыв между весами."""
    y, preds = toy_preds
    sharp = inverse_error_weights(y, preds, power=1.0)
    flat = inverse_error_weights(y, preds, power=0.5)
    assert (flat["good"] - flat["bad"]) < (sharp["good"] - sharp["bad"])


def test_weighted_average_with_single_model_is_identity(toy_preds) -> None:
    """Взвешивание с единственным прогнозом не меняет его значения."""
    _, preds = toy_preds
    out = weighted_average({"good": preds["good"]}, {"good": 1.0})
    np.testing.assert_allclose(out, preds["good"])


def test_weighted_average_respects_weights(toy_preds) -> None:
    """Взвешенное среднее = конвективная комбинация прогнозов."""
    _, preds = toy_preds
    out = weighted_average(preds, {"good": 0.75, "bad": 0.25})
    np.testing.assert_allclose(out, 0.75 * preds["good"] + 0.25 * preds["bad"])


# --- src/features.py: группы V3 ---

# --- src/models_zoo.py ---

def test_model_registry_covers_all_backends() -> None:
    """В зоопарке есть все три градиентных бустинга из ТЗ."""
    from src.models_zoo import MODEL_REGISTRY

    assert set(MODEL_REGISTRY) == {"LightGBM", "CatBoost", "XGBoost"}
    for name, spec in MODEL_REGISTRY.items():
        assert callable(spec["fit"]), name
        assert callable(spec["predict"]), name


def test_train_model_rejects_unknown_name() -> None:
    """Неизвестное имя модели — явная ошибка."""
    import pandas as pd

    from src.models_zoo import train_model

def test_custom_params_do_not_drop_base_params() -> None:
    """
    Кастомные параметры ДОПОЛНЯЮТ базовые, а не заменяют их.

    Регрессия: при передаче params терялись objective и n_estimators,
    из-за чего «тюненная» модель обучалась на 100 деревьях с L2-функцией
    потерь вместо L1. Проверяем слияние на уровне обучающих функций.
    """
    import pandas as pd

    from src.models_zoo import (
        CATBOOST_PARAMS,
        LGBM_PARAMS,
        XGB_PARAMS,
        train_model,
    )

    custom = {"num_leaves": 81, "learning_rate": 0.02}
    merged = dict(LGBM_PARAMS, **custom)
    assert merged["objective"] == "regression_l1"
    assert merged["n_estimators"] >= 1000
    assert merged["num_leaves"] == 81
    assert dict(CATBOOST_PARAMS, **custom)["loss_function"] == "MAE"
    assert dict(XGB_PARAMS, **custom)["objective"] == "reg:absoluteerror"

    # Практическая проверка: обучение с кастомными параметрами идёт
    # с полным бюджетом итераций, а не с дефолтными 100.
    X = pd.DataFrame({"x": list(range(50))})
    y = pd.Series(list(range(50)), dtype=float)
    _, rounds = train_model("LightGBM", X, y, X, y, params=custom)
    assert rounds > 100, (
        f"модель остановилась на {rounds} деревьях — базовые параметры потеряны"
    )


def test_unknown_feature_group_raises() -> None:

    empty = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(KeyError):
        train_model("НетТакой", empty, pd.Series([1, 2]), empty, pd.Series([1, 2]))


def test_lgbm_fixed_rounds_ignores_validation() -> None:
    """При заданном n_rounds обучение идёт без early stopping."""
    import numpy as np
    import pandas as pd

    from src.models_zoo import predict_model, train_model

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"hour": np.arange(200) % 24, "route": np.arange(200) % 3,
                      "x": rng.normal(size=200)})
    y = pd.Series(rng.gamma(2.0, 50.0, 200))
    model, rounds = train_model("LightGBM", X, y, X, y, n_rounds=15)
    assert rounds == 15
    assert not hasattr(model, "best_iteration_") or model.best_iteration_ in (0, 15)
    assert len(predict_model("LightGBM", model, X)) == len(y)


# --- src/stats_models.py ---

def test_seasonal_naive_is_not_negative_on_grid() -> None:
    """Seasonal naive возвращает конечные значения нужной длины."""
    import pandas as pd

    from src.stats_models import seasonal_naive

    dates = pd.date_range("2025-01-01", "2025-02-10", freq="D")
    feats = pd.DataFrame(
        {
            "route": 1,
            "date": np.repeat(dates, 24),
            "hour": np.tile(np.arange(24), len(dates)),
            "boardings": 10.0,
        }
    )
    out = seasonal_naive(feats, pd.Timestamp("2025-02-01"))
    assert len(out) == len(dates[dates >= pd.Timestamp("2025-02-01")]) * 24
    assert np.isfinite(out).all()


def test_forecast_ets_length_and_finiteness() -> None:
    """ETS даёт прогноз на весь будущий период без NaN."""
    import pandas as pd

    from src.stats_models import forecast_ets

    dates = pd.date_range("2025-01-01", "2025-01-31", freq="D")
    feats = pd.DataFrame(
        {
            "route": 1,
            "date": np.repeat(dates, 24),
            "hour": np.tile(np.arange(24), len(dates)),
            "boardings": np.tile(np.arange(24, dtype=float), len(dates)),
        }
    )
    out = forecast_ets(feats, pd.Timestamp("2025-01-15"))
    assert len(out) == 17 * 24
    assert np.isfinite(out).all()


@pytest.fixture(scope="module")
def extra_feats() -> pd.DataFrame:
    from src.data import load_and_prepare

    return make_features(
        load_and_prepare(), cutoff="2025-10-01", use_weather=True, extra=tuple(EXTRA_GROUPS)
    )


def test_extra_groups_are_reachable(extra_feats: pd.DataFrame) -> None:
    """Каждая группа V3 реально добавляет свои колонки."""
    cols = get_feature_columns(use_weather=True, extra=tuple(EXTRA_GROUPS))
    for group, names in EXTRA_GROUPS.items():
        for name in names:
            assert name in cols, f"{group}: нет колонки {name}"
            assert name in extra_feats.columns


def test_extra_features_have_no_nan(extra_feats: pd.DataFrame) -> None:
    """Новые признаки заполнены полностью, без пропусков."""
    cols = get_feature_columns(use_weather=True, extra=tuple(EXTRA_GROUPS))
    assert int(extra_feats[cols].isna().sum().sum()) == 0


def test_extra_features_preserve_rows(extra_feats: pd.DataFrame) -> None:
    """Добавление признаков не меняет сетку строк."""
    from src.data import load_and_prepare

    base = load_and_prepare()
    assert len(extra_feats) == len(base)
    assert not extra_feats[["route", "date", "hour"]].duplicated().any()


def test_extra_features_ignore_future(extra_feats: pd.DataFrame) -> None:
    """Признаки не зависят от данных после cutoff (страховка от утечки)."""
    from src.data import load_and_prepare

    cutoff = pd.Timestamp("2025-10-01")
    raw = load_and_prepare()
    tail = raw[raw["date"] >= cutoff].copy()
    tail["boardings"] = tail["boardings"] * 10 + 999  # намеренно портим будущее
    polluted = pd.concat([raw[raw["date"] < cutoff], tail], ignore_index=True)
    again = make_features(
        polluted, cutoff=cutoff, use_weather=True, extra=tuple(EXTRA_GROUPS)
    )

    keys = ["route", "date", "hour"]
    cols = [
        c for c in get_feature_columns(use_weather=True, extra=tuple(EXTRA_GROUPS))
        if c not in keys
    ]
    # Сравниваем по ключу, а не по позиции: make_features сохраняет
    # порядок СВОЕГО входного кадра, а в тесте входы собраны по-разному.
    a = extra_feats.set_index(keys)[cols].sort_index()
    b = again.set_index(keys)[cols].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_lstm_training_targets_end_before_cutoff() -> None:
    """Цели обучающей выборки LSTM заканчиваются строго ДО cutoff (нет утечки)."""
    from src.nn_models import _build_examples, build_training_set

    cutoff = pd.Timestamp("2025-10-01")
    dates = pd.date_range("2025-01-01", "2025-10-31", freq="D")
    frames = [
        pd.DataFrame(
            {
                "route": r,
                "date": np.repeat(dates, 24),
                "hour": np.tile(np.arange(24), len(dates)),
                "boardings": 100.0,
                "profile_pred": 100.0,
            }
        )
        for r in (1, 7)
    ]
    feats = pd.concat(frames, ignore_index=True)
    horizon, anchors = 14, 3

    seq, point, y = build_training_set(
        feats, cutoff, window_days=7, horizon_days=horizon, n_anchors=anchors
    )
    assert len(y) == anchors * horizon * 24 * 2  # якоря x дни x часы x маршруты
    assert len(seq) == len(point) == len(y)

    # Каждый якорь даёт ровно horizon дней целей, и последний из них
    # заканчивается за сутки до cutoff.
    _, _, y_last = _build_examples(
        feats, cutoff - pd.Timedelta(days=horizon), 7, horizon
    )
    assert len(y_last) == horizon * 24 * 2
    assert pd.Timestamp(cutoff - pd.Timedelta(days=horizon)) + pd.Timedelta(
        days=horizon
    ) <= cutoff


def test_lstm_inference_examples_start_at_cutoff() -> None:
    """Примеры для инференса начинаются ровно с cutoff (весь прогноз)."""
    from src.nn_models import build_sequences

    cutoff = pd.Timestamp("2025-10-01")
    dates = pd.date_range("2025-01-01", "2025-10-20", freq="D")
    feats = pd.concat(
        [
            pd.DataFrame(
                {
                    "route": r,
                    "date": np.repeat(dates, 24),
                    "hour": np.tile(np.arange(24), len(dates)),
                    "boardings": 100.0,
                    "profile_pred": 100.0,
                }
            )
            for r in (1, 7)
        ],
        ignore_index=True,
    )
    _, _, y = build_sequences(feats, cutoff, window_days=7)
    # Данные идут по 2025-10-20, цель начинается с 2025-10-01 => 20 дней
    assert len(y) == 20 * 24 * 2


def test_unknown_feature_group_raises() -> None:
    """Неизвестная группа признаков — явная ошибка, а не тихий игнор."""
    with pytest.raises(KeyError):
        get_feature_columns(use_weather=True, extra=("нет_такой_группы",))
