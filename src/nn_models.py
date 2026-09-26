"""
Нейросетевые модели последовательностей (PyTorch, CPU).

Задача специфична: прогноз на 2 месяца вперёд без «свежих» данных, поэтому
классическую схему «окно -> следующий час» применить нельзя — в момент
инференса для декабря мы не знаем ноябрь. Вместо этого NN получает
историю каждого маршрута, сжатую до окна перед cutoff, и учится
предсказывать уровень будущего часа по календарю, погоде и профилю.

Архитектура: LSTM (контекст последних дней) + скалярное self-attention
над состояниями + полносвязная «голова» на фиче профиля.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Гиперпараметры по умолчанию (подбирались вручную по скорости/качеству)
DEFAULT_HP: dict[str, Any] = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.1,
    "learning_rate": 0.01,
    "batch_size": 512,
    "epochs": 8,
    "weight_decay": 1e-4,
    "window_days": 21,
    # Сколько исторических якорей используется для обучения и какой
    # горизонт каждый из них закрывает. Обучение полностью отделено от
    # прогнозного периода: якоря стоят строго ДО cutoff.
    "horizon_days": 14,
    "n_anchors": 10,
    "anchor_step_days": 7,
}

# Признаки, по которым строится окно истории (безопасные: без утечки)
SEQUENCE_FEATURES: list[str] = ["boardings"]


class Attention(nn.Module):
    """Скалярное self-attention: усреднение состояний с обучаемыми весами."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(states).squeeze(-1), dim=1)
        return (states * weights.unsqueeze(-1)).sum(dim=1)


class BoardingsLSTM(nn.Module):
    """LSTM по окну истории + attention-пулинг + голова на календарных фичах."""

    def __init__(self, n_seq: int, n_feat: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_seq + 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, seq: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        states, _ = self.lstm(seq)
        pooled = self.attention(states)
        flat = seq[:, -1, :]
        return self.head(torch.cat([pooled, flat, point], dim=1)).squeeze(-1)

def build_sequences(
    feats: pd.DataFrame, cutoff: pd.Timestamp, window_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Примеры для ИНФЕРЕНСА: окно последних `window_days` дней, заканчивающееся
    строго ДО `cutoff`, и цели — все строки начиная с `cutoff`.

    Окно одно и то же для всех строк: это ровно та информация, которая
    доступна в реальной задаче (прогноз на 2 месяца без свежих данных).

    :return: (seq [N, T, 1], point [N, 1], y [N]).
    """
    return _build_examples(
        feats, anchor=pd.Timestamp(cutoff), window_days=int(window_days),
        horizon_days=None,
    )


def _build_examples(
    feats: pd.DataFrame,
    anchor: pd.Timestamp,
    window_days: int,
    horizon_days: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Строит (окно перед якорем, профиль, цель) для заданного якоря.

    :param anchor: момент, на который «заканчивается» известная история.
    :param horizon_days: None — брать все строки начиная с якоря (инференс);
        целое число — ограничить горизонт столькими днями (обучение).
    """
    window = window_days * 24
    hist = feats[feats["date"] < anchor].sort_values(["route", "date", "hour"])
    hist = hist[hist["date"] >= anchor - pd.Timedelta(days=window_days)]

    series: dict[int, np.ndarray] = {}
    for route, grp in hist.groupby("route", sort=False):
        values = grp.sort_values(["date", "hour"])["boardings"].to_numpy(dtype=np.float32)
        if len(values) < window:
            values = np.pad(values, (window - len(values), 0))
        series[int(route)] = values[-window:]

    end = anchor + pd.Timedelta(days=horizon_days) if horizon_days else None
    fut = feats[(feats["date"] >= anchor) & (feats["date"] < end if end is not None else True)]
    fut = fut.sort_values(["route", "date", "hour"])
    seq = np.stack([series[int(r)] for r in fut["route"]], axis=0)[:, :, None]
    point = fut["profile_pred"].to_numpy(dtype=np.float32)[:, None]
    y = fut["boardings"].to_numpy(dtype=np.float32)
    return seq, point, y


def build_training_set(
    feats: pd.DataFrame,
    cutoff: pd.Timestamp,
    window_days: int,
    horizon_days: int = 14,
    n_anchors: int = 8,
    step_days: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Обучающая выборка из НЕСКОЛЬКИХ исторических якорей перед `cutoff`.

    Это ключевой момент против утечки: сеть учится предсказывать `horizon_days`
    вперёд от якоря, используя только окно ДО якоря. Так тренировка
    повторяет условия инференса (прогноз вперёд без свежих данных), а
    строки валидационного периода в обучение не попадают.

    :param n_anchors: сколько якорей брать (каждая даёт window/horizon примеров).
    :param step_days: шаг между якорями в днях.
    """
    cutoff = pd.Timestamp(cutoff)
    seqs, points, ys = [], [], []
    for k in range(n_anchors):
        # Отступ нужен, чтобы цель последнего якоря закончилась ДО cutoff:
        # иначе самый свежий якорь заходил бы в прогнозный период.
        anchor = cutoff - pd.Timedelta(days=horizon_days) - pd.Timedelta(days=k * step_days)
        s, p, y = _build_examples(feats, anchor, window_days, horizon_days)
        if len(y):
            seqs.append(s)
            points.append(p)
            ys.append(y)
    if not ys:
        raise ValueError("Не удалось построить обучающие примеры: слишком короткая история")
    return np.concatenate(seqs), np.concatenate(points), np.concatenate(ys)



def fit_lstm(
    feats: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    hp: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    """
    Обучает LSTM+attention. Loss — L1 (совпадает с WAPE по построению).

    Обучающие примеры берутся ТОЛЬКО из прошлого относительно `cutoff`
    (несколько исторических якорей), поэтому строки прогнозного периода
    в обучение не попадают — утечки нет.

    :return: обученная модель и статистика нормализации/обучения.
    """
    params = {**DEFAULT_HP, **(hp or {})}
    cutoff = pd.Timestamp(cutoff)

    seq, point, y = build_training_set(
        feats,
        cutoff=cutoff,
        window_days=params["window_days"],
        horizon_days=params["horizon_days"],
        n_anchors=params["n_anchors"],
        step_days=params["anchor_step_days"],
    )
    logger.info(
        "LSTM: обучающих примеров %d (якорей=%d, горизонт=%d дн.)",
        len(y), params["n_anchors"], params["horizon_days"],
    )

    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    # Стандартизация по обучению (окно и цель)
    seq_mean, seq_std = float(seq.mean()), float(seq.std() + 1e-6)
    y_mean, y_std = float(y.mean()), float(y.std() + 1e-6)

    x = torch.tensor((seq - seq_mean) / seq_std, dtype=torch.float32)
    p = torch.tensor((point - point.mean()) / (point.std() + 1e-6), dtype=torch.float32)
    t = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)

    model = BoardingsLSTM(
        n_seq=1, n_feat=x.shape[-1],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )
    loss_fn = nn.L1Loss()

    n = len(x)
    batch = min(params["batch_size"], n)
    last_loss = float("nan")
    for epoch in range(params["epochs"]):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx], p[idx]), t[idx])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        last_loss = total / n
        logger.info("LSTM epoch %d/%d train L1(z)=%.4f", epoch + 1, params["epochs"], last_loss)

    model.eval()
    stats = {
        "train_l1_z": last_loss,
        "n_rows": float(n),
        "epochs": float(params["epochs"]),
        "window_days": params["window_days"],
        "y_mean": y_mean,
        "y_std": y_std,
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "point_mean": float(point.mean()),
        "point_std": float(point.std() + 1e-6),
    }
    return model, stats


def predict_lstm(model: nn.Module, stats: dict[str, float], seq: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Предсказание LSTM в исходном масштабе пассажиропотока."""
    x = torch.tensor((seq - stats["seq_mean"]) / stats["seq_std"], dtype=torch.float32)
    p = torch.tensor(
        (point - stats["point_mean"]) / stats["point_std"], dtype=torch.float32
    )
    with torch.no_grad():
        z = model(x, p).numpy()
    return z * stats["y_std"] + stats["y_mean"]

