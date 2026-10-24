"""Causal hourly features, train-only normalization and purged labels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from learning.config import LearningConfig


@dataclass
class PreparedData:
    """Chronological arrays and partition boundaries; indices are decision closes."""

    frame: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_end: int
    validation_end: int
    test_start: int
    volatility: np.ndarray


def validate_candles(frame: pd.DataFrame) -> None:
    """Require complete sorted hourly OHLCV with physically valid candles."""
    columns = ["open", "high", "low", "close", "volume"]
    if not isinstance(frame.index, pd.DatetimeIndex) or not set(columns) <= set(frame):
        raise ValueError("Hourly OHLCV with a DatetimeIndex is required")
    if len(frame) < 2 or frame.index.has_duplicates or frame.index.hasnans:
        raise ValueError("Nonempty unique timestamps are required")
    if not (frame.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Expected continuous hourly candles without gaps")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Prices must be positive and volume nonnegative, all finite")
    if (
        (frame.high < frame[["open", "close", "low"]].max(axis=1))
        | (frame.low > frame[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("Invalid OHLC candle geometry")


def causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute only past/current information, without backwards filling warmup."""
    close = frame.close
    returns = np.log(close).diff()
    output = pd.DataFrame(index=frame.index)
    for period in (1, 4, 12, 24, 72, 168):
        output[f"return_{period}"] = np.log(close / close.shift(period))
    for period in (12, 24, 72):
        output[f"volatility_{period}"] = returns.rolling(period).std()
        output[f"ma_deviation_{period}"] = close / close.rolling(period).mean() - 1
    output["range"] = (frame.high - frame.low) / close
    output["body"] = (close - frame.open) / frame.open
    output["upper_wick"] = (frame.high - frame[["open", "close"]].max(axis=1)) / close
    output["lower_wick"] = (frame[["open", "close"]].min(axis=1) - frame.low) / close
    output["log_volume_change"] = np.log1p(frame.volume).diff()
    output["volume_ratio"] = frame.volume / frame.volume.rolling(24).mean().replace(0, np.nan) - 1
    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    output["rsi"] = gains / (gains + losses).replace(0, np.nan)
    output["rsi"] = output["rsi"].fillna(0.5)
    tr = pd.concat(
        [
            frame.high - frame.low,
            (frame.high - close.shift()).abs(),
            (frame.low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    output["atr_ratio"] = tr.rolling(14).mean() / close
    output["hour_sin"] = np.sin(2 * np.pi * frame.index.hour / 24)
    output["hour_cos"] = np.cos(2 * np.pi * frame.index.hour / 24)
    return output


def transform_features(features: pd.DataFrame, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply stored normalization; zero marks unavailable historical warmup."""
    values = features.to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    normalized = np.clip((values - mean) / scale, -10, 10)
    normalized[missing] = 0
    return np.c_[normalized, missing.astype(float)].astype(np.float32)


def prepare_data(frame: pd.DataFrame, config: LearningConfig) -> PreparedData:
    """Fit training-only scaler and purge future labels from partition boundaries."""
    validate_candles(frame)
    n = len(frame)
    train_end = int(n * config.train_ratio)
    validation_end = int(n * (config.train_ratio + config.validation_ratio))
    warmup = 168 + config.lookback - 1
    horizon = max(config.horizons)
    if (
        train_end - horizon <= warmup
        or min(validation_end - train_end, n - validation_end) <= horizon + 1
    ):
        raise ValueError("Insufficient data for warmup and purged chronological partitions")
    raw = causal_features(frame)
    mean = raw.iloc[168:train_end].mean().to_numpy()
    scale = raw.iloc[168:train_end].std().to_numpy()
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    mean = np.nan_to_num(mean)
    labels = np.stack(
        [(frame.close.shift(-h) / frame.open.shift(-1) - 1).to_numpy() for h in config.horizons],
        axis=1,
    ).astype(np.float32)
    names = list(raw.columns)
    volatility = raw.volatility_24.fillna(0).to_numpy(dtype=np.float32)
    return PreparedData(
        frame,
        transform_features(raw, mean, scale),
        labels,
        mean,
        scale,
        names + [f"{name}_missing" for name in names],
        np.arange(warmup, train_end - horizon),
        np.arange(train_end, validation_end - horizon),
        train_end,
        validation_end,
        validation_end,
        volatility,
    )
