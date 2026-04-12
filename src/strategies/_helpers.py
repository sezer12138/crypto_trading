"""
Strategy Helper Functions

Provides shared computation logic commonly used in strategies to avoid code duplication.
Includes: signal forward fill, crossover detection, RSI calculation, event-based signal
conversion, and trend filter.
"""

import numpy as np
import pandas as pd
from strategies.constants import DEFAULT_RSI_PERIOD


def forward_fill_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate position state column based on signal column

    Forward fills 0 values in the signal column, representing maintaining the current
    position when no new signal occurs. i.e., hold position after a buy signal until
    a sell signal appears.

    Args:
        df: DataFrame containing a 'signal' column

    Returns:
        DataFrame with a 'position' column added.
        position values: 1=In position, 0=No position, -1=Sell (instantaneous)

    Example:
        >>> df['signal'] = [0, 1, 0, 0, -1, 0]
        >>> df = forward_fill_position(df)
        >>> df['position'].tolist()
        [0, 1, 1, 1, -1, -1]
    """
    df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)
    return df


def detect_crossover(
    df: pd.DataFrame,
    fast_col: str,
    slow_col: str,
) -> pd.DataFrame:
    """
    Detect crossover signals between two columns (Golden Cross / Death Cross)

    Golden Cross: fast_col crosses above slow_col from below, generating a buy signal.
    Death Cross: fast_col crosses below slow_col from above, generating a sell signal.

    Args:
        df: DataFrame containing the two data columns
        fast_col: Fast line column name (e.g., 'ma_short', 'macd')
        slow_col: Slow line column name (e.g., 'ma_long', 'macd_signal')

    Returns:
        DataFrame with a 'signal' column added
        signal: 1=Golden Cross (Buy), -1=Death Cross (Sell), 0=No signal

    Example:
        >>> df = detect_crossover(df, 'ma_short', 'ma_long')
        >>> buy_signals = df[df['signal'] == 1]
    """
    df["signal"] = 0

    # Golden Cross: fast line crosses above slow line
    df.loc[
        (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1)),
        "signal",
    ] = 1

    # Death Cross: fast line crosses below slow line
    df.loc[
        (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1)),
        "signal",
    ] = -1

    return df


def convert_to_event_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert state-based signals to event-based signals.

    State-based strategies set signal=1 or -1 on every bar where a condition is true,
    causing over-trading. This helper keeps only the first bar of each state change,
    converting consecutive identical signals into a single event.

    Args:
        df: DataFrame containing a 'signal' column with state-based signals

    Returns:
        DataFrame with the 'signal' column modified to contain only event-based signals.
        Only the first bar where signal changes from 0 to 1, or 0 to -1, or from one
        direction to another will retain the signal value.

    Example:
        >>> df['signal'] = [0, 0, 1, 1, 1, -1, -1, 0, 1, 1]
        >>> df = convert_to_event_signals(df)
        >>> df['signal'].tolist()
        [0, 0, 1, 0, 0, -1, 0, 0, 1, 0]
    """
    signal = df["signal"].values.copy()
    prev = 0
    for i in range(len(signal)):
        if signal[i] == prev:
            signal[i] = 0
        elif signal[i] != 0:
            prev = signal[i]
        # If signal[i] == 0, reset prev to 0
        else:
            prev = 0
    df["signal"] = signal.astype(int)
    return df


def add_trend_filter(
    df: pd.DataFrame,
    trend_window: int = 50,
    trend_tolerance: float = 0.03,
) -> pd.DataFrame:
    """
    Add a trend filter column to suppress mean-reversion signals in strong trends.

    When price is far from the moving average (indicating a strong trend), mean-reversion
    signals are unreliable. This filter identifies ranging/neutral conditions.

    Args:
        df: DataFrame containing a 'close' column
        trend_window: Window for trend MA calculation (default 50)
        trend_tolerance: Maximum allowed deviation from MA for ranging market (default 0.03 = 3%)

    Returns:
        DataFrame with a 'trend_filter' column added.
        trend_filter: True where market is ranging (safe for mean-reversion),
                      False where market is trending (suppress signals).
    """
    trend_ma = df["close"].rolling(window=trend_window).mean()
    deviation = (df["close"] - trend_ma) / trend_ma.replace(0, float("nan"))
    df["trend_filter"] = deviation.abs() < trend_tolerance
    return df


def calculate_rsi(prices: pd.Series, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)

    RSI measures the speed and magnitude of price changes, ranging from 0 to 100.
    Typically RSI > 70 is considered overbought, RSI < 30 is considered oversold.

    Calculation steps:
        1. Calculate price changes (delta)
        2. Separate gains and losses
        3. Calculate average gain/loss (rolling mean)
        4. RS = Average gain / Average loss
        5. RSI = 100 - 100/(1+RS)

    Args:
        prices: Price series (typically closing prices)
        period: RSI calculation period (default 14)

    Returns:
        RSI value series, range [0, 100], first period-1 values are NaN
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    # Division-by-zero guard: when loss is 0, RSI is 100 (all gains)
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    return rsi
