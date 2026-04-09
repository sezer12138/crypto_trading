"""
策略辅助函数

提供策略中常用的共享计算逻辑，避免代码重复。
包括: 信号前向填充、交叉检测、RSI 计算等。
"""

import pandas as pd
from strategies.constants import DEFAULT_RSI_PERIOD


def forward_fill_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据信号列生成持仓状态列

    将 signal 列中的 0 值向前填充，表示在没有新信号时维持当前持仓。
    即：买入信号后持续持仓，直到出现卖出信号。

    Args:
        df: 包含 'signal' 列的 DataFrame

    Returns:
        添加了 'position' 列的 DataFrame。
        position 值: 1=持仓中, 0=空仓, -1=卖出(瞬间)

    Example:
        >>> df['signal'] = [0, 1, 0, 0, -1, 0]
        >>> df = forward_fill_position(df)
        >>> df['position'].tolist()
        [0, 1, 1, 1, -1, -1]
    """
    df["position"] = df["signal"].replace(to_replace=0, method="ffill")
    return df


def detect_crossover(
    df: pd.DataFrame,
    fast_col: str,
    slow_col: str,
) -> pd.DataFrame:
    """
    检测两列之间的交叉信号（金叉/死叉）

    金叉：fast_col 从下方穿过 slow_col（上穿），产生买入信号。
    死叉：fast_col 从上方穿过 slow_col（下穿），产生卖出信号。

    Args:
        df: 包含两列数据的 DataFrame
        fast_col: 快线列名（如 'ma_short', 'macd'）
        slow_col: 慢线列名（如 'ma_long', 'macd_signal'）

    Returns:
        添加了 'signal' 列的 DataFrame
        signal: 1=金叉(买入), -1=死叉(卖出), 0=无信号

    Example:
        >>> df = detect_crossover(df, 'ma_short', 'ma_long')
        >>> buy_signals = df[df['signal'] == 1]
    """
    df["signal"] = 0

    # 金叉：快线上穿慢线
    df.loc[
        (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1)),
        "signal",
    ] = 1

    # 死叉：快线下穿慢线
    df.loc[
        (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1)),
        "signal",
    ] = -1

    return df


def calculate_rsi(prices: pd.Series, period: int = DEFAULT_RSI_PERIOD) -> pd.Series:
    """
    计算相对强弱指标 (RSI)

    RSI 衡量价格变动的速度和幅度，范围 0-100。
    通常 RSI > 70 视为超买，RSI < 30 视为超卖。

    计算步骤:
        1. 计算价格变动 (delta)
        2. 分离上涨 (gain) 和下跌 (loss)
        3. 计算平均涨跌幅 (滚动均值)
        4. RS = 平均涨幅 / 平均跌幅
        5. RSI = 100 - 100/(1+RS)

    Args:
        prices: 价格序列 (通常为收盘价)
        period: RSI 计算周期 (默认 14)

    Returns:
        RSI 值序列，范围 [0, 100]，前 period-1 个值为 NaN
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
