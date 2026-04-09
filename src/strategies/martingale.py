"""
马丁格尔策略

亏损后加倍下单，直到获利。属于高风险博弈策略，仅用于回测研究。

⚠️ 风险警告:
    - 马丁格尔策略在连续亏损时会指数级增加仓位
    - 第 N 次加倍时，单次仓位为初始的 2^N 倍
    - 连续亏损多次后，累计损失可能非常巨大
    - 此策略仅用于回测研究，切勿用于实盘交易

使用示例:
    >>> from strategies import get_strategy
    >>> strategy = get_strategy('martingale', base_amount=0.001, multiplier=2.0)
    >>> result_df = strategy.generate_signals(df)
"""

import pandas as pd
from strategies._base import TradingStrategy


class MartingaleStrategy(TradingStrategy):
    """
    马丁格尔策略 (Martingale Strategy)

    核心逻辑:
    1. 初始买入 base_amount
    2. 如果价格下跌达到止损比例，加倍买入 (乘以 multiplier)
    3. 重复加倍，直到达到 max_steps 次或价格回升到止盈目标
    4. 达到最大加倍次数后强制止损离场

    ⚠️ 此策略风险极高，仅用于回测研究。

    Args:
        base_amount: 初始买入量 (默认 0.001)
        multiplier: 加倍倍数 (默认 2.0)
        max_steps: 最大加倍次数 (默认 5)
        target_profit: 止盈目标比例 (默认 0.01 = 1%)
        stop_loss: 单步止损触发比例 (默认 0.05 = 5%)

    生成的指标列:
        position: 当前持仓强度 (步数+1)，0 表示空仓
    """

    def __init__(
        self,
        base_amount: float = 0.001,
        multiplier: float = 2.0,
        max_steps: int = 5,
        target_profit: float = 0.01,
        stop_loss: float = 0.05,
    ):
        super().__init__("Martingale_Strategy")
        self.base_amount = base_amount
        self.multiplier = multiplier
        self.max_steps = max_steps
        self.target_profit = target_profit
        self.stop_loss = stop_loss

    def _update_martingale_position(
        self,
        current_price: float,
        entry_price: float,
        current_step: int,
        in_position: bool,
    ) -> tuple:
        """
        根据当前价格更新马丁格尔仓位

        检查止盈/止损条件，决定是否平仓或加倍。

        Args:
            current_price: 当前价格
            entry_price: 平均入场价格
            current_step: 当前加倍步数
            in_position: 是否持有仓位

        Returns:
            (signal, new_entry_price, new_step, new_in_position) 元组
            signal: 0=无操作, 1=加倍买入, -1=平仓
        """
        if not in_position:
            return 1, current_price, 0, True

        price_change = (current_price - entry_price) / entry_price

        # 达到目标止盈
        if price_change >= self.target_profit:
            return -1, 0.0, 0, False

        # 亏损达到止损触发线 (随步数增加，触发线收紧)
        stop_threshold = self.stop_loss / (current_step + 1)
        if price_change <= -stop_threshold:
            if current_step < self.max_steps:
                # 加倍买入，更新平均入场价
                total_weight = sum(self.multiplier ** j for j in range(current_step + 2))
                last_weight = self.multiplier ** (current_step + 1)
                new_entry_price = (
                    entry_price * (total_weight - last_weight) + current_price * last_weight
                ) / total_weight
                return 1, new_entry_price, current_step + 1, True
            else:
                # 超过最大加倍次数，止损离场
                return -1, 0.0, 0, False

        return 0, entry_price, current_step, True

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号

        逐行执行马丁格尔逻辑:
        1. 无仓位时初始买入
        2. 持仓时检查止盈/止损条件
        3. 亏损时加倍买入，止盈时全部卖出

        Args:
            df: 包含 'close' 列的 DataFrame

        Returns:
            添加了 signal, position 列的 DataFrame
        """
        df = df.copy()
        df["signal"] = 0
        df["position"] = 0.0

        current_step = 0
        entry_price = 0.0
        in_position = False

        for i in range(1, len(df)):
            current_price = df["close"].iloc[i]

            signal, entry_price, current_step, in_position = self._update_martingale_position(
                current_price, entry_price, current_step, in_position
            )

            df.loc[df.index[i], "signal"] = signal
            df.loc[df.index[i], "position"] = (current_step + 1) if in_position else 0

        return df
