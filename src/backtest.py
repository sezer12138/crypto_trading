"""
Backtest Engine
回测引擎 - 模拟历史交易并计算收益指标

本模块提供完整的回测功能，包括：
    - 模拟真实交易环境（手续费、滑点）
    - 详细的交易记录和决策日志
    - 完整的绩效指标计算
    - 支持仓位管理

Classes:
    Trade: 单笔交易记录
    BacktestResult: 回测结果容器
    BacktestEngine: 回测引擎主类

Example:
    >>> from backtest import BacktestEngine
    >>> from strategies import MovingAverageCrossStrategy
    >>> 
    >>> engine = BacktestEngine(initial_capital=10000.0)
    >>> strategy = MovingAverageCrossStrategy()
    >>> result = engine.run_backtest(df, strategy, coin='BTC')
    >>> print(f"总收益: {result.metrics['total_return_pct']:.2f}%")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)

# 确保日志目录存在
Path("logs").mkdir(parents=True, exist_ok=True)


@dataclass
class Trade:
    """
    单笔交易记录
    
    记录一次完整的交易操作，包括时间、价格、数量等信息。
    
    Attributes:
        timestamp: 交易时间
        action: 交易动作 ('buy' 或 'sell')
        price: 成交价格
        quantity: 交易数量
        value: 交易金额
        coin: 交易币种
        strategy_signal: 策略信号值 (1=买入, -1=卖出)
        
    Example:
        >>> trade = Trade(
        ...     timestamp=datetime.now(),
        ...     action='buy',
        ...     price=50000.0,
        ...     quantity=0.1,
        ...     value=5000.0,
        ...     coin='BTC',
        ...     strategy_signal=1
        ... )
    """
    timestamp: datetime
    action: str  # 'buy' or 'sell'
    price: float
    quantity: float
    value: float
    coin: str
    strategy_signal: int
    
    def to_dict(self) -> Dict[str, Any]:
        """将交易记录转换为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "value": self.value,
            "coin": self.coin,
            "signal": self.strategy_signal,
        }


@dataclass
class BacktestResult:
    """
    回测结果容器
    
    存储回测的全部结果数据，包括交易记录、权益曲线、绩效指标等。
    
    Attributes:
        trades: 交易记录列表
        daily_returns: 日收益率序列
        cumulative_returns: 累计收益率序列
        equity_curve: 权益曲线（资金变化）
        metrics: 绩效指标字典
        decision_log: 决策日志列表
        
    Methods:
        add_trade: 添加交易记录
        add_decision: 添加决策记录
        calculate_metrics: 计算绩效指标
        save_logs: 保存日志到文件
    """
    trades: List[Trade] = field(default_factory=list)
    daily_returns: Optional[pd.Series] = None
    cumulative_returns: Optional[pd.Series] = None
    equity_curve: Optional[pd.Series] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_trade(self, trade: Trade) -> None:
        """
        添加交易记录
        
        Args:
            trade: Trade 对象
        """
        self.trades.append(trade)
    
    def add_decision(
        self, 
        timestamp: datetime, 
        decision: str, 
        reason: str, 
        **kwargs
    ) -> None:
        """
        记录每一步决策
        
        Args:
            timestamp: 决策时间
            decision: 决策类型 ('hold', 'buy', 'sell')
            reason: 决策原因说明
            **kwargs: 其他相关数据（如 price, cash, position 等）
        """
        self.decision_log.append({
            "timestamp": timestamp.isoformat(),
            "decision": decision,
            "reason": reason,
            **kwargs
        })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        计算回测绩效指标
        
        计算的指标包括：
        - total_return_pct: 总收益率 (%)
        - annual_return_pct: 年化收益率 (%)
        - volatility_pct: 年化波动率 (%)
        - sharpe_ratio: 夏普比率
        - max_drawdown_pct: 最大回撤 (%)
        - win_rate_pct: 胜率 (%)
        - total_trades: 总交易次数
        - trades_per_month: 月均交易次数
        
        Returns:
            包含各项指标的字典
        """
        if self.daily_returns is None or len(self.daily_returns) == 0:
            logger.warning("没有日收益数据，无法计算指标")
            return {}
        
        returns = self.daily_returns.dropna()
        
        if len(returns) == 0:
            logger.warning("日收益数据为空")
            return {}
        
        # 基础指标
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100
        
        # 计算时间跨度（天数）
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days <= 0:
            logger.warning("数据时间跨度无效")
            return {}
        
        # 年化收益
        annual_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100
        
        # 年化波动率
        volatility = returns.std() * np.sqrt(365) * 100
        
        # 夏普比率 (假设无风险利率 2%)
        risk_free_rate = 0.02
        if volatility > 0:
            sharpe_ratio = (annual_return / 100 - risk_free_rate) / (volatility / 100)
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # 胜率（统计卖出时的盈利情况）
        sell_trades = [t for t in self.trades if t.action == "sell"]
        if len(sell_trades) > 0:
            # 通过查找对应的买入记录计算盈亏
            profitable_sells = 0
            for sell in sell_trades:
                # 找到前一笔买入
                buy_idx = None
                for i, trade in enumerate(self.trades):
                    if trade == sell:
                        # 查找之前的买入
                        for j in range(i-1, -1, -1):
                            if self.trades[j].action == "buy":
                                buy_idx = j
                                break
                        break
                
                if buy_idx is not None:
                    buy = self.trades[buy_idx]
                    if sell.price > buy.price:
                        profitable_sells += 1
            
            win_rate = (profitable_sells / len(sell_trades)) * 100
        else:
            win_rate = 0.0
        
        # 交易统计
        num_trades = len(self.trades)
        trades_per_month = num_trades / (days / 30) if days > 0 else 0
        
        self.metrics = {
            "total_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return, 2),
            "volatility_pct": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 2),
            "total_trades": num_trades,
            "trades_per_month": round(trades_per_month, 2),
        }
        
        return self.metrics
    
    def save_logs(self, filepath: str) -> None:
        """
        保存决策日志到 JSON 文件
        
        Args:
            filepath: 保存路径
        """
        log_data = {
            "metrics": self.metrics,
            "trades": [t.to_dict() for t in self.trades],
            "decisions": self.decision_log,
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"💾 决策日志已保存: {filepath}")


class BacktestEngine:
    """
    回测引擎主类
    
    模拟历史交易环境，执行策略并计算收益。
    支持手续费、滑点、仓位管理等真实交易因素。
    
    Args:
        initial_capital: 初始资金（默认 10000.0）
        commission_rate: 手续费率（默认 0.001 = 0.1%）
        slippage: 滑点率（默认 0.001 = 0.1%）
        position_size: 仓位比例（默认 0.95 = 95%）
        
    Attributes:
        initial_capital: 初始资金
        commission_rate: 手续费率
        slippage: 滑点率
        position_size: 仓位比例
        cash: 当前现金
        position: 当前持仓数量
        position_value: 当前持仓市值
        
    Example:
        >>> engine = BacktestEngine(
        ...     initial_capital=10000.0,
        ...     commission_rate=0.001,
        ...     slippage=0.001
        ... )
        >>> result = engine.run_backtest(df, strategy, coin='BTC')
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% 手续费
        slippage: float = 0.001,  # 0.1% 滑点
        position_size: float = 0.95,  # 仓位比例
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.position_size = position_size
        
        # 状态变量
        self.cash = initial_capital
        self.position = 0.0  # 持仓数量
        self.position_value = 0.0  # 持仓市值
        
        logger.info("🚀 回测引擎初始化")
        logger.info(f"   初始资金: ${initial_capital:,.2f}")
        logger.info(f"   手续费: {commission_rate * 100:.2f}%")
        logger.info(f"   滑点: {slippage * 100:.2f}%")
        logger.info(f"   仓位比例: {position_size * 100:.0f}%")
    
    def run_backtest(
        self, 
        df: pd.DataFrame, 
        strategy: object, 
        coin: str = "BTC"
    ) -> BacktestResult:
        """
        运行回测
        
        使用给定策略在历史数据上执行回测。
        
        Args:
            df: 包含价格数据的 DataFrame，需要有 'close' 列
            strategy: 交易策略对象，需实现 generate_signals 方法
            coin: 币种名称（用于日志和记录）
            
        Returns:
            BacktestResult 对象，包含完整的回测结果
            
        Raises:
            ValueError: 如果输入数据无效
        """
        result = BacktestResult()
        
        # 验证输入数据
        if df.empty:
            raise ValueError("输入数据为空")
        if "close" not in df.columns:
            raise ValueError("数据缺少 'close' 列")
        
        # 生成信号
        df = strategy.generate_signals(df.copy())
        
        if "signal" not in df.columns:
            raise ValueError("策略未生成 'signal' 列")
        
        # 初始化权益曲线
        equity_curve = []
        
        logger.info(f"📊 开始回测 {coin}...")
        logger.info(f"   策略: {strategy.name}")
        logger.info(f"   数据点数: {len(df)}")
        
        for timestamp, row in df.iterrows():
            price = row["close"]
            signal = int(row["signal"])
            
            # 当前总资产
            total_value = self.cash + self.position * price
            equity_curve.append(total_value)
            
            # 决策逻辑
            if signal == 1 and self.position == 0:
                # 买入信号
                self._execute_buy(timestamp, price, coin, result, row)
                
            elif signal == -1 and self.position > 0:
                # 卖出信号
                self._execute_sell(timestamp, price, coin, result, row)
            
            # 记录每一步状态
            result.add_decision(
                timestamp=timestamp,
                decision="hold" if signal == 0 else ("buy" if signal == 1 else "sell"),
                reason=f"Signal: {signal}, Price: {price:.2f}, "
                       f"Cash: {self.cash:.2f}, Position: {self.position:.6f}",
                price=price,
                cash=self.cash,
                position=self.position,
                total_value=total_value,
                signal=signal,
            )
        
        # 最后平仓（如果还有持仓）
        if self.position > 0:
            final_price = df["close"].iloc[-1]
            self._execute_sell(
                df.index[-1], final_price, coin, result, 
                df.iloc[-1], force=True
            )
        
        # 计算结果
        result.equity_curve = pd.Series(equity_curve, index=df.index)
        result.daily_returns = result.equity_curve.pct_change().dropna()
        result.cumulative_returns = (
            result.equity_curve / result.equity_curve.iloc[0] - 1
        ) * 100
        
        # 计算指标
        result.calculate_metrics()
        
        # 输出结果
        logger.info("✅ 回测完成")
        logger.info(f"   最终资产: ${result.equity_curve.iloc[-1]:,.2f}")
        logger.info(f"   总收益率: {result.metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"   交易次数: {len(result.trades) // 2}")
        
        return result
    
    def _execute_buy(
        self, 
        timestamp: datetime, 
        price: float, 
        coin: str, 
        result: BacktestResult, 
        row: pd.Series
    ) -> None:
        """
        执行买入操作
        
        Args:
            timestamp: 交易时间
            price: 当前价格
            coin: 币种
            result: BacktestResult 对象
            row: 当前数据行
        """
        # 考虑滑点（买入时价格上升）
        executed_price = price * (1 + self.slippage)
        
        # 计算可买入金额（扣除手续费）
        position_value = self.cash * self.position_size
        commission = position_value * self.commission_rate
        quantity = (position_value - commission) / executed_price
        
        # 更新状态
        self.position = quantity
        self.position_value = position_value
        self.cash -= position_value
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            action="buy",
            price=executed_price,
            quantity=quantity,
            value=position_value,
            coin=coin,
            strategy_signal=int(row["signal"]),
        )
        result.add_trade(trade)
        
        logger.debug(
            f"   买入 @ ${executed_price:.2f}, "
            f"数量: {quantity:.6f}, "
            f"金额: ${position_value:.2f}"
        )
    
    def _execute_sell(
        self,
        timestamp: datetime,
        price: float,
        coin: str,
        result: BacktestResult,
        row: pd.Series,
        force: bool = False,
    ) -> None:
        """
        执行卖出操作
        
        Args:
            timestamp: 交易时间
            price: 当前价格
            coin: 币种
            result: BacktestResult 对象
            row: 当前数据行
            force: 是否为强制平仓（默认 False）
        """
        # 考虑滑点（卖出时价格下降）
        executed_price = price * (1 - self.slippage)
        
        # 获取持仓数量（在更新前保存）
        sell_quantity = self.position
        
        # 计算卖出价值
        sell_value = sell_quantity * executed_price
        commission = sell_value * self.commission_rate
        net_value = sell_value - commission
        
        # 更新状态
        self.cash += net_value
        self.position = 0.0
        self.position_value = 0.0
        
        # 记录交易
        signal_value = int(row["signal"]) if not force else -2
        trade = Trade(
            timestamp=timestamp,
            action="sell",
            price=executed_price,
            quantity=sell_quantity,
            value=net_value,
            coin=coin,
            strategy_signal=signal_value,
        )
        result.add_trade(trade)
        
        logger.debug(
            f"   卖出 @ ${executed_price:.2f}, "
            f"数量: {sell_quantity:.6f}, "
            f"净收入: ${net_value:.2f}"
        )
    
    def reset(self) -> None:
        """重置引擎状态到初始值"""
        self.cash = self.initial_capital
        self.position = 0.0
        self.position_value = 0.0
        logger.info("🔄 回测引擎已重置")


if __name__ == "__main__":
    # 简单测试
    from strategies import MovingAverageCrossStrategy
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    # 运行回测
    strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run_backtest(df, strategy, coin="TEST")
    
    print("\n📈 回测结果:")
    for key, value in result.metrics.items():
        print(f"   {key}: {value}")
