"""
Backtest Engine
回测引擎 - 模拟历史交易并计算收益
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


class Trade:
    """单笔交易记录"""
    def __init__(
        self,
        timestamp: datetime,
        action: str,  # 'buy' or 'sell'
        price: float,
        quantity: float,
        value: float,
        coin: str,
        strategy_signal: int
    ):
        self.timestamp = timestamp
        self.action = action
        self.price = price
        self.quantity = quantity
        self.value = value
        self.coin = coin
        self.strategy_signal = strategy_signal
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'price': self.price,
            'quantity': self.quantity,
            'value': self.value,
            'coin': self.coin,
            'signal': self.strategy_signal
        }


class BacktestResult:
    """回测结果"""
    def __init__(self):
        self.trades: List[Trade] = []
        self.daily_returns: pd.Series = None
        self.cumulative_returns: pd.Series = None
        self.equity_curve: pd.Series = None
        self.metrics: Dict = {}
        self.decision_log: List[Dict] = []
    
    def add_trade(self, trade: Trade):
        self.trades.append(trade)
    
    def add_decision(self, timestamp: datetime, decision: str, reason: str, **kwargs):
        """记录每一步决策"""
        self.decision_log.append({
            'timestamp': timestamp.isoformat(),
            'decision': decision,
            'reason': reason,
            **kwargs
        })
    
    def calculate_metrics(self) -> Dict:
        """计算回测指标"""
        if self.daily_returns is None or len(self.daily_returns) == 0:
            return {}
        
        returns = self.daily_returns.dropna()
        
        # 基础指标
        total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100
        
        # 年化收益
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else 0
        
        # 波动率
        volatility = returns.std() * np.sqrt(365) * 100
        
        # 夏普比率 (假设无风险利率 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0
        
        # 最大回撤
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # 胜率
        win_trades = sum(1 for t in self.trades if t.action == 'sell' and t.value > 0)
        total_trades = len([t for t in self.trades if t.action == 'sell'])
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 交易次数
        num_trades = len(self.trades)
        
        self.metrics = {
            'total_return_pct': round(total_return, 2),
            'annual_return_pct': round(annual_return, 2),
            'volatility_pct': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate_pct': round(win_rate, 2),
            'total_trades': num_trades,
            'trades_per_month': round(num_trades / (days/30), 2) if days > 0 else 0
        }
        
        return self.metrics
    
    def save_logs(self, filepath: str):
        """保存决策日志"""
        log_data = {
            'metrics': self.metrics,
            'trades': [t.to_dict() for t in self.trades],
            'decisions': self.decision_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"💾 决策日志已保存: {filepath}")


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1% 手续费
        slippage: float = 0.001,  # 0.1% 滑点
        position_size: float = 0.95  # 仓位比例
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.position_size = position_size
        
        self.cash = initial_capital
        self.position = 0  # 持仓数量
        self.position_value = 0  # 持仓市值
        
        logger.info(f"🚀 回测引擎初始化")
        logger.info(f"   初始资金: ${initial_capital:,.2f}")
        logger.info(f"   手续费: {commission_rate*100}%")
        logger.info(f"   滑点: {slippage*100}%")
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy,
        coin: str = 'BTC'
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            df: 包含 price 和 signal 的 DataFrame
            strategy: 交易策略对象
            coin: 币种名称
        
        Returns:
            BacktestResult 对象
        """
        result = BacktestResult()
        
        # 生成信号
        df = strategy.generate_signals(df)
        
        # 初始化权益曲线
        equity_curve = []
        
        logger.info(f"📊 开始回测 {coin}...")
        logger.info(f"   策略: {strategy.name}")
        logger.info(f"   数据点数: {len(df)}")
        
        for timestamp, row in df.iterrows():
            price = row['close']
            signal = row['signal']
            
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
                decision='hold' if signal == 0 else ('buy' if signal == 1 else 'sell'),
                reason=f"Signal: {signal}, Price: {price:.2f}, Cash: {self.cash:.2f}, Position: {self.position:.6f}",
                price=price,
                cash=self.cash,
                position=self.position,
                total_value=total_value,
                signal=signal
            )
        
        # 最后平仓
        if self.position > 0:
            final_price = df['close'].iloc[-1]
            self._execute_sell(
                df.index[-1], final_price, coin, result, 
                df.iloc[-1], force=True
            )
        
        # 计算结果
        result.equity_curve = pd.Series(equity_curve, index=df.index)
        result.daily_returns = result.equity_curve.pct_change().dropna()
        result.cumulative_returns = (result.equity_curve / result.equity_curve.iloc[0] - 1) * 100
        
        # 计算指标
        result.calculate_metrics()
        
        logger.info(f"✅ 回测完成")
        logger.info(f"   最终资产: ${result.equity_curve.iloc[-1]:,.2f}")
        logger.info(f"   总收益率: {result.metrics.get('total_return_pct', 0):.2f}%")
        logger.info(f"   交易次数: {len(result.trades)//2}")
        
        return result
    
    def _execute_buy(
        self, 
        timestamp: datetime, 
        price: float, 
        coin: str, 
        result: BacktestResult,
        row: pd.Series
    ):
        """执行买入"""
        # 考虑滑点
        executed_price = price * (1 + self.slippage)
        
        # 计算可买入数量（扣除手续费）
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
            action='buy',
            price=executed_price,
            quantity=quantity,
            value=position_value,
            coin=coin,
            strategy_signal=row['signal']
        )
        result.add_trade(trade)
        
        logger.debug(f"   买入 @ ${executed_price:.2f}, 数量: {quantity:.6f}")
    
    def _execute_sell(
        self, 
        timestamp: datetime, 
        price: float, 
        coin: str, 
        result: BacktestResult,
        row: pd.Series,
        force: bool = False
    ):
        """执行卖出"""
        # 考虑滑点
        executed_price = price * (1 - self.slippage)
        
        # 计算卖出价值
        sell_value = self.position * executed_price
        commission = sell_value * self.commission_rate
        net_value = sell_value - commission
        
        # 更新状态
        self.cash += net_value
        self.position = 0
        self.position_value = 0
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            action='sell',
            price=executed_price,
            quantity=self.position,
            value=net_value,
            coin=coin,
            strategy_signal=row['signal'] if not force else -2  # -2 表示强制平仓
        )
        result.add_trade(trade)
        
        logger.debug(f"   卖出 @ ${executed_price:.2f}, 收入: ${net_value:.2f}")
    
    def reset(self):
        """重置引擎状态"""
        self.cash = self.initial_capital
        self.position = 0
        self.position_value = 0


if __name__ == "__main__":
    # 简单测试
    from strategies import MovingAverageCrossStrategy
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # 运行回测
    strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run_backtest(df, strategy, coin='TEST')
    
    print("\n📈 回测结果:")
    for key, value in result.metrics.items():
        print(f"   {key}: {value}")
