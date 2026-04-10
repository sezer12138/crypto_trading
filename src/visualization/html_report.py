"""
HTML 报告生成器

生成完整的回测 HTML 报告，包含策略详情、交易记录、收益分析和总结结论。
"""

import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# 策略描述信息
STRATEGY_DESCRIPTIONS = {
    "ma_cross": {
        "name": "双均线交叉策略",
        "name_en": "Moving Average Cross",
        "type": "趋势跟踪",
        "description": "基于短期均线上穿/下穿长期均线产生买卖信号。当短期均线上穿长期均线时买入，下穿时卖出。",
        "params": ["short_window (短期均线周期)", "long_window (长期均线周期)"],
        "suitable_market": "趋势明显的市场",
        "risk_level": "中等",
    },
    "rsi": {
        "name": "RSI 超买超卖策略",
        "name_en": "RSI Overbought/Oversold",
        "type": "均值回归",
        "description": "基于 RSI 指标判断超买超卖区域。RSI 低于超卖阈值时买入，高于超买阈值时卖出。",
        "params": ["period (RSI 周期)", "oversold (超卖阈值)", "overbought (超买阈值)"],
        "suitable_market": "震荡市场",
        "risk_level": "中等",
    },
    "bollinger": {
        "name": "布林带策略",
        "name_en": "Bollinger Bands",
        "type": "均值回归",
        "description": "利用布林带上下轨判断价格偏离程度。价格触及下轨买入，触及上轨卖出。",
        "params": ["window (移动窗口)", "num_std (标准差倍数)"],
        "suitable_market": "震荡市场",
        "risk_level": "中等",
    },
    "multi_factor": {
        "name": "多因子组合策略",
        "name_en": "Multi-Factor Strategy",
        "type": "综合策略",
        "description": "结合均线趋势、RSI、成交量和波动率多个因子，通过加权评分生成信号。",
        "params": ["ma_short", "ma_long", "rsi_period", "volume_threshold"],
        "suitable_market": "多种市场环境",
        "risk_level": "中等",
    },
    "mean_reversion": {
        "name": "均值回归策略",
        "name_en": "Mean Reversion",
        "type": "统计套利",
        "description": "基于价格偏离均值的程度进行反向操作。价格大幅偏离均值时预期回归。",
        "params": ["window (窗口期)", "entry_z (入场 Z 分数)", "exit_z (出场 Z 分数)"],
        "suitable_market": "震荡市场",
        "risk_level": "中等偏高",
    },
    "macd": {
        "name": "MACD 趋势策略",
        "name_en": "MACD Strategy",
        "type": "趋势跟踪",
        "description": "基于 MACD 线与信号线的交叉产生买卖信号。MACD 上穿信号线买入，下穿卖出。",
        "params": ["fast_period (快线周期)", "slow_period (慢线周期)", "signal_period (信号线周期)"],
        "suitable_market": "趋势市场",
        "risk_level": "中等",
    },
    "breakout": {
        "name": "突破策略",
        "name_en": "Breakout Strategy",
        "type": "趋势跟踪",
        "description": "当价格突破 N 周期最高价时买入，突破 N 周期最低价时卖出。",
        "params": ["window (突破窗口)", "confirmation (是否确认突破)"],
        "suitable_market": "趋势启动阶段",
        "risk_level": "中等偏高",
    },
    "vwap": {
        "name": "VWAP 均值回归策略",
        "name_en": "VWAP Strategy",
        "type": "均值回归",
        "description": "利用成交量加权平均价格判断价格偏离。价格低于 VWAP 买入，高于 VWAP 卖出。",
        "params": ["window (VWAP 窗口)", "deviation (偏离阈值)"],
        "suitable_market": "日内交易/震荡市场",
        "risk_level": "中等",
    },
    "momentum": {
        "name": "动量策略",
        "name_en": "Momentum Strategy",
        "type": "趋势跟踪",
        "description": "基于价格变化率 (ROC) 和动量指标判断趋势强度，顺势交易。",
        "params": ["roc_period (ROC 周期)", "momentum_period (动量周期)", "threshold (阈值)"],
        "suitable_market": "趋势市场",
        "risk_level": "中等",
    },
    "atr_stop": {
        "name": "ATR 动态止损策略",
        "name_en": "ATR Stop Loss Strategy",
        "type": "趋势跟踪",
        "description": "利用 ATR 计算动态止损位，在保护利润的同时让利润奔跑。",
        "params": ["atr_period (ATR 周期)", "multiplier (ATR 倍数)", "trend_ma (趋势均线)"],
        "suitable_market": "趋势市场",
        "risk_level": "中等",
    },
    "stochastic": {
        "name": "随机指标策略",
        "name_en": "Stochastic Strategy",
        "type": "均值回归",
        "description": "利用 K 线和 D 线的交叉以及超买超卖区域产生信号。",
        "params": ["k_period (K 周期)", "d_period (D 周期)"],
        "suitable_market": "震荡市场",
        "risk_level": "中等",
    },
    "grid": {
        "name": "网格交易策略",
        "name_en": "Grid Trading Strategy",
        "type": "震荡套利",
        "description": "在预设价格区间内设置多个买卖网格，低买高卖赚取差价。",
        "params": ["lower_price (下限)", "upper_price (上限)", "grid_num (网格数)", "amount_per_grid (每格数量)"],
        "suitable_market": "震荡市场",
        "risk_level": "中等偏高（单边行情风险大）",
    },
    "martingale": {
        "name": "马丁格尔策略",
        "name_en": "Martingale Strategy",
        "type": "高风险博弈",
        "description": "亏损后加倍下注，直到获利后回到初始下注额。⚠️ 高风险策略，需要充足资金。",
        "params": ["base_amount (基础仓位)", "multiplier (倍数)", "max_steps (最大步数)", "target_profit (目标利润)", "stop_loss (止损)"],
        "suitable_market": "仅用于测试，不建议实盘",
        "risk_level": "极高 ⚠️",
    },
}


class HTMLReportGenerator:
    """HTML 报告生成器"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_single_report(
        self,
        result: Any,
        df: pd.DataFrame,
        strategy_name: str,
        coin: str,
        days: int,
        interval: str,
        capital: float,
        output_path: str,
        chart_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        生成单策略回测 HTML 报告

        Args:
            result: BacktestResult 对象
            df: 价格数据
            strategy_name: 策略名称
            coin: 币种
            days: 回测天数
            interval: 时间粒度
            capital: 初始资金
            output_path: 输出路径
            chart_paths: 图表路径字典

        Returns:
            生成的 HTML 文件路径
        """
        strategy_info = STRATEGY_DESCRIPTIONS.get(strategy_name, {})

        # 读取图表并转为 base64
        charts_base64 = {}
        if chart_paths:
            for name, path in chart_paths.items():
                if Path(path).exists():
                    charts_base64[name] = self._image_to_base64(path)

        html_content = self._generate_single_html(
            result=result,
            df=df,
            strategy_name=strategy_name,
            strategy_info=strategy_info,
            coin=coin,
            days=days,
            interval=interval,
            capital=capital,
            charts_base64=charts_base64,
        )

        # 保存 HTML 文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"📄 HTML 报告已生成: {output_file}")
        return str(output_file)

    def generate_comparison_report(
        self,
        results: Dict[str, Any],
        coin: str,
        days: int,
        interval: str,
        capital: float,
        output_path: str,
        chart_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        生成多策略对比 HTML 报告

        Args:
            results: 策略结果字典
            coin: 币种
            days: 回测天数
            interval: 时间粒度
            capital: 初始资金
            output_path: 输出路径
            chart_paths: 图表路径字典

        Returns:
            生成的 HTML 文件路径
        """
        # 读取图表并转为 base64
        charts_base64 = {}
        if chart_paths:
            for name, path in chart_paths.items():
                if Path(path).exists():
                    charts_base64[name] = self._image_to_base64(path)

        html_content = self._generate_comparison_html(
            results=results,
            coin=coin,
            days=days,
            interval=interval,
            capital=capital,
            charts_base64=charts_base64,
        )

        # 保存 HTML 文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"📄 HTML 对比报告已生成: {output_file}")
        return str(output_file)

    def _image_to_base64(self, image_path: str) -> str:
        """将图片转换为 base64 编码"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _generate_single_html(
        self,
        result: Any,
        df: pd.DataFrame,
        strategy_name: str,
        strategy_info: Dict,
        coin: str,
        days: int,
        interval: str,
        capital: float,
        charts_base64: Dict[str, str],
    ) -> str:
        """生成单策略报告 HTML 内容"""

        metrics = result.metrics
        trades = result.trades

        # 计算额外统计
        buy_trades = [t for t in trades if t.action == "buy"]
        sell_trades = [t for t in trades if t.action == "sell"]

        # 计算平均持仓时间
        avg_holding_time = self._calculate_avg_holding_time(trades)

        # 计算最大单笔盈亏
        max_profit, max_loss = self._calculate_max_profit_loss(trades)

        # 生成结论
        conclusion = self._generate_conclusion(metrics, strategy_name)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回测报告 - {strategy_name.upper()} - {coin.upper()}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e4;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #0f3460;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            color: #00d9ff;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }}
        .header .subtitle {{
            color: #888;
            font-size: 1.1em;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            color: #00d9ff;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0, 217, 255, 0.3);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .card h2::before {{
            content: '';
            width: 4px;
            height: 24px;
            background: #00d9ff;
            border-radius: 2px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .metric-item {{
            background: rgba(0, 0, 0, 0.2);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s ease;
        }}
        .metric-item:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-value.positive {{ color: #00ff88; }}
        .metric-value.negative {{ color: #ff4757; }}
        .metric-value.neutral {{ color: #00d9ff; }}
        .metric-label {{
            color: #888;
            font-size: 0.9em;
        }}
        .strategy-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .info-label {{
            color: #888;
        }}
        .info-value {{
            color: #fff;
            font-weight: 500;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        th {{
            background: rgba(0, 217, 255, 0.1);
            color: #00d9ff;
            font-weight: 600;
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        .buy {{ color: #00ff88; }}
        .sell {{ color: #ff4757; }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }}
        .conclusion {{
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 255, 136, 0.1) 100%);
            border-left: 4px solid #00d9ff;
            padding: 20px;
            border-radius: 0 10px 10px 0;
        }}
        .conclusion h3 {{
            color: #00d9ff;
            margin-bottom: 15px;
        }}
        .conclusion p {{
            line-height: 1.8;
            color: #ccc;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-success {{ background: rgba(0, 255, 136, 0.2); color: #00ff88; }}
        .badge-warning {{ background: rgba(255, 193, 7, 0.2); color: #ffc107; }}
        .badge-danger {{ background: rgba(255, 71, 87, 0.2); color: #ff4757; }}
        .footer {{
            text-align: center;
            padding: 30px 0;
            color: #666;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 40px;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-stat {{
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-stat .value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #00d9ff;
        }}
        .summary-stat .label {{
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }}
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .summary-stats {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 回测报告</h1>
            <div class="subtitle">
                {strategy_info.get('name', strategy_name)} | {coin.upper()} | {days} 天 | {interval} 周期
            </div>
            <div class="subtitle" style="margin-top: 10px; font-size: 0.9em;">
                生成时间: {self.timestamp}
            </div>
        </div>

        <!-- 核心指标 -->
        <div class="card">
            <h2>核心绩效指标</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value {'positive' if metrics.get('total_return_pct', 0) >= 0 else 'negative'}">
                        {metrics.get('total_return_pct', 0):.2f}%
                    </div>
                    <div class="metric-label">总收益率</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value {'positive' if metrics.get('annual_return_pct', 0) >= 0 else 'negative'}">
                        {metrics.get('annual_return_pct', 0):.2f}%
                    </div>
                    <div class="metric-label">年化收益率</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value neutral">{metrics.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">夏普比率</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value negative">{metrics.get('max_drawdown_pct', 0):.2f}%</div>
                    <div class="metric-label">最大回撤</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value neutral">{metrics.get('win_rate_pct', 0):.2f}%</div>
                    <div class="metric-label">胜率</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value neutral">{metrics.get('total_trades', 0)}</div>
                    <div class="metric-label">总交易次数</div>
                </div>
            </div>
        </div>

        <!-- 策略信息 -->
        <div class="card">
            <h2>策略详情</h2>
            <div class="strategy-info">
                <div class="info-item">
                    <span class="info-label">策略名称</span>
                    <span class="info-value">{strategy_info.get('name', strategy_name)}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">英文名称</span>
                    <span class="info-value">{strategy_info.get('name_en', strategy_name)}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">策略类型</span>
                    <span class="info-value">{strategy_info.get('type', 'N/A')}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">风险等级</span>
                    <span class="info-value">{strategy_info.get('risk_level', 'N/A')}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">适用市场</span>
                    <span class="info-value">{strategy_info.get('suitable_market', 'N/A')}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">回测币种</span>
                    <span class="info-value">{coin.upper()}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">回测天数</span>
                    <span class="info-value">{days} 天</span>
                </div>
                <div class="info-item">
                    <span class="info-label">K线周期</span>
                    <span class="info-value">{interval}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">初始资金</span>
                    <span class="info-value">${capital:,.2f}</span>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                <strong style="color: #00d9ff;">策略描述：</strong>
                <p style="margin-top: 10px; color: #ccc; line-height: 1.6;">
                    {strategy_info.get('description', '暂无描述')}
                </p>
            </div>
        </div>

        <!-- 交易统计 -->
        <div class="card">
            <h2>交易统计</h2>
            <div class="summary-stats">
                <div class="summary-stat">
                    <div class="value">{len(buy_trades)}</div>
                    <div class="label">买入次数</div>
                </div>
                <div class="summary-stat">
                    <div class="value">{len(sell_trades)}</div>
                    <div class="label">卖出次数</div>
                </div>
                <div class="summary-stat">
                    <div class="value">{avg_holding_time}</div>
                    <div class="label">平均持仓时间</div>
                </div>
                <div class="summary-stat">
                    <div class="value" style="color: #00ff88;">+{max_profit:.2f}%</div>
                    <div class="label">最大单笔盈利</div>
                </div>
                <div class="summary-stat">
                    <div class="value" style="color: #ff4757;">{max_loss:.2f}%</div>
                    <div class="label">最大单笔亏损</div>
                </div>
                <div class="summary-stat">
                    <div class="value">{metrics.get('trades_per_month', 0):.1f}</div>
                    <div class="label">月均交易次数</div>
                </div>
                <div class="summary-stat">
                    <div class="value">{metrics.get('volatility_pct', 0):.2f}%</div>
                    <div class="label">年化波动率</div>
                </div>
                <div class="summary-stat">
                    <div class="value">${result.equity_curve.iloc[-1]:,.2f}</div>
                    <div class="label">最终资产</div>
                </div>
            </div>
        </div>

        <!-- 图表 -->
        {self._generate_charts_html(charts_base64)}

        <!-- 交易记录 -->
        <div class="card">
            <h2>交易记录详情</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>时间</th>
                            <th>操作</th>
                            <th>价格</th>
                            <th>数量</th>
                            <th>金额</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_trades_table(trades)}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 总结与结论 -->
        <div class="card">
            <h2>总结与结论</h2>
            <div class="conclusion">
                <h3>📈 回测分析结论</h3>
                {conclusion}
            </div>
        </div>

        <div class="footer">
            <p>⚠️ 免责声明：历史回测结果不代表未来收益，投资有风险，入市需谨慎。</p>
            <p style="margin-top: 10px;">Generated by Crypto Trading Backtest System | {self.timestamp}</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _generate_comparison_html(
        self,
        results: Dict[str, Any],
        coin: str,
        days: int,
        interval: str,
        capital: float,
        charts_base64: Dict[str, str],
    ) -> str:
        """生成多策略对比报告 HTML 内容"""

        # 按夏普比率排序
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].metrics.get("sharpe_ratio", 0),
            reverse=True,
        )

        best_strategy = sorted_results[0][0] if sorted_results else "N/A"
        best_metrics = sorted_results[0][1].metrics if sorted_results else {}

        # 生成对比表格
        comparison_table = self._generate_comparison_table(sorted_results)

        # 生成策略详情卡片
        strategy_cards = self._generate_strategy_cards(sorted_results[:5])

        # 生成结论
        conclusion = self._generate_comparison_conclusion(sorted_results, coin, days)

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>策略对比报告 - {coin.upper()}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e4;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #0f3460;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            color: #00d9ff;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }}
        .header .subtitle {{
            color: #888;
            font-size: 1.1em;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            color: #00d9ff;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0, 217, 255, 0.3);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .card h2::before {{
            content: '';
            width: 4px;
            height: 24px;
            background: #00d9ff;
            border-radius: 2px;
        }}
        .best-strategy {{
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
            border: 2px solid rgba(0, 255, 136, 0.3);
            text-align: center;
            padding: 30px;
        }}
        .best-strategy h3 {{
            color: #00ff88;
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        .best-strategy .name {{
            font-size: 2.5em;
            color: #fff;
            font-weight: bold;
            margin: 10px 0;
        }}
        .best-strategy .metrics {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 20px;
        }}
        .best-strategy .metric {{
            text-align: center;
        }}
        .best-strategy .metric .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff88;
        }}
        .best-strategy .metric .label {{
            color: #888;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        th {{
            background: rgba(0, 217, 255, 0.1);
            color: #00d9ff;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        tr.best {{
            background: rgba(0, 255, 136, 0.1);
        }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4757; }}
        .neutral {{ color: #00d9ff; }}
        .rank {{
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            border-radius: 50%;
            background: rgba(0, 217, 255, 0.2);
            color: #00d9ff;
            font-weight: bold;
        }}
        .rank-1 {{ background: rgba(255, 215, 0, 0.3); color: #ffd700; }}
        .rank-2 {{ background: rgba(192, 192, 192, 0.3); color: #c0c0c0; }}
        .rank-3 {{ background: rgba(205, 127, 50, 0.3); color: #cd7f32; }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }}
        .strategy-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        .strategy-card {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #00d9ff;
        }}
        .strategy-card.top-1 {{
            border-left-color: #ffd700;
            background: rgba(255, 215, 0, 0.05);
        }}
        .strategy-card.top-2 {{
            border-left-color: #c0c0c0;
        }}
        .strategy-card.top-3 {{
            border-left-color: #cd7f32;
        }}
        .strategy-card h4 {{
            color: #fff;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .strategy-card .mini-metrics {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .strategy-card .mini-metric {{
            text-align: center;
            padding: 8px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 5px;
        }}
        .strategy-card .mini-metric .value {{
            font-weight: bold;
        }}
        .strategy-card .mini-metric .label {{
            font-size: 0.75em;
            color: #888;
        }}
        .conclusion {{
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 255, 136, 0.1) 100%);
            border-left: 4px solid #00d9ff;
            padding: 20px;
            border-radius: 0 10px 10px 0;
        }}
        .conclusion h3 {{
            color: #00d9ff;
            margin-bottom: 15px;
        }}
        .conclusion p {{
            line-height: 1.8;
            color: #ccc;
        }}
        .footer {{
            text-align: center;
            padding: 30px 0;
            color: #666;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 40px;
        }}
        @media (max-width: 768px) {{
            .best-strategy .metrics {{
                flex-direction: column;
                gap: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 策略对比报告</h1>
            <div class="subtitle">
                {coin.upper()} | {days} 天 | {interval} 周期 | {len(results)} 个策略
            </div>
            <div class="subtitle" style="margin-top: 10px; font-size: 0.9em;">
                生成时间: {self.timestamp}
            </div>
        </div>

        <!-- 最佳策略 -->
        <div class="card best-strategy">
            <h3>🏆 最佳策略</h3>
            <div class="name">{STRATEGY_DESCRIPTIONS.get(best_strategy, {}).get('name', best_strategy.upper())}</div>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{'+' if best_metrics.get('total_return_pct', 0) >= 0 else ''}{best_metrics.get('total_return_pct', 0):.2f}%</div>
                    <div class="label">总收益率</div>
                </div>
                <div class="metric">
                    <div class="value">{best_metrics.get('sharpe_ratio', 0):.2f}</div>
                    <div class="label">夏普比率</div>
                </div>
                <div class="metric">
                    <div class="value">{best_metrics.get('win_rate_pct', 0):.2f}%</div>
                    <div class="label">胜率</div>
                </div>
                <div class="metric">
                    <div class="value">{best_metrics.get('max_drawdown_pct', 0):.2f}%</div>
                    <div class="label">最大回撤</div>
                </div>
            </div>
        </div>

        <!-- 策略排名表格 -->
        <div class="card">
            <h2>策略排名对比</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th>策略</th>
                            <th>总收益</th>
                            <th>年化收益</th>
                            <th>夏普比率</th>
                            <th>最大回撤</th>
                            <th>胜率</th>
                            <th>交易次数</th>
                        </tr>
                    </thead>
                    <tbody>
                        {comparison_table}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 图表 -->
        {self._generate_comparison_charts_html(charts_base64)}

        <!-- Top 5 策略详情 -->
        <div class="card">
            <h2>Top 5 策略详情</h2>
            <div class="strategy-cards">
                {strategy_cards}
            </div>
        </div>

        <!-- 总结与结论 -->
        <div class="card">
            <h2>总结与结论</h2>
            <div class="conclusion">
                <h3>📈 策略对比分析结论</h3>
                {conclusion}
            </div>
        </div>

        <div class="footer">
            <p>⚠️ 免责声明：历史回测结果不代表未来收益，投资有风险，入市需谨慎。</p>
            <p style="margin-top: 10px;">Generated by Crypto Trading Backtest System | {self.timestamp}</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _generate_charts_html(self, charts_base64: Dict[str, str]) -> str:
        """生成图表 HTML"""
        if not charts_base64:
            return ""

        html_parts = ['<div class="card"><h2>可视化图表</h2>']

        chart_titles = {
            "equity": "权益曲线与回撤分析",
            "signals": "价格走势与交易信号",
            "monthly": "月度收益热力图",
        }

        for name, base64_data in charts_base64.items():
            title = chart_titles.get(name, name)
            html_parts.append(f"""
                <div class="chart-container">
                    <h3 style="color: #888; margin-bottom: 15px;">{title}</h3>
                    <img src="data:image/png;base64,{base64_data}" alt="{title}">
                </div>
            """)

        html_parts.append('</div>')
        return ''.join(html_parts)

    def _generate_comparison_charts_html(self, charts_base64: Dict[str, str]) -> str:
        """生成对比图表 HTML"""
        if not charts_base64:
            return ""

        html_parts = ['<div class="card"><h2>策略对比图表</h2>']

        chart_titles = {
            "metrics": "核心指标对比",
            "ranking": "策略排名",
            "equity": "权益曲线对比 (Top 5)",
        }

        for name, base64_data in charts_base64.items():
            title = chart_titles.get(name, name)
            html_parts.append(f"""
                <div class="chart-container">
                    <h3 style="color: #888; margin-bottom: 15px;">{title}</h3>
                    <img src="data:image/png;base64,{base64_data}" alt="{title}">
                </div>
            """)

        html_parts.append('</div>')
        return ''.join(html_parts)

    def _generate_trades_table(self, trades: List) -> str:
        """生成交易记录表格 HTML"""
        rows = []
        for i, trade in enumerate(trades, 1):
            action_class = "buy" if trade.action == "buy" else "sell"
            action_text = "买入 🟢" if trade.action == "buy" else "卖出 🔴"
            rows.append(f"""
                <tr>
                    <td>{i}</td>
                    <td>{trade.timestamp.strftime('%Y-%m-%d %H:%M')}</td>
                    <td class="{action_class}">{action_text}</td>
                    <td>${trade.price:,.2f}</td>
                    <td>{trade.quantity:.6f}</td>
                    <td>${trade.value:,.2f}</td>
                </tr>
            """)
        return ''.join(rows)

    def _generate_comparison_table(self, sorted_results: List) -> str:
        """生成对比表格 HTML"""
        rows = []
        for rank, (name, result) in enumerate(sorted_results, 1):
            m = result.metrics
            rank_class = f"rank-{rank}" if rank <= 3 else ""
            row_class = "best" if rank == 1 else ""

            return_class = "positive" if m.get('total_return_pct', 0) >= 0 else "negative"
            sharpe_class = "positive" if m.get('sharpe_ratio', 0) >= 1 else ("neutral" if m.get('sharpe_ratio', 0) >= 0 else "negative")

            rows.append(f"""
                <tr class="{row_class}">
                    <td><span class="rank {rank_class}">{rank}</span></td>
                    <td><strong>{STRATEGY_DESCRIPTIONS.get(name, {}).get('name', name)}</strong><br><small style="color: #888;">{name}</small></td>
                    <td class="{return_class}">{m.get('total_return_pct', 0):.2f}%</td>
                    <td class="{return_class}">{m.get('annual_return_pct', 0):.2f}%</td>
                    <td class="{sharpe_class}">{m.get('sharpe_ratio', 0):.2f}</td>
                    <td class="negative">{m.get('max_drawdown_pct', 0):.2f}%</td>
                    <td>{m.get('win_rate_pct', 0):.2f}%</td>
                    <td>{m.get('total_trades', 0)}</td>
                </tr>
            """)
        return ''.join(rows)

    def _generate_strategy_cards(self, top_results: List) -> str:
        """生成策略卡片 HTML"""
        cards = []
        for rank, (name, result) in enumerate(top_results, 1):
            m = result.metrics
            info = STRATEGY_DESCRIPTIONS.get(name, {})
            top_class = f"top-{rank}" if rank <= 3 else ""

            return_class = "positive" if m.get('total_return_pct', 0) >= 0 else "negative"

            cards.append(f"""
                <div class="strategy-card {top_class}">
                    <h4>
                        <span>{info.get('name', name)}</span>
                        <span class="rank {f'rank-{rank}' if rank <= 3 else ''}">#{rank}</span>
                    </h4>
                    <p style="color: #888; font-size: 0.85em; margin-bottom: 15px;">{info.get('type', 'N/A')}</p>
                    <div class="mini-metrics">
                        <div class="mini-metric">
                            <div class="value {return_class}">{m.get('total_return_pct', 0):.2f}%</div>
                            <div class="label">总收益</div>
                        </div>
                        <div class="mini-metric">
                            <div class="value neutral">{m.get('sharpe_ratio', 0):.2f}</div>
                            <div class="label">夏普</div>
                        </div>
                        <div class="mini-metric">
                            <div class="value">{m.get('win_rate_pct', 0):.1f}%</div>
                            <div class="label">胜率</div>
                        </div>
                        <div class="mini-metric">
                            <div class="value negative">{m.get('max_drawdown_pct', 0):.1f}%</div>
                            <div class="label">回撤</div>
                        </div>
                    </div>
                </div>
            """)
        return ''.join(cards)

    def _calculate_avg_holding_time(self, trades: List) -> str:
        """计算平均持仓时间"""
        if len(trades) < 2:
            return "N/A"

        holding_times = []
        for i in range(len(trades)):
            if trades[i].action == "sell":
                # 找到对应的买入
                for j in range(i - 1, -1, -1):
                    if trades[j].action == "buy":
                        delta = trades[i].timestamp - trades[j].timestamp
                        holding_times.append(delta.total_seconds() / 3600)  # 小时
                        break

        if not holding_times:
            return "N/A"

        avg_hours = sum(holding_times) / len(holding_times)
        if avg_hours >= 24:
            return f"{avg_hours / 24:.1f} 天"
        else:
            return f"{avg_hours:.1f} 小时"

    def _calculate_max_profit_loss(self, trades: List) -> tuple:
        """计算最大单笔盈亏"""
        max_profit = 0.0
        max_loss = 0.0

        for i in range(len(trades)):
            if trades[i].action == "sell":
                for j in range(i - 1, -1, -1):
                    if trades[j].action == "buy":
                        profit_pct = (trades[i].price - trades[j].price) / trades[j].price * 100
                        if profit_pct > max_profit:
                            max_profit = profit_pct
                        if profit_pct < max_loss:
                            max_loss = profit_pct
                        break

        return max_profit, max_loss

    def _generate_conclusion(self, metrics: Dict, strategy_name: str) -> str:
        """生成单策略结论"""
        total_return = metrics.get('total_return_pct', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown_pct', 0)
        win_rate = metrics.get('win_rate_pct', 0)

        strategy_info = STRATEGY_DESCRIPTIONS.get(strategy_name, {})
        strategy_cn_name = strategy_info.get('name', strategy_name)

        # 评价等级
        if sharpe >= 2:
            sharpe_eval = "优秀"
            sharpe_class = "badge-success"
        elif sharpe >= 1:
            sharpe_eval = "良好"
            sharpe_class = "badge-success"
        elif sharpe >= 0:
            sharpe_eval = "一般"
            sharpe_class = "badge-warning"
        else:
            sharpe_eval = "较差"
            sharpe_class = "badge-danger"

        conclusion = f"""
        <p><strong>策略表现：</strong>
        {strategy_cn_name} 在本次回测中{'实现盈利' if total_return >= 0 else '出现亏损'}，
        总收益率为 <span class="{'positive' if total_return >= 0 else 'negative'}">{total_return:.2f}%</span>。
        夏普比率为 <span class="neutral">{sharpe:.2f}</span>，风险调整后收益表现
        <span class="badge {sharpe_class}">{sharpe_eval}</span>。
        </p>

        <p style="margin-top: 15px;"><strong>风险分析：</strong>
        最大回撤为 <span class="negative">{max_dd:.2f}%</span>，
        {'风险控制较好' if max_dd > -15 else ('回撤幅度较大，建议关注风险控制' if max_dd > -25 else '回撤严重，风险较高')}。
        胜率为 {win_rate:.2f}%，{'交易胜率较高' if win_rate >= 50 else '胜率偏低，需优化策略参数'}。
        </p>

        <p style="margin-top: 15px;"><strong>投资建议：</strong>
        """

        if sharpe >= 1.5 and total_return > 0:
            conclusion += "该策略在当前参数下表现良好，可考虑进一步优化参数或增加资金规模进行测试。"
        elif sharpe >= 0 and total_return > 0:
            conclusion += "该策略有一定盈利能力，但风险调整后收益一般，建议优化策略参数或结合其他策略使用。"
        else:
            conclusion += "该策略在当前参数下表现不佳，建议调整策略参数或选择其他策略。"

        conclusion += "</p>"

        # 添加适用性说明
        suitable_market = strategy_info.get('suitable_market', 'N/A')
        conclusion += f"""
        <p style="margin-top: 15px;"><strong>适用场景：</strong>
        该策略适用于 {suitable_market}。请注意，历史回测结果不代表未来收益，实盘交易前请充分测试并做好风险管理。
        </p>
        """

        return conclusion

    def _generate_comparison_conclusion(self, sorted_results: List, coin: str, days: int) -> str:
        """生成多策略对比结论"""
        if not sorted_results:
            return "<p>无有效回测结果</p>"

        best_name, best_result = sorted_results[0]
        best_info = STRATEGY_DESCRIPTIONS.get(best_name, {})
        best_m = best_result.metrics

        # 统计盈利策略数量
        profitable = sum(1 for _, r in sorted_results if r.metrics.get('total_return_pct', 0) > 0)

        # 找出夏普比率最高的策略
        best_sharpe_name = best_name
        best_sharpe = best_m.get('sharpe_ratio', 0)

        # 找出回撤最小的策略
        min_dd_name, min_dd_result = min(sorted_results, key=lambda x: x[1].metrics.get('max_drawdown_pct', 0))
        min_dd = min_dd_result.metrics.get('max_drawdown_pct', 0)

        conclusion = f"""
        <p><strong>整体表现：</strong>
        在 {coin.upper()} 的 {days} 天回测中，共测试了 {len(sorted_results)} 个策略，
        其中 <span class="positive">{profitable}</span> 个策略实现盈利，
        {len(sorted_results) - profitable} 个策略出现亏损。
        </p>

        <p style="margin-top: 15px;"><strong>最佳策略：</strong>
        <span class="positive">{best_info.get('name', best_name)}</span> 表现最优，
        总收益率 {best_m.get('total_return_pct', 0):.2f}%，
        夏普比率 {best_m.get('sharpe_ratio', 0):.2f}，
        最大回撤 {best_m.get('max_drawdown_pct', 0):.2f}%。
        </p>

        <p style="margin-top: 15px;"><strong>风险最低：</strong>
        {STRATEGY_DESCRIPTIONS.get(min_dd_name, {}).get('name', min_dd_name)} 回撤控制最好，
        最大回撤仅 {min_dd:.2f}%。
        </p>

        <p style="margin-top: 15px;"><strong>策略类型分析：</strong>
        """

        # 分析不同类型策略的表现
        trend_strategies = ["ma_cross", "macd", "breakout", "momentum", "atr_stop"]
        mean_reversion_strategies = ["rsi", "bollinger", "mean_reversion", "vwap", "stochastic"]

        trend_avg = sum(r.metrics.get('sharpe_ratio', 0) for n, r in sorted_results if n in trend_strategies) / max(len([n for n, _ in sorted_results if n in trend_strategies]), 1)
        mr_avg = sum(r.metrics.get('sharpe_ratio', 0) for n, r in sorted_results if n in mean_reversion_strategies) / max(len([n for n, _ in sorted_results if n in mean_reversion_strategies]), 1)

        if trend_avg > mr_avg:
            conclusion += f"趋势跟踪类策略平均夏普比率 ({trend_avg:.2f}) 优于均值回归类 ({mr_avg:.2f})，当前市场可能更适合趋势策略。"
        else:
            conclusion += f"均值回归类策略平均夏普比率 ({mr_avg:.2f}) 优于趋势跟踪类 ({trend_avg:.2f})，当前市场可能更适合震荡策略。"

        conclusion += "</p>"

        conclusion += f"""
        <p style="margin-top: 15px;"><strong>投资建议：</strong>
        建议优先考虑 {best_info.get('name', best_name)} 策略，同时可结合回撤控制较好的
        {STRATEGY_DESCRIPTIONS.get(min_dd_name, {}).get('name', min_dd_name)} 进行组合配置，
        以平衡收益与风险。
        </p>

        <p style="margin-top: 15px; color: #ff4757;">
        ⚠️ <strong>重要提示：</strong>历史回测结果不代表未来收益，市场环境变化可能导致策略表现差异。
        实盘交易前请进行充分的模拟测试，并严格执行风险管理规则。
        </p>
        """

        return conclusion
