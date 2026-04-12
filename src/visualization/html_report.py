"""
HTML Report Generator

Generates complete backtest HTML reports with strategy details, trade records, return analysis, and summary conclusions.
"""

import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Strategy descriptions
STRATEGY_DESCRIPTIONS = {
    "ma_cross": {
        "name": "Moving Average Cross",
        "name_en": "Moving Average Cross",
        "type": "Trend Following",
        "description": "Generates buy/sell signals based on the short-term moving average crossing above/below the long-term moving average. Buys when the short MA crosses above the long MA, sells when it crosses below.",
        "params": ["short_window (Short MA period)", "long_window (Long MA period)"],
        "suitable_market": "Markets with clear trends",
        "risk_level": "Medium",
    },
    "rsi": {
        "name": "RSI Overbought/Oversold",
        "name_en": "RSI Overbought/Oversold",
        "type": "Mean Reversion",
        "description": "Uses the RSI indicator to identify overbought and oversold zones. Buys when RSI falls below the oversold threshold, sells when it rises above the overbought threshold.",
        "params": ["period (RSI period)", "oversold (Oversold threshold)", "overbought (Overbought threshold)"],
        "suitable_market": "Range-bound markets",
        "risk_level": "Medium",
    },
    "bollinger": {
        "name": "Bollinger Bands",
        "name_en": "Bollinger Bands",
        "type": "Mean Reversion",
        "description": "Uses Bollinger Band upper and lower bands to gauge price deviation. Buys when price touches the lower band, sells when it touches the upper band.",
        "params": ["window (Moving window)", "num_std (Standard deviation multiplier)"],
        "suitable_market": "Range-bound markets",
        "risk_level": "Medium",
    },
    "multi_factor": {
        "name": "Multi-Factor Strategy",
        "name_en": "Multi-Factor Strategy",
        "type": "Comprehensive Strategy",
        "description": "Combines multiple factors including MA trend, RSI, volume, and volatility, generating signals through weighted scoring.",
        "params": ["ma_short", "ma_long", "rsi_period", "volume_threshold"],
        "suitable_market": "Various market conditions",
        "risk_level": "Medium",
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "name_en": "Mean Reversion",
        "type": "Statistical Arbitrage",
        "description": "Takes contrarian positions based on the degree of price deviation from the mean. Expects reversion when price deviates significantly from the mean.",
        "params": ["window (Window period)", "entry_z (Entry Z-score)", "exit_z (Exit Z-score)"],
        "suitable_market": "Range-bound markets",
        "risk_level": "Medium-High",
    },
    "macd": {
        "name": "MACD Strategy",
        "name_en": "MACD Strategy",
        "type": "Trend Following",
        "description": "Generates buy/sell signals based on MACD line and signal line crossovers. Buys when MACD crosses above the signal line, sells when it crosses below.",
        "params": ["fast_period (Fast line period)", "slow_period (Slow line period)", "signal_period (Signal line period)"],
        "suitable_market": "Trending markets",
        "risk_level": "Medium",
    },
    "breakout": {
        "name": "Breakout Strategy",
        "name_en": "Breakout Strategy",
        "type": "Trend Following",
        "description": "Buys when price breaks above the N-period high, sells when it breaks below the N-period low.",
        "params": ["window (Breakout window)", "confirmation (Whether to confirm breakout)"],
        "suitable_market": "Trend initiation phase",
        "risk_level": "Medium-High",
    },
    "vwap": {
        "name": "VWAP Strategy",
        "name_en": "VWAP Strategy",
        "type": "Mean Reversion",
        "description": "Uses Volume Weighted Average Price to gauge price deviation. Buys when price is below VWAP, sells when above VWAP.",
        "params": ["window (VWAP window)", "deviation (Deviation threshold)"],
        "suitable_market": "Intraday/Range-bound markets",
        "risk_level": "Medium",
    },
    "momentum": {
        "name": "Momentum Strategy",
        "name_en": "Momentum Strategy",
        "type": "Trend Following",
        "description": "Uses Rate of Change (ROC) and momentum indicators to assess trend strength and trade in the direction of the trend.",
        "params": ["roc_period (ROC period)", "momentum_period (Momentum period)", "threshold (Threshold)"],
        "suitable_market": "Trending markets",
        "risk_level": "Medium",
    },
    "atr_stop": {
        "name": "ATR Stop Loss Strategy",
        "name_en": "ATR Stop Loss Strategy",
        "type": "Trend Following",
        "description": "Uses ATR to calculate dynamic stop loss levels, protecting profits while allowing them to grow.",
        "params": ["atr_period (ATR period)", "multiplier (ATR multiplier)", "trend_ma (Trend MA)"],
        "suitable_market": "Trending markets",
        "risk_level": "Medium",
    },
    "stochastic": {
        "name": "Stochastic Strategy",
        "name_en": "Stochastic Strategy",
        "type": "Mean Reversion",
        "description": "Generates signals using K-line and D-line crossovers along with overbought and oversold zones.",
        "params": ["k_period (K period)", "d_period (D period)"],
        "suitable_market": "Range-bound markets",
        "risk_level": "Medium",
    },
    "grid": {
        "name": "Grid Trading Strategy",
        "name_en": "Grid Trading Strategy",
        "type": "Range Arbitrage",
        "description": "Sets up multiple buy/sell grids within a predefined price range, buying low and selling high to capture spreads.",
        "params": ["lower_price (Lower bound)", "upper_price (Upper bound)", "grid_num (Number of grids)", "amount_per_grid (Amount per grid)"],
        "suitable_market": "Range-bound markets",
        "risk_level": "Medium-High (high risk in trending markets)",
    },
    "martingale": {
        "name": "Martingale Strategy",
        "name_en": "Martingale Strategy",
        "type": "High-Risk Gambling",
        "description": "Doubles the bet after each loss, returning to the initial bet after a win. ⚠️ High-risk strategy requiring sufficient capital.",
        "params": ["base_amount (Base position)", "multiplier (Multiplier)", "max_steps (Maximum steps)", "target_profit (Target profit)", "stop_loss (Stop loss)"],
        "suitable_market": "Testing only, not recommended for live trading",
        "risk_level": "Very High ⚠️",
    },
}


class HTMLReportGenerator:
    """HTML Report Generator"""

    # Shared CSS styles (used by both single-strategy and comparison reports)
    _SHARED_CSS = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e4; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; padding: 30px 0; border-bottom: 2px solid #0f3460; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; color: #00d9ff; margin-bottom: 10px; text-shadow: 0 0 20px rgba(0, 217, 255, 0.3); }
        .header .subtitle { color: #888; font-size: 1.1em; }
        .card {
            background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 25px;
            margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px);
        }
        .card h2 {
            color: #00d9ff; margin-bottom: 20px; padding-bottom: 10px;
            border-bottom: 1px solid rgba(0, 217, 255, 0.3); display: flex; align-items: center; gap: 10px;
        }
        .card h2::before { content: ''; width: 4px; height: 24px; background: #00d9ff; border-radius: 2px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
        th { background: rgba(0, 217, 255, 0.1); color: #00d9ff; font-weight: 600; }
        tr:hover { background: rgba(255, 255, 255, 0.05); }
        .positive { color: #00ff88; } .negative { color: #ff4757; } .neutral { color: #00d9ff; }
        .chart-container { text-align: center; margin: 20px 0; }
        .chart-container img { max-width: 100%; border-radius: 10px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }
        .conclusion {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 255, 136, 0.1) 100%);
            border-left: 4px solid #00d9ff; padding: 20px; border-radius: 0 10px 10px 0;
        }
        .conclusion h3 { color: #00d9ff; margin-bottom: 15px; }
        .conclusion p { line-height: 1.8; color: #ccc; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 500; }
        .badge-success { background: rgba(0, 255, 136, 0.2); color: #00ff88; }
        .badge-warning { background: rgba(255, 193, 7, 0.2); color: #ffc107; }
        .badge-danger { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
        .footer { text-align: center; padding: 30px 0; color: #666; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 40px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .metric-item { background: rgba(0, 0, 0, 0.2); padding: 20px; border-radius: 10px; text-align: center; transition: transform 0.3s ease; }
        .metric-item:hover { transform: translateY(-5px); }
        .metric-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .metric-label { color: #888; font-size: 0.9em; }
        @media (max-width: 768px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } .summary-stats { grid-template-columns: repeat(2, 1fr); } }
    """

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
        Generate single-strategy backtest HTML report

        Args:
            result: BacktestResult object
            df: Price data
            strategy_name: Strategy name
            coin: Coin symbol
            days: Backtest days
            interval: Time interval
            capital: Initial capital
            output_path: Output path
            chart_paths: Chart path dictionary

        Returns:
            Generated HTML file path
        """
        strategy_info = STRATEGY_DESCRIPTIONS.get(strategy_name, {})

        # Read charts and convert to base64
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

        # Save HTML file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"📄 HTML report generated: {output_file}")
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
        Generate multi-strategy comparison HTML report

        Args:
            results: Strategy results dictionary
            coin: Coin symbol
            days: Backtest days
            interval: Time interval
            capital: Initial capital
            output_path: Output path
            chart_paths: Chart path dictionary

        Returns:
            Generated HTML file path
        """
        # Read charts and convert to base64
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

        # Save HTML file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"📄 HTML comparison report generated: {output_file}")
        return str(output_file)

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 encoding"""
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
        """Generate single-strategy report HTML content"""
        metrics = result.metrics
        trades = result.trades
        buy_trades = [t for t in trades if t.action == "buy"]
        sell_trades = [t for t in trades if t.action == "sell"]
        avg_holding_time = self._calculate_avg_holding_time(trades)
        max_profit, max_loss = self._calculate_max_profit_loss(trades)
        conclusion = self._generate_conclusion(metrics, strategy_name)

        metrics_html = self._build_metrics_card(metrics)
        strategy_html = self._build_strategy_info_card(strategy_info, strategy_name, coin, days, interval, capital)
        stats_html = self._build_trade_stats_card(metrics, buy_trades, sell_trades, avg_holding_time, max_profit, max_loss, result)
        charts_html = self._generate_charts_html(charts_base64)
        trades_html = self._build_trades_card(trades)
        conclusion_html = self._build_conclusion_card(conclusion)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {strategy_name.upper()} - {coin.upper()}</title>
    <style>{self._SHARED_CSS}
        .strategy-info {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
        .info-item {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }}
        .info-label {{ color: #888; }} .info-value {{ color: #fff; font-weight: 500; }}
        .summary-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px; }}
        .summary-stat {{ background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 8px; text-align: center; }}
        .summary-stat .value {{ font-size: 1.3em; font-weight: bold; color: #00d9ff; }}
        .summary-stat .label {{ font-size: 0.85em; color: #888; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Backtest Report</h1>
            <div class="subtitle">{strategy_info.get('name', strategy_name)} | {coin.upper()} | {days} Days | {interval} Interval</div>
            <div class="subtitle" style="margin-top: 10px; font-size: 0.9em;">Generated: {self.timestamp}</div>
        </div>
        {metrics_html}
        {strategy_html}
        {stats_html}
        {charts_html}
        {trades_html}
        {conclusion_html}
        <div class="footer">
            <p>⚠️ Disclaimer: Historical backtest results do not guarantee future returns. Trading involves risk, proceed with caution.</p>
            <p style="margin-top: 10px;">Generated by Crypto Trading Backtest System | {self.timestamp}</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _build_metrics_card(self, metrics: Dict) -> str:
        """Build core metrics card"""
        m = metrics
        return f"""
        <div class="card">
            <h2>Key Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value {'positive' if m.get('total_return_pct', 0) >= 0 else 'negative'}">{m.get('total_return_pct', 0):.2f}%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value {'positive' if m.get('annual_return_pct', 0) >= 0 else 'negative'}">{m.get('annual_return_pct', 0):.2f}%</div>
                    <div class="metric-label">Annualized Return</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value neutral">{m.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value negative">{m.get('max_drawdown_pct', 0):.2f}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value neutral">{m.get('win_rate_pct', 0):.2f}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value neutral">{m.get('total_trades', 0)}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
            </div>
        </div>"""

    def _build_strategy_info_card(self, strategy_info: Dict, strategy_name: str, coin: str, days: int, interval: str, capital: float) -> str:
        """Build strategy details card"""
        si = strategy_info
        return f"""
        <div class="card">
            <h2>Strategy Details</h2>
            <div class="strategy-info">
                <div class="info-item"><span class="info-label">Strategy Name</span><span class="info-value">{si.get('name', strategy_name)}</span></div>
                <div class="info-item"><span class="info-label">English Name</span><span class="info-value">{si.get('name_en', strategy_name)}</span></div>
                <div class="info-item"><span class="info-label">Strategy Type</span><span class="info-value">{si.get('type', 'N/A')}</span></div>
                <div class="info-item"><span class="info-label">Risk Level</span><span class="info-value">{si.get('risk_level', 'N/A')}</span></div>
                <div class="info-item"><span class="info-label">Suitable Market</span><span class="info-value">{si.get('suitable_market', 'N/A')}</span></div>
                <div class="info-item"><span class="info-label">Backtest Coin</span><span class="info-value">{coin.upper()}</span></div>
                <div class="info-item"><span class="info-label">Backtest Days</span><span class="info-value">{days} Days</span></div>
                <div class="info-item"><span class="info-label">K-Line Interval</span><span class="info-value">{interval}</span></div>
                <div class="info-item"><span class="info-label">Initial Capital</span><span class="info-value">${capital:,.2f}</span></div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                <strong style="color: #00d9ff;">Strategy Description:</strong>
                <p style="margin-top: 10px; color: #ccc; line-height: 1.6;">{si.get('description', 'No description available')}</p>
            </div>
        </div>"""

    def _build_trade_stats_card(self, metrics: Dict, buy_trades: List, sell_trades: List, avg_holding_time: str, max_profit: float, max_loss: float, result: Any) -> str:
        """Build trade statistics card"""
        m = metrics
        return f"""
        <div class="card">
            <h2>Trade Statistics</h2>
            <div class="summary-stats">
                <div class="summary-stat"><div class="value">{len(buy_trades)}</div><div class="label">Buy Count</div></div>
                <div class="summary-stat"><div class="value">{len(sell_trades)}</div><div class="label">Sell Count</div></div>
                <div class="summary-stat"><div class="value">{avg_holding_time}</div><div class="label">Avg Holding Time</div></div>
                <div class="summary-stat"><div class="value" style="color: #00ff88;">+{max_profit:.2f}%</div><div class="label">Max Single Profit</div></div>
                <div class="summary-stat"><div class="value" style="color: #ff4757;">{max_loss:.2f}%</div><div class="label">Max Single Loss</div></div>
                <div class="summary-stat"><div class="value">{m.get('trades_per_month', 0):.1f}</div><div class="label">Monthly Avg Trades</div></div>
                <div class="summary-stat"><div class="value">{m.get('volatility_pct', 0):.2f}%</div><div class="label">Annualized Volatility</div></div>
                <div class="summary-stat"><div class="value">${result.equity_curve.iloc[-1]:,.2f}</div><div class="label">Final Equity</div></div>
            </div>
        </div>"""

    def _build_trades_card(self, trades: List) -> str:
        """Build trade records card"""
        return f"""
        <div class="card">
            <h2>Trade Records</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead><tr><th>#</th><th>Time</th><th>Action</th><th>Price</th><th>Quantity</th><th>Amount</th></tr></thead>
                    <tbody>{self._generate_trades_table(trades)}</tbody>
                </table>
            </div>
        </div>"""

    def _build_conclusion_card(self, conclusion: str) -> str:
        """Build conclusion card"""
        return f"""
        <div class="card">
            <h2>Summary & Conclusions</h2>
            <div class="conclusion"><h3>📈 Backtest Analysis Conclusions</h3>{conclusion}</div>
        </div>"""

    def _generate_comparison_html(
        self,
        results: Dict[str, Any],
        coin: str,
        days: int,
        interval: str,
        capital: float,
        charts_base64: Dict[str, str],
    ) -> str:
        """Generate multi-strategy comparison report HTML content"""
        sorted_results = sorted(results.items(), key=lambda x: x[1].metrics.get("sharpe_ratio", 0), reverse=True)
        best_strategy = sorted_results[0][0] if sorted_results else "N/A"
        best_metrics = sorted_results[0][1].metrics if sorted_results else {}
        comparison_table = self._generate_comparison_table(sorted_results)
        strategy_cards = self._generate_strategy_cards(sorted_results[:5])
        conclusion = self._generate_comparison_conclusion(sorted_results, coin, days)

        best_html = self._build_best_strategy_card(best_strategy, best_metrics)
        table_html = self._build_comparison_table_card(comparison_table)
        charts_html = self._generate_comparison_charts_html(charts_base64)
        cards_html = self._build_top_cards(strategy_cards)
        conclusion_html = self._build_conclusion_card(conclusion)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategy Comparison Report - {coin.upper()}</title>
    <style>{self._SHARED_CSS}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        th, td {{ text-align: center; }}
        th {{ position: sticky; top: 0; }}
        .best-strategy {{
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
            border: 2px solid rgba(0, 255, 136, 0.3); text-align: center; padding: 30px;
        }}
        .best-strategy h3 {{ color: #00ff88; font-size: 1.5em; margin-bottom: 15px; }}
        .best-strategy .name {{ font-size: 2.5em; color: #fff; font-weight: bold; margin: 10px 0; }}
        .best-strategy .metrics {{ display: flex; justify-content: center; gap: 40px; margin-top: 20px; }}
        .best-strategy .metric {{ text-align: center; }}
        .best-strategy .metric .value {{ font-size: 1.8em; font-weight: bold; color: #00ff88; }}
        .best-strategy .metric .label {{ color: #888; font-size: 0.9em; }}
        tr.best {{ background: rgba(0, 255, 136, 0.1); }}
        .rank {{
            display: inline-block; width: 30px; height: 30px; line-height: 30px;
            border-radius: 50%; background: rgba(0, 217, 255, 0.2); color: #00d9ff; font-weight: bold;
        }}
        .rank-1 {{ background: rgba(255, 215, 0, 0.3); color: #ffd700; }}
        .rank-2 {{ background: rgba(192, 192, 192, 0.3); color: #c0c0c0; }}
        .rank-3 {{ background: rgba(205, 127, 50, 0.3); color: #cd7f32; }}
        .strategy-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
        .strategy-card {{ background: rgba(0, 0, 0, 0.2); border-radius: 10px; padding: 20px; border-left: 4px solid #00d9ff; }}
        .strategy-card.top-1 {{ border-left-color: #ffd700; background: rgba(255, 215, 0, 0.05); }}
        .strategy-card.top-2 {{ border-left-color: #c0c0c0; }}
        .strategy-card.top-3 {{ border-left-color: #cd7f32; }}
        .strategy-card h4 {{ color: #fff; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }}
        .strategy-card .mini-metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
        .strategy-card .mini-metric {{ text-align: center; padding: 8px; background: rgba(0, 0, 0, 0.2); border-radius: 5px; }}
        .strategy-card .mini-metric .value {{ font-weight: bold; }}
        .strategy-card .mini-metric .label {{ font-size: 0.75em; color: #888; }}
        @media (max-width: 768px) {{ .best-strategy .metrics {{ flex-direction: column; gap: 20px; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Strategy Comparison Report</h1>
            <div class="subtitle">{coin.upper()} | {days} Days | {interval} Interval | {len(results)} Strategies</div>
            <div class="subtitle" style="margin-top: 10px; font-size: 0.9em;">Generated: {self.timestamp}</div>
        </div>
        {best_html}
        {table_html}
        {charts_html}
        {cards_html}
        {conclusion_html}
        <div class="footer">
            <p>⚠️ Disclaimer: Historical backtest results do not guarantee future returns. Trading involves risk, proceed with caution.</p>
            <p style="margin-top: 10px;">Generated by Crypto Trading Backtest System | {self.timestamp}</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _build_best_strategy_card(self, best_strategy: str, best_metrics: Dict) -> str:
        """Build best strategy card"""
        bm = best_metrics
        return f"""
        <div class="card best-strategy">
            <h3>🏆 Best Strategy</h3>
            <div class="name">{STRATEGY_DESCRIPTIONS.get(best_strategy, {}).get('name', best_strategy.upper())}</div>
            <div class="metrics">
                <div class="metric"><div class="value">{'+' if bm.get('total_return_pct', 0) >= 0 else ''}{bm.get('total_return_pct', 0):.2f}%</div><div class="label">Total Return</div></div>
                <div class="metric"><div class="value">{bm.get('sharpe_ratio', 0):.2f}</div><div class="label">Sharpe Ratio</div></div>
                <div class="metric"><div class="value">{bm.get('win_rate_pct', 0):.2f}%</div><div class="label">Win Rate</div></div>
                <div class="metric"><div class="value">{bm.get('max_drawdown_pct', 0):.2f}%</div><div class="label">Max Drawdown</div></div>
            </div>
        </div>"""

    def _build_comparison_table_card(self, comparison_table: str) -> str:
        """Build comparison ranking table card"""
        return f"""
        <div class="card">
            <h2>Strategy Ranking Comparison</h2>
            <div style="overflow-x: auto;">
                <table>
                    <thead><tr><th>Rank</th><th>Strategy</th><th>Total Return</th><th>Annual Return</th><th>Sharpe Ratio</th><th>Max Drawdown</th><th>Win Rate</th><th>Trades</th></tr></thead>
                    <tbody>{comparison_table}</tbody>
                </table>
            </div>
        </div>"""

    def _build_top_cards(self, strategy_cards: str) -> str:
        """Build Top 5 strategy detail cards"""
        return f"""
        <div class="card">
            <h2>Top 5 Strategy Details</h2>
            <div class="strategy-cards">{strategy_cards}</div>
        </div>"""

    def _generate_charts_html(self, charts_base64: Dict[str, str]) -> str:
        """Generate charts HTML"""
        if not charts_base64:
            return ""

        html_parts = ['<div class="card"><h2>Visualization Charts</h2>']

        chart_titles = {
            "equity": "Equity Curve & Drawdown Analysis",
            "signals": "Price Trends & Trading Signals",
            "monthly": "Monthly Returns Heatmap",
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
        """Generate comparison charts HTML"""
        if not charts_base64:
            return ""

        html_parts = ['<div class="card"><h2>Strategy Comparison Charts</h2>']

        chart_titles = {
            "metrics": "Core Metrics Comparison",
            "ranking": "Strategy Ranking",
            "equity": "Equity Curve Comparison (Top 5)",
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
        """Generate trade records table HTML"""
        rows = []
        for i, trade in enumerate(trades, 1):
            action_class = "buy" if trade.action == "buy" else "sell"
            action_text = "Buy 🟢" if trade.action == "buy" else "Sell 🔴"
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
        """Generate comparison table HTML"""
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
        """Generate strategy cards HTML"""
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
                            <div class="label">Return</div>
                        </div>
                        <div class="mini-metric">
                            <div class="value neutral">{m.get('sharpe_ratio', 0):.2f}</div>
                            <div class="label">Sharpe</div>
                        </div>
                        <div class="mini-metric">
                            <div class="value">{m.get('win_rate_pct', 0):.1f}%</div>
                            <div class="label">Win Rate</div>
                        </div>
                        <div class="mini-metric">
                            <div class="value negative">{m.get('max_drawdown_pct', 0):.1f}%</div>
                            <div class="label">Drawdown</div>
                        </div>
                    </div>
                </div>
            """)
        return ''.join(cards)

    def _calculate_avg_holding_time(self, trades: List) -> str:
        """Calculate average holding time -- O(n) single pass"""
        if len(trades) < 2:
            return "N/A"

        holding_times = []
        last_buy_time = None
        for trade in trades:
            if trade.action == "buy":
                last_buy_time = trade.timestamp
            elif trade.action == "sell" and last_buy_time is not None:
                delta = trade.timestamp - last_buy_time
                holding_times.append(delta.total_seconds() / 3600)

        if not holding_times:
            return "N/A"

        avg_hours = sum(holding_times) / len(holding_times)
        if avg_hours >= 24:
            return f"{avg_hours / 24:.1f} days"
        else:
            return f"{avg_hours:.1f} hours"

    def _calculate_max_profit_loss(self, trades: List) -> tuple:
        """Calculate max single-trade profit and loss -- O(n) single pass"""
        max_profit = 0.0
        max_loss = 0.0
        last_buy_price = None

        for trade in trades:
            if trade.action == "buy":
                last_buy_price = trade.price
            elif trade.action == "sell" and last_buy_price is not None:
                profit_pct = (trade.price - last_buy_price) / last_buy_price * 100
                if profit_pct > max_profit:
                    max_profit = profit_pct
                if profit_pct < max_loss:
                    max_loss = profit_pct

        return max_profit, max_loss

    def _generate_conclusion(self, metrics: Dict, strategy_name: str) -> str:
        """Generate single-strategy conclusion"""
        total_return = metrics.get('total_return_pct', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown_pct', 0)
        win_rate = metrics.get('win_rate_pct', 0)

        strategy_info = STRATEGY_DESCRIPTIONS.get(strategy_name, {})
        strategy_display_name = strategy_info.get('name', strategy_name)

        # Rating levels
        if sharpe >= 2:
            sharpe_eval = "Excellent"
            sharpe_class = "badge-success"
        elif sharpe >= 1:
            sharpe_eval = "Good"
            sharpe_class = "badge-success"
        elif sharpe >= 0:
            sharpe_eval = "Average"
            sharpe_class = "badge-warning"
        else:
            sharpe_eval = "Poor"
            sharpe_class = "badge-danger"

        conclusion = f"""
        <p><strong>Strategy Performance:</strong>
        {strategy_display_name} {'generated a profit' if total_return >= 0 else 'incurred a loss'} in this backtest,
        with a total return of <span class="{'positive' if total_return >= 0 else 'negative'}">{total_return:.2f}%</span>.
        The Sharpe ratio is <span class="neutral">{sharpe:.2f}</span>, and the risk-adjusted return is
        <span class="badge {sharpe_class}">{sharpe_eval}</span>.
        </p>

        <p style="margin-top: 15px;"><strong>Risk Analysis:</strong>
        The maximum drawdown is <span class="negative">{max_dd:.2f}%</span>,
        {'indicating good risk control' if max_dd > -15 else ('indicating a relatively large drawdown; risk management improvements are recommended' if max_dd > -25 else 'indicating a severe drawdown with high risk')}.
        The win rate is {win_rate:.2f}%, {'which is relatively high' if win_rate >= 50 else 'which is below average; strategy parameter optimization is recommended'}.
        </p>

        <p style="margin-top: 15px;"><strong>Investment Advice:</strong>
        """

        if sharpe >= 1.5 and total_return > 0:
            conclusion += "This strategy performs well with current parameters. Consider further parameter optimization or testing with larger capital."
        elif sharpe >= 0 and total_return > 0:
            conclusion += "This strategy has some profitability, but the risk-adjusted return is average. Consider optimizing strategy parameters or combining with other strategies."
        else:
            conclusion += "This strategy underperforms with current parameters. Consider adjusting parameters or selecting a different strategy."

        conclusion += "</p>"

        # Add applicability note
        suitable_market = strategy_info.get('suitable_market', 'N/A')
        conclusion += f"""
        <p style="margin-top: 15px;"><strong>Applicable Scenarios:</strong>
        This strategy is suitable for {suitable_market}. Please note that historical backtest results do not guarantee future returns. Conduct thorough testing and implement proper risk management before live trading.
        </p>
        """

        return conclusion

    def _generate_comparison_conclusion(self, sorted_results: List, coin: str, days: int) -> str:
        """Generate multi-strategy comparison conclusion"""
        if not sorted_results:
            return "<p>No valid backtest results</p>"

        best_name, best_result = sorted_results[0]
        best_info = STRATEGY_DESCRIPTIONS.get(best_name, {})
        best_m = best_result.metrics

        # Count profitable strategies
        profitable = sum(1 for _, r in sorted_results if r.metrics.get('total_return_pct', 0) > 0)

        # Find strategy with highest Sharpe ratio
        best_sharpe_name = best_name
        best_sharpe = best_m.get('sharpe_ratio', 0)

        # Find strategy with smallest drawdown (max_drawdown_pct is negative, closest to 0 is best)
        min_dd_name, min_dd_result = max(sorted_results, key=lambda x: x[1].metrics.get('max_drawdown_pct', 0))
        min_dd = min_dd_result.metrics.get('max_drawdown_pct', 0)

        conclusion = f"""
        <p><strong>Overall Performance:</strong>
        In the {days}-day backtest for {coin.upper()}, a total of {len(sorted_results)} strategies were tested,
        of which <span class="positive">{profitable}</span> generated profits
        and {len(sorted_results) - profitable} incurred losses.
        </p>

        <p style="margin-top: 15px;"><strong>Best Strategy:</strong>
        <span class="positive">{best_info.get('name', best_name)}</span> performed the best,
        with a total return of {best_m.get('total_return_pct', 0):.2f}%,
        a Sharpe ratio of {best_m.get('sharpe_ratio', 0):.2f},
        and a maximum drawdown of {best_m.get('max_drawdown_pct', 0):.2f}%.
        </p>

        <p style="margin-top: 15px;"><strong>Lowest Risk:</strong>
        {STRATEGY_DESCRIPTIONS.get(min_dd_name, {}).get('name', min_dd_name)} had the best drawdown control,
        with a maximum drawdown of only {min_dd:.2f}%.
        </p>

        <p style="margin-top: 15px;"><strong>Strategy Type Analysis:</strong>
        """

        # Analyze performance by strategy type
        trend_strategies = ["ma_cross", "macd", "breakout", "momentum", "atr_stop"]
        mean_reversion_strategies = ["rsi", "bollinger", "mean_reversion", "vwap", "stochastic"]

        trend_avg = sum(r.metrics.get('sharpe_ratio', 0) for n, r in sorted_results if n in trend_strategies) / max(len([n for n, _ in sorted_results if n in trend_strategies]), 1)
        mr_avg = sum(r.metrics.get('sharpe_ratio', 0) for n, r in sorted_results if n in mean_reversion_strategies) / max(len([n for n, _ in sorted_results if n in mean_reversion_strategies]), 1)

        if trend_avg > mr_avg:
            conclusion += f"Trend-following strategies have a higher average Sharpe ratio ({trend_avg:.2f}) than mean-reversion strategies ({mr_avg:.2f}), suggesting the current market may be more suitable for trend strategies."
        else:
            conclusion += f"Mean-reversion strategies have a higher average Sharpe ratio ({mr_avg:.2f}) than trend-following strategies ({trend_avg:.2f}), suggesting the current market may be more suitable for range-bound strategies."

        conclusion += "</p>"

        conclusion += f"""
        <p style="margin-top: 15px;"><strong>Investment Advice:</strong>
        It is recommended to prioritize the {best_info.get('name', best_name)} strategy,
        and consider combining it with {STRATEGY_DESCRIPTIONS.get(min_dd_name, {}).get('name', min_dd_name)}
        which has better drawdown control for a balanced portfolio that manages both returns and risk.
        </p>

        <p style="margin-top: 15px; color: #ff4757;">
        ⚠️ <strong>Important Notice:</strong> Historical backtest results do not guarantee future returns. Changes in market conditions may cause strategy performance to differ.
        Please conduct thorough simulated testing before live trading and strictly follow risk management rules.
        </p>
        """

        return conclusion
