# Crypto Trading Data Fetcher & Backtest System

中高频加密货币交易策略回测系统，支持 BTC/ETH/SOL 的历史数据获取、策略回测和收益可视化。

## 📁 项目结构

```
crypto_trading/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── setup.sh                     # 安装脚本
├── quick_start.py               # 快速测试脚本
├── run_backtest.py              # 主回测程序 ⭐
│
├── config/
│   └── settings.yaml           # 配置文件
│
├── src/                         # 源代码
│   ├── main.py                 # 实时数据获取主入口
│   ├── data_fetcher.py         # 实时数据获取
│   ├── websocket_client.py     # WebSocket 实时流
│   ├── historical_data.py      # 历史数据获取 ⭐
│   ├── strategies.py           # 交易策略集合 ⭐
│   ├── backtest.py             # 回测引擎 ⭐
│   ├── utils.py                # 工具函数
│   └── visualization.py        # 可视化模块 ⭐
│
├── data/                        # 数据目录
│   ├── raw/                    # 原始数据
│   ├── historical/             # 历史K线数据
│   └── processed/              # 处理后数据
│
├── logs/                        # 日志文件
│   ├── backtest.log            # 回测日志
│   └── backtest_*.json         # 决策记录
│
└── results/                     # 回测结果
    ├── *_equity.png            # 权益曲线
    ├── *_signals.png           # 交易信号图
    ├── *_monthly.png           # 月度收益图
    └── strategy_comparison.png # 策略对比
```

## 🎯 核心功能

### 1. 历史数据获取 (`src/historical_data.py`)

**功能：** 从 Binance 获取 BTC/ETH/SOL 的历史K线数据

**数据源确认：**
- ✅ Binance API (api.binance.com) - 免费，支持2年历史数据
- 支持时间粒度：1m, 5m, 15m, 1h, 4h, 1d
- 自动分页处理，突破1000条限制

**主要方法：**
```python
fetcher = HistoricalDataFetcher()

# 获取2小时K线数据（2年）
df = fetcher.fetch_historical_data(
    coin='btc',        # 币种
    interval='1h',     # 时间粒度
    days=730,          # 天数
    save_path='data/historical/btc_1h_730d.csv'
)

# 批量获取多个币种
results = fetcher.get_all_coins_historical(
    coins=['btc', 'eth', 'sol'],
    interval='1h',
    days=730
)
```

### 2. 交易策略 (`src/strategies.py`)

**已实现策略：**

#### 2.1 双均线交叉策略 (MovingAverageCrossStrategy)
- **类型：** 中频趋势跟踪
- **逻辑：** 短均线上穿长均线买入，下穿卖出
- **参数：** short_window=10, long_window=30

#### 2.2 RSI 超买超卖策略 (RSIStrategy)
- **类型：** 中高频均值回归
- **逻辑：** RSI<30 买入，RSI>70 卖出
- **参数：** period=14, oversold=30, overbought=70

#### 2.3 布林带策略 (BollingerBandsStrategy)
- **类型：** 中高频通道突破
- **逻辑：** 价格触及下轨买入，触及上轨卖出
- **参数：** window=20, num_std=2.0

#### 2.4 多因子组合策略 (MultiFactorStrategy) ⭐推荐
- **类型：** 高频综合策略
- **逻辑：** 结合均线趋势 + RSI + 成交量 + 波动率
- **因子权重：**
  - 均线趋势 (30%)
  - RSI 归一化 (30%)
  - 成交量确认 (20%)
  - 波动率过滤 (20%)

#### 2.5 均值回归策略 (MeanReversionStrategy)
- **类型：** 高频统计套利
- **逻辑：** 价格偏离均值过大时反向操作
- **参数：** window=20, entry_z=2.0, exit_z=0.5

#### 2.6 MACD趋势策略 (MACDStrategy)
- **类型：** 中频趋势跟踪
- **逻辑：** MACD线上穿信号线买入，下穿卖出
- **参数：** fast=12, slow=26, signal=9

#### 2.7 突破策略 (BreakoutStrategy)
- **类型：** 高频趋势跟踪
- **逻辑：** 价格突破N周期高点买入，突破N周期低点卖出
- **参数：** window=20, confirmation=True

#### 2.8 VWAP均值回归策略 (VWAPStrategy)
- **类型：** 高频均值回归
- **逻辑：** 价格低于VWAP买入，高于VWAP卖出
- **参数：** window=20, deviation=0.01

#### 2.9 动量策略 (MomentumStrategy)
- **类型：** 中频趋势跟踪
- **逻辑：** 基于价格变化率(ROC)和动量指标
- **参数：** roc_period=10, momentum_period=14, threshold=0.02

#### 2.10 ATR动态止损策略 (ATRStopLossStrategy)
- **类型：** 高频趋势跟踪
- **逻辑：** 基于ATR的动态止损和止盈
- **参数：** atr_period=14, multiplier=2.0, trend_ma=50

#### 2.11 随机指标策略 (StochasticStrategy)
- **类型：** 中高频均值回归
- **逻辑：** K线上穿D线且低于20买入，K线下穿D线且高于80卖出
- **参数：** k_period=14, d_period=3

#### 2.12 网格交易策略 (GridStrategy)
- **类型：** 中高频震荡市策略
- **逻辑：** 在预设价格区间内，每隔一定间距挂单买入和卖出
- **适用市场：** 震荡行情
- **参数：** lower_price, upper_price, grid_num, amount_per_grid
- **风险提示：** 单边行情可能穿网亏损
- **使用：** `python run_backtest.py --strategy grid`

#### 2.13 马丁格尔策略 (MartingaleStrategy) ⚠️ 高风险
- **类型：** 高风险博弈策略
- **逻辑：** 亏损后加倍下注，直到获利后回到初始下注额
- **适用场景：** 极小资金测试，不适合实盘
- **参数：** base_amount, multiplier, max_steps, target_profit, stop_loss
- **风险提示：** 需要充足资金，连续亏损可能爆仓
- **使用：** `python run_backtest.py --strategy martingale`

**使用方式：**
```python
from strategies import get_strategy

# 获取策略
strategy = get_strategy('multi_factor')

# 或自定义参数
strategy = MultiFactorStrategy(
    ma_short=5,
    ma_long=20,
    rsi_period=14,
    volume_threshold=1.5
)

# 生成信号
df_with_signals = strategy.generate_signals(df)
```

### 3. 回测引擎 (`src/backtest.py`)

**功能：** 模拟历史交易，计算收益和风险指标

**特点：**
- ✅ 模拟真实交易环境（手续费 0.1%，滑点 0.1%）
- ✅ 详细的交易记录和决策日志
- ✅ 完整的绩效指标计算
- ✅ 支持仓位管理

**使用方式：**
```python
from backtest import BacktestEngine

engine = BacktestEngine(
    initial_capital=10000.0,  # 初始资金
    commission_rate=0.001,    # 手续费
    slippage=0.001,          # 滑点
    position_size=0.95       # 仓位比例
)

# 运行回测
result = engine.run_backtest(df, strategy, coin='BTC')

# 查看指标
print(result.metrics)
# {
#     'total_return_pct': 156.5,      # 总收益率
#     'annual_return_pct': 78.2,      # 年化收益
#     'sharpe_ratio': 1.85,           # 夏普比率
#     'max_drawdown_pct': -25.3,      # 最大回撤
#     'win_rate_pct': 62.5            # 胜率
# }

# 保存决策日志
result.save_logs('logs/backtest_decisions.json')
```

**决策日志结构：**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "decision": "buy",
  "reason": "Signal: 1, Price: 43250.50, Cash: 5000.00, Position: 0",
  "price": 43250.50,
  "cash": 5000.00,
  "position": 0.1156,
  "total_value": 10000.00,
  "signal": 1
}
```

### 4. 可视化模块 (`src/visualization.py`)

**功能：** 生成专业的收益曲线、交易信号、月度收益热力图和多策略对比

**图表类型：**

| 图表 | 说明 | 文件名 |
|------|------|--------|
| **权益曲线图** | 资金变化、累计收益、回撤分析 | `*_equity.png` |
| **价格信号图** | 价格走势、买卖信号、成交量 | `*_signals.png` |
| **月度收益热力图** | 各月收益表现可视化 | `*_monthly.png` |
| **策略对比-核心指标** | 2×3 布局展示6大核心指标 | `comparison_metrics_*.png` |
| **策略排名** | 按夏普比率排序的水平条形图 | `comparison_ranking_*.png` |
| **权益曲线对比** | Top 5 策略的权益曲线对比 | `comparison_equity_*.png` |

**改进亮点（解决11策略拥挤问题）：**

1. **核心指标对比图** - 使用 2×3 子图布局，每个指标独立展示，避免柱状图拥挤
2. **策略排名图** - 水平条形图展示所有策略排名，更清晰易读
3. **权益曲线对比** - 只展示 Top 5 策略的权益曲线，避免线条重叠

**使用方式：**
```python
from visualization import Visualizer

viz = Visualizer(style='seaborn-v0_8-darkgrid')

# 生成单次回测报告
viz.create_full_report(
    result=result,
    df=df,
    strategy_name='Multi_Factor',
    coin='BTC',
    output_dir='results'
)

# 生成策略对比报告（针对多策略对比优化）
results = {
    'ma_cross': result1,
    'rsi': result2,
    'multi_factor': result3,
    # ... 更多策略
}
viz.create_comparison_report(results, coin='BTC', output_dir='results')

# 单独使用各个图表函数
viz.plot_equity_curve(result, title='My Strategy', save_path='equity.png')
viz.plot_metrics_comparison(results, save_path='metrics.png')
viz.plot_strategy_ranking(results, metric='sharpe_ratio', save_path='ranking.png')
viz.plot_equity_comparison(results, top_n=5, save_path='equity_comp.png')
```

**可视化参数说明：**

```python
Visualizer(
    style='seaborn-v0_8',  # Matplotlib 样式
    fig_dpi=300            # 图表分辨率
)
```

支持的样式：`'seaborn-v0_8'`, `'seaborn-v0_8-darkgrid'`, `'ggplot'`, `'bmh'` 等

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ~/code/crypto_trading
pip install -r requirements.txt
```

### 2. 运行回测

```bash
# 默认回测 BTC 使用多因子策略（2年数据）
python run_backtest.py

# 回测 ETH 使用均线策略
python run_backtest.py --coin eth --strategy ma_cross

# 回测所有币种
python run_backtest.py --coin all

# 对比所有策略性能
python run_backtest.py --coin btc --compare

# 使用1小时数据回测2年
python run_backtest.py --interval 1h --days 730 --capital 50000
```

### 3. 查看结果

回测完成后，会在以下位置生成结果：
- **日志：** `logs/backtest_*.json` - 详细决策记录
- **图表：** `results/*.png` - 可视化图表
- **数据：** `data/historical/*.csv` - 历史数据

## 📊 回测示例

### 运行命令
```bash
python run_backtest.py --coin btc --strategy multi_factor --days 730
```

### 预期输出
```
🚀 开始回测 | 币种: BTC | 策略: multi_factor
============================================================
📥 下载历史数据...
✅ 成功获取 BTCUSDT 1h 数据: 1000 条
   进度: 50.0%
✅ 共获取 17520 条数据
💾 数据已保存至: data/historical/btc_1h_730d.csv

📊 回测结果:
   策略: multi_factor
   总收益率: 156.50%
   年化收益: 78.25%
   夏普比率: 1.85
   最大回撤: -25.30%
   胜率: 62.50%
   交易次数: 245

💾 决策日志已保存: logs/backtest_btc_multi_factor_20240208_121530.json
✅ 完整报告已生成: results/
```

## 📈 策略对比结果示例

```
============================================================
📊 策略对比结果
============================================================

Strategy        Return%      CAGR%        Sharpe    MaxDD%     Win%       Trades
-------------------------------------------------------------------------------------
ma_cross          89.50%       44.75%       1.20      -35.50%    55.20%      128
rsi               45.30%       22.65%       0.85      -28.40%    48.50%      256
bollinger        112.80%       56.40%       1.45      -22.10%    58.30%      189
multi_factor     156.50%       78.25%       1.85      -25.30%    62.50%      245
mean_reversion    67.20%       33.60%       1.05      -30.80%    52.10%      312
macd             134.60%       67.30%       1.65      -28.50%    59.80%      198
breakout         178.20%       89.10%       2.05      -22.40%    64.20%      156
vwap              76.40%       38.20%       1.15      -26.80%    56.50%      289
momentum         145.30%       72.65%       1.75      -24.60%    61.30%      212
atr_stop          98.70%       49.35%       1.35      -31.20%    57.80%      245
stochastic        82.50%       41.25%       1.10      -29.50%    54.60%      267
```

## ⚙️ 配置说明

### 修改策略参数

编辑 `src/strategies.py` 中的策略类：

```python
class MultiFactorStrategy(TradingStrategy):
    def __init__(
        self,
        ma_short: int = 5,           # 短期均线
        ma_long: int = 20,           # 长期均线
        rsi_period: int = 14,        # RSI周期
        volume_threshold: float = 1.5 # 成交量倍数
    ):
        ...
```

### 修改回测参数

编辑 `run_backtest.py` 中的默认参数：

```python
parser.add_argument(
    '--capital',
    type=float,
    default=10000.0,    # 修改初始资金
    help='初始资金'
)
```

## 📝 项目做了什么？

1. **数据层：** 从 Binance 获取高质量的2年历史K线数据
2. **策略层：** 实现了13种经典中高频策略，包含趋势跟踪、均值回归、网格交易和马丁格尔
3. **回测层：** 完整的交易模拟，考虑手续费、滑点、仓位管理
4. **记录层：** 详细记录每一步决策，便于分析和优化
5. **可视化层：** 专业的收益曲线、回撤分析、策略对比

## 🔧 高级用法

### 自定义策略

```python
from strategies import TradingStrategy

class MyStrategy(TradingStrategy):
    def __init__(self):
        super().__init__("My_Custom_Strategy")
    
    def generate_signals(self, df):
        df = df.copy()
        # 你的交易逻辑
        df['signal'] = ...
        return df

# 使用
strategy = MyStrategy()
result = engine.run_backtest(df, strategy, coin='BTC')
```

### 批量优化参数

```python
import itertools

best_result = None
best_params = None

for short, long in itertools.product([5, 10, 15], [20, 30, 50]):
    strategy = MovingAverageCrossStrategy(short, long)
    result = engine.run_backtest(df, strategy, coin='BTC')
    
    if best_result is None or result.metrics['sharpe_ratio'] > best_result.metrics['sharpe_ratio']:
        best_result = result
        best_params = (short, long)

print(f"最优参数: short={best_params[0]}, long={best_params[1]}")
```

## 🧪 Testing & Code Quality

### Run Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_crypto_trading.py -v

# Run only fast tests (exclude slow/integration)
pytest tests/ -v -m "not slow and not integration"
```

### Code Quality Standards

本项目遵循以下代码质量标准：

#### 1. 类型提示 (Type Hints)
所有函数都包含完整的类型注解：
```python
def run_backtest(
    self, 
    df: pd.DataFrame, 
    strategy: object, 
    coin: str = "BTC"
) -> BacktestResult:
    ...
```

#### 2. 文档字符串 (Docstrings)
使用 Google 风格的文档字符串：
```python
def calculate_metrics(self) -> Dict[str, float]:
    """
    计算回测绩效指标
    
    计算的指标包括：
    - total_return_pct: 总收益率 (%)
    - sharpe_ratio: 夏普比率
    - max_drawdown_pct: 最大回撤 (%)
    
    Returns:
        包含各项指标的字典
        
    Example:
        >>> result.calculate_metrics()
        {'total_return_pct': 156.5, 'sharpe_ratio': 1.85}
    """
```

#### 3. 代码格式化 (Black)

```bash
# Install black
pip install black

# Format all code (line length 100)
black src/ tests/ --line-length 100

# Check formatting without changes
black --check src/ tests/ --line-length 100
```

#### 4. 代码结构

```
src/
├── __init__.py
├── backtest.py          # 回测引擎 (BacktestEngine, BacktestResult)
├── strategies.py        # 交易策略集合
├── visualization.py     # 可视化模块
├── data_fetcher.py      # 实时数据获取
├── historical_data.py   # 历史数据获取
├── websocket_client.py  # WebSocket 客户端
└── utils.py            # 工具函数

tests/
├── __init__.py
└── test_crypto_trading.py
    ├── TestDataFetcher      # 数据获取测试
    ├── TestStrategies       # 策略测试
    ├── TestBacktest         # 回测引擎测试
    └── TestIntegration      # 集成测试
```

#### 5. 日志规范

使用分级日志系统：
```python
import logging

logger = logging.getLogger(__name__)

logger.info("🚀 回测引擎初始化")      # 关键流程
logger.debug("买入 @ $50000")         # 详细信息
logger.warning("数据可能缺失")        # 警告
logger.error("无法获取数据")          # 错误
```

## ⚠️ 风险提示

1. **历史回测不等于未来收益** - 过往表现不代表未来
2. **过拟合风险** - 优化参数可能导致过度拟合历史数据
3. **市场变化** - 策略可能在不同市场周期表现差异巨大
4. **实盘差异** - 实际交易还有网络延迟、流动性等问题

## 📚 依赖说明

```
pandas       - 数据处理
numpy        - 数值计算
requests     - API请求
matplotlib   - 可视化
pyyaml       - 配置解析
pytest       - 测试框架
black        - 代码格式化
```

## 🤝 贡献

欢迎提交 PR 添加新策略或优化功能！

## 📄 License

MIT License - 仅供学习和研究使用
