"""
Visualization Parameter Constants

Centralized definition of all visualization-related default parameter values.
"""

# ============================================================
# Chart Base Settings
# ============================================================
DEFAULT_FIG_DPI = 300                   # Chart save resolution
DEFAULT_STYLE = "seaborn-v0_8"         # Default matplotlib style

# ============================================================
# Chart Sizes
# ============================================================
DEFAULT_EQUITY_FIGSIZE = (14, 10)       # Equity curve chart size
DEFAULT_PRICE_FIGSIZE = (14, 8)         # Price signal chart size
DEFAULT_MONTHLY_FIGSIZE = (12, 7)       # Monthly returns chart size
DEFAULT_COMPARISON_FIGSIZE = (18, 10)   # Strategy comparison chart size
DEFAULT_RANKING_FIGSIZE_BASE = (10, 6)  # Ranking chart base size
DEFAULT_EQUITY_COMPARISON_FIGSIZE = (14, 8)  # Equity comparison chart size

# ============================================================
# Subplot Ratios
# ============================================================
PRICE_HEIGHT_RATIO = 3                  # Price subplot to volume subplot height ratio

# ============================================================
# Monthly Returns Heatmap
# ============================================================
MONTHLY_RETURN_VMIN = -20               # Heatmap color minimum value (%)
MONTHLY_RETURN_VMAX = 20                # Heatmap color maximum value (%)
COLOR_TEXT_THRESHOLD = 10               # Text color switch threshold (white/black)

# ============================================================
# Strategy Comparison
# ============================================================
DEFAULT_TOP_N_COMPARISON = 6            # Default number of strategies shown in comparison chart
DEFAULT_TOP_N_EQUITY = 5                # Default number of strategies shown in equity comparison
