"""
Crypto Trading Data Fetcher - Real-time cryptocurrency price tracking.
Fetches BTC, ETH, SOL trading data from multiple sources.
"""

import requests
import json
import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "crypto_fetcher.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PolymarketPaperTrader:
    """Paper trading bot for Polymarket prediction markets."""

    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    CLOB_API_URL = "https://clob.polymarket.com"

    def __init__(self, starting_cash: float):
        """Initialize the paper trader with starting capital.

        Args:
            starting_cash: Initial cash balance for trading
        """
        self.session = requests.Session()

        # Portfolio State
        self.cash_balance = starting_cash
        self.positions = {
            "yes": {"holdings": 0.0, "average_cost_basis": 0.0, "token_id": None},
            "no": {"holdings": 0.0, "average_cost_basis": 0.0, "token_id": None},
        }

        # Price storage for both entry (buy) and exit (sell) sides
        self.last_known_prices = {
            "yes": {"buy": 0.0, "sell": 0.0},
            "no": {"buy": 0.0, "sell": 0.0},
        }
        self.market_question = ""
        self.market_slug = ""
        self.market_end_time = ""
        self.remaining_time = timedelta(0)

    @property
    def total_account_value(self) -> float:
        """Calculate total account value (cash + market value of positions)."""
        yes_market_value = (
            self.positions["yes"]["holdings"] * self.last_known_prices["yes"]["sell"]
        )
        no_market_value = (
            self.positions["no"]["holdings"] * self.last_known_prices["no"]["sell"]
        )
        return self.cash_balance + yes_market_value + no_market_value

    def get_current_15m_market_slug(self, market: str) -> str:
        """Generate market slug for 15-minute prediction market."""
        now = time.time()
        next_settlement = math.floor(now / 900) * 900
        return f"{market}-updown-15m-{int(next_settlement)}"

    def get_current_hourly_market_slug(self, market: str) -> str:
        """Generate market slug for hourly prediction market."""
        et_tz = ZoneInfo("America/New_York")
        now_et = datetime.now(et_tz)
        current_hour_et = now_et.replace(minute=0, second=0, microsecond=0)
        formatted_hour = current_hour_et.strftime("%I%p").lstrip("0").lower()
        formatted_date = current_hour_et.strftime("%B-%d").lower()
        return f"{market}-up-or-down-{formatted_date}-{formatted_hour}-et".replace(" ", "-")

    def get_current_daily_market_slug(self, market: str) -> str:
        """Generate market slug for daily prediction market."""
        et_tz = ZoneInfo("America/New_York")
        current_time = datetime.now(et_tz)

        if current_time.hour < 12:
            market_date = current_time
        else:
            market_date = current_time + timedelta(days=1)

        month_name = market_date.strftime("%B").lower()
        day = market_date.day
        return f"{market}-up-or-down-on-{month_name}-{day}"

    def load_market_data(self, search_keyword: str, time_choice: str) -> bool:
        """Search for an active market and initialize token identifiers.

        Args:
            search_keyword: Keyword to search for in market questions

        Returns:
            True if market found and loaded successfully, False otherwise
        """
        slug = ""
        if time_choice == "15m":
            slug = self.get_current_15m_market_slug(search_keyword)
        elif time_choice == "hourly":
            slug = self.get_current_hourly_market_slug(search_keyword)
        elif time_choice == "daily":
            slug = self.get_current_daily_market_slug(search_keyword)
        else:
            logger.error(f"Invalid time choice: {time_choice}")
            return False

        logger.info(f"Using market slug: {slug}")

        params = {"limit": 1, "active": "true", "closed": "false", "slug": slug}
        response = self.session.get(f"{self.GAMMA_API_URL}/markets", params=params)
        response.raise_for_status()
        markets = response.json()

        for market in markets:
            self.market_question = market["question"]
            self.market_slug = market["slug"]
            market_end = market.get("endDate", "")
            logger.info(f"Market ends at: {market_end} UTC")
            self.market_end_time = market_end

            token_ids = json.loads(market["clobTokenIds"])
            self.positions["yes"]["token_id"] = token_ids[0]
            self.positions["no"]["token_id"] = token_ids[1]

            logger.info(f"Market loaded: {self.market_question}")
            return True

        logger.error(f"Error loading market: {slug}")
        return False

    def update_prices(self):
        """Update local price cache for both buy and sell sides."""
        for outcome in ["yes", "no"]:
            token_id = self.positions[outcome]["token_id"]
            if not token_id:
                continue

            for side in ["buy", "sell"]:
                try:
                    response = self.session.get(
                        f"{self.CLOB_API_URL}/price",
                        params={"token_id": token_id, "side": side},
                        timeout=2,
                    )
                    if response.status_code == 200:
                        price = float(response.json().get("price", 0))
                        self.last_known_prices[outcome][side] = price
                except Exception:
                    continue

    def calculate_execution_price(
        self, token_id: str, side: str, quantity: float
    ) -> Tuple[float, float]:
        """Simulate order execution against the order book.

        Args:
            token_id: Token identifier for the outcome
            side: 'buy' or 'sell'
            quantity: Number of shares to trade

        Returns:
            Tuple of (average_price, filled_quantity)
        """
        response = self.session.get(
            f"{self.CLOB_API_URL}/book", params={"token_id": token_id}
        )
        order_book = response.json()

        target_side = "asks" if side == "buy" else "bids"
        orders = order_book.get(target_side, [])
        orders.sort(key=lambda x: float(x["price"]), reverse=(side == "sell"))

        filled_quantity = 0.0
        total_cost = 0.0

        for order in orders:
            if filled_quantity >= quantity:
                break

            price = float(order["price"])
            available = float(order["size"])

            take_amount = min(available, quantity - filled_quantity)
            filled_quantity += take_amount
            total_cost += take_amount * price

        if filled_quantity > 0:
            average_price = total_cost / filled_quantity
            return average_price, filled_quantity

        logger.error("Order book matching error")
        return 0.0, 0.0

    def execute_trade(self, outcome: str, side: str, quantity: float) -> bool:
        """Execute a trade and update account state.

        Args:
            outcome: 'yes' or 'no'
            side: 'buy' or 'sell'
            quantity: Number of shares to trade
        """
        outcome = outcome.lower()
        side = side.lower()
        token_id = self.positions[outcome]["token_id"]

        avg_price, filled = self.calculate_execution_price(token_id, side, quantity)

        if filled <= 0:
            logger.warning(f"Trade failed: No liquidity for {outcome.upper()}")
            return False

        if side == "buy":
            total_cost = filled * avg_price
            if total_cost > self.cash_balance:
                logger.error(
                    f"Insufficient funds: Need ${total_cost:.2f}, have ${self.cash_balance:.2f}"
                )
                return False

            current_holdings = self.positions[outcome]["holdings"]
            current_avg = self.positions[outcome]["average_cost_basis"]

            new_holdings = current_holdings + filled
            self.positions[outcome]["average_cost_basis"] = (
                (current_holdings * current_avg) + total_cost
            ) / new_holdings

            self.positions[outcome]["holdings"] = new_holdings
            self.cash_balance -= total_cost

        elif side == "sell":
            if self.positions[outcome]["holdings"] < filled:
                logger.error(f"Insufficient shares: Cannot sell {filled} shares")
                return False

            revenue = filled * avg_price
            self.cash_balance += revenue
            self.positions[outcome]["holdings"] -= filled

            if self.positions[outcome]["holdings"] == 0:
                self.positions[outcome]["average_cost_basis"] = 0.0

        self.update_prices()
        logger.warning(
            f"\n✅ SUCCESS: {side.upper()} {filled:.2f} {outcome.upper()} @ ${avg_price:.4f}"
        )
        return True

    def clear_holdings_when_closed(
        self, outcome: str, side: str, quantity: float
    ) -> bool:
        """Clear positions if market is closed."""
        self.update_prices()
        assert outcome in ["yes", "no"], "Invalid outcome"
        assert side == "sell", "Can only clear holdings by selling"
        assert quantity > 0, "Quantity must be positive"
        assert (
            self.positions[outcome]["holdings"] >= quantity
        ), "Insufficient holdings to clear"

        revenue = quantity * self.last_known_prices[outcome]["sell"]
        self.cash_balance += revenue
        self.positions[outcome]["holdings"] -= quantity
        if self.positions[outcome]["holdings"] == 0:
            self.positions[outcome]["average_cost_basis"] = 0.0

        logger.warning(
            f"\n✅ SUCCESS: CLEARED {quantity:.2f} {outcome.upper()} @ ${self.last_known_prices[outcome]['sell']:.4f} DUE TO MARKET CLOSURE"
        )
        return True

    def display_dashboard(self):
        """Display current account status in terminal."""
        print("=" * 70)
        print("📊 POLYMARKET PAPER TRADING DASHBOARD")
        print(f"Market: {self.market_question}")
        print(
            f"Time:   {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"Slug:   {self.market_slug}")
        print("-" * 70)
        print(f"Total Account Value:        ${self.total_account_value:,.2f}")
        print(f"Available Cash:             ${self.cash_balance:,.2f}")
        print("-" * 70)

        for outcome in ["yes", "no"]:
            data = self.positions[outcome]
            buy_price = self.last_known_prices[outcome]["buy"]
            sell_price = self.last_known_prices[outcome]["sell"]
            pnl = (sell_price - data["average_cost_basis"]) * data["holdings"]

            print(f"OUTCOME: {outcome.upper()}")
            print(f"  Current Buy Price:        ${buy_price:.4f}")
            print(f"  Current Sell Price:       ${sell_price:.4f}")
            print(f"  Shares Owned:             {data['holdings']:.2f}")
            print(f"  Average Cost Basis:       ${data['average_cost_basis']:.4f}")
            print(f"  Unrealized P&L:           ${pnl:+.2f}")
            print("-" * 40)

        print("Updates every 1 second | Press Ctrl+C to stop")
        print("=" * 70)

    def strategy_simple_momentum(self):
        """Buy below 0.45, sell above 0.50 for both outcomes."""
        yes_buy = self.last_known_prices["yes"]["buy"]
        if 0 < yes_buy < 0.45 and self.positions["yes"]["holdings"] == 0:
            logger.info("Strategy: YES price below $0.45, buying")
            self.execute_trade(outcome="yes", side="buy", quantity=100.0)

        no_buy = self.last_known_prices["no"]["buy"]
        if 0 < no_buy < 0.45 and self.positions["no"]["holdings"] == 0:
            logger.info("Strategy: NO price below $0.45, buying")
            self.execute_trade(outcome="no", side="buy", quantity=100.0)

        yes_sell = self.last_known_prices["yes"]["sell"]
        if yes_sell > 0.50 and self.positions["yes"]["holdings"] > 0:
            logger.info("Strategy: YES price above $0.50, selling")
            self.execute_trade(
                outcome="yes",
                side="sell",
                quantity=self.positions["yes"]["holdings"],
            )

        no_sell = self.last_known_prices["no"]["sell"]
        if no_sell > 0.50 and self.positions["no"]["holdings"] > 0:
            logger.info("Strategy: NO price above $0.50, selling")
            self.execute_trade(
                outcome="no",
                side="sell",
                quantity=self.positions["no"]["holdings"],
            )

    def strategy_arbitrage(self):
        """Exploit arbitrage when probabilities don't sum to 1."""
        yes_buy = self.last_known_prices["yes"]["buy"]
        no_buy = self.last_known_prices["no"]["buy"]
        yes_sell = self.last_known_prices["yes"]["sell"]
        no_sell = self.last_known_prices["no"]["sell"]

        if (
            (yes_buy + no_buy) < 1.0
            and self.positions["yes"]["holdings"] == 0
            and self.positions["no"]["holdings"] == 0
        ):
            logger.info(f"Arbitrage: Buy prices sum to {yes_buy + no_buy:.4f}, buying both")
            self.execute_trade(outcome="yes", side="buy", quantity=100.0 * yes_buy)
            self.execute_trade(outcome="no", side="buy", quantity=100.0 * no_buy)

        if (
            (yes_sell + no_sell) > 1.0
            and self.positions["yes"]["holdings"] > 0
            and self.positions["no"]["holdings"] > 0
        ):
            logger.info(
                f"Arbitrage: Sell prices sum to {yes_sell + no_sell:.4f}, selling both"
            )
            self.execute_trade(
                outcome="yes",
                side="sell",
                quantity=self.positions["yes"]["holdings"],
            )
            self.execute_trade(
                outcome="no",
                side="sell",
                quantity=self.positions["no"]["holdings"],
            )

    def half_strategy(self):
        """Buy price less than $0.5 outcome."""
        yes_buy = self.last_known_prices["yes"]["buy"]
        no_buy = self.last_known_prices["no"]["buy"]
        if yes_buy < 0.5 and self.positions["yes"]["holdings"] == 0:
            logger.info("Strategy: YES price below $0.5, buying")
            self.execute_trade(outcome="yes", side="buy", quantity=100.0)
        if no_buy < 0.5 and self.positions["no"]["holdings"] == 0:
            logger.info("Strategy: NO price below $0.5, buying")
            self.execute_trade(outcome="no", side="buy", quantity=100.0)

    def strategy_at_close(self):
        """Buy price greater than $0.8 outcome if market is about to close."""
        if self.market_end_time:
            self.remaining_time = (
                datetime.fromisoformat(self.market_end_time.replace("Z", "+00:00"))
                - datetime.now(timezone.utc)
            )
        else:
            self.remaining_time = None
            print("Remaining Time: N/A (Market end time not set)")
            assert False, "Market end time not set"

        if (
            self.remaining_time <= timedelta(minutes=5)
            and self.positions["yes"]["holdings"] + self.positions["no"]["holdings"] == 0
        ):
            yes_buy = self.last_known_prices["yes"]["buy"]
            no_buy = self.last_known_prices["no"]["buy"]
            if yes_buy > 0.8 and self.positions["yes"]["holdings"] == 0:
                logger.info("Strategy: YES price above $0.8, buying")
                self.execute_trade(outcome="yes", side="buy", quantity=100.0)
            if no_buy > 0.8 and self.positions["no"]["holdings"] == 0:
                logger.info("Strategy: NO price above $0.8, buying")
                self.execute_trade(outcome="no", side="buy", quantity=100.0)

    def strategy_at_close_balance(self):
        """Buy price greater than $0.9 outcome if market is about to close."""
        if self.market_end_time:
            self.remaining_time = (
                datetime.fromisoformat(self.market_end_time.replace("Z", "+00:00"))
                - datetime.now(timezone.utc)
            )
        else:
            self.remaining_time = None
            print("Remaining Time: N/A (Market end time not set)")
            assert False, "Market end time not set"

        if (
            self.remaining_time <= timedelta(minutes=5)
            and self.positions["yes"]["holdings"] + self.positions["no"]["holdings"] == 0
        ):
            yes_buy = self.last_known_prices["yes"]["buy"]
            no_buy = self.last_known_prices["no"]["buy"]

            if yes_buy > 0.9 and self.positions["yes"]["holdings"] == 0:
                logger.info("Strategy: YES price above $0.9, buying")
                self.execute_trade(outcome="yes", side="buy", quantity=100.0)
                logger.info("Strategy: NO price below $0.1, buying")
                self.execute_trade(outcome="no", side="buy", quantity=5.0)
            if no_buy > 0.9 and self.positions["no"]["holdings"] == 0:
                logger.info("Strategy: NO price above $0.9, buying")
                self.execute_trade(outcome="no", side="buy", quantity=100.0)
                logger.info("Strategy: YES price below $0.1, buying")
                self.execute_trade(outcome="yes", side="buy", quantity=5.0)


def main():
    """Main entry point for the paper trading bot."""
    parser = argparse.ArgumentParser(description="Polymarket Paper Trading Bot")
    parser.add_argument(
        "--cash", type=float, default=1000.0, help="Starting cash balance"
    )
    parser.add_argument(
        "--keyword", type=str, default="bitcoin", help="Market search keyword"
    )
    parser.add_argument(
        "--time-choice",
        type=str,
        default="15m",
        choices=["15m", "hourly", "daily"],
        help="Timeframe for market prediction",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="at_close",
        choices=["arbitrage", "half_strategy", "momentum", "at_close", "at_close_balance"],
        help="Trading strategy to use",
    )
    args = parser.parse_args()

    trader = PolymarketPaperTrader(args.cash)

    if not trader.load_market_data(args.keyword, args.time_choice):
        logger.error("No matching market found")
        return

    if args.strategy == "at_close":
        strategy_func = trader.strategy_at_close
    elif args.strategy == "at_close_balance":
        strategy_func = trader.strategy_at_close_balance
    elif args.strategy == "arbitrage":
        strategy_func = trader.strategy_arbitrage
    elif args.strategy == "half_strategy":
        strategy_func = trader.half_strategy
    else:
        strategy_func = trader.strategy_simple_momentum

    logger.info(f"Starting paper trading with {args.strategy} strategy")

    try:
        while True:
            current_slug = ""
            if args.time_choice == "15m":
                current_slug = trader.get_current_15m_market_slug(args.keyword)
            elif args.time_choice == "hourly":
                current_slug = trader.get_current_hourly_market_slug(args.keyword)
            elif args.time_choice == "daily":
                current_slug = trader.get_current_daily_market_slug(args.keyword)

            if current_slug != trader.market_slug:
                logger.info("Market period changed, reloading market data...")
                for outcome in ["yes", "no"]:
                    if trader.positions[outcome]["holdings"] > 0:
                        logger.info(f"Closing {outcome.upper()} position before market switch")
                        assert trader.clear_holdings_when_closed(
                            outcome=outcome,
                            side="sell",
                            quantity=trader.positions[outcome]["holdings"],
                        )
                        logger.warning(
                            f"Total Account Value after closing: ${trader.total_account_value:,.2f}"
                        )
                        logger.warning(
                            f"Available Cash after closing: ${trader.cash_balance:,.2f}"
                        )
                if not trader.load_market_data(args.keyword, args.time_choice):
                    logger.error("Failed to load new market, continuing with current market")
                else:
                    logger.info("Successfully switched to new market period")

            trader.update_prices()
            strategy_func()
            trader.display_dashboard()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n✋ Trading stopped by user")
        print(f"Final Account Value: ${trader.total_account_value:,.2f}")


if __name__ == "__main__":
    main()
