"""Phase 2 backtest harness integrating RL, Transformer, and ensemble.

This module runs backtests that use:
- Phase 1 (Ensemble + Kelly + Multi-timeframe)
- Phase 2 (PPO trader + Transformer trader + Meta-learner)

It reports improvements and metrics for comparison.
"""
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    final_portfolio_value: float
    total_pnl: float
    phase: str  # "Phase 1" or "Phase 2"
    
    def __str__(self):
        return f"""
╔════════════════════════════════════════════╗
║ {self.phase} BACKTEST RESULTS                ║
╠════════════════════════════════════════════╣
║ Total Trades:         {self.total_trades:>6}           ║
║ Win Rate:             {self.win_rate:>6.1%}           ║
║ Avg Win / Loss:       ${self.avg_win:>7.2f} / ${self.avg_loss:>7.2f}  ║
║ Profit Factor:        {self.profit_factor:>6.2f}            ║
║ Total Return:         {self.total_return:>6.1%}           ║
║ Sharpe Ratio:         {self.sharpe_ratio:>6.2f}            ║
║ Max Drawdown:         {self.max_drawdown:>6.1%}           ║
║ Final Portfolio:      ${self.final_portfolio_value:>11,.0f}    ║
║ Total PnL:            ${self.total_pnl:>11,.0f}    ║
╚════════════════════════════════════════════╝
"""


class Phase2Backtester:
    """Simulates Phase 2 trading using PPO, Transformer, and ensemble models."""
    
    def __init__(self, ohlcv: np.ndarray, initial_capital: float = 10000,
                 phase: str = "Phase 2", use_rl_model: bool = True,
                 use_transformer: bool = True, use_metalearner: bool = True):
        """
        Args:
            ohlcv: Historical OHLCV data
            initial_capital: Starting portfolio value
            phase: "Phase 1" or "Phase 2"
            use_rl_model: Use PPO trader
            use_transformer: Use Transformer trader
            use_metalearner: Use MetaLearner for allocation
        """
        self.ohlcv = ohlcv
        self.initial_capital = initial_capital
        self.phase = phase
        self.use_rl_model = use_rl_model
        self.use_transformer = use_transformer
        self.use_metalearner = use_metalearner
        
        # Backtest state
        self.current_step = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.trades = []
        self.max_portfolio = initial_capital
    
    def run(self) -> BacktestMetrics:
        """Run the full backtest with stop-loss, take-profit, and trailing stops."""
        logger.info(f"Starting {self.phase} backtest with {len(self.ohlcv)} bars")

        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.best_price = 0.0

        for step in range(len(self.ohlcv) - 1):
            self.current_step = step
            price = self.ohlcv[step, 3]

            # Check exit conditions for open positions
            if self.position > 0:
                # Update trailing stop
                if price > self.best_price:
                    self.best_price = price
                    closes = self.ohlcv[max(0, step - 14):step + 1, 3]
                    atr = np.mean(np.abs(np.diff(closes))) if len(closes) > 1 else price * 0.02
                    new_trail = price - 1.2 * atr
                    self.trailing_stop = max(self.trailing_stop, new_trail)

                # Check stops
                if price <= self.stop_loss:
                    self._close_position(step)
                elif price >= self.take_profit:
                    self._close_position(step)
                elif self.trailing_stop > self.stop_loss and price <= self.trailing_stop:
                    self._close_position(step)

            # Generate signals for new entries
            if self.position == 0:
                signal = self._generate_signal(step)
                if signal == "BUY":
                    self._open_position(step)
            elif self.position > 0:
                signal = self._generate_signal(step)
                if signal == "SELL":
                    self._close_position(step)

            self._update_portfolio(step)

        if self.position > 0:
            self._close_position(len(self.ohlcv) - 1)

        return self._calculate_metrics()
    
    def _generate_signal(self, step: int) -> str:
        """Generate trading signal using Phase 1 + Phase 2 models."""
        phase1_signal = self._phase1_signal(step)

        if self.phase == "Phase 1":
            return phase1_signal

        phase2_signal = self._phase2_signal(step)

        # Phase 2 is more aggressive: either signal triggers, with preference for agreement
        if phase1_signal == phase2_signal:
            return phase1_signal
        elif phase2_signal != "HOLD":
            return phase2_signal  # Phase 2 ML models can trade independently
        else:
            return phase1_signal
    
    def _phase1_signal(self, step: int) -> str:
        """Phase 1 signal: RSI + MACD + trend with responsive thresholds."""
        if step < 30:
            return "HOLD"

        closes = self.ohlcv[max(0, step - 14):step + 1, 3]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        if losses == 0:
            rsi = 100 if gains > 0 else 50
        else:
            rsi = 100 - (100 / (1 + gains / losses))

        ema_short = self.ohlcv[step - 8:step + 1, 3].mean()
        ema_long = self.ohlcv[max(0, step - 21):step + 1, 3].mean()
        trend = "UP" if ema_short > ema_long else "DOWN"

        momentum = (closes[-1] - closes[0]) / closes[0] if len(closes) > 1 else 0

        # More responsive entry criteria
        if rsi < 35 and trend == "UP":
            return "BUY"
        elif rsi < 40 and momentum > 0.01 and trend == "UP":
            return "BUY"
        elif rsi > 65 and trend == "DOWN":
            return "SELL"
        elif rsi > 60 and momentum < -0.01 and trend == "DOWN":
            return "SELL"
        else:
            return "HOLD"
    
    def _phase2_signal(self, step: int) -> str:
        """Phase 2 signal: RL + Transformer + MetaLearner with better thresholds."""
        if step < 50:
            return "HOLD"

        rl_signal = self._mock_rl_prediction(step)
        transformer_signal = self._mock_transformer_prediction(step)
        mean_rev_signal = self._mock_mean_reversion(step)
        confidence = self._mock_metalearner_confidence(step)

        buy_votes = sum(1 for s in [rl_signal, transformer_signal, mean_rev_signal] if s == "BUY")
        sell_votes = sum(1 for s in [rl_signal, transformer_signal, mean_rev_signal] if s == "SELL")

        if buy_votes >= 2 and confidence > 0.5:
            return "BUY"
        elif sell_votes >= 2 and confidence > 0.5:
            return "SELL"
        elif buy_votes >= 1 and confidence > 0.7:
            return "BUY"
        elif sell_votes >= 1 and confidence > 0.7:
            return "SELL"
        else:
            return "HOLD"

    def _mock_rl_prediction(self, step: int) -> str:
        """RL model: buys dips in uptrends, sells rallies in downtrends."""
        closes = self.ohlcv[max(0, step - 20):step + 1, 3]
        returns = np.diff(closes) / closes[:-1]
        recent_return = returns[-1] if len(returns) > 0 else 0
        trend = np.mean(returns) if len(returns) > 1 else 0

        if trend > 0 and recent_return < -0.005:
            return "BUY"  # Buy the dip in uptrend
        elif trend < 0 and recent_return > 0.005:
            return "SELL"  # Sell the rally in downtrend
        elif trend > 0.002:
            return "BUY"
        elif trend < -0.002:
            return "SELL"
        else:
            return "HOLD"

    def _mock_transformer_prediction(self, step: int) -> str:
        """Transformer: multi-scale momentum with breakout detection."""
        closes = self.ohlcv[max(0, step - 50):step + 1, 3]
        short_mom = (closes[-1] - closes[-min(10, len(closes))]) / closes[-min(10, len(closes))]
        long_mom = (closes[-1] - closes[0]) / closes[0]

        # Breakout detection: price near 20-period high/low
        recent = closes[-min(20, len(closes)):]
        near_high = closes[-1] >= np.percentile(recent, 90)
        near_low = closes[-1] <= np.percentile(recent, 10)

        if near_high and short_mom > 0.005:
            return "BUY"  # Breakout
        elif near_low and short_mom < -0.005:
            return "SELL"
        elif short_mom > 0.01 and long_mom > 0:
            return "BUY"
        elif short_mom < -0.01 and long_mom < 0:
            return "SELL"
        else:
            return "HOLD"

    def _mock_mean_reversion(self, step: int) -> str:
        """Mean-reversion model for oversold bounces."""
        closes = self.ohlcv[max(0, step - 14):step + 1, 3]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        rsi = 100 - (100 / (1 + gains / (losses + 1e-6)))

        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        else:
            return "HOLD"

    def _mock_metalearner_confidence(self, step: int) -> float:
        """MetaLearner confidence with trend-regime awareness."""
        if step < 30:
            return 0.5

        closes = self.ohlcv[max(0, step - 14):step + 1, 3]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        rsi = 100 - (100 / (1 + gains / (losses + 1e-6)))

        # Trend strength via ADX approximation
        ema_s = self.ohlcv[step - 8:step + 1, 3].mean()
        ema_l = self.ohlcv[max(0, step - 21):step + 1, 3].mean()
        trend_strength = abs(ema_s - ema_l) / ema_l

        if rsi < 30 or rsi > 70:
            return 0.9
        elif trend_strength > 0.02:
            return 0.75
        elif rsi < 40 or rsi > 60:
            return 0.65
        else:
            return 0.5
    
    def _open_position(self, step: int):
        """Open a position with volatility-scaled sizing and stop-loss."""
        price = self.ohlcv[step, 3]
        if self.cash <= 0 or price <= 0:
            return

        closes = self.ohlcv[max(0, step - 14):step + 1, 3]
        atr = np.mean(np.abs(np.diff(closes))) if len(closes) > 1 else price * 0.02

        risk_per_share = max(1.5 * atr, price * 0.005)
        risk_budget = self.cash * 0.02

        position_size = min(
            risk_budget / risk_per_share,
            (self.cash * 0.30) / price  # Max 30% of cash per trade
        )

        if position_size * price < 10:
            return

        fee = position_size * price * 0.001
        self.cash -= (position_size * price + fee)
        self.position = position_size
        self.entry_price = price
        self.stop_loss = price - 1.5 * atr
        self.take_profit = price + 3.0 * atr
        self.trailing_stop = self.stop_loss
        self.best_price = price

    def _close_position(self, step: int):
        """Close position and record trade."""
        if self.position == 0:
            return

        price = self.ohlcv[step, 3]
        trade_value = self.position * price
        pnl = self.position * (price - self.entry_price)
        fee = trade_value * 0.001
        net_pnl = pnl - fee
        entry_value = self.position * self.entry_price

        self.cash += trade_value - fee
        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": price,
            "position_size": self.position,
            "pnl": net_pnl,
            "return": net_pnl / entry_value if entry_value > 0 else 0
        })

        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.best_price = 0.0
    
    def _update_portfolio(self, step: int):
        """Update portfolio value."""
        price = self.ohlcv[step, 3]
        position_value = self.position * price if self.position > 0 else 0
        self.portfolio_value = self.cash + position_value
        self.max_portfolio = max(self.max_portfolio, self.portfolio_value)
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate performance metrics."""
        if not self.trades:
            total_trades = 0
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        else:
            total_trades = len(self.trades)
            pnls = [t["pnl"] for t in self.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            profit_factor = sum(wins) / (-sum(losses) + 1e-6) if losses else 0.0
        
        total_pnl = self.portfolio_value - self.initial_capital
        total_return = total_pnl / self.initial_capital
        max_drawdown = (self.max_portfolio - self.portfolio_value) / self.max_portfolio
        
        # Sharpe ratio (simplified: daily returns)
        if self.trades:
            returns = np.array([t["return"] for t in self.trades])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return BacktestMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=abs(avg_loss),
            profit_factor=profit_factor,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            final_portfolio_value=self.portfolio_value,
            total_pnl=total_pnl,
            phase=self.phase
        )


def compare_phases(ohlcv: np.ndarray, initial_capital: float = 10000) -> None:
    """Run Phase 1 vs Phase 2 backtests and show comparison."""
    
    print("\n" + "="*50)
    print("PHASE 1 vs PHASE 2 BACKTEST COMPARISON")
    print("="*50)
    
    # Phase 1 backtest
    bt1 = Phase2Backtester(ohlcv, initial_capital, phase="Phase 1", 
                            use_rl_model=False, use_transformer=False)
    metrics1 = bt1.run()
    print(metrics1)
    
    # Phase 2 backtest
    bt2 = Phase2Backtester(ohlcv, initial_capital, phase="Phase 2",
                            use_rl_model=True, use_transformer=True)
    metrics2 = bt2.run()
    print(metrics2)
    
    # Improvements
    print("\n╔════════════════════════════════════════════╗")
    print("║ IMPROVEMENTS (Phase 2 vs Phase 1)          ║")
    print("╠════════════════════════════════════════════╣")
    print(f"║ Win Rate:          {metrics2.win_rate - metrics1.win_rate:+6.1%}            ║")
    print(f"║ Profit Factor:     {metrics2.profit_factor - metrics1.profit_factor:+6.2f}            ║")
    print(f"║ Total Return:      {metrics2.total_return - metrics1.total_return:+6.1%}            ║")
    print(f"║ Sharpe Ratio:      {metrics2.sharpe_ratio - metrics1.sharpe_ratio:+6.2f}            ║")
    print(f"║ Final Portfolio:   ${metrics2.final_portfolio_value - metrics1.final_portfolio_value:+11,.0f}    ║")
    print("╚════════════════════════════════════════════╝\n")
    
    return metrics1, metrics2


if __name__ == "__main__":
    # Generate sample data
    from data.data_connector import DataConnector
    
    dc = DataConnector()
    ohlcv = dc.load_ohlcv("BTC/USDT", timeframe="1h",
                          start_date="2023-06-01", end_date="2023-12-31")
    
    metrics1, metrics2 = compare_phases(ohlcv, initial_capital=10000)
    
    # Save results
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "backtest_comparison.json", "w") as f:
        json.dump({
            "phase_1": asdict(metrics1),
            "phase_2": asdict(metrics2)
        }, f, indent=2)
    
    print(f"Results saved to {results_dir / 'backtest_comparison.json'}")
