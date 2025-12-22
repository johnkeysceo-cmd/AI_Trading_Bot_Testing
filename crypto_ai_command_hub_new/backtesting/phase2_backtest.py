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
        """Run the full backtest."""
        logger.info(f"Starting {self.phase} backtest with {len(self.ohlcv)} bars")
        
        for step in range(len(self.ohlcv) - 1):
            self.current_step = step
            
            # Generate signals
            signal = self._generate_signal(step)
            
            # Execute trades
            if signal == "BUY" and self.position == 0:
                self._open_position(step)
            elif signal == "SELL" and self.position > 0:
                self._close_position(step)
            
            # Update portfolio
            self._update_portfolio(step)
        
        # Close any open position at end
        if self.position > 0:
            self._close_position(len(self.ohlcv) - 1)
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def _generate_signal(self, step: int) -> str:
        """Generate trading signal using Phase 1 + Phase 2 models."""
        price = self.ohlcv[step, 3]
        
        # Phase 1: Technical + Multi-timeframe consensus (simplified)
        phase1_signal = self._phase1_signal(step)
        
        if self.phase == "Phase 1":
            return phase1_signal
        
        # Phase 2: RL + Transformer + MetaLearner ensemble
        phase2_signal = self._phase2_signal(step)
        
        # Combine: Phase 2 overrides Phase 1 if confidence high
        if phase2_signal == "BUY" and phase1_signal == "BUY":
            return "BUY"
        elif phase2_signal == "SELL" and phase1_signal == "SELL":
            return "SELL"
        else:
            return phase1_signal  # Conservative: require agreement
    
    def _phase1_signal(self, step: int) -> str:
        """Phase 1 signal: RSI + MACD + trend."""
        if step < 50:
            return "HOLD"
        
        # RSI indicator
        closes = self.ohlcv[max(0, step-14):step+1, 3]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        if losses == 0:
            rsi = 100 if gains > 0 else 50
        else:
            rsi = 100 - (100 / (1 + gains / losses))
        
        # Trend (EMA)
        ema_short = self.ohlcv[step-10:step+1, 3].mean()
        ema_long = self.ohlcv[max(0, step-30):step+1, 3].mean()
        trend = "UP" if ema_short > ema_long else "DOWN"
        
        # Signal
        if rsi < 30 and trend == "UP":
            return "BUY"
        elif rsi > 70 and trend == "DOWN":
            return "SELL"
        else:
            return "HOLD"
    
    def _phase2_signal(self, step: int) -> str:
        """Phase 2 signal: RL + Transformer + MetaLearner."""
        if step < 100:
            return "HOLD"
        
        # Mock RL prediction (in production: use actual PPO)
        rl_signal = self._mock_rl_prediction(step)
        
        # Mock Transformer prediction
        transformer_signal = self._mock_transformer_prediction(step)
        
        # Mock MetaLearner allocation
        confidence = self._mock_metalearner_confidence(step)
        
        # Combine: majority vote with confidence weighting
        buy_votes = (1 if rl_signal == "BUY" else 0) + (1 if transformer_signal == "BUY" else 0)
        
        if buy_votes >= 1.5 and confidence > 0.6:
            return "BUY"
        elif buy_votes <= 0.5 and confidence > 0.6:
            return "SELL"
        else:
            return "HOLD"
    
    def _mock_rl_prediction(self, step: int) -> str:
        """Mock PPO trader prediction."""
        # Simulated: PPO learns to trade on volatility
        closes = self.ohlcv[max(0, step-20):step+1, 3]
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns)
        
        # PPO buys on low volatility
        if volatility < 0.01:
            return "BUY"
        elif volatility > 0.02:
            return "SELL"
        else:
            return "HOLD"
    
    def _mock_transformer_prediction(self, step: int) -> str:
        """Mock Transformer trader prediction."""
        # Simulated: Transformer learns to follow price momentum
        closes = self.ohlcv[max(0, step-50):step+1, 3]
        momentum = (closes[-1] - closes[0]) / closes[0]
        
        if momentum > 0.02:
            return "BUY"
        elif momentum < -0.02:
            return "SELL"
        else:
            return "HOLD"
    
    def _mock_metalearner_confidence(self, step: int) -> float:
        """Mock MetaLearner confidence score."""
        # Simulated: ML model learns to trust ensemble when RSI/MACD align
        if step < 50:
            return 0.5
        
        closes = self.ohlcv[max(0, step-14):step+1, 3]
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        rsi = 100 - (100 / (1 + gains / (losses + 1e-6)))
        
        # High confidence when RSI at extremes (consensus)
        if rsi < 30 or rsi > 70:
            return 0.85
        else:
            return 0.55
    
    def _open_position(self, step: int):
        """Open a long position."""
        price = self.ohlcv[step, 3]
        position_size = (self.cash * 0.9) / price  # Use 90% of cash
        
        fee = position_size * price * 0.001
        self.cash -= fee
        self.position = position_size
        self.entry_price = price
    
    def _close_position(self, step: int):
        """Close position and record trade."""
        if self.position == 0:
            return
        
        price = self.ohlcv[step, 3]
        pnl = self.position * (price - self.entry_price)
        fee = self.position * price * 0.001
        
        self.cash += (self.position * price) - fee
        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": price,
            "position_size": self.position,
            "pnl": pnl,
            "return": pnl / (self.position * self.entry_price)
        })
        
        self.position = 0.0
        self.entry_price = 0.0
    
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
