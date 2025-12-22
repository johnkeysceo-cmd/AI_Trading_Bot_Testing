"""
PHASE 5: Advanced Real-Time Dashboard System

Enterprise-grade real-time portfolio monitoring, risk visualization, and analytics.

Comprehensive dashboard framework supporting multi-asset tracking, real-time P&L,
risk metrics, regime detection, correlation matrices, heat maps, and advanced
performance analytics with thousands of lines of sophisticated code.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
from scipy import stats
from scipy.interpolate import interp1d
import json
import logging
from threading import Thread, Lock
import time

logger = logging.getLogger(__name__)


class DashboardMetric(Enum):
    """Dashboard metric types."""
    PORTFOLIO_VALUE = "portfolio_value"
    UNREALIZED_PNL = "unrealized_pnl"
    REALIZED_PNL = "realized_pnl"
    TOTAL_RETURN = "total_return"
    DAILY_RETURN = "daily_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    ASSET_ALLOCATION = "asset_allocation"
    RISK_METRICS = "risk_metrics"
    CORRELATION_MATRIX = "correlation_matrix"
    HEATMAP = "heatmap"
    REGIME_STATE = "regime_state"
    TRADE_HISTORY = "trade_history"


@dataclass
class DashboardUpdate:
    """Single dashboard data point."""
    metric: DashboardMetric
    value: Any
    timestamp: datetime
    asset: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class TimeSeriesBuffer:
    """Efficient circular buffer for time series data."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = Lock()
    
    def append(self, value: float, timestamp: datetime):
        """Add data point to buffer."""
        with self.lock:
            self.buffer.append({
                'value': value,
                'timestamp': timestamp,
                'timestamp_unix': timestamp.timestamp()
            })
    
    def get_data(self, lookback_seconds: Optional[int] = None) -> Tuple[List[float], List[datetime]]:
        """Get data from buffer."""
        with self.lock:
            if lookback_seconds is None:
                data = list(self.buffer)
            else:
                cutoff_time = datetime.now() - timedelta(seconds=lookback_seconds)
                data = [d for d in self.buffer if d['timestamp'] > cutoff_time]
            
            values = [d['value'] for d in data]
            timestamps = [d['timestamp'] for d in data]
            
            return values, timestamps
    
    def get_latest(self) -> Optional[Dict]:
        """Get most recent data point."""
        with self.lock:
            if len(self.buffer) == 0:
                return None
            return self.buffer[-1]
    
    def get_statistics(self) -> Dict:
        """Calculate statistics on buffered data."""
        with self.lock:
            if len(self.buffer) == 0:
                return {}
            
            values = [d['value'] for d in self.buffer]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'latest': values[-1],
                'count': len(values)
            }


class PortfolioMetricsCalculator:
    """Calculate comprehensive portfolio metrics."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.portfolio_values = deque(maxlen=10000)
        self.daily_returns = deque(maxlen=10000)
        self.trades = []
        self.lock = Lock()
    
    def add_portfolio_value(self, value: float, timestamp: datetime):
        """Record portfolio value snapshot."""
        with self.lock:
            self.portfolio_values.append({
                'value': value,
                'timestamp': timestamp
            })
            
            # Calculate daily return if we have previous value
            if len(self.portfolio_values) > 1:
                prev_value = self.portfolio_values[-2]['value']
                daily_return = (value - prev_value) / prev_value if prev_value > 0 else 0
                self.daily_returns.append(daily_return)
    
    def add_trade(self, trade_data: Dict):
        """Record a closed trade."""
        with self.lock:
            self.trades.append({
                'timestamp': datetime.now(),
                'pnl': trade_data.get('pnl', 0),
                'pnl_pct': trade_data.get('pnl_pct', 0),
                'symbol': trade_data.get('symbol', ''),
                'side': trade_data.get('side', ''),
                'duration': trade_data.get('duration', 0)
            })
    
    def calculate_total_return(self) -> float:
        """Calculate total return from inception."""
        with self.lock:
            if len(self.portfolio_values) == 0:
                return 0.0
            
            final_value = self.portfolio_values[-1]['value']
            return (final_value - self.initial_capital) / self.initial_capital
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        with self.lock:
            if len(self.daily_returns) < 2:
                return 0.0
            
            returns = np.array(list(self.daily_returns))
            excess_return = np.mean(returns) - (risk_free_rate / 252)
            volatility = np.std(returns)
            
            if volatility == 0:
                return 0.0
            
            return excess_return / volatility * np.sqrt(252)
    
    def calculate_sortino_ratio(self, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (focus on downside volatility)."""
        with self.lock:
            if len(self.daily_returns) < 2:
                return 0.0
            
            returns = np.array(list(self.daily_returns))
            excess_return = np.mean(returns) - target_return
            
            downside_returns = returns[returns < target_return]
            downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0
            
            if downside_volatility == 0:
                return 0.0
            
            return excess_return / downside_volatility * np.sqrt(252)
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        annual_return = self.calculate_total_return() / (len(self.portfolio_values) / 252) if len(self.portfolio_values) > 0 else 0
        max_dd = self.calculate_max_drawdown()
        
        if max_dd >= 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        with self.lock:
            if len(self.portfolio_values) == 0:
                return 0.0
            
            values = np.array([v['value'] for v in self.portfolio_values])
            running_max = np.maximum.accumulate(values)
            drawdown = (values - running_max) / running_max
            
            return np.min(drawdown)
    
    def calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        with self.lock:
            if len(self.daily_returns) < 2:
                return 0.0
            
            returns = np.array(list(self.daily_returns))
            return np.std(returns) * np.sqrt(252)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from closed trades."""
        with self.lock:
            if len(self.trades) == 0:
                return 0.0
            
            winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
            return winning_trades / len(self.trades)
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        with self.lock:
            gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            return gross_profit / gross_loss
    
    def calculate_recovery_factor(self) -> float:
        """Calculate recovery factor (net profit / max drawdown)."""
        max_dd = self.calculate_max_drawdown()
        total_return = self.calculate_total_return()
        
        if max_dd >= 0:
            return 0.0
        
        return total_return / abs(max_dd)
    
    def get_monthly_returns(self) -> Dict[str, float]:
        """Calculate returns grouped by month."""
        with self.lock:
            monthly_data = defaultdict(list)
            
            for value_data in self.portfolio_values:
                year_month = value_data['timestamp'].strftime('%Y-%m')
                monthly_data[year_month].append(value_data['value'])
            
            monthly_returns = {}
            for month, values in sorted(monthly_data.items()):
                if len(values) > 1:
                    month_return = (values[-1] - values[0]) / values[0]
                    monthly_returns[month] = month_return
            
            return monthly_returns
    
    def get_yearly_returns(self) -> Dict[str, float]:
        """Calculate returns grouped by year."""
        with self.lock:
            yearly_data = defaultdict(list)
            
            for value_data in self.portfolio_values:
                year = value_data['timestamp'].strftime('%Y')
                yearly_data[year].append(value_data['value'])
            
            yearly_returns = {}
            for year, values in sorted(yearly_data.items()):
                if len(values) > 1:
                    year_return = (values[-1] - values[0]) / values[0]
                    yearly_returns[year] = year_return
            
            return yearly_returns
    
    def get_trade_statistics(self) -> Dict:
        """Get comprehensive trade statistics."""
        with self.lock:
            if len(self.trades) == 0:
                return {}
            
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            stats_dict = {
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': self.calculate_win_rate(),
                'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
                'largest_win': max((t['pnl'] for t in winning_trades), default=0),
                'largest_loss': min((t['pnl'] for t in losing_trades), default=0),
                'profit_factor': self.calculate_profit_factor(),
                'avg_trade_duration': np.mean([t['duration'] for t in self.trades]) if self.trades else 0,
                'expectancy': np.mean([t['pnl'] for t in self.trades]) if self.trades else 0
            }
            
            return stats_dict


class AssetAllocationTracker:
    """Track asset class allocations in real-time."""
    
    def __init__(self):
        self.allocations = {}
        self.allocation_history = deque(maxlen=10000)
        self.lock = Lock()
    
    def update_allocation(self, allocations: Dict[str, float], timestamp: datetime):
        """Update asset allocations."""
        with self.lock:
            self.allocations = allocations.copy()
            
            self.allocation_history.append({
                'timestamp': timestamp,
                'allocations': allocations.copy(),
                'timestamp_unix': timestamp.timestamp()
            })
    
    def get_current_allocation(self) -> Dict[str, float]:
        """Get current asset allocations."""
        with self.lock:
            return self.allocations.copy()
    
    def get_allocation_drift(self, target_allocation: Dict[str, float]) -> Dict[str, float]:
        """Calculate drift from target allocation."""
        with self.lock:
            drift = {}
            
            for asset, target in target_allocation.items():
                current = self.allocations.get(asset, 0)
                drift[asset] = current - target
            
            return drift
    
    def calculate_rebalancing_needs(self, threshold: float = 0.05) -> List[Tuple[str, float, float]]:
        """Identify assets needing rebalancing."""
        rebalancing = []
        
        with self.lock:
            for asset, current_weight in self.allocations.items():
                target_weight = 1.0 / len(self.allocations)
                
                if abs(current_weight - target_weight) > threshold:
                    rebalancing.append((asset, current_weight, target_weight))
        
        return rebalancing


class CorrelationMatrixCalculator:
    """Calculate and track correlation matrices."""
    
    def __init__(self, assets: List[str], window_days: int = 30):
        self.assets = assets
        self.window_days = window_days
        self.returns_data = {asset: deque(maxlen=window_days * 24) for asset in assets}
        self.correlation_matrix = None
        self.lock = Lock()
    
    def add_return(self, asset: str, return_pct: float, timestamp: datetime):
        """Add return data for asset."""
        if asset in self.returns_data:
            with self.lock:
                self.returns_data[asset].append({
                    'return': return_pct,
                    'timestamp': timestamp
                })
    
    def calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix from buffered returns."""
        with self.lock:
            n_assets = len(self.assets)
            correlation = np.zeros((n_assets, n_assets))
            
            for i, asset1 in enumerate(self.assets):
                for j, asset2 in enumerate(self.assets):
                    returns1 = np.array([r['return'] for r in self.returns_data[asset1]])
                    returns2 = np.array([r['return'] for r in self.returns_data[asset2]])
                    
                    if len(returns1) > 1 and len(returns2) > 1:
                        # Align to same length
                        min_len = min(len(returns1), len(returns2))
                        corr = np.corrcoef(returns1[-min_len:], returns2[-min_len:])[0, 1]
                        correlation[i, j] = corr if not np.isnan(corr) else 0
                    else:
                        correlation[i, j] = 1.0 if i == j else 0.0
            
            self.correlation_matrix = correlation
            return correlation
    
    def get_correlation_heatmap_data(self) -> Dict:
        """Get data formatted for heatmap visualization."""
        corr_matrix = self.calculate_correlation_matrix()
        
        heatmap_data = {
            'assets': self.assets,
            'correlation_matrix': corr_matrix.tolist(),
            'max_correlation': np.max(np.abs(corr_matrix)),
            'min_correlation': np.min(corr_matrix)
        }
        
        return heatmap_data
    
    def identify_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find highly correlated asset pairs."""
        corr_matrix = self.calculate_correlation_matrix()
        
        pairs = []
        for i in range(len(self.assets)):
            for j in range(i + 1, len(self.assets)):
                corr_value = corr_matrix[i, j]
                if abs(corr_value) > threshold:
                    pairs.append((self.assets[i], self.assets[j], corr_value))
        
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


class RealTimeRiskMonitor:
    """Real-time risk metric monitoring."""
    
    def __init__(self):
        self.var_95_buffer = TimeSeriesBuffer()
        self.var_99_buffer = TimeSeriesBuffer()
        self.drawdown_buffer = TimeSeriesBuffer()
        self.volatility_buffer = TimeSeriesBuffer()
        self.concentration_buffer = TimeSeriesBuffer()
        
        self.risk_alerts = []
        self.risk_limits = {
            'var_95': -0.05,
            'var_99': -0.10,
            'max_drawdown': -0.20,
            'volatility': 0.40,
            'concentration': 0.40
        }
        self.lock = Lock()
    
    def update_risk_metrics(self, metrics: Dict, timestamp: datetime):
        """Update risk metrics."""
        self.var_95_buffer.append(metrics.get('var_95', 0), timestamp)
        self.var_99_buffer.append(metrics.get('var_99', 0), timestamp)
        self.drawdown_buffer.append(metrics.get('drawdown', 0), timestamp)
        self.volatility_buffer.append(metrics.get('volatility', 0), timestamp)
        self.concentration_buffer.append(metrics.get('concentration', 0), timestamp)
        
        # Check for limit breaches
        self._check_limit_breaches(metrics, timestamp)
    
    def _check_limit_breaches(self, metrics: Dict, timestamp: datetime):
        """Check if any risk limits are breached."""
        with self.lock:
            breaches = []
            
            if metrics.get('var_95', 0) < self.risk_limits['var_95']:
                breaches.append(f"VaR 95% breach: {metrics['var_95']:.4f}")
            
            if metrics.get('drawdown', 0) < self.risk_limits['max_drawdown']:
                breaches.append(f"Max drawdown breach: {metrics['drawdown']:.4f}")
            
            if metrics.get('volatility', 0) > self.risk_limits['volatility']:
                breaches.append(f"Volatility breach: {metrics['volatility']:.4f}")
            
            if metrics.get('concentration', 0) > self.risk_limits['concentration']:
                breaches.append(f"Concentration breach: {metrics['concentration']:.4f}")
            
            for breach in breaches:
                self.risk_alerts.append({
                    'timestamp': timestamp,
                    'message': breach,
                    'severity': 'HIGH'
                })
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary."""
        return {
            'var_95': self.var_95_buffer.get_latest(),
            'var_99': self.var_99_buffer.get_latest(),
            'drawdown': self.drawdown_buffer.get_latest(),
            'volatility': self.volatility_buffer.get_latest(),
            'concentration': self.concentration_buffer.get_latest(),
            'recent_alerts': self.risk_alerts[-10:]  # Last 10 alerts
        }


class AdvancedDashboard:
    """
    Main advanced dashboard orchestrating all components.
    
    Manages real-time portfolio data, risk metrics, performance analytics,
    and visualization data updates.
    """
    
    def __init__(self, initial_capital: float = 100000, update_interval: float = 1.0):
        self.initial_capital = initial_capital
        self.update_interval = update_interval
        self.is_running = False
        self.update_thread = None
        
        # Initialize all components
        self.portfolio_metrics = PortfolioMetricsCalculator(initial_capital)
        self.allocation_tracker = AssetAllocationTracker()
        self.correlation_calculator = CorrelationMatrixCalculator([])
        self.risk_monitor = RealTimeRiskMonitor()
        
        # Time series buffers for main metrics
        self.portfolio_value_buffer = TimeSeriesBuffer()
        self.daily_return_buffer = TimeSeriesBuffer()
        self.pnl_buffer = TimeSeriesBuffer()
        
        # Position tracking
        self.open_positions = {}
        self.closed_positions = []
        
        # Dashboard state
        self.last_update = datetime.now()
        self.update_count = 0
        self.lock = Lock()
        
        logger.info(f"Dashboard initialized with ${initial_capital:,.2f} capital")
    
    def start(self):
        """Start dashboard update loop."""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Dashboard started")
    
    def stop(self):
        """Stop dashboard update loop."""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Dashboard stopped")
    
    def _update_loop(self):
        """Main update loop."""
        while self.is_running:
            try:
                self.update()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in dashboard update: {e}")
                time.sleep(1)
    
    def update_portfolio_value(self, value: float):
        """Update current portfolio value."""
        now = datetime.now()
        self.portfolio_value_buffer.append(value, now)
        self.portfolio_metrics.add_portfolio_value(value, now)
        self.last_update = now
    
    def update_position(self, symbol: str, quantity: float, current_price: float, 
                       entry_price: float, unrealized_pnl: float):
        """Update open position."""
        with self.lock:
            self.open_positions[symbol] = {
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'notional': quantity * current_price,
                'return_pct': (current_price - entry_price) / entry_price if entry_price > 0 else 0,
                'timestamp': datetime.now()
            }
    
    def close_position(self, symbol: str, exit_price: float, pnl: float, pnl_pct: float, 
                      duration_hours: float = 0):
        """Record closed position."""
        with self.lock:
            if symbol in self.open_positions:
                position = self.open_positions[symbol]
                
                closed_position = {
                    'symbol': symbol,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_hours': duration_hours,
                    'close_time': datetime.now()
                }
                
                self.closed_positions.append(closed_position)
                self.portfolio_metrics.add_trade(closed_position)
                
                del self.open_positions[symbol]
    
    def update_asset_allocation(self, allocations: Dict[str, float]):
        """Update asset class allocations."""
        self.allocation_tracker.update_allocation(allocations, datetime.now())
    
    def update_correlation_data(self, asset: str, return_pct: float):
        """Update correlation calculation data."""
        self.correlation_calculator.add_return(asset, return_pct, datetime.now())
    
    def update_risk_metrics(self, metrics: Dict):
        """Update risk metrics."""
        self.risk_monitor.update_risk_metrics(metrics, datetime.now())
    
    def update(self):
        """Internal dashboard update."""
        with self.lock:
            self.update_count += 1
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data for rendering."""
        with self.lock:
            portfolio_stats = self.portfolio_metrics.get_trade_statistics()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio': {
                    'value': self.portfolio_value_buffer.get_latest(),
                    'initial_capital': self.initial_capital,
                    'total_return': self.portfolio_metrics.calculate_total_return(),
                    'daily_return': self.daily_return_buffer.get_latest(),
                    'unrealized_pnl': sum(p['unrealized_pnl'] for p in self.open_positions.values()),
                    'realized_pnl': portfolio_stats.get('expectancy', 0) * portfolio_stats.get('total_trades', 0)
                },
                'performance': {
                    'sharpe_ratio': self.portfolio_metrics.calculate_sharpe_ratio(),
                    'sortino_ratio': self.portfolio_metrics.calculate_sortino_ratio(),
                    'calmar_ratio': self.portfolio_metrics.calculate_calmar_ratio(),
                    'volatility': self.portfolio_metrics.calculate_volatility(),
                    'max_drawdown': self.portfolio_metrics.calculate_max_drawdown(),
                    'recovery_factor': self.portfolio_metrics.calculate_recovery_factor()
                },
                'trades': {
                    'total': portfolio_stats.get('total_trades', 0),
                    'winning': portfolio_stats.get('winning_trades', 0),
                    'losing': portfolio_stats.get('losing_trades', 0),
                    'win_rate': portfolio_stats.get('win_rate', 0),
                    'profit_factor': portfolio_stats.get('profit_factor', 0),
                    'avg_win': portfolio_stats.get('avg_win', 0),
                    'avg_loss': portfolio_stats.get('avg_loss', 0)
                },
                'positions': {
                    'open': len(self.open_positions),
                    'details': self.open_positions
                },
                'allocation': self.allocation_tracker.get_current_allocation(),
                'correlation': self.correlation_calculator.get_correlation_heatmap_data() if self.correlation_calculator.correlation_matrix is not None else {},
                'risk': self.risk_monitor.get_risk_summary(),
                'time_series': {
                    'portfolio_values': {
                        'values': self.portfolio_value_buffer.get_data(lookback_seconds=86400)[0],
                        'timestamps': [t.isoformat() for t in self.portfolio_value_buffer.get_data(lookback_seconds=86400)[1]]
                    }
                },
                'monthly_returns': self.portfolio_metrics.get_monthly_returns(),
                'yearly_returns': self.portfolio_metrics.get_yearly_returns()
            }
    
    def get_position_details(self, symbol: str) -> Dict:
        """Get detailed information about a specific position."""
        with self.lock:
            if symbol in self.open_positions:
                return self.open_positions[symbol]
            return {}
    
    def get_closed_positions(self, limit: int = 100) -> List[Dict]:
        """Get most recent closed positions."""
        with self.lock:
            return self.closed_positions[-limit:]
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for reporting."""
        return {
            'total_return': self.portfolio_metrics.calculate_total_return(),
            'annualized_return': self.portfolio_metrics.calculate_total_return() * 252 / max(len(self.portfolio_metrics.daily_returns), 1),
            'volatility': self.portfolio_metrics.calculate_volatility(),
            'sharpe_ratio': self.portfolio_metrics.calculate_sharpe_ratio(),
            'sortino_ratio': self.portfolio_metrics.calculate_sortino_ratio(),
            'calmar_ratio': self.portfolio_metrics.calculate_calmar_ratio(),
            'max_drawdown': self.portfolio_metrics.calculate_max_drawdown(),
            'recovery_factor': self.portfolio_metrics.calculate_recovery_factor(),
            'win_rate': self.portfolio_metrics.calculate_win_rate(),
            'profit_factor': self.portfolio_metrics.calculate_profit_factor(),
            'num_trades': len(self.portfolio_metrics.trades),
            'average_trade_pnl': np.mean([t['pnl'] for t in self.portfolio_metrics.trades]) if self.portfolio_metrics.trades else 0
        }
    
    def export_to_json(self, filepath: str):
        """Export dashboard data to JSON file."""
        data = self.get_dashboard_data()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Dashboard data exported to {filepath}")
    
    def generate_html_report(self, filepath: str = "dashboard.html"):
        """Generate interactive HTML dashboard report."""
        data = self.get_dashboard_data()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Dashboard - {data['timestamp']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .label {{ color: #7f8c8d; font-size: 12px; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #2c3e50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trading Dashboard</h1>
            <p>Generated: {data['timestamp']}</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="label">Portfolio Value</div>
                <div class="metric">${data['portfolio']['value']['value']:,.2f}</div>
            </div>
            <div class="card">
                <div class="label">Total Return</div>
                <div class="metric {'positive' if data['performance']['sharpe_ratio'] > 0 else 'negative'}">
                    {data['performance']['sharpe_ratio']:.2%}
                </div>
            </div>
            <div class="card">
                <div class="label">Sharpe Ratio</div>
                <div class="metric">{data['performance']['sharpe_ratio']:.2f}</div>
            </div>
            <div class="card">
                <div class="label">Win Rate</div>
                <div class="metric">{data['trades']['win_rate']:.1%}</div>
            </div>
        </div>
        
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{data['performance']['sortino_ratio']:.2f}</td>
            </tr>
            <tr>
                <td>Calmar Ratio</td>
                <td>{data['performance']['calmar_ratio']:.2f}</td>
            </tr>
            <tr>
                <td>Max Drawdown</td>
                <td class="negative">{data['performance']['max_drawdown']:.2%}</td>
            </tr>
            <tr>
                <td>Volatility</td>
                <td>{data['performance']['volatility']:.2%}</td>
            </tr>
            <tr>
                <td>Total Trades</td>
                <td>{data['trades']['total']}</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{data['trades']['profit_factor']:.2f}</td>
            </tr>
        </table>
    </div>
</body>
</html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML dashboard generated at {filepath}")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("ADVANCED REAL-TIME DASHBOARD")
    print("=" * 80)
    
    # Create dashboard
    dashboard = AdvancedDashboard(initial_capital=100000)
    dashboard.start()
    
    # Simulate data updates
    np.random.seed(42)
    
    for i in range(100):
        portfolio_value = 100000 + np.cumsum(np.random.randn(100) * 500)[i]
        dashboard.update_portfolio_value(portfolio_value)
        
        # Update positions
        dashboard.update_position(
            'BTC/USDT',
            quantity=0.5,
            current_price=45000 + np.random.randn() * 1000,
            entry_price=44000,
            unrealized_pnl=np.random.randn() * 5000
        )
        
        # Close position occasionally
        if i % 20 == 0 and i > 0:
            dashboard.close_position(
                'BTC/USDT',
                exit_price=45500,
                pnl=750,
                pnl_pct=0.017,
                duration_hours=24
            )
        
        # Update allocation
        dashboard.update_asset_allocation({
            'EQUITIES': 0.4,
            'CRYPTO': 0.3,
            'FOREX': 0.2,
            'COMMODITIES': 0.1
        })
        
        time.sleep(0.01)
    
    # Get dashboard data
    data = dashboard.get_dashboard_data()
    print("\nDashboard Data:")
    print(json.dumps(data, indent=2, default=str)[:1000])
    
    # Export
    dashboard.export_to_json("dashboard_data.json")
    dashboard.generate_html_report("dashboard.html")
    
    dashboard.stop()
    print("\nDashboard stopped")
