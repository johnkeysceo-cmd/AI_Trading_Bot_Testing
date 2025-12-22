"""Comprehensive Backtesting Engine with Real Market Conditions
Full simulation with realistic fees, slippage, and market microstructure"""

import numpy as np, json, logging, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

@dataclass
class Order:
    order_id: str
    timestamp: float
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    size: float
    price: float
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    fee_paid: float = 0.0
    
    def to_dict(self):
        return {
            'order_id': self.order_id,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type.value,
            'size': float(self.size),
            'price': float(self.price),
            'status': self.status.value,
            'filled_size': float(self.filled_size),
            'average_fill_price': float(self.average_fill_price),
            'fee_paid': float(self.fee_paid)
        }

@dataclass
class Trade:
    trade_id: str
    entry_timestamp: float
    exit_timestamp: float
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    fees_paid: float
    duration_days: float
    
    def to_dict(self):
        return {
            'trade_id': self.trade_id,
            'entry_timestamp': self.entry_timestamp,
            'exit_timestamp': self.exit_timestamp,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': float(self.entry_price),
            'exit_price': float(self.exit_price),
            'size': float(self.size),
            'pnl': float(self.pnl),
            'pnl_pct': float(self.pnl_pct),
            'fees_paid': float(self.fees_paid),
            'duration_days': float(self.duration_days)
        }

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    entry_timestamp: float
    unrealized_pnl: float = 0.0
    
    def get_current_value(self, current_price: float) -> float:
        if self.side == 'long':
            return self.size * current_price
        else:  # short
            return -self.size * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        if self.side == 'long':
            return self.size * (current_price - self.entry_price)
        else:  # short
            return self.size * (self.entry_price - current_price)

@dataclass
class BacktestMetrics:
    timestamp: float
    portfolio_value: float
    cash: float
    open_positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float
    sharpe_ratio: float
    
    def to_dict(self):
        return {
            'timestamp': float(self.timestamp),
            'portfolio_value': float(self.portfolio_value),
            'cash': float(self.cash),
            'open_positions_value': float(self.open_positions_value),
            'unrealized_pnl': float(self.unrealized_pnl),
            'realized_pnl': float(self.realized_pnl),
            'drawdown': float(self.drawdown),
            'sharpe_ratio': float(self.sharpe_ratio)
        }

class PortfolioManager:
    """Manages portfolio state, positions, and cash"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.portfolio_value_history = deque(maxlen=10000)
        self.realized_pnl = 0.0
    
    def add_position(self, symbol: str, side: str, size: float,
                    entry_price: float, timestamp: float):
        """Open a new position"""
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_timestamp=timestamp
        )
        logger.debug(f"Opened {side} position: {symbol} {size} @ ${entry_price}")
    
    def close_position(self, symbol: str, exit_price: float,
                      timestamp: float, fees_paid: float):
        """Close an open position"""
        
        if symbol not in self.positions:
            logger.warning(f"No position to close: {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate PnL
        if position.side == 'long':
            pnl = position.size * (exit_price - position.entry_price)
        else:  # short
            pnl = position.size * (position.entry_price - exit_price)
        
        pnl_after_fees = pnl - fees_paid
        pnl_pct = pnl_after_fees / (position.size * position.entry_price) if position.size > 0 else 0
        
        duration = (timestamp - position.entry_timestamp) / (24 * 3600)
        
        # Create trade record
        trade = Trade(
            trade_id=f"{symbol}_{timestamp}",
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=timestamp,
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees_paid=fees_paid,
            duration_days=duration
        )
        
        self.closed_trades.append(trade)
        self.realized_pnl += pnl_after_fees
        
        del self.positions[symbol]
        
        logger.debug(f"Closed {position.side} position: {symbol} PnL={pnl_after_fees:.2f} ({pnl_pct*100:.2f}%)")
        
        return trade
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value += position.get_current_value(current_prices[symbol])
        
        return self.cash + position_value
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total unrealized PnL"""
        pnl = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                pnl += position.get_unrealized_pnl(current_prices[symbol])
        
        return pnl
    
    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total PnL (realized + unrealized)"""
        return self.realized_pnl + self.get_unrealized_pnl(current_prices)

class ExecutionEngine:
    """Simulates realistic order execution with slippage and fees"""
    
    def __init__(self, maker_fee: float = 0.001, taker_fee: float = 0.0025):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
    
    def submit_order(self, symbol: str, side: str, size: float,
                    price: float, order_type: OrderType = OrderType.MARKET,
                    timestamp: float = None) -> Order:
        """Submit an order"""
        
        if timestamp is None:
            timestamp = time.time()
        
        order_id = f"ORDER_{self.order_counter}_{timestamp}"
        self.order_counter += 1
        
        order = Order(
            order_id=order_id,
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            price=price
        )
        
        self.orders[order_id] = order
        
        logger.debug(f"Order submitted: {order_id} {side} {size} {symbol} @ ${price}")
        
        return order
    
    def execute_order(self, order: Order, current_price: float,
                     is_maker: bool = False) -> Tuple[float, float, float]:
        """Execute order and return (filled_size, fill_price, fee)"""
        
        # Calculate execution price with slippage
        if order.order_type == OrderType.MARKET:
            # Market orders get slippage
            slippage = np.random.uniform(0.0005, 0.002)  # 0.05% - 0.2%
            if order.side == 'buy':
                execution_price = current_price * (1 + slippage)
            else:  # sell
                execution_price = current_price * (1 - slippage)
        else:
            execution_price = order.price
        
        # Fill order
        filled_size = order.size
        
        # Calculate fee
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = order.size * execution_price * fee_rate
        
        order.filled_size = filled_size
        order.average_fill_price = execution_price
        order.fee_paid = fee
        order.status = OrderStatus.FILLED
        
        return filled_size, execution_price, fee

class BacktestingEngine:
    """Complete backtesting system with realistic market conditions"""
    
    def __init__(self, initial_capital: float = 10000.0,
                 maker_fee: float = 0.001, taker_fee: float = 0.0025):
        self.initial_capital = initial_capital
        self.portfolio = PortfolioManager(initial_capital)
        self.execution = ExecutionEngine(maker_fee, taker_fee)
        
        self.metrics_history: List[BacktestMetrics] = []
        self.portfolio_values = deque(maxlen=10000)
        
        self.max_portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.peak_timestamp = time.time()
    
    def calculate_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown percentage"""
        if self.peak_value == 0:
            return 0
        return (self.peak_value - current_value) / self.peak_value
    
    def simulate_trading_signal(self, symbol: str, signal: float, strength: float,
                               current_prices: Dict[str, float], timestamp: float,
                               kelly_fraction: float = 0.1):
        """Simulate trading based on signal (real trading logic)"""
        
        if symbol not in current_prices:
            return None
        
        current_price = current_prices[symbol]
        
        # Calculate position size using Kelly criterion
        portfolio_value = self.portfolio.get_total_value(current_prices)
        position_size = (portfolio_value * kelly_fraction * strength) / current_price
        
        # Check if we already have a position
        if symbol in self.portfolio.positions:
            position = self.portfolio.positions[symbol]
            
            # Check for exit signals
            if signal < -0.3:  # Strong exit signal
                fees = position.size * current_price * self.execution.taker_fee
                self.portfolio.cash += position.size * current_price - fees
                
                self.portfolio.close_position(symbol, current_price, timestamp, fees)
        
        else:
            # Open new position
            if signal > 0.3:  # Strong buy signal
                # Buy
                order = self.execution.submit_order(
                    symbol, 'buy', position_size, current_price,
                    order_type=OrderType.MARKET, timestamp=timestamp
                )
                
                filled_size, fill_price, fee = self.execution.execute_order(
                    order, current_price, is_maker=False
                )
                
                cost = filled_size * fill_price + fee
                
                if self.portfolio.cash >= cost:
                    self.portfolio.cash -= cost
                    self.portfolio.add_position(
                        symbol, 'long', filled_size, fill_price, timestamp
                    )
    
    def record_metrics(self, current_prices: Dict[str, float], timestamp: float):
        """Record portfolio metrics at current timestamp"""
        
        portfolio_value = self.portfolio.get_total_value(current_prices)
        position_value = portfolio_value - self.portfolio.cash
        unrealized_pnl = self.portfolio.get_unrealized_pnl(current_prices)
        
        # Update peak
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.peak_timestamp = timestamp
        
        drawdown = self.calculate_drawdown(portfolio_value)
        
        # Calculate Sharpe ratio
        if len(self.portfolio_values) > 1:
            returns = np.diff(list(self.portfolio_values)) / np.array(list(self.portfolio_values)[:-1])
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        metrics = BacktestMetrics(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            cash=self.portfolio.cash,
            open_positions_value=position_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.portfolio.realized_pnl,
            drawdown=drawdown,
            sharpe_ratio=sharpe
        )
        
        self.metrics_history.append(metrics)
        self.portfolio_values.append(portfolio_value)
    
    def get_backtest_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results"""
        
        if not self.metrics_history:
            return {}
        
        portfolio_values = list(self.portfolio_values)
        final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
        
        total_return = (final_value - self.initial_capital) / self.initial_capital
        annual_return = total_return  # For simplicity, adjust based on actual duration
        
        # Calculate volatility
        returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        annual_volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio
        sharpe = annual_return / (annual_volatility + 1e-8) if annual_volatility > 0 else 0
        
        # Maximum drawdown
        max_dd = np.max([m.drawdown for m in self.metrics_history])
        
        # Trade statistics
        total_trades = len(self.portfolio.closed_trades)
        winning_trades = sum(1 for t in self.portfolio.closed_trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in self.portfolio.closed_trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.portfolio.closed_trades if t.pnl < 0]) if losing_trades > 0 else 0
        
        profit_factor = (winning_trades * abs(avg_win)) / (losing_trades * abs(avg_loss) + 1e-8) if losing_trades > 0 else np.inf
        
        total_fees = sum(t.fees_paid for t in self.portfolio.closed_trades)
        
        return {
            'initial_capital': float(self.initial_capital),
            'final_portfolio_value': float(final_value),
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'total_fees_paid': float(total_fees),
            'realized_pnl': float(self.portfolio.realized_pnl),
            'closed_trades': [t.to_dict() for t in self.portfolio.closed_trades[:100]],  # Last 100 trades
            'metrics': [m.to_dict() for m in self.metrics_history[-500:]],  # Last 500 metrics
        }

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("BACKTESTING ENGINE - Real Market Conditions")
    logger.info("=" * 80)
    
    backtest = BacktestingEngine(initial_capital=10000.0)
    
    # Simulate trading
    logger.info("\nSimulating trades...")
    
    for i in range(100):
        # Simulate signals
        signal = np.sin(i * 0.1) * 0.5
        strength = abs(signal)
        
        # Simulate price movement
        current_prices = {
            'BTC': 42000 + np.random.normal(0, 500),
            'ETH': 2500 + np.random.normal(0, 50),
            'XRP': 0.75 + np.random.normal(0, 0.01)
        }
        
        timestamp = time.time() + i * 3600
        
        for symbol in ['BTC', 'ETH']:
            backtest.simulate_trading_signal(symbol, signal, strength, current_prices, timestamp)
        
        backtest.record_metrics(current_prices, timestamp)
    
    results = backtest.get_backtest_results()
    
    logger.info("\nBACKTEST RESULTS:")
    logger.info(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    logger.info(f"  Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
    logger.info(f"  Total Return: {results['total_return']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    logger.info(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
    logger.info(f"  Total Trades: {results['total_trades']}")
    logger.info(f"  Win Rate: {results['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"  Total Fees: ${results['total_fees_paid']:,.2f}")
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n✓ Backtest complete")
