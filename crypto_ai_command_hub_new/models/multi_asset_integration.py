"""
PHASE 4: Multi-Asset Class Integration Engine

Unified framework for managing stocks, cryptocurrencies, forex, and commodities.

Handles cross-asset correlations, multi-currency risk, execution routing,
and consolidated portfolio optimization across all asset classes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize, LinearConstraint, Bounds
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITIES = "equities"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITIES = "commodities"
    FIXED_INCOME = "fixed_income"
    DERIVATIVES = "derivatives"


class Exchange(Enum):
    """Exchange enumeration."""
    # Equities
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    LSE = "lse"
    TSE = "tse"
    
    # Crypto
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    FTX = "ftx"
    
    # Forex
    EURUSD = "eurusd"
    GBPUSD = "gbpusd"
    JPYUSD = "jpyusd"
    
    # Commodities
    CME = "cme"
    COMEX = "comex"
    NYMEX = "nymex"


@dataclass
class Asset:
    """Asset specification."""
    ticker: str
    asset_class: AssetClass
    exchange: Exchange
    
    # Trading parameters
    min_order_size: float
    max_order_size: float
    tick_size: float
    multiplier: float  # Contract multiplier for futures
    
    # Currency
    base_currency: str
    quote_currency: str
    
    # Risk parameters
    daily_volume_limit: float  # Maximum % of daily volume to trade
    position_limit: float  # Maximum position size
    margin_requirement: float
    
    # Transaction costs
    commission_pct: float
    bid_ask_spread: float
    
    # Risk metrics
    beta: float = 1.0
    volatility: float = 0.0
    var_95: float = 0.0
    
    def __str__(self):
        return f"{self.ticker} ({self.asset_class.value} @ {self.exchange.value})"


@dataclass
class Position:
    """Open position."""
    asset: Asset
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    
    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Portfolio weight
    portfolio_weight: float = 0.0
    
    @property
    def notional_value(self) -> float:
        """Notional position value."""
        return self.quantity * self.current_price * self.asset.multiplier
    
    @property
    def returns(self) -> float:
        """Position return %."""
        return (self.current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0
    
    def update_price(self, new_price: float):
        """Update position price and P&L."""
        self.current_price = new_price
        self.unrealized_pnl = self.notional_value - (self.quantity * self.entry_price * self.asset.multiplier)


@dataclass
class ExecutionRoute:
    """Execution routing specification."""
    asset: Asset
    exchange: Exchange
    execution_algorithm: str  # VWAP, TWAP, ICEBERG, etc.
    max_participation_rate: float
    urgency_level: int  # 1-10, higher = more urgent
    client_id: str = ""
    
    def __str__(self):
        return f"{self.asset.ticker} via {self.exchange.value} ({self.execution_algorithm})"


class CurrencyConverter:
    """Handle multi-currency conversions and risk."""
    
    def __init__(self, exchange_rates: Dict[str, float]):
        """
        Initialize with exchange rates.
        
        exchange_rates: {'EURUSD': 1.0850, 'GBPUSD': 1.2750, ...}
        """
        self.exchange_rates = exchange_rates.copy()
        self.historical_rates = {pair: [rate] for pair, rate in exchange_rates.items()}
        
        # Track FX volatility
        self.fx_volatility = {pair: 0.01 for pair in exchange_rates.keys()}
    
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert between currencies."""
        
        if from_currency == to_currency:
            return amount
        
        pair = f"{from_currency}{to_currency}"
        reverse_pair = f"{to_currency}{from_currency}"
        
        if pair in self.exchange_rates:
            return amount * self.exchange_rates[pair]
        elif reverse_pair in self.exchange_rates:
            return amount / self.exchange_rates[reverse_pair]
        else:
            # Try through USD
            if from_currency != "USD":
                usd_amount = self.convert(amount, from_currency, "USD")
            else:
                usd_amount = amount
            
            if to_currency != "USD":
                return self.convert(usd_amount, "USD", to_currency)
            else:
                return usd_amount
    
    def update_rates(self, new_rates: Dict[str, float]):
        """Update exchange rates and track history."""
        
        for pair, rate in new_rates.items():
            if pair in self.exchange_rates:
                self.historical_rates[pair].append(rate)
                self.exchange_rates[pair] = rate
        
        # Update FX volatility
        for pair in self.exchange_rates.keys():
            if len(self.historical_rates[pair]) > 20:
                recent_rates = np.array(self.historical_rates[pair][-20:])
                returns = np.diff(np.log(recent_rates))
                self.fx_volatility[pair] = np.std(returns)
    
    def get_fx_exposure(self, positions: List[Position], base_currency: str = "USD") -> Dict[str, float]:
        """Calculate FX exposure for portfolio."""
        
        fx_exposure = {}
        
        for position in positions:
            if position.asset.base_currency != base_currency:
                pair = f"{position.asset.base_currency}{base_currency}"
                notional = position.notional_value
                
                if pair not in fx_exposure:
                    fx_exposure[pair] = 0.0
                
                fx_exposure[pair] += notional
        
        return fx_exposure
    
    def get_fx_hedging_costs(self, exposures: Dict[str, float]) -> float:
        """Calculate cost to hedge FX exposures."""
        
        total_cost = 0.0
        
        for pair, notional in exposures.items():
            if pair in self.fx_volatility:
                # Hedge cost proportional to volatility and notional
                cost = notional * self.fx_volatility[pair] * 0.01  # 1% per vol point
                total_cost += cost
        
        return total_cost


class MultiAssetPortfolio:
    """
    Unified portfolio management across asset classes.
    """
    
    def __init__(self, base_currency: str = "USD"):
        self.base_currency = base_currency
        self.positions: List[Position] = []
        self.assets: Dict[str, Asset] = {}
        
        # Trading history
        self.trade_history = []
        
        # Risk tracking
        self.total_portfolio_value = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Asset class allocations
        self.asset_class_weights = {}
        
        # Currency management
        self.currency_converter = None
        
        # Correlation matrix across assets
        self.correlation_matrix = None
        
        logger_name = f"Portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def add_asset(self, asset: Asset):
        """Register asset for trading."""
        
        self.assets[asset.ticker] = asset
        logger.debug(f"Added asset: {asset}")
    
    def open_position(
        self,
        asset: Asset,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """Open a new position."""
        
        position = Position(
            asset=asset,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions.append(position)
        
        # Log trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'asset': asset.ticker,
            'side': 'BUY' if quantity > 0 else 'SELL',
            'quantity': abs(quantity),
            'price': entry_price,
            'notional': position.notional_value,
            'type': 'OPEN'
        })
        
        logger.info(f"Opened position: {position.asset.ticker} x{quantity} @ {entry_price}")
        
        return position
    
    def close_position(self, position: Position, exit_price: float) -> float:
        """Close an open position and realize P&L."""
        
        pnl = (exit_price - position.entry_price) * position.quantity * position.asset.multiplier
        position.realized_pnl = pnl
        
        # Log trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'asset': position.asset.ticker,
            'side': 'SELL' if position.quantity > 0 else 'BUY',
            'quantity': abs(position.quantity),
            'price': exit_price,
            'notional': position.quantity * exit_price * position.asset.multiplier,
            'pnl': pnl,
            'type': 'CLOSE'
        })
        
        self.positions.remove(position)
        self.realized_pnl += pnl
        
        logger.info(f"Closed position: {position.asset.ticker}, P&L: {pnl:.2f}")
        
        return pnl
    
    def update_positions(self, price_updates: Dict[str, float]):
        """Update all positions with new prices."""
        
        total_notional = 0.0
        total_unrealized = 0.0
        
        for position in self.positions:
            if position.asset.ticker in price_updates:
                new_price = price_updates[position.asset.ticker]
                position.update_price(new_price)
            
            total_notional += position.notional_value
            total_unrealized += position.unrealized_pnl
        
        self.total_portfolio_value = total_notional
        self.unrealized_pnl = total_unrealized
        
        # Update weights
        for position in self.positions:
            position.portfolio_weight = position.notional_value / (total_notional + 1e-10)
        
        # Update asset class weights
        self._update_asset_class_weights()
    
    def _update_asset_class_weights(self):
        """Calculate portfolio weights by asset class."""
        
        self.asset_class_weights = {}
        
        for position in self.positions:
            asset_class = position.asset.asset_class
            
            if asset_class not in self.asset_class_weights:
                self.asset_class_weights[asset_class] = 0.0
            
            self.asset_class_weights[asset_class] += position.portfolio_weight
    
    def calculate_cross_asset_correlations(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate correlation matrix across assets."""
        
        correlation = returns_data.corr().values
        self.correlation_matrix = correlation
        
        return correlation
    
    def optimize_allocation(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation across asset classes and assets.
        """
        
        assets = list(expected_returns.keys())
        n = len(assets)
        
        returns = np.array([expected_returns[a] for a in assets])
        
        def objective(weights):
            """Minimize portfolio variance."""
            portfolio_var = weights @ covariance_matrix @ weights.T
            return portfolio_var
        
        def returns_constraint(weights):
            """Target expected return."""
            return np.sum(weights * returns)
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]
        
        # Add custom constraints if provided
        if constraints:
            for name, constraint_func in constraints.items():
                cons.append({'type': 'ineq', 'fun': constraint_func})
        
        # Bounds: no short selling
        bnds = [(0, 1) for _ in range(n)]
        
        # Initial guess: equal weight
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bnds,
            constraints=cons
        )
        
        optimal_weights = dict(zip(assets, result.x))
        
        return optimal_weights
    
    def calculate_portfolio_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk for entire portfolio."""
        
        returns = []
        weights = []
        
        for position in self.positions:
            # Placeholder: would use actual return distribution
            position_returns = np.random.randn(1000) * position.asset.volatility
            returns.append(position_returns)
            weights.append(position.portfolio_weight)
        
        if len(returns) == 0:
            return 0.0
        
        # Weighted portfolio returns
        portfolio_returns = np.zeros(1000)
        for ret, weight in zip(returns, weights):
            portfolio_returns += ret * weight
        
        # VaR at confidence level
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        return float(var)
    
    def calculate_portfolio_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        
        var = self.calculate_portfolio_var(confidence)
        
        returns = []
        weights = []
        
        for position in self.positions:
            position_returns = np.random.randn(10000) * position.asset.volatility
            returns.append(position_returns)
            weights.append(position.portfolio_weight)
        
        if len(returns) == 0:
            return 0.0
        
        portfolio_returns = np.zeros(10000)
        for ret, weight in zip(returns, weights):
            portfolio_returns += ret * weight
        
        # CVaR: expected return conditional on being in VaR tail
        cvar = np.mean(portfolio_returns[portfolio_returns <= var])
        
        return float(cvar)
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        
        summary = {
            'total_notional': self.total_portfolio_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.unrealized_pnl + self.realized_pnl,
            'num_positions': len(self.positions),
            'asset_class_breakdown': dict(self.asset_class_weights),
            'var_95': self.calculate_portfolio_var(0.95),
            'cvar_95': self.calculate_portfolio_cvar(0.95),
            'return_pct': (self.unrealized_pnl + self.realized_pnl) / (self.total_portfolio_value + 1e-10) * 100
        }
        
        return summary


class ExecutionRouter:
    """
    Route trades to optimal exchanges based on liquidity and costs.
    """
    
    def __init__(self):
        self.execution_routes: Dict[str, List[ExecutionRoute]] = {}
        self.execution_history = []
    
    def register_route(self, asset: Asset, route: ExecutionRoute):
        """Register execution route for asset."""
        
        ticker = asset.ticker
        
        if ticker not in self.execution_routes:
            self.execution_routes[ticker] = []
        
        self.execution_routes[ticker].append(route)
    
    def select_best_route(
        self,
        asset: Asset,
        quantity: float,
        side: str
    ) -> ExecutionRoute:
        """Select best execution route based on liquidity and costs."""
        
        routes = self.execution_routes.get(asset.ticker, [])
        
        if not routes:
            # Default route
            return ExecutionRoute(
                asset=asset,
                exchange=asset.exchange,
                execution_algorithm="VWAP",
                max_participation_rate=0.2,
                urgency_level=5
            )
        
        # Score routes based on estimated costs
        best_route = None
        best_score = float('inf')
        
        for route in routes:
            # Estimate execution cost
            commission = asset.commission_pct * quantity * (quantity * asset.tick_size)  # Placeholder
            participation_cost = quantity * asset.bid_ask_spread / 2
            
            # Total cost
            total_cost = commission + participation_cost
            
            if total_cost < best_score:
                best_score = total_cost
                best_route = route
        
        return best_route if best_route else routes[0]
    
    def execute_order(
        self,
        asset: Asset,
        quantity: float,
        price: float,
        side: str = "BUY"
    ) -> Dict:
        """Execute order via optimal route."""
        
        route = self.select_best_route(asset, quantity, side)
        
        # Simulate execution
        execution_detail = {
            'timestamp': datetime.now(),
            'asset': asset.ticker,
            'side': side,
            'quantity': quantity,
            'target_price': price,
            'executed_price': price * (1 + np.random.randn() * asset.bid_ask_spread),
            'execution_route': str(route),
            'notional': quantity * price * asset.multiplier,
            'commission': quantity * price * asset.commission_pct,
            'status': 'EXECUTED'
        }
        
        self.execution_history.append(execution_detail)
        
        return execution_detail


class CrossAssetOptimizer:
    """
    Optimize portfolio across multiple asset classes with correlation.
    """
    
    def __init__(self, portfolio: MultiAssetPortfolio):
        self.portfolio = portfolio
        
        # Asset class statistics
        self.asset_class_stats = {}
    
    def estimate_cross_asset_correlations(
        self,
        returns_data: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Estimate correlations across asset classes."""
        
        assets = list(returns_data.keys())
        n = len(assets)
        
        # Create returns matrix
        max_len = max(len(r) for r in returns_data.values())
        returns_matrix = np.zeros((max_len, n))
        
        for i, (asset, returns) in enumerate(returns_data.items()):
            returns_matrix[:len(returns), i] = returns
        
        # Calculate correlation
        correlation = np.corrcoef(returns_matrix.T)
        
        return correlation
    
    def optimize_across_asset_classes(
        self,
        expected_returns: Dict[str, float],
        covariances: Dict[str, float],
        target_volatility: float = 0.15
    ) -> Dict[str, float]:
        """
        Optimize allocation across asset classes with diversification target.
        """
        
        asset_classes = list(expected_returns.keys())
        n = len(asset_classes)
        
        returns = np.array([expected_returns[ac] for ac in asset_classes])
        vols = np.array([np.sqrt(covariances.get(ac, 0.01)) for ac in asset_classes])
        
        def objective(weights):
            """Minimize variance with return target."""
            # Simplified variance calculation
            portfolio_vol = np.sum(weights * vols)
            portfolio_return = np.sum(weights * returns)
            
            # Penalize deviation from target return
            return_penalty = 100 * (portfolio_return - 0.08)**2
            
            return portfolio_vol + return_penalty
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bnds = [(0, 0.4) for _ in range(n)]  # Max 40% in any class
        
        x0 = np.ones(n) / n
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bnds,
            constraints=cons
        )
        
        return dict(zip(asset_classes, result.x))


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("MULTI-ASSET CLASS INTEGRATION ENGINE")
    print("=" * 80)
    
    # Create portfolio
    portfolio = MultiAssetPortfolio(base_currency="USD")
    
    # Create assets
    stock = Asset(
        ticker="AAPL",
        asset_class=AssetClass.EQUITIES,
        exchange=Exchange.NASDAQ,
        min_order_size=1,
        max_order_size=100000,
        tick_size=0.01,
        multiplier=1.0,
        base_currency="USD",
        quote_currency="USD",
        daily_volume_limit=0.05,
        position_limit=10000,
        margin_requirement=0.5,
        commission_pct=0.001,
        bid_ask_spread=0.01,
        beta=1.2,
        volatility=0.25
    )
    
    crypto = Asset(
        ticker="BTC",
        asset_class=AssetClass.CRYPTO,
        exchange=Exchange.BINANCE,
        min_order_size=0.001,
        max_order_size=100,
        tick_size=1.0,
        multiplier=1.0,
        base_currency="USDT",
        quote_currency="USD",
        daily_volume_limit=0.1,
        position_limit=10,
        margin_requirement=1.0,
        commission_pct=0.001,
        bid_ask_spread=5.0,
        beta=2.5,
        volatility=0.8
    )
    
    # Add to portfolio
    portfolio.add_asset(stock)
    portfolio.add_asset(crypto)
    
    # Open positions
    stock_pos = portfolio.open_position(stock, 100, 150.0, stop_loss=145.0, take_profit=160.0)
    crypto_pos = portfolio.open_position(crypto, 0.5, 45000.0, stop_loss=42000.0, take_profit=50000.0)
    
    # Update prices
    portfolio.update_positions({"AAPL": 152.0, "BTC": 46000.0})
    
    # Print summary
    summary = portfolio.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    for key, val in summary.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    # Currency conversion
    fx_converter = CurrencyConverter({"EURUSD": 1.0850, "GBPUSD": 1.2750})
    usd_amount = fx_converter.convert(100, "EUR", "USD")
    print(f"\n100 EUR = {usd_amount:.2f} USD")
    
    # Routing
    router = ExecutionRouter()
    router.register_route(stock, ExecutionRoute(
        asset=stock,
        exchange=Exchange.NASDAQ,
        execution_algorithm="VWAP",
        max_participation_rate=0.2,
        urgency_level=5
    ))
    
    order = router.execute_order(stock, 100, 150.0, "BUY")
    print(f"\nOrder executed: {order['asset']} x{order['quantity']} @ {order['executed_price']:.2f}")
