#!/usr/bin/env python3
"""Live-data-driven simulation runner.

This script pulls historical/realtime data from the built-in
`RealtimeMarketEngine` (CoinGecko + Kraken adapters) and runs the
`ULTIMATE_10_10_TradingBot` in paper/simulation mode locally.

Important safety: This script will NOT execute real orders unless you
explicitly set the environment variable `ALLOW_REAL_ORDERS=true` and
provide exchange API keys in `config/api_keys.json`. Do NOT paste API
keys into this file or into shared chats.
"""

import os
import argparse
import logging
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from models.realtime_market_engine import RealtimeMarketEngine
from models.ULTIMATE_10_10_TRADING_BOT import ULTIMATE_10_10_TradingBot
from scripts.online_trainer import OnlineTrainer

import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('live_simulator')


def convert_marketdata_to_ohlcv(candles):
    """Convert MarketData objects to bot's expected OHLCV dicts."""
    ohlcv = []
    for c in candles:
        try:
            ohlcv.append({'o': float(c.open), 'h': float(c.high), 'l': float(c.low), 'c': float(c.close), 'v': float(getattr(c, 'volume', 0) or 0)})
        except Exception:
            # If passed as tuple-like (timestamp, o, h, l, c)
            try:
                ts, o, h, l, cval = c
                ohlcv.append({'o': float(o), 'h': float(h), 'l': float(l), 'c': float(cval), 'v': 0})
            except Exception:
                continue
    return ohlcv


def main(symbols, days, allow_real_orders):
    engine = RealtimeMarketEngine()

    logger.info(f"Loading {days} days of historical data for: {', '.join(symbols)}")

    # Populate engine cache (CoinGecko daily prices)
    for s in symbols:
        loaded = engine.load_historical_data(s, days=days)
        logger.info(f"  → {s}: {loaded} daily candles loaded")

    # Prepare bot and inject OHLCV/price data
    bot = ULTIMATE_10_10_TradingBot(initial_capital=200.0, use_real_data=False, lookback_days=days)

    # Convert engine cache to bot-compatible structure
    for s in symbols:
        candles = engine.cache.get_candles(s, '1d', limit=days)
        if not candles:
            logger.warning(f"No candles found for {s} — skipping")
            bot.ohlcv_data[s] = []
            bot.price_data[s] = []
            continue

        ohlcv = convert_marketdata_to_ohlcv(candles)
        bot.ohlcv_data[s] = ohlcv
        bot.price_data[s] = [bar['c'] for bar in ohlcv]

    # Safety: prevent accidental real orders
    if allow_real_orders:
        logger.warning("ALLOW_REAL_ORDERS is set: ensure config/api_keys.json is populated and you accept risk.")
    else:
        logger.info("Running in PAPER mode: real orders disabled (set ALLOW_REAL_ORDERS=true to override locally).")

    # Prepare results directory
    results_dir = ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer (records features/outcomes during simulation)
    trainer = OnlineTrainer(results_dir=results_dir)

    # Attach trainer hooks: the simplest approach is to swap-in the bot.meta_learner
    # or use a callback. We will poll closed_trades after run and record features
    # for offline retraining. For more advanced integration, the trainer can be
    # called inside the bot's trade lifecycle.

    # Run simulation — the bot uses injected OHLCV/price data
    start = time.time()
    bot.run_simulation()
    elapsed = time.time() - start
    logger.info(f"Simulation completed in {elapsed:.1f}s")

    # Telemetry: collect basic metrics
    telemetry = {
        'timestamp': datetime.utcnow().isoformat(),
        'symbols': symbols,
        'days': days,
        'elapsed_sec': elapsed,
        'final_capital': bot.capital,
        'initial_capital': bot.initial_capital,
        'num_closed_trades': len(bot.closed_trades),
    }

    # Compute trade-level metrics
    wins = sum(1 for t in bot.closed_trades if t['pnl'] > 0)
    losses = len(bot.closed_trades) - wins
    telemetry.update({'wins': wins, 'losses': losses})

    total_fees = 0.0
    trade_cadence = []
    for t in bot.closed_trades:
        # approximate fee as entry*size*binance_fee
        fee_est = (t.get('entry', 0) * t.get('size', 0)) * bot.binance_fee
        total_fees += abs(fee_est)
        trade_cadence.append(t.get('day_opened', 0))
    telemetry.update({'total_fees': total_fees, 'trade_cadence_days': trade_cadence})

    # Save telemetry
    tele_path = results_dir / f"telemetry_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(tele_path, 'w') as f:
        json.dump(telemetry, f, indent=2)
    logger.info(f"Telemetry saved to {tele_path}")

    # Record features/outcomes for trainer (basic: use closed_trades records)
    for t in bot.closed_trades:
        # Minimal feature set scaffold: you can extend to include full feature dicts
        sample = {
            'symbol': t['symbol'],
            'entry': t['entry'],
            'exit': t['exit'],
            'size': t['size'],
            'pnl': t['pnl'],
            'pnl_pct': t.get('pnl_pct', 0),
            'reason': t.get('reason', '')
        }
        trainer.record_sample(sample)

    # Persist training dataset and run an offline retrain checkpoint (safe)
    trainer.save_dataset()
    trainer.retrain_offline()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run ULTIMATE bot using live historical prices (CoinGecko)')
    p.add_argument('--symbols', '-s', nargs='+', default=['BTC', 'ETH'], help='Symbols to simulate (default: BTC ETH)')
    p.add_argument('--days', '-d', type=int, default=30, help='Days of history to pull (daily candles)')
    args = p.parse_args()

    allow_real = os.environ.get('ALLOW_REAL_ORDERS', 'false').lower() in ('1', 'true', 'yes')

    main(args.symbols, args.days, allow_real)
