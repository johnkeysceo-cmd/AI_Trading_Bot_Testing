#!/usr/bin/env python3
"""Continuous live runner: polls CoinGecko and drives the ULTIMATE bot in PAPER mode.

Runs until interrupted (Ctrl+C). Uses `config/api_keys.json` CoinGecko key when present.
"""

import os
import time
import logging
import argparse
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from models.realtime_market_engine import RealtimeMarketEngine
from models.ULTIMATE_10_10_TRADING_BOT import ULTIMATE_10_10_TradingBot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('live_runner')


def convert_marketdata_to_ohlcv(candles):
    ohlcv = []
    for c in candles:
        try:
            ohlcv.append({'o': float(c.open), 'h': float(c.high), 'l': float(c.low), 'c': float(c.close), 'v': float(getattr(c, 'volume', 0) or 0)})
        except Exception:
            try:
                ts, o, h, l, cval = c
                ohlcv.append({'o': float(o), 'h': float(h), 'l': float(l), 'c': float(cval), 'v': 0})
            except Exception:
                continue
    return ohlcv


def main(symbols, lookback_days, interval, initial_capital, entry_threshold=None, min_confidence=None, move_threshold=0.0, aggressive=False, force_on_move=False):
    engine = RealtimeMarketEngine()

    logger.info(f"Preloading {lookback_days} days history for: {', '.join(symbols)}")
    # preload
    for s in symbols:
        engine.load_historical_data(s, days=lookback_days)

    # Initialize bot (we feed it live ticks)
    bot = ULTIMATE_10_10_TradingBot(initial_capital=initial_capital, use_real_data=False, lookback_days=lookback_days)

    # Override thresholds if requested (for live testing)
    if entry_threshold is not None:
        bot.entry_threshold = float(entry_threshold)
    if min_confidence is not None:
        bot.min_confidence = float(min_confidence)
    if aggressive:
        bot.entry_threshold = min(bot.entry_threshold, 0.15)
        bot.min_confidence = min(bot.min_confidence, 0.25)

    # Prepare trade log
    trades_log_path = Path(ROOT) / 'results' / 'live_trades.jsonl'
    trades_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Populate bot historic buckets from engine cache
    for s in symbols:
        candles = engine.cache.get_candles(s, '1d', limit=lookback_days)
        if not candles:
            logger.warning(f"No cached candles for {s} — bot will start with synthetic history")
            continue
        ohlcv = convert_marketdata_to_ohlcv(candles)
        bot.ohlcv_data[s] = ohlcv
        bot.price_data[s] = [bar['c'] for bar in ohlcv]
        bot.volumes[s] = [bar.get('v', 0) for bar in ohlcv]

    logger.info("Starting continuous live loop (paper mode). Press Ctrl+C to stop.")

    tick = 0
    try:
        while True:
            tick += 1
            current_prices = engine.get_current_prices()

            # build per-symbol current price map (fallback to last known)
            prices = {}
            for s in symbols:
                p = current_prices.get(s)
                if not p:
                    # fallback to last known
                    last = bot.price_data.get(s)
                    p = last[-1] if last else None
                prices[s] = p

            # Update bot data stores with synthetic OHLCV for this tick (daily resolution)
            for s in symbols:
                price = prices.get(s)
                if price is None:
                    continue
                bar = {'o': price, 'h': price, 'l': price, 'c': price, 'v': 0}
                bot.ohlcv_data[s].append(bar)
                bot.price_data[s].append(price)
                bot.volumes[s].append(0)

                # keep history bounded
                if len(bot.price_data[s]) > max(500, lookback_days * 2):
                    bot.price_data[s] = bot.price_data[s][-500:]
                    bot.ohlcv_data[s] = bot.ohlcv_data[s][-500:]
                    bot.volumes[s] = bot.volumes[s][-500:]

            # Run signal & exit logic per symbol (reuse bot methods)
            # create a synthetic day index
            day = len(bot.price_data[symbols[0]]) - 1

            # Close aged positions and check exit conditions
            try:
                bot.check_exit_conditions(day, prices)
            except Exception as e:
                logger.exception(f"Error in check_exit_conditions: {e}")

            # Decision logic
            for s in symbols:
                try:
                    if len(bot.positions) >= bot.max_positions:
                        break
                    if s in bot.positions:
                        continue

                    if len(bot.ohlcv_data[s]) < 20:
                        continue

                    ohlcv_window = bot.ohlcv_data[s][-60:]
                    volumes_window = bot.volumes[s][-60:]

                    features = bot.feature_engineer.engineer_features(ohlcv_window, volumes_window)
                    signal, confidence = bot.ensemble.get_ensemble_signal(features)

                    # Debugging: log feature snapshot and ensemble details occasionally
                    try:
                        if tick <= 5 or abs(signal) > 0.15:
                            logger.info(f"DEBUG {s} tick={tick} signal={signal:.4f} conf={confidence:.4f} threshold={min(bot.meta_learner.get_adaptive_threshold(), bot.entry_threshold):.3f}")
                            # attempt to log model-level votes if available
                            try:
                                x = bot.ensemble.xgboost_model.score_features(features)
                                l = bot.ensemble.lightgbm_model.score_features(features)
                                n = bot.ensemble.neural_model.predict(features)
                                logger.info(f"  votes: xg={x:.3f} lgb={l:.3f} nn={n:.3f} vol={1.0 if features.get('volume_ratio',1)>1.2 else (-1.0 if features.get('volume_ratio',1)<0.8 else 0.0)} trend={1.0 if features.get('trend_strength',0)>0.05 else (-1.0 if features.get('trend_strength',0)<-0.05 else 0.0)}")
                            except Exception:
                                pass
                            # log a compact features subset
                            compact = {k: features.get(k) for k in ['rsi','momentum','volatility','volume_ratio','trend_strength','macd_signal','bb_position']}
                            logger.info(f"  features: {compact}")
                    except Exception:
                        pass
                    adaptive_threshold = bot.meta_learner.get_adaptive_threshold()
                    is_optimal_time, quantum_amp = bot.quantum_timer.get_optimal_entry_time(abs(signal), (tick * 1) % 24, tick % 7)

                    threshold = min(adaptive_threshold, bot.entry_threshold)
                    quantum_ok = is_optimal_time or (quantum_amp > 0.20 and confidence > 0.55)
                    # Optionally require a price move before entering
                    allow_by_move = True
                    if move_threshold and len(bot.price_data.get(s, [])) >= 2:
                        prev = bot.price_data[s][-2]
                        curr = prices[s]
                        if prev and curr:
                            pct = abs(curr - prev) / prev
                            allow_by_move = pct >= float(move_threshold)

                    # If configured, force a trade when move threshold is met
                    if force_on_move and allow_by_move and s not in bot.positions:
                        # determine direction by last move
                        prev = bot.price_data[s][-2] if len(bot.price_data[s]) >= 2 else None
                        curr = prices[s]
                        if prev and curr:
                            side_signal = 0.6 if curr > prev else -0.6
                            bot.execute_trade(s, side_signal, prices[s], features, day)
                            pos = bot.positions.get(s)
                            if pos:
                                entry_record = {
                                    'timestamp': time.time(),
                                    'symbol': s,
                                    'side': pos.get('side'),
                                    'entry': pos.get('entry'),
                                    'size': pos.get('size'),
                                    'confidence': float(confidence),
                                    'signal': float(side_signal),
                                    'forced_by_move': True
                                }
                                with open(trades_log_path, 'a') as tf:
                                    tf.write(json.dumps(entry_record) + '\n')
                            continue

                    if abs(signal) > threshold and confidence > bot.min_confidence and quantum_ok and allow_by_move:
                        bot.execute_trade(s, signal, prices[s], features, day)
                        # log the new position
                        pos = bot.positions.get(s)
                        if pos:
                            entry_record = {
                                'timestamp': time.time(),
                                'symbol': s,
                                'side': pos.get('side'),
                                'entry': pos.get('entry'),
                                'size': pos.get('size'),
                                'confidence': float(confidence),
                                'signal': float(signal)
                            }
                            with open(trades_log_path, 'a') as tf:
                                tf.write(json.dumps(entry_record) + '\n')
                except Exception as e:
                    logger.exception(f"Error evaluating signal for {s}: {e}")

            # Brief status
            logger.info(f"Tick {tick}: Prices: {', '.join([f'{s}=${prices.get(s,0):.2f}' for s in symbols])} | Positions: {len(bot.positions)} | Capital: ${bot.capital:.2f}")

            # Persist closed trades after each tick
            try:
                if bot.closed_trades:
                    closed_path = Path(ROOT) / 'results' / 'closed_trades.jsonl'
                    with open(closed_path, 'w') as cf:
                        json.dump(bot.closed_trades, cf, indent=2)
            except Exception:
                pass

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Shutdown requested (Ctrl+C). Printing final report...")
        bot.print_final_report()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', '-s', nargs='+', default=['BTC', 'ETH'], help='Symbols to follow')
    p.add_argument('--lookback-days', '-l', type=int, default=30, help='Days of history to preload')
    p.add_argument('--interval', '-i', type=int, default=30, help='Polling interval in seconds')
    p.add_argument('--capital', '-c', type=float, default=200.0, help='Initial paper capital')
    p.add_argument('--entry-threshold', type=float, default=None, help='Override bot entry threshold (lower = more trades)')
    p.add_argument('--min-confidence', type=float, default=None, help='Override bot min confidence (lower = more trades)')
    p.add_argument('--move-threshold', type=float, default=0.0, help='Require pct move (decimal) before entry, e.g. 0.01 = 1%')
    p.add_argument('--aggressive', action='store_true', help='Set aggressive defaults for more frequent entries')
    p.add_argument('--force-on-move', action='store_true', help='Force a trade when a move exceeds move-threshold (paper mode)')
    args = p.parse_args()

    main(args.symbols, args.lookback_days, args.interval, args.capital,
         entry_threshold=args.entry_threshold, min_confidence=args.min_confidence,
         move_threshold=args.move_threshold, aggressive=args.aggressive,
         force_on_move=args.force_on_move)
