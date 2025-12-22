#!/usr/bin/env python3
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from models.ULTIMATE_10_10_TRADING_BOT import ULTIMATE_10_10_TradingBot

def run_once(seed:int, out_dir:Path):
    bot = ULTIMATE_10_10_TradingBot(initial_capital=200.0, use_real_data=True, lookback_days=30,
                                     entry_threshold=0.10, min_confidence=0.0, max_positions=3,
                                     random_seed=seed)
    bot.run_simulation()
    # read result file
    res_file = Path('ultimate_10_10_results.json')
    if res_file.exists():
        return json.load(open(res_file))
    return None

if __name__ == '__main__':
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    runs = 100
    summaries = []
    for seed in range(runs):
        res = run_once(seed, out_dir)
        if res is None:
            continue
        stats = res.get('stats', {})
        summaries.append(stats)
    # aggregate
    finals = [s.get('final', 0) for s in summaries]
    pnls = [s.get('pnl', 0) for s in summaries]
    trades = [s.get('trades', 0) for s in summaries]
    win_rates = [s.get('wr', 0) for s in summaries if s.get('wr') is not None]

    report = {
        'runs': len(summaries),
        'final_capital_mean': round(statistics.mean(finals),2) if finals else None,
        'final_capital_median': round(statistics.median(finals),2) if finals else None,
        'final_capital_std': round(statistics.pstdev(finals),2) if finals else None,
        'pnl_mean': round(statistics.mean(pnls),2) if pnls else None,
        'pnl_median': round(statistics.median(pnls),2) if pnls else None,
        'trades_mean': round(statistics.mean(trades),2) if trades else None,
        'win_rate_mean': round(statistics.mean(win_rates),2) if win_rates else None,
        'detailed': summaries,
    }
    with open(out_dir / 'monte_carlo_results.json','w') as f:
        json.dump(report, f, indent=2)
    print('Monte Carlo complete. Summary:')
    print(json.dumps({k:v for k,v in report.items() if k!='detailed'}, indent=2))
