#!/usr/bin/env python3
"""Online trainer scaffold: records feature/outcome samples and can retrain offline.

This is a minimal, safe scaffold. It records samples to a JSONL dataset and
provides a retrain_offline() method that fits a simple model (RandomForest)
and saves a checkpoint. Extend features and model as needed.
"""

import json
from pathlib import Path
import joblib
import logging
from typing import Dict, Any

logger = logging.getLogger('online_trainer')


class OnlineTrainer:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.dataset_path = self.results_dir / 'trainer_dataset.jsonl'
        self.model_path = self.results_dir / 'trainer_model.joblib'
        self._buffer = []

    def record_sample(self, sample: Dict[str, Any]):
        """Record a training sample (feature dict + outcome fields)."""
        self._buffer.append(sample)

    def save_dataset(self):
        """Append buffered samples to JSONL dataset and clear buffer."""
        if not self._buffer:
            logger.info('No samples to save.')
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.dataset_path, 'a', encoding='utf-8') as f:
            for s in self._buffer:
                f.write(json.dumps(s) + '\n')

        logger.info(f'Saved {len(self._buffer)} samples to {self.dataset_path}')
        self._buffer = []

    def load_dataset(self):
        """Load dataset into memory (simple JSONL loader)."""
        if not self.dataset_path.exists():
            return []
        samples = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    continue
        return samples

    def retrain_offline(self):
        """Simple offline retrain that fits a RandomForest on pnl sign.

        This is intentionally minimal — it's a scaffold to plug into a larger
        retraining pipeline. It treats `pnl > 0` as label=1 else 0 and uses
        `entry` and `size` features as a toy example.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
        except Exception as e:
            logger.warning('scikit-learn not installed — skipping retrain. Install scikit-learn to enable retraining.')
            return

        samples = self.load_dataset()
        if len(samples) < 10:
            logger.info('Not enough samples for retrain (need >=10).')
            return

        X = []
        y = []
        for s in samples:
            entry = s.get('entry', 0) or 0
            size = s.get('size', 0) or 0
            pnl = s.get('pnl', 0) or 0
            X.append([float(entry), float(size)])
            y.append(1 if float(pnl) > 0 else 0)

        X = np.array(X)
        y = np.array(y)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)

        joblib.dump(clf, self.model_path)
        logger.info(f'Retrained model saved to {self.model_path}')

    def load_model(self):
        if self.model_path.exists():
            try:
                return joblib.load(self.model_path)
            except Exception:
                return None
        return None


if __name__ == '__main__':
    print('OnlineTrainer module — import from scripts.live_simulator')
