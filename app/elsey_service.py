"""Inference API for the team's selected 70/15/15 notebook experiment."""
import hashlib
import json
import math
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .service import MODEL_NAMES, CONDITIONS


class ElseyInferenceService:
    def __init__(self, root):
        root = Path(root).resolve()
        bundle = root / 'integration/elsey/demo_artifacts'
        manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
        self.features = manifest['features']
        self.classes = manifest['classes']
        self.ranges = manifest['ranges']
        self.examples = manifest['examples']
        self.fingerprint = manifest['raw_sha256']
        self.models, self.metadata = {}, {}
        self.lock = threading.Lock()
        for name in MODEL_NAMES:
            for condition in CONDITIONS:
                meta = manifest['models'][f'{name}_{condition}']
                path = bundle / meta['file']
                if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                    raise ValueError('Selected notebook model hash mismatch: ' + str(path))
                estimator = joblib.load(path)
                if estimator.n_features_in_ != len(self.features) or list(estimator.classes_) != self.classes:
                    raise ValueError('Selected notebook model input/output schema mismatch')
                self.models[name, condition] = estimator
                self.metadata[name, condition] = meta

    def bootstrap(self):
        return {
            'features': [{'name': feature, 'min': self.ranges[feature][0],
                          'max': self.ranges[feature][1]} for feature in self.features],
            'examples': self.examples, 'classes': self.classes,
            'models': MODEL_NAMES, 'seed': 17, 'freeze_fingerprint': self.fingerprint,
            'experiment': 'Group 8 selected notebook (70/15/15 split)',
        }

    def predict(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Request must be a JSON object.')
        name, condition = payload.get('model'), payload.get('condition')
        if name not in MODEL_NAMES:
            raise ValueError('Select LR, RF or MLP.')
        if condition not in (*CONDITIONS, 'compare'):
            raise ValueError('Select clean, random or compare.')
        features = payload.get('features')
        if not isinstance(features, dict):
            raise ValueError('Features must be a JSON object with all 16 numeric fields.')
        missing = set(self.features) - features.keys()
        extra = features.keys() - set(self.features)
        if missing or extra:
            raise ValueError(f'Feature fields differ. Missing: {sorted(missing)}; unexpected: {sorted(extra)}')
        values, notices = [], []
        for feature in self.features:
            raw = features[feature]
            try:
                if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                    raise ValueError()
                value = float(raw)
                if not math.isfinite(value) or abs(value) > 1e12:
                    raise ValueError()
            except (ValueError, OverflowError):
                raise ValueError(feature + ': enter a finite number with magnitude at most 1e12.') from None
            values.append(value)
            lo, hi = self.ranges[feature]
            if value < lo or value > hi:
                notices.append(feature + ' is outside the training range; prediction may be unreliable.')
        x = pd.DataFrame([values], columns=self.features)
        predictions = []
        with self.lock, threadpool_limits(limits=1):
            for key in (list(CONDITIONS) if condition == 'compare' else [condition]):
                estimator, meta = self.models[name, key], self.metadata[name, key]
                probabilities = estimator.predict_proba(x)[0]
                prediction = estimator.predict(x)[0]
                steps = getattr(estimator[-1], 'n_iter_', None)
                iterations = int(np.asarray(steps).max()) if steps is not None else None
                predictions.append({
                    'class': str(prediction), 'condition': key, 'label_p': CONDITIONS[key],
                    'run_id': f'{name}_p{CONDITIONS[key]*100:03}_s17',
                    'model': name, 'seed': 17,
                    'probabilities': [{'class': str(label), 'probability': float(probability)}
                                      for label, probability in zip(estimator.classes_, probabilities)],
                    'training_label_accuracy': meta['train_noisy_accuracy'],
                    'validation_macro_f1': meta['validation_macro_f1'],
                    'params': meta['params'], 'iterations': iterations,
                    'stop_reason': 'Notebook iteration budget',
                    'model_sha256': meta['sha256'],
                })
        return {'predictions': predictions, 'range_warnings': notices,
                'probability_note': 'Raw, uncalibrated model probabilities; not calibrated confidence.',
                'freeze_fingerprint': self.fingerprint}
