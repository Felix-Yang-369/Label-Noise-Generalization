"""Read-only inference from the six existing, checksum-verified Pipelines."""
import json
import math
import threading
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.core import hash_file, hash_object, verify_cached_run
from src.pipeline import validate_environment

MODEL_NAMES = {'LR':'Logistic Regression','RF':'Random Forest','MLP':'Multilayer Perceptron'}
CONDITIONS = {'clean':0, 'random':1}

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

class InferenceService:
    def __init__(self, root):
        root = Path(root).resolve()
        validate_environment(root)
        protocol = read_json(root/'config/protocol.json')
        frozen = read_json(root/'config/FROZEN.json')
        inputs = frozen['inputs']
        if hash_object(inputs) != frozen['fingerprint']:
            raise ValueError('Frozen context integrity check failed')
        if hash_object(protocol) != inputs['protocol_hash']:
            raise ValueError('Protocol integrity check failed')
        for name, expected in inputs['code_hashes'].items():
            if hash_file(root/'src'/name) != expected:
                raise ValueError('Scientific source integrity check failed')
        for rel, expected in [('environment.json',inputs['environment_hash']),
                ('data/processed/split_manifest.json',inputs['split_hash']),
                ('config/selected_params.json',inputs['selected_params_hash'])]:
            if hash_file(root/rel) != expected:
                raise ValueError('Frozen input integrity check failed: '+rel)
        manifest = read_json(root/'data/processed/split_manifest.json')
        self.features = protocol['dataset']['features']
        self.classes = protocol['dataset']['classes']
        selected = read_json(root/'config/selected_params.json')['selected']
        self.fingerprint = frozen['fingerprint']
        self.models = {}
        self.metadata = {}
        self.lock = threading.Lock()
        for name in MODEL_NAMES:
            for condition, p in CONDITIONS.items():
                run_id = f'{name}_p{p*100:03}_s17'
                directory = root/'artifacts/runs'/run_id
                meta = read_json(directory/'meta.json')
                # Check the independently sealed metadata before trusting its model hash.
                fp = hash_object({'freeze':self.fingerprint,'run_id':run_id,'noise_hash':meta['noise_hash']})
                if not verify_cached_run(directory,fp):
                    raise ValueError('Incomplete model artifact: '+run_id)
                if any(meta.get(key)!=value for key,value in {
                    'run_id':run_id,'model':name,'p':p,'model_seed':17,'noise_seed':17,
                    'phase':'main','freeze_fingerprint':self.fingerprint,
                    'params':selected[name]['params']}.items()):
                    raise ValueError('Model metadata integrity check failed: '+run_id)
                model = joblib.load(directory/'model.joblib')
                if model.n_features_in_ != len(self.features) or not np.array_equal(
                        np.sort(model.classes_),np.arange(len(self.classes))):
                    raise ValueError('Model feature/class schema differs: '+run_id)
                self.models[name,condition] = model
                self.metadata[name,condition] = meta
        frames = {}
        # No holdout files or test metrics are read by the inference demo.
        for split in ['train','validation']:
            path = root/f'data/processed/{split}.csv'
            if hash_file(path)!=manifest['file_hashes'][path.name]:
                raise ValueError('Demo data integrity check failed: '+split)
            frames[split] = pd.read_csv(path,float_precision='round_trip')
        self.ranges = {name:(float(frames['train'][name].min()),float(frames['train'][name].max()))
                       for name in self.features}
        self.examples = []
        for class_name in self.classes:
            row = frames['validation'].query('Class == @class_name').sort_values('row_id').iloc[0]
            self.examples.append({'row_id':int(row.row_id),'class':class_name,'source':'validation',
                                  'features':{name:float(row[name]) for name in self.features}})

    def bootstrap(self):
        return {'features':[{'name':name,'min':self.ranges[name][0],'max':self.ranges[name][1]}
                            for name in self.features], 'examples':self.examples,
                'classes':self.classes, 'models':MODEL_NAMES, 'seed':17,
                'freeze_fingerprint':self.fingerprint}

    def predict(self, payload):
        if not isinstance(payload,dict):
            raise ValueError('Request must be a JSON object.')
        name, condition = payload.get('model'), payload.get('condition')
        if not isinstance(name,str) or name not in MODEL_NAMES:
            raise ValueError('Select LR, RF or MLP.')
        if not isinstance(condition,str) or condition not in (*CONDITIONS,'compare'):
            raise ValueError('Select clean, random or compare.')
        features = payload.get('features')
        if not isinstance(features,dict):
            raise ValueError('Features must be a JSON object with all 16 numeric fields.')
        missing, extra = set(self.features)-features.keys(), features.keys()-set(self.features)
        if missing or extra:
            raise ValueError('Feature fields differ. Missing: '+', '.join(sorted(missing))+
                             '; unexpected: '+', '.join(sorted(extra)))
        values, notices = [], []
        for feature in self.features:
            value = features[feature]
            try:
                if isinstance(value,bool) or not isinstance(value,(str,int,float)):
                    raise ValueError()
                value = float(value)
                if not math.isfinite(value) or abs(value)>1e12:
                    raise ValueError()
            except (ValueError,OverflowError):
                raise ValueError(feature+': enter a finite number with magnitude at most 1e12.') from None
            values.append(value)
            lo, hi = self.ranges[feature]
            if value<lo or value>hi:
                notices.append(f'{feature} is outside the training range [{lo:.6g}, {hi:.6g}]. '
                               'The value was used without clipping; prediction may be unreliable.')
        x = np.array([values],dtype=float)
        predictions = []
        conditions = list(CONDITIONS) if condition=='compare' else [condition]
        # threadpool_limits changes shared runtime state; serialize these short local predictions.
        with self.lock, threadpool_limits(limits=1):
            for key in conditions:
                estimator, meta = self.models[name,key], self.metadata[name,key]
                probabilities = estimator.predict_proba(x)[0]
                prediction = int(estimator.predict(x)[0])
                if not np.isfinite(probabilities).all():
                    raise ValueError('Input produced non-finite probabilities; use values in the training range.')
                predictions.append({'class':self.classes[prediction], 'condition':key,
                    'label_p':CONDITIONS[key], 'run_id':meta['run_id'], 'model':name,'seed':17,
                    'probabilities':[{'class':self.classes[int(label)],'probability':float(probability)}
                        for label,probability in zip(estimator.classes_,probabilities,strict=True)],
                    'training_label_accuracy':meta['train_noisy']['accuracy'],
                    'validation_macro_f1':meta['validation']['macro_f1'], 'params':meta['params'],
                    'iterations':meta['iterations'],'stop_reason':meta['stop_reason'],
                    'model_sha256':meta['artifact_hashes']['model.joblib']})
        return {'predictions':predictions,'range_warnings':notices,
                'probability_note':'Raw, uncalibrated model probabilities; not calibrated confidence.',
                'freeze_fingerprint':self.fingerprint}
