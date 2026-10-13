"""Shared scientific contracts for cleaning, perturbation and estimation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def hash_object(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def prepare_frame(frame,features,classes,test_seed=171,validation_seed=172):
    required=features+['Class']
    if any(c not in frame for c in required):raise ValueError('Missing required columns')
    df=frame[required].copy()
    df.insert(0,'row_id',np.arange(len(df),dtype=np.int64)+2)
    df[features]=df[features].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(df[features].to_numpy(dtype=float)).all():
        raise ValueError('All feature values must be finite')
    if df.Class.isna().any() or set(df.Class)!=set(classes):
        raise ValueError('Dataset must contain exactly the declared classes')
    duplicate_count=int(df.duplicated(subset=required).sum())
    if (df.groupby(features,dropna=False).Class.nunique()>1).any():
        raise ValueError('Identical features have conflicting labels')
    df=df.drop_duplicates(subset=required,keep='first').reset_index(drop=True)
    remaining,test=train_test_split(df,test_size=.20,stratify=df.Class,random_state=test_seed)
    train,validation=train_test_split(remaining,test_size=.25,stratify=remaining.Class,random_state=validation_seed)
    splits={'train':train.reset_index(drop=True),'validation':validation.reset_index(drop=True),'test':test.reset_index(drop=True)}
    identity_sets=[set(v.row_id) for v in splits.values()]
    if any(identity_sets[i]&identity_sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Row identity leakage')
    if sum(map(len,identity_sets))!=len(df):raise ValueError('Split does not cover all rows')
    if any(set(v.Class)!=set(classes) for v in splits.values()):raise ValueError('A split misses a class')
    # Clean data contain one row per feature vector, so disjoint row IDs imply disjoint features.
    audit={'original_rows':len(frame),'exact_duplicate_rows':duplicate_count,'cleaned_rows':len(df),
           'missing_cells':0,'nonfinite_cells':0,'conflicting_feature_groups':0,
           'class_counts_original':{str(k):int(v) for k,v in frame.Class.value_counts().sort_index().items()},
           'class_counts_cleaned':{str(k):int(v) for k,v in df.Class.value_counts().sort_index().items()}}
    return df,splits,audit


def noise_labels(clean,p,seed,n_classes):
    y=np.asarray(clean,dtype=np.int64)
    if not 0<=p<=1 or n_classes<2 or y.ndim!=1 or np.any((y<0)|(y>=n_classes)):
        raise ValueError('Invalid noise probability or label domain')
    rng=np.random.default_rng(seed)
    selected=rng.random(len(y))<p
    replacement=rng.integers(0,n_classes,size=len(y))
    noisy=np.where(selected,replacement,y)
    return {'noisy':noisy,'selected':selected,'changed':noisy!=y,'replacement':replacement}


def scores(truth,prediction,labels):
    return {'macro_f1':float(f1_score(truth,prediction,labels=labels,average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(truth,prediction)),
            'balanced_accuracy':float(balanced_accuracy_score(truth,prediction))}


def select_candidate(model,rows,tolerance):
    if not rows:raise ValueError('No tuning candidates')
    best=max(r['validation_macro_f1'] for r in rows)
    near=[r for r in rows if best-r['validation_macro_f1']<=tolerance+1e-12]
    def key(row):
        p=row['params']
        if model=='LR':return (p['C'],)
        if model=='RF':return (float('inf') if p['max_depth'] is None else p['max_depth'],-p['min_samples_leaf'])
        if model=='MLP':return (-p['alpha'],)
        raise ValueError('Unknown model')
    return min(near,key=key)


def make_model(model,spec,params,seed):
    settings={**spec['fixed'],**params,'random_state':seed}
    if 'hidden_layer_sizes' in settings:settings['hidden_layer_sizes']=tuple(settings['hidden_layer_sizes'])
    if model=='LR':estimator=LogisticRegression(**settings)
    elif model=='RF':estimator=RandomForestClassifier(**settings)
    elif model=='MLP':estimator=MLPClassifier(**settings)
    else:raise ValueError('Unknown model')
    steps=[] if model=='RF' else [('scale',StandardScaler())]
    return Pipeline(steps+[('model',estimator)])


def verify_cached_run(directory,fingerprint):
    directory=Path(directory);path=directory/'meta.json'
    if not path.exists():return False
    meta=json.loads(path.read_text(encoding='utf-8'))
    if meta.get('fingerprint')!=fingerprint:raise ValueError('Cached run fingerprint differs; use a fresh output directory')
    if meta.get('status')!='complete':return False
    seal=directory/'meta.sha256'
    if not seal.is_file() or seal.read_text(encoding='utf-8').strip()!=hash_file(path):
        raise ValueError('Cached run metadata integrity check failed')
    if not meta.get('artifact_hashes'):raise ValueError('Cached run integrity metadata missing')
    for name,expected in meta['artifact_hashes'].items():
        path=directory/name
        if not path.is_file() or hash_file(path)!=expected:raise ValueError('Cached run integrity check failed: '+name)
    return True
