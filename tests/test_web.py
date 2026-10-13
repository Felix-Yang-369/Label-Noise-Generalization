"""Breaks caught: wrong artifact/feature/class order, invalid input, forged results."""
import importlib
import importlib.util
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
FEATURES = json.loads((ROOT/'config/protocol.json').read_text())['dataset']['features']

def factory(root=ROOT):
    assert importlib.util.find_spec('app') is not None, 'Real inference app has not been implemented'
    return importlib.import_module('app').create_app(root)

@pytest.fixture(scope='module')
def client():
    app = factory()
    app.config['TESTING'] = True
    return app.test_client()

def values():
    row = pd.read_csv(ROOT/'data/processed/validation.csv', float_precision='round_trip').iloc[0]
    return {name:float(row[name]) for name in FEATURES}

@pytest.mark.parametrize('model', ['LR', 'RF', 'MLP'])
@pytest.mark.parametrize('condition,p', [('clean', 0), ('random', 1)])
def test_http_matches_independently_loaded_pipeline(client, model, condition, p):
    features = values()
    # Deliberately reverse JSON order: a server that trusts request order must fail.
    response = client.post('/api/predict', json={'model':model, 'condition':condition,
                           'features':dict(reversed(list(features.items())))})
    assert response.status_code == 200
    result = response.json['predictions'][0]
    saved = joblib.load(ROOT/f'artifacts/runs/{model}_p{p*100:03}_s17/model.joblib')
    x = np.array([[features[name] for name in FEATURES]])
    expected = saved.predict_proba(x)[0]
    assert result['class'] == CLASSES[int(saved.predict(x)[0])]
    assert [v['class'] for v in result['probabilities']] == [CLASSES[int(c)] for c in saved.classes_]
    np.testing.assert_allclose([v['probability'] for v in result['probabilities']], expected, rtol=0, atol=1e-12)
    assert result['run_id'] == f'{model}_p{p*100:03}_s17'
    assert result['label_p'] == p
    assert 'uncalibrated' in response.json['probability_note'].lower()

def test_compare_uses_identical_features_for_both_models(client):
    payload = {'model':'RF','condition':'compare','features':values()}
    both = client.post('/api/predict',json=payload)
    assert both.status_code == 200
    for actual,condition in zip(both.json['predictions'], ['clean','random'], strict=True):
        one = client.post('/api/predict', json={**payload,'condition':condition})
        assert actual == one.json['predictions'][0]

@pytest.mark.parametrize('bad', [None, '', 'NaN', 'Infinity', float('nan'), float('inf'), True, [], {}, 1e300, 10**100, 'not a number'])
def test_invalid_numeric_values_never_return_prediction(client,bad):
    features=values();features['Area']=bad
    response=client.post('/api/predict',json={'model':'LR','condition':'clean','features':features})
    assert response.status_code == 400
    assert 'predictions' not in response.json
    assert 'Area' in response.json['error']

@pytest.mark.parametrize('change', ['missing','extra','unknown_model','unknown_condition','features_array','body_array'])
def test_rejects_invalid_request_schema(client,change):
    payload={'model':'LR','condition':'clean','features':values()}
    if change=='missing':payload['features'].pop('Perimeter')
    elif change=='extra':payload['features']['row_id']=1
    elif change=='unknown_model':payload['model']='../../model'
    elif change=='unknown_condition':payload['condition']='p025'
    elif change=='features_array':payload['features']=list(payload['features'].values())
    elif change=='body_array':payload=[]
    response=client.post('/api/predict',json=payload)
    assert response.status_code==400 and 'predictions' not in response.json

def test_malformed_or_non_json_request_is_explained(client):
    for content,kind in [('oops','application/json'),('Area=1','text/plain')]:
        response=client.post('/api/predict',data=content,content_type=kind)
        assert response.status_code==400 and response.is_json

def test_outside_training_range_warns_without_clipping(client):
    features=values();features['Area']=-1
    response=client.post('/api/predict',json={'model':'LR','condition':'clean','features':features})
    assert response.status_code==200
    assert any('Area' in v for v in response.json['range_warnings'])
    model=joblib.load(ROOT/'artifacts/runs/LR_p000_s17/model.joblib')
    assert response.json['predictions'][0]['class']==CLASSES[int(model.predict([[features[n] for n in FEATURES]])[0])]

def test_examples_use_validation_rows_chosen_without_predictions(client):
    data=client.get('/api/bootstrap').json
    assert [v['name'] for v in data['features']]==FEATURES
    val=pd.read_csv(ROOT/'data/processed/validation.csv',float_precision='round_trip')
    train=pd.read_csv(ROOT/'data/processed/train.csv',float_precision='round_trip')
    assert len(data['examples'])==7
    for example in data['examples']:
        rows=val[val.Class==example['class']]
        expected=rows.sort_values('row_id').iloc[0]
        assert example['row_id']==int(expected.row_id)
        assert example['source']=='validation'
        assert example['features']=={n:float(expected[n]) for n in FEATURES}
    for item in data['features']:
        assert item['min']==float(train[item['name']].min())
        assert item['max']==float(train[item['name']].max())

def test_corrupt_model_refuses_startup_before_deserialization(tmp_path):
    for rel in ['config','src']:
        shutil.copytree(ROOT/rel,tmp_path/rel,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'environment.json',tmp_path/'environment.json')
    shutil.copytree(ROOT/'data/processed',tmp_path/'data/processed')
    shutil.copytree(ROOT/'artifacts/runs/LR_p000_s17',tmp_path/'artifacts/runs/LR_p000_s17')
    with (tmp_path/'artifacts/runs/LR_p000_s17/model.joblib').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='integrity|checksum'):
        factory(tmp_path)

def test_web_shell_and_assets_are_available(client):
    response=client.get('/')
    assert response.status_code==200
    for url in ['/static/demo.js','/static/style.css']:
        assert client.get(url).status_code==200
    assert client.get('/api/health').json['models_loaded']==6
    assert client.application.debug is False
