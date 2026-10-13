"""Independent post-experiment audit; holdout read only after final receipt exists."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_metrics(truth,pred):
    cm=np.zeros((7,7),dtype=int)
    np.add.at(cm,(truth,pred),1)
    tp=cm.diagonal();denom=cm.sum(axis=0)+cm.sum(axis=1)
    f1=np.divide(2*tp,denom,out=np.zeros(7,dtype=float),where=denom!=0)
    recall=np.divide(tp,cm.sum(axis=1),out=np.zeros(7,dtype=float),where=cm.sum(axis=1)!=0)
    return {'accuracy':tp.sum()/cm.sum(),'macro_f1':f1.mean(),'balanced_accuracy':recall.mean()}


@pytest.mark.skipif(not (ROOT/'results'/'FINAL_EVALUATION.json').exists(),reason='Final holdout is still sealed')
def test_actual_45_runs_against_original_data_and_independent_metrics():
    receipt=json.loads((ROOT/'results'/'FINAL_EVALUATION.json').read_text())
    matrix=json.loads((ROOT/'config'/'main_matrix.json').read_text())
    protocol=json.loads((ROOT/'config'/'protocol.json').read_text())
    frozen=json.loads((ROOT/'config'/'FROZEN.json').read_text())
    selected=json.loads((ROOT/'config'/'selected_params.json').read_text())
    features=protocol['dataset']['features'];classes=protocol['dataset']['classes']
    source=pd.read_excel(ROOT/'data'/'raw'/'Dry_Bean_Dataset.xlsx')
    sets={name:pd.read_csv(ROOT/'data'/'processed'/f'{name}.csv',float_precision='round_trip') for name in ['train','validation','test']}
    assert [len(sets[n]) for n in ['train','validation','test']]==[8125,2709,2709]
    combined=pd.concat(sets.values())
    assert combined.row_id.nunique()==13543
    assert not combined.duplicated(subset=features).any()
    for split in sets.values():
        assert np.array_equal(split[features].to_numpy(),source.iloc[split.row_id.to_numpy()-2][features].to_numpy())
        assert np.array_equal(split.Class,source.iloc[split.row_id.to_numpy()-2].Class)
    metrics=pd.read_csv(ROOT/'results'/'test_metrics.csv')
    assert len(matrix)==len(metrics)==receipt['run_count']==45
    assert set(metrics.run_id)=={r['run_id'] for r in matrix}
    for name,expected in receipt['file_hashes'].items():assert sha(ROOT/'results'/name)==expected
    mapping={c:i for i,c in enumerate(classes)}
    ytest=sets['test'].Class.map(mapping).to_numpy();Xtest=sets['test'][features].to_numpy(dtype=float)
    train_mean=sets['train'][features].to_numpy(dtype=float).mean(axis=0)
    ytrain=sets['train'].Class.map(mapping).to_numpy()
    for run in matrix:
        directory=ROOT/'artifacts'/'runs'/run['run_id']
        meta=json.loads((directory/'meta.json').read_text())
        assert meta['status']=='complete' and meta['freeze_fingerprint']==frozen['fingerprint']
        assert sha(directory/'meta.json')==(directory/'meta.sha256').read_text().strip()
        assert meta['started_utc']>=frozen['frozen_utc'] and meta['completed_utc']<=receipt['evaluated_utc']
        for name,expected in meta['artifact_hashes'].items():assert sha(directory/name)==expected
        train=pd.read_csv(directory/'train_predictions.csv')
        val=pd.read_csv(directory/'validation_predictions.csv')
        test=pd.read_csv(ROOT/'results'/'test_predictions'/f'{run["run_id"]}.csv')
        assert np.array_equal(train.row_id,sets['train'].row_id)
        assert np.array_equal(val.row_id,sets['validation'].row_id)
        assert np.array_equal(test.row_id,sets['test'].row_id)
        assert np.array_equal(train.clean_label,ytrain)
        assert np.array_equal(test.clean_label,ytest)
        noise=pd.read_csv(ROOT/'data'/'noise'/f'p{int(run["p"]*100):03d}_s{run["noise_seed"]}.csv')
        assert np.array_equal(train.fit_label,noise.noisy_label)
        assert meta['changed_fraction']==pytest.approx(np.mean(train.fit_label!=train.clean_label))
        for truth,recorded in [(train.fit_label.to_numpy(),meta['train_noisy']),
                               (train.clean_label.to_numpy(),meta['train_clean'])]:
            actual=independent_metrics(truth,train.prediction.to_numpy())
            for name,v in actual.items():assert recorded[name]==pytest.approx(v,abs=1e-12)
        actual=independent_metrics(ytest,test.prediction.to_numpy())
        row=metrics[metrics.run_id==run['run_id']].iloc[0]
        for name,v in actual.items():assert row['test_'+name]==pytest.approx(v,abs=1e-12)
        assert np.allclose(test[[f'prob_{i}' for i in range(7)]].sum(axis=1),1,atol=1e-10)
        assert row.clean_target_gap==pytest.approx(meta['train_clean']['accuracy']-actual['accuracy'])
        if run['model_seed']==17 and run['p'] in [0,1]:
            est=joblib.load(directory/'model.joblib')
            with threadpool_limits(limits=1):
                assert np.array_equal(est.predict(Xtest),test.prediction)
                assert np.allclose(est.predict_proba(Xtest),test[[f'prob_{i}' for i in range(7)]].to_numpy(),atol=1e-12)
            for name,v in selected['selected'][run['model']]['params'].items():
                assert est.named_steps['model'].get_params()[name]==v
            if run['model']!='RF':assert np.allclose(est.named_steps['scale'].mean_,train_mean,rtol=1e-12,atol=1e-10)
    expected=metrics.groupby(['model','p'])['test_macro_f1'].agg(['mean','std'])
    summary=pd.read_csv(ROOT/'results'/'model_noise_summary.csv').set_index(['model','p'])
    assert np.allclose(summary.test_macro_f1_mean,expected['mean'])
    assert np.allclose(summary.test_macro_f1_std,expected['std'])
