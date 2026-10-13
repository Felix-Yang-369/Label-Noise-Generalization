import importlib
import json
import numpy as np
import pytest
from src.core import make_model


def runner():
    try:return importlib.import_module('src.pipeline')
    except ModuleNotFoundError as exc:
        if exc.name=='src.pipeline':pytest.fail('Execution pipeline is not implemented yet')
        raise


def test_fit_persists_both_training_targets_and_reloadable_predictions(tmp_path):
    m=runner()
    X=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.],[4.,4.],[5.,5.]])
    clean=np.array([0,0,0,1,1,1]);noisy=1-clean
    est=make_model('LR',{'fixed':{'max_iter':1000,'solver':'lbfgs'}},{'C':1},17)
    meta=m.fit_and_save(tmp_path,est,X,noisy,clean,X,clean,
                        np.arange(2,8),np.arange(20,26),[0,1],'fixed',{'run_id':'toy'})
    assert meta['status']=='complete'
    assert meta['train_noisy']['accuracy'] > meta['train_clean']['accuracy']
    assert not any('test' in k for k in meta)
    assert (tmp_path/'model.joblib').is_file()
    import pandas as pd,joblib
    rows=pd.read_csv(tmp_path/'train_predictions.csv')
    assert rows.row_id.tolist()==list(range(2,8))
    assert rows.clean_label.tolist()==clean.tolist()
    assert rows.fit_label.tolist()==noisy.tolist()
    assert np.array_equal(rows.prediction,joblib.load(tmp_path/'model.joblib').predict(X))
    assert m.fit_and_save(tmp_path,est,X,noisy,clean,X,clean,np.arange(2,8),
                          np.arange(20,26),[0,1],'fixed',{'run_id':'toy'})==meta


def test_final_evaluation_is_blocked_until_every_predeclared_fit_completes(tmp_path):
    m=runner()
    protocol={'metric_labels':list(range(7))}
    matrix=[{'run_id':'missing'}]
    with pytest.raises(ValueError,match='complete'):
        m.final_evaluate(tmp_path,protocol,matrix)
    assert not (tmp_path/'results'/'test_metrics.csv').exists()


def test_completed_subset_cannot_open_final_holdout(tmp_path):
    m=runner();(tmp_path/'config').mkdir()
    declared=[{'run_id':'first'},{'run_id':'second'}]
    (tmp_path/'config'/'main_matrix.json').write_text(json.dumps(declared))
    p=tmp_path/'artifacts'/'runs'/'first';p.mkdir(parents=True)
    (p/'meta.json').write_text(json.dumps({'status':'complete'}))
    with pytest.raises(ValueError,match='matrix'):
        m.final_evaluate(tmp_path,{},[declared[0]])
    assert not (tmp_path/'results'/'test_metrics.csv').exists()


def test_live_environment_mismatch_is_rejected_before_resume(tmp_path):
    m=runner()
    (tmp_path/'environment.json').write_text(json.dumps({'python':'different interpreter'}))
    with pytest.raises(ValueError,match='environment'):
        m.validate_environment(tmp_path)
