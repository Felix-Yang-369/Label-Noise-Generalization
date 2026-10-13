"""Scientific-contract tests: catch leakage, invalid noise, wrong metric targets."""
import importlib
import numpy as np
import pandas as pd
import pytest


def api():
    try:
        return importlib.import_module('src.core')
    except ModuleNotFoundError as exc:
        if exc.name in ('src.core', 'src'):
            pytest.fail('Scientific pipeline API is not implemented yet')
        raise


def fixture_frame():
    return pd.DataFrame({'a':np.arange(210,dtype=float),
                         'b':np.arange(210,dtype=float)**2,
                         'Class':np.repeat(list('ABCDEFG'),30)})


def test_exact_duplicates_removed_before_split_keep_original_identity():
    m=api()
    df=fixture_frame()
    df=pd.concat([df,df.iloc[[0,1]]],ignore_index=True)
    cleaned,splits,audit=m.prepare_frame(df,['a','b'],list('ABCDEFG'))
    assert len(cleaned)==210
    assert audit['exact_duplicate_rows']==2
    assert cleaned.iloc[0].row_id==2
    assert {k:len(v) for k,v in splits.items()}=={'train':126,'validation':42,'test':42}
    identities=[set(v.row_id) for v in splits.values()]
    assert not (identities[0]&identities[1] or identities[0]&identities[2] or identities[1]&identities[2])
    assert set().union(*identities)==set(range(2,212))
    assert all(set(v.Class)==set('ABCDEFG') for v in splits.values())


def test_conflicting_labels_for_identical_features_are_rejected():
    m=api();df=fixture_frame()
    conflict=df.iloc[[0]].copy();conflict['Class']='G'
    with pytest.raises(ValueError,match='conflict'):
        m.prepare_frame(pd.concat([df,conflict]),['a','b'],list('ABCDEFG'))


def test_missing_or_nonfinite_feature_is_rejected():
    m=api();df=fixture_frame();df.loc[0,'a']=np.inf
    with pytest.raises(ValueError,match='finite'):
        m.prepare_frame(df,['a','b'],list('ABCDEFG'))


def test_noise_zero_preserves_labels_and_full_noise_allows_matches():
    m=api();y=np.arange(1001)%7
    zero=m.noise_labels(y,0,17,7)
    full=m.noise_labels(y,1,17,7)
    assert np.array_equal(zero['noisy'],y)
    assert not zero['selected'].any()
    assert full['selected'].all()
    assert 80 < np.sum(full['noisy']==y) < 210
    counts=np.bincount(full['noisy'],minlength=7)
    assert np.all((counts>90)&(counts<200))
    assert np.array_equal(full['noisy'],m.noise_labels(y,1,17,7)['noisy'])


def test_noise_is_nested_and_invalid_probability_is_rejected():
    m=api();y=np.arange(1001)%7
    low=m.noise_labels(y,.25,29,7);high=m.noise_labels(y,.75,29,7)
    assert np.all(high['selected'][low['selected']])
    assert np.array_equal(low['noisy'][low['selected']],high['noisy'][low['selected']])
    with pytest.raises(ValueError):m.noise_labels(y,-.1,17,7)


def test_macro_f1_is_unweighted_class_average():
    m=api();result=m.scores([0,0,1,1,2,2],[0,1,1,1,2,0],[0,1,2])
    assert result['macro_f1']==pytest.approx((.5+.8+2/3)/3)
    assert result['accuracy']==pytest.approx(4/6)
    assert result['balanced_accuracy']==pytest.approx(2/3)


def test_hyperparameter_ties_choose_simpler_candidate_not_first_row():
    m=api()
    rows=[{'params':{'C':10},'validation_macro_f1':.9},
          {'params':{'C':.1},'validation_macro_f1':.8995},
          {'params':{'C':1},'validation_macro_f1':.8998}]
    assert m.select_candidate('LR',rows,.001)['params']=={'C':.1}
    rows=[{'params':{'max_depth':None,'min_samples_leaf':1},'validation_macro_f1':.9},
          {'params':{'max_depth':8,'min_samples_leaf':5},'validation_macro_f1':.8997}]
    assert m.select_candidate('RF',rows,.001)['params']=={'max_depth':8,'min_samples_leaf':5}
    rows=[{'params':{'alpha':0},'validation_macro_f1':.9},
          {'params':{'alpha':.01},'validation_macro_f1':.8995}]
    assert m.select_candidate('MLP',rows,.001)['params']=={'alpha':.01}


def test_saved_pipeline_scaler_uses_training_only_and_reloads(tmp_path):
    m=api()
    spec={'fixed':{'solver':'lbfgs','max_iter':1000},'initial_default':{'C':1}}
    estimator=m.make_model('LR',spec,{'C':1},17)
    X=np.array([[0.,1.],[1.,2.],[2.,3.],[3.,4.],[4.,5.],[5.,6.]])
    y=np.array([0,0,0,1,1,1])
    estimator.fit(X,y)
    assert np.allclose(estimator.named_steps['scale'].mean_,[2.5,3.5])
    import joblib
    path=tmp_path/'model.joblib';joblib.dump(estimator,path)
    loaded=joblib.load(path)
    assert np.array_equal(loaded.predict([[1000.,2000.]]),estimator.predict([[1000.,2000.]]))
    assert np.allclose(loaded.named_steps['scale'].mean_,[2.5,3.5])


def test_resume_refuses_changed_config_or_corrupt_artifact(tmp_path):
    import json,hashlib
    m=api();artifact=tmp_path/'model.joblib';artifact.write_bytes(b'known model bytes')
    meta={'status':'complete','fingerprint':'fixed',
          'artifact_hashes':{'model.joblib':hashlib.sha256(b'known model bytes').hexdigest()}}
    (tmp_path/'meta.json').write_text(json.dumps(meta))
    (tmp_path/'meta.sha256').write_text(hashlib.sha256((tmp_path/'meta.json').read_bytes()).hexdigest())
    assert m.verify_cached_run(tmp_path,'fixed') is True
    with pytest.raises(ValueError,match='fingerprint'):
        m.verify_cached_run(tmp_path,'changed')
    artifact.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='integrity'):
        m.verify_cached_run(tmp_path,'fixed')


def test_macro_f1_counts_absent_classes_as_zero_when_predeclared():
    m=api();result=m.scores([0,0],[0,0],[0,1,2])
    assert result['macro_f1']==pytest.approx(1/3)


def test_resume_rejects_mutated_cached_scores_even_with_valid_model(tmp_path):
    import json,hashlib
    m=api();(tmp_path/'model.joblib').write_bytes(b'unchanged model')
    meta={'status':'complete','fingerprint':'fixed','validation':{'macro_f1':.7},
          'artifact_hashes':{'model.joblib':hashlib.sha256(b'unchanged model').hexdigest()}}
    path=tmp_path/'meta.json';path.write_text(json.dumps(meta))
    (tmp_path/'meta.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    meta['validation']['macro_f1']=.99;path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='metadata integrity'):
        m.verify_cached_run(tmp_path,'fixed')
