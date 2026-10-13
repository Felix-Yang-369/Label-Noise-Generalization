"""Staged, resumable experiment execution with a sealed final holdout."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import contextlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report,confusion_matrix
from threadpoolctl import threadpool_limits,threadpool_info
from .core import (hash_file,hash_object,write_json,prepare_frame,noise_labels,
                   make_model,scores,select_candidate,verify_cached_run)

ROOT=Path(__file__).resolve().parents[1]


def now():return datetime.now(timezone.utc).isoformat()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load_split(root,name,protocol):
    df=pd.read_csv(root/'data'/'processed'/f'{name}.csv',float_precision='round_trip')
    mapping={c:i for i,c in enumerate(protocol['dataset']['classes'])}
    return df,df[protocol['dataset']['features']].to_numpy(dtype=float),df.Class.map(mapping).to_numpy(dtype=np.int64)


def source_hashes(root):
    return {name:hash_file(root/'src'/name) for name in ('core.py','pipeline.py')}


def base_context(root,protocol):
    validate_environment(root)
    manifest=read_json(root/'data'/'processed'/'split_manifest.json')
    for name,expected in manifest['file_hashes'].items():
        if hash_file(root/'data'/'processed'/name)!=expected:raise ValueError('Processed data integrity failed: '+name)
    return {'protocol_hash':hash_object(protocol),'split_hash':hash_file(root/'data'/'processed'/'split_manifest.json'),
            'data_hash':manifest['raw_archive_sha256'],'code_hashes':source_hashes(root),
            'environment_hash':hash_file(root/'environment.json'),'blas_threads':1}


def fit_and_save(directory,estimator,X,yfit,yclean,Xval,yval,train_ids,val_ids,labels,fingerprint,details):
    directory=Path(directory)
    if verify_cached_run(directory,fingerprint):return read_json(directory/'meta.json')
    directory.mkdir(parents=True,exist_ok=True)
    meta={**details,'status':'running','fingerprint':fingerprint,'started_utc':now()}
    write_json(directory/'meta.json',meta)
    start=time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
            warnings.simplefilter('always')
            estimator.fit(X,yfit)
        elapsed=time.perf_counter()-start
        predtrain=estimator.predict(X);predval=estimator.predict(Xval)
        joblib.dump(estimator,directory/'model.joblib',compress=3)
        pd.DataFrame({'row_id':train_ids,'clean_label':yclean,'fit_label':yfit,'prediction':predtrain}).to_csv(directory/'train_predictions.csv',index=False)
        pd.DataFrame({'row_id':val_ids,'clean_label':yval,'prediction':predval}).to_csv(directory/'validation_predictions.csv',index=False)
        inner=estimator.named_steps['model']
        iterations=getattr(inner,'n_iter_',None)
        if hasattr(iterations,'tolist'):iterations=iterations.tolist()
        curve=[float(v) for v in getattr(inner,'loss_curve_',[])]
        write_json(directory/'training_trace.json',{'iterations':iterations,'loss_curve':curve})
        max_iter=getattr(inner,'max_iter',None)
        stop='not_applicable' if iterations is None else ('max_iter_reached' if np.max(iterations)>=max_iter else 'training_objective_or_convergence_stop')
        meta.update(status='complete',completed_utc=now(),fit_seconds=elapsed,
                    train_noisy=scores(yfit,predtrain,labels),train_clean=scores(yclean,predtrain,labels),
                    validation=scores(yval,predval,labels),iterations=iterations,stop_reason=stop,
                    classes=inner.classes_.tolist(),warnings=[{'category':w.category.__name__,'message':str(w.message)} for w in caught],
                    scaler_mean=estimator.named_steps['scale'].mean_.tolist() if 'scale' in estimator.named_steps else None)
        meta['artifact_hashes']={name:hash_file(directory/name) for name in ('model.joblib','train_predictions.csv','validation_predictions.csv','training_trace.json')}
        write_json(directory/'meta.json',meta)
        seal=directory/'meta.sha256.tmp'
        seal.write_text(hash_file(directory/'meta.json'),encoding='utf-8')
        seal.replace(directory/'meta.sha256')
        return meta
    except Exception as exc:
        meta.update(status='failed',error=repr(exc),failed_utc=now())
        write_json(directory/'meta.json',meta)
        raise


def prepare_data(root,protocol):
    original=root.parent/'10_Technical_Plan'/'data'
    raw=root/'data'/'raw';raw.mkdir(parents=True,exist_ok=True)
    archive=raw/'UCI_Dry_Bean_Original.zip'
    if not archive.exists():shutil.copy2(original/'UCI_Dry_Bean_Original.zip',archive)
    if hash_file(archive)!=protocol['dataset']['archive_sha256']:raise ValueError('Original ZIP hash mismatch')
    xlsx=raw/'Dry_Bean_Dataset.xlsx'
    if not xlsx.exists():shutil.copy2(original/'original'/'DryBeanDataset'/'Dry_Bean_Dataset.xlsx',xlsx)
    # Require copied XLSX to match the corresponding member of the verified archive.
    import zipfile,hashlib
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Original ZIP CRC failed')
        member=[n for n in z.namelist() if n.endswith('.xlsx')]
        if len(member)!=1 or hashlib.sha256(z.read(member[0])).hexdigest()!=hash_file(xlsx):
            raise ValueError('XLSX differs from verified archive')
    df=pd.read_excel(xlsx)
    cleaned,splits,audit=prepare_frame(df,protocol['dataset']['features'],protocol['dataset']['classes'],
                                     protocol['split']['first_random_state'],protocol['split']['second_random_state'])
    if len(cleaned)!=protocol['dataset']['rows_after_exact_dedup']:raise ValueError('Unexpected deduplicated sample count')
    counts={name:len(frame) for name,frame in splits.items()}
    if counts!=protocol['split']['expected_counts']:raise ValueError('Unexpected split counts')
    processed=root/'data'/'processed';processed.mkdir(parents=True,exist_ok=True)
    existing=processed/'split_manifest.json'
    if existing.exists() and (root/'config'/'FROZEN.json').exists():
        manifest=read_json(existing)
        if manifest['raw_archive_sha256']!=hash_file(archive):raise ValueError('Frozen dataset differs')
        for name,h in manifest['file_hashes'].items():
            if hash_file(processed/name)!=h:raise ValueError('Frozen split files changed')
        print('prepare: verified existing frozen split',flush=True);return
    cleaned.to_csv(processed/'cleaned.csv',index=False)
    for name,frame in splits.items():frame.to_csv(processed/f'{name}.csv',index=False)
    manifest={'created_utc':now(),'raw_archive_sha256':hash_file(archive),'raw_xlsx_sha256':hash_file(xlsx),
              'features':protocol['dataset']['features'],'classes':protocol['dataset']['classes'],
              'row_id_rule':protocol['dataset']['row_id'],'counts':counts,
              'seeds':{'test':171,'validation':172},'rows':{name:frame.row_id.tolist() for name,frame in splits.items()},
              'class_counts':{name:{str(k):int(v) for k,v in frame.Class.value_counts().sort_index().items()} for name,frame in splits.items()},
              'file_hashes':{name:hash_file(processed/name) for name in ['cleaned.csv','train.csv','validation.csv','test.csv']}}
    write_json(existing,manifest);write_json(processed/'audit.json',audit)
    print('prepare:',counts,'duplicates:',audit['exact_duplicate_rows'],flush=True)


def live_environment_signature():
    frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    return {'python':sys.version,'scikit_learn':sklearn.__version__,'numpy':np.__version__,
            'pandas':pd.__version__,'dependencies':frozen.splitlines()}


def validate_environment(root):
    old=read_json(root/'environment.json');current=live_environment_signature()
    for name,value in current.items():
        if old.get(name)!=value:raise ValueError('Live environment differs from saved environment: '+name)


def environment(root):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):sklearn.show_versions()
    (root/'logs'/'sklearn_versions.txt').write_text(buf.getvalue(),encoding='utf-8')
    signature=live_environment_signature()
    frozen='\n'.join(signature['dependencies'])+'\n'
    (root/'requirements.lock.txt').write_text(frozen,encoding='utf-8')
    current={**signature,'executable':sys.executable,'platform':platform.platform(),
             'cpu_logical':os.cpu_count(),'threadpools':threadpool_info(),'compute_policy':{'blas_threads':1,'RF_n_jobs':4,'fits':'sequential'}}
    path=root/'environment.json'
    if path.exists() and (root/'config'/'FROZEN.json').exists():
        old=read_json(path)
        for name in ['python','scikit_learn','numpy','pandas','dependencies']:
            if old[name]!=current[name]:raise ValueError('Frozen environment changed: '+name)
    else:write_json(path,current)


def create_noise(root,protocol):
    train,_,y=load_split(root,'train',protocol)
    directory=root/'data'/'noise';directory.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in protocol['noise']['seeds']:
        for p in protocol['noise']['p']:
            noise=noise_labels(y,p,seed,len(protocol['dataset']['classes']))
            name=f'p{int(p*100):03d}_s{seed}.csv';path=directory/name
            df=pd.DataFrame({'row_id':train.row_id,'clean_label':y,'noisy_label':noise['noisy'],
                             'selected_mask':noise['selected'],'changed_mask':noise['changed']})
            if not path.exists():df.to_csv(path,index=False)
            else:
                old=pd.read_csv(path)
                if not old.equals(df):raise ValueError('Existing noise labels differ: '+name)
            records.append({'p':p,'seed':seed,'rows':len(y),'selected_fraction':float(noise['selected'].mean()),
                            'changed_fraction':float(noise['changed'].mean()),'file':name,'sha256':hash_file(path)})
    manifest=directory/'noise_manifest.json'
    if manifest.exists() and read_json(manifest)!=records:raise ValueError('Noise manifest differs')
    write_json(manifest,records)


def smoke(root,protocol):
    train,X,y=load_split(root,'train',protocol);val,V,v=load_split(root,'validation',protocol)
    for model,p in [('LR',0),('RF',0),('MLP',0),('MLP',1)]:
        spec=json.loads(json.dumps(protocol['models'][model]))
        if model=='RF':spec['fixed']['n_estimators']=50
        else:spec['fixed']['max_iter']=20
        noisy=noise_labels(y,p,17,7)['noisy']
        context=base_context(root,protocol);rid=f'{model}_p{int(p*100):03d}'
        fingerprint=hash_object({'context':context,'spec':spec,'run_id':rid,'smoke':True})
        meta=fit_and_save(root/'artifacts'/'smoke'/rid,make_model(model,spec,spec['initial_default'],17),
                          X,noisy,y,V,v,train.row_id,val.row_id,protocol['metric_labels'],fingerprint,
                          {'run_id':rid,'phase':'smoke','model':model,'p':p,'seed':17,'context':context,'excluded_from_research':True})
        print('smoke:',rid,'seconds:',round(meta['fit_seconds'],2),flush=True)


def tune(root,protocol):
    selected_path=root/'config'/'selected_params.json'
    if (root/'config'/'FROZEN.json').exists():
        check_freeze(root,protocol);print('tuning: frozen configuration already verified',flush=True);return
    train,X,y=load_split(root,'train',protocol);val,V,v=load_split(root,'validation',protocol)
    context=base_context(root,protocol);rows=[];chosen={}
    for model,spec in protocol['models'].items():
        candidates=[]
        for i,params in enumerate(ParameterGrid(spec['tune_grid'])):
            rid=f'{model}_candidate_{i+1:02d}'
            fp=hash_object({'context':context,'model':model,'params':params,'seed':17,'phase':'tuning'})
            meta=fit_and_save(root/'artifacts'/'tuning'/rid,make_model(model,spec,params,17),X,y,y,V,v,
                              train.row_id,val.row_id,protocol['metric_labels'],fp,
                              {'run_id':rid,'phase':'tuning','model':model,'params':params,'seed':17,'context':context})
            row={'model':model,'run_id':rid,'params':params,'validation_macro_f1':meta['validation']['macro_f1'],
                 'fit_seconds':meta['fit_seconds'],'warnings':len(meta['warnings'])}
            rows.append(row);candidates.append(row)
            print('tuning:',rid,'val Macro-F1:',round(row['validation_macro_f1'],6),'seconds:',round(meta['fit_seconds'],2),flush=True)
        chosen[model]=select_candidate(model,candidates,protocol['tuning']['tie_tolerance'])
    if len(rows)!=protocol['tuning']['total_fits']:raise ValueError('Wrong candidate fit count')
    (root/'results').mkdir(exist_ok=True)
    pd.DataFrame([{**r,'params':json.dumps(r['params'])} for r in rows]).to_csv(root/'results'/'tuning.csv',index=False)
    write_json(selected_path,{'selected_utc':now(),'context':context,'criterion':'clean validation Macro-F1, predeclared 0.001 complexity tie rules',
                              'selected':chosen,'all_candidates':rows})


def frozen_inputs(root,protocol):
    return {**base_context(root,protocol),'selected_params_hash':hash_file(root/'config'/'selected_params.json'),
            'main_matrix_hash':hash_file(root/'config'/'main_matrix.json'),
            'noise_manifest_hash':hash_file(root/'data'/'noise'/'noise_manifest.json')}


def freeze(root,protocol):
    validate_main_matrix(read_json(root/'config'/'main_matrix.json'),protocol)
    selected=read_json(root/'config'/'selected_params.json')
    if selected['context']!=base_context(root,protocol):raise ValueError('Selected parameters came from different inputs')
    current=frozen_inputs(root,protocol);path=root/'config'/'FROZEN.json'
    if path.exists():
        if read_json(path)['inputs']!=current:raise ValueError('Frozen experiment inputs changed')
    else:write_json(path,{'frozen_utc':now(),'inputs':current,'fingerprint':hash_object(current),
                         'test_metrics_accessed':False,'note':'All conditions fixed before any final holdout evaluation.'})
    print('freeze: scientific configuration saved',flush=True)


def check_freeze(root,protocol):
    frozen=read_json(root/'config'/'FROZEN.json')
    if frozen['inputs']!=frozen_inputs(root,protocol):raise ValueError('Frozen experiment inputs changed')
    return frozen


def validate_main_matrix(matrix,protocol):
    expected={(model,p,seed,seed) for model in protocol['models']
              for p in protocol['noise']['p'] for seed in protocol['noise']['seeds']}
    actual={(r['model'],r['p'],r['noise_seed'],r['model_seed']) for r in matrix}
    if actual!=expected or len(matrix)!=len(expected) or len({r['run_id'] for r in matrix})!=len(matrix):
        raise ValueError('Main matrix must contain every predeclared condition exactly once')
    for r in matrix:
        if r['run_id']!=f'{r["model"]}_p{int(r["p"]*100):03d}_s{r["noise_seed"]}':
            raise ValueError('Main matrix run identity differs from condition')


def main_runs(root,protocol,matrix):
    if matrix!=read_json(root/'config'/'main_matrix.json'):raise ValueError('Main matrix differs from canonical frozen matrix')
    validate_main_matrix(matrix,protocol)
    frozen=check_freeze(root,protocol);selected=read_json(root/'config'/'selected_params.json')['selected']
    train,X,y=load_split(root,'train',protocol);val,V,v=load_split(root,'validation',protocol)
    noise_manifest=read_json(root/'data'/'noise'/'noise_manifest.json')
    for i,run in enumerate(matrix):
        name=f'p{int(run["p"]*100):03d}_s{run["noise_seed"]}.csv';path=root/'data'/'noise'/name
        expected=next(r for r in noise_manifest if r['file']==name)
        if hash_file(path)!=expected['sha256']:raise ValueError('Noise file integrity failed')
        noise=pd.read_csv(path)
        if not np.array_equal(noise.row_id,train.row_id) or not np.array_equal(noise.clean_label,y):raise ValueError('Noise row order differs')
        model=run['model'];params=selected[model]['params']
        fp=hash_object({'freeze':frozen['fingerprint'],'run_id':run['run_id'],'noise_hash':expected['sha256']})
        meta=fit_and_save(root/'artifacts'/'runs'/run['run_id'],make_model(model,protocol['models'][model],params,run['model_seed']),
                         X,noise.noisy_label.to_numpy(),y,V,v,train.row_id,val.row_id,protocol['metric_labels'],fp,
                         {**run,'phase':'main','params':params,'freeze_fingerprint':frozen['fingerprint'],'noise_hash':expected['sha256'],
                          'selected_fraction':expected['selected_fraction'],'changed_fraction':expected['changed_fraction']})
        print(f'main {i+1}/{len(matrix)}:',run['run_id'],'train noisy:',round(meta['train_noisy']['accuracy'],4),
              'val F1:',round(meta['validation']['macro_f1'],4),'seconds:',round(meta['fit_seconds'],2),flush=True)
    summaries=[read_json(root/'artifacts'/'runs'/r['run_id']/'meta.json') for r in matrix]
    write_json(root/'results'/'main_training_metrics.json',summaries)


def final_evaluate(root,protocol,matrix):
    # Gate precedes even opening the holdout or checking its path.
    for run in matrix:
        path=root/'artifacts'/'runs'/run['run_id']/'meta.json'
        if not path.exists() or read_json(path).get('status')!='complete':
            raise ValueError('Every predeclared run must be complete before final evaluation')
    if matrix!=read_json(root/'config'/'main_matrix.json'):
        raise ValueError('Final matrix differs from canonical frozen matrix')
    validate_main_matrix(matrix,protocol)
    frozen=check_freeze(root,protocol)
    noise_manifest=read_json(root/'data'/'noise'/'noise_manifest.json')
    for run in matrix:
        noise=next(r for r in noise_manifest if r['p']==run['p'] and r['seed']==run['noise_seed'])
        fp=hash_object({'freeze':frozen['fingerprint'],'run_id':run['run_id'],'noise_hash':noise['sha256']})
        verify_cached_run(root/'artifacts'/'runs'/run['run_id'],fp)
    results=root/'results';receipt=results/'FINAL_EVALUATION.json'
    if receipt.exists():
        old=read_json(receipt)
        if old['freeze_fingerprint']!=frozen['fingerprint']:raise ValueError('Final results use different frozen experiment')
        for name,expected in old['file_hashes'].items():
            if hash_file(results/name)!=expected:raise ValueError('Final metric integrity failed')
        print('final evaluation: verified cached final metrics',flush=True);return
    test,X,y=load_split(root,'test',protocol);labels=protocol['metric_labels']
    records=[]
    for run in matrix:
        directory=root/'artifacts'/'runs'/run['run_id'];meta=read_json(directory/'meta.json')
        estimator=joblib.load(directory/'model.joblib')
        with threadpool_limits(limits=1):pred=estimator.predict(X);proba=estimator.predict_proba(X)
        output=results/'test_predictions'/f'{run["run_id"]}.csv';output.parent.mkdir(exist_ok=True)
        prediction=pd.DataFrame({'row_id':test.row_id,'clean_label':y,'prediction':pred})
        for i,c in enumerate(estimator.named_steps['model'].classes_):prediction[f'prob_{c}']=proba[:,i]
        prediction.to_csv(output,index=False)
        metric=scores(y,pred,labels)
        records.append({'run_id':run['run_id'],'model':run['model'],'p':run['p'],'seed':run['model_seed'],
                        'changed_fraction':meta['changed_fraction'],'selected_fraction':meta['selected_fraction'],
                        'train_acc_noisy':meta['train_noisy']['accuracy'],'train_acc_clean':meta['train_clean']['accuracy'],
                        'validation_macro_f1':meta['validation']['macro_f1'],
                        'test_macro_f1':metric['macro_f1'],'test_balanced_accuracy':metric['balanced_accuracy'],
                        'test_accuracy':metric['accuracy'],'clean_target_gap':meta['train_clean']['accuracy']-metric['accuracy'],
                        'fit_seconds':meta['fit_seconds'],'iterations':json.dumps(meta['iterations']),
                        'warning_count':len(meta['warnings']),'stop_reason':meta['stop_reason']})
        write_json(results/'class_reports'/f'{run["run_id"]}.json',
                   {'classification':classification_report(y,pred,labels=labels,target_names=protocol['dataset']['classes'],output_dict=True,zero_division=0),
                    'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),'classes':protocol['dataset']['classes']})
    df=pd.DataFrame(records);baseline=df[df.p==0].set_index(['model','seed']).test_macro_f1
    df['delta_macro_f1_vs_clean']=[r.test_macro_f1-baseline.loc[(r.model,r.seed)] for r in df.itertuples()]
    df.to_csv(results/'test_metrics.csv',index=False)
    columns=['changed_fraction','train_acc_noisy','train_acc_clean','test_macro_f1','test_balanced_accuracy','test_accuracy','clean_target_gap','fit_seconds','delta_macro_f1_vs_clean']
    grouped=df.groupby(['model','p'])[columns].agg(['mean','std'])
    grouped.columns=['_'.join(c) for c in grouped.columns];grouped.reset_index().to_csv(results/'model_noise_summary.csv',index=False)
    train,_,yclean=load_split(root,'train',protocol);majority=int(np.bincount(yclean).argmax())
    write_json(results/'baselines.json',{'majority_class':protocol['dataset']['classes'][majority],
              'clean_training_majority_test':scores(y,np.full_like(y,majority),labels),'uniform_guess_expected_accuracy':1/7,
              'note':'Majority class selected from clean training labels; a reference, not a fourth experimental model.'})
    files=[p for p in results.rglob('*') if p.is_file() and p.name!='FINAL_EVALUATION.json']
    write_json(receipt,{'evaluated_utc':now(),'freeze_fingerprint':frozen['fingerprint'],'test_rows':len(test),
                       'run_count':len(records),'test_revealed':True,'file_hashes':{p.relative_to(results).as_posix():hash_file(p) for p in files}})
    print('final evaluation: all',len(records),'runs evaluated on',len(test),'clean holdout rows',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','eda','smoke','tune','train','evaluate','report','all'])
    args=parser.parse_args();root=ROOT;protocol=read_json(root/'config'/'protocol.json');matrix=read_json(root/'config'/'main_matrix.json')
    stages=['prepare','eda','smoke','tune','train','evaluate','report'] if args.stage=='all' else [args.stage]
    if args.stage in ('prepare','all'):environment(root)
    for stage in stages:
        if stage=='prepare':prepare_data(root,protocol);create_noise(root,protocol)
        elif stage=='eda':
            from .visuals import eda
            eda(root,protocol)
        elif stage=='smoke':smoke(root,protocol)
        elif stage=='tune':tune(root,protocol);freeze(root,protocol)
        elif stage=='train':main_runs(root,protocol,matrix)
        elif stage=='evaluate':final_evaluate(root,protocol,matrix)
        elif stage=='report':
            from .visuals import research_report
            research_report(root,protocol)


if __name__=='__main__':main()
