import importlib
import numpy as np
import pandas as pd
import pytest


def test_eda_statistics_use_only_training_rows(tmp_path):
    try:m=importlib.import_module('src.visuals')
    except ModuleNotFoundError as exc:
        if exc.name=='src.visuals':pytest.fail('Training-only EDA is not implemented yet')
        raise
    path=tmp_path/'data'/'processed';path.mkdir(parents=True)
    pd.DataFrame({'row_id':np.arange(2,72),'Area':np.arange(70),'Perimeter':2*np.arange(70),
                  'Compactness':np.arange(70)/100,'Class':np.repeat(list('ABCDEFG'),10)}).to_csv(path/'train.csv',index=False)
    (path/'test.csv').write_text('DO NOT READ: held-out sentinel')
    protocol={'dataset':{'features':['Area','Perimeter','Compactness'],'classes':list('ABCDEFG')}}
    m.eda(tmp_path,protocol)
    stats=pd.read_csv(tmp_path/'reports'/'eda'/'feature_statistics.csv',index_col=0)
    assert stats.loc['Area','mean']==pytest.approx(34.5)
    assert stats.loc['Perimeter','max']==138
    assert all((tmp_path/'reports'/'eda'/name).is_file() for name in
               ['class_counts.png','correlation.png','feature_distributions.png'])
