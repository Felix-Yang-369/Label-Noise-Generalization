"""Training-only EDA and figures from completed, frozen experiments."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .pipeline import load_split,read_json
from .core import hash_file,write_json

COLORS={'LR':'#2866a8','RF':'#23907c','MLP':'#d97431'}
plt.rcParams.update({'font.size':10,'figure.dpi':110,'savefig.dpi':170,'axes.spines.top':False,'axes.spines.right':False})


def save_figure(fig,path):
    fig.savefig(path,bbox_inches='tight');plt.close(fig)


def eda(root,protocol):
    train,X,y=load_split(root,'train',protocol)
    output=root/'reports'/'eda';output.mkdir(parents=True,exist_ok=True)
    features=protocol['dataset']['features'];classes=protocol['dataset']['classes']
    stats=train[features].describe().T
    stats['median']=train[features].median();stats['missing']=train[features].isna().sum();stats['skew']=train[features].skew()
    stats.to_csv(output/'feature_statistics.csv')
    corr=train[features].corr();corr.to_csv(output/'feature_correlation.csv')
    counts=train.Class.value_counts().reindex(classes)
    counts.rename('train_count').to_csv(output/'class_counts.csv')
    pairs=sorted([(features[i],features[j],float(corr.iloc[i,j])) for i in range(len(features))
                  for j in range(i+1,len(features))],key=lambda item:abs(item[2]),reverse=True)
    notes=[f'仅分析训练集 {len(train):,} 条；验证和测试未用于探索或决定特征。',
           f'训练集最多类 {counts.idxmax()}={int(counts.max())}，最少类 {counts.idxmin()}={int(counts.min())}；数量比为 {counts.max()/counts.min():.2f}。因此主要比较采用等权 Macro-F1，并报告逐类 Recall。',
           '尺寸与形状特征存在强相关；相关性图数值按两位小数显示，1.00 不代表两个特征完全相同。最高绝对相关的五组如下：']
    notes += [f'{a} / {b}: Pearson r={value:.6f}' for a,b,value in pairs[:5]]
    notes += ['原始特征尺度差异明显，LR / MLP 使用仅训练集拟合的 StandardScaler；RF 使用原数值。',
              '按照已固定设计保留全部原始特征，不依据测试表现删列或添加 PCA。离群值不默认删除；箱线图仅隐藏散点以便观察主体分布。']
    (output/'EDA_Notes.md').write_text('# 训练集 EDA 解读\n\n'+'\n\n'.join(notes),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,4))
    bars=ax.bar(classes,counts,color='#2866a8')
    ax.bar_label(bars,padding=3);ax.set_ylim(0,counts.max()*1.18)
    ax.set_ylabel('Training examples');ax.set_title(f'Training-set class distribution (n={len(train):,})')
    ax.tick_params(axis='x',labelrotation=25);save_figure(fig,output/'class_counts.png')
    n=len(features);fig,ax=plt.subplots(figsize=(11,10))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(n),features,rotation=90);ax.set_yticks(range(n),features)
    ax.set_title('Training-only Pearson feature correlation')
    for i in range(n):
        for j in range(n):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=6.7,
                                color='white' if abs(corr.iloc[i,j])>.6 else '#222')
    fig.colorbar(im,ax=ax,fraction=.035,pad=.025);save_figure(fig,output/'correlation.png')
    columns=[c for c in ['Area','Perimeter','Compactness'] if c in features]
    fig,axes=plt.subplots(1,len(columns),figsize=(15,4.8),squeeze=False)
    for ax,c in zip(axes[0],columns):
        ax.boxplot([train.loc[train.Class==label,c] for label in classes],tick_labels=classes,showfliers=False)
        ax.tick_params(axis='x',labelrotation=70);ax.set_title(c);ax.set_ylabel('Original feature value')
    fig.suptitle('Training-only feature distributions by class (outliers hidden in plot only)')
    save_figure(fig,output/'feature_distributions.png')
    write_json(output/'eda_metadata.json',{'training_rows':len(train),'train_csv_sha256':hash_file(root/'data'/'processed'/'train.csv'),
               'eda_code_sha256':hash_file(Path(__file__)),
               'scope':'training split only; validation and test were not read for EDA',
               'features_retained':features,'missing_cells':int(train[features].isna().sum().sum()),
               'class_counts':{k:int(v) for k,v in counts.items()},'feature_decision':'keep all 16 original features; train-fitted StandardScaler for LR/MLP',
               'outliers':'not removed; boxplot hides fliers for readability only'})
    body=f'<h1>ECS171 - Training-set EDA</h1><p>Only {len(train):,} training rows; original labels, no test exploration.</p>'
    body+=''.join('<p>'+html.escape(note)+'</p>' for note in notes)
    body+=''.join(f'<h2>{title}</h2><img src="{name}" style="max-width:100%">' for title,name in
                 [('Class balance','class_counts.png'),('Feature correlation','correlation.png'),('Feature distributions','feature_distributions.png')])
    body+='<h2>Feature statistics</h2>'+stats.round(5).to_html()
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1150px;margin:32px auto;font:16px/1.6 system-ui}table{border-collapse:collapse;font-size:13px}td,th{padding:6px}</style>'+body,encoding='utf-8')
    print('EDA: training-only statistics and three figures generated',flush=True)


def research_report(root,protocol):
    receipt=read_json(root/'results'/'FINAL_EVALUATION.json')
    if receipt['run_count']!=45:raise ValueError('The full main experiment must be evaluated before reporting')
    for name,expected in receipt['file_hashes'].items():
        if hash_file(root/'results'/name)!=expected:raise ValueError('Final metric integrity failed')
    df=pd.read_csv(root/'results'/'test_metrics.csv')
    summary=pd.read_csv(root/'results'/'model_noise_summary.csv')
    selected=read_json(root/'config'/'selected_params.json')['selected']
    baselines=read_json(root/'results'/'baselines.json')
    output=root/'reports'/'experiments';output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5))
    for model,color in COLORS.items():
        rows=summary[summary.model==model].sort_values('p')
        ax.errorbar(rows.p,rows.test_macro_f1_mean,yerr=rows.test_macro_f1_std,label=model,color=color,marker='o',capsize=3)
    ax.set(xlabel='Training-label randomization probability p',ylabel='Clean test Macro-F1',ylim=(0,1.02),xticks=protocol['noise']['p'])
    ax.set_title('Clean test performance (mean +/- sample SD; 3 seeds)');ax.legend();ax.grid(alpha=.2)
    save_figure(fig,output/'01_test_macro_f1.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharex=True,sharey=True)
    for model,color in COLORS.items():
        rows=summary[summary.model==model].sort_values('p')
        for ax,metric in zip(axes,['train_acc_noisy','test_accuracy']):
            ax.errorbar(rows.p,rows[metric+'_mean'],yerr=rows[metric+'_std'],label=model,color=color,marker='o',capsize=3)
    axes[0].set_title('Training accuracy against perturbed labels');axes[1].set_title('Test accuracy against original labels')
    for ax in axes:ax.set(xlabel='Randomization probability p',ylim=(0,1.02),xticks=protocol['noise']['p']);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Accuracy');save_figure(fig,output/'02_fit_and_clean_test.png')
    best_score=max(v['validation_macro_f1'] for v in selected.values())
    display_model=next(m for m in ['LR','RF','MLP'] if best_score-selected[m]['validation_macro_f1']<=.001+1e-12)
    classes=protocol['dataset']['classes'];fig,axes=plt.subplots(1,2,figsize=(15,6.2),gridspec_kw={'wspace':.45})
    for ax,p in zip(axes,[0,1]):
        record=read_json(root/'results'/'class_reports'/f'{display_model}_p{int(p*100):03d}_s17.json')
        cm=np.asarray(record['confusion_matrix'])
        normalized=cm/cm.sum(axis=1,keepdims=True)
        im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
        for i in range(7):
            for j in range(7):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if normalized[i,j]>.5 else 'black')
        ax.set_xticks(range(7),classes,rotation=60,ha='right');ax.set_yticks(range(7),classes)
        ax.set(xlabel='Predicted class',ylabel='True class',title=f'{display_model}, p={p}, seed=17 (row-normalized color)')
    axes[1].set_ylabel('')
    fig.colorbar(im,ax=list(axes),fraction=.025,pad=.035,label='Row fraction');save_figure(fig,output/'03_confusion_matrices.png')
    fig,ax=plt.subplots(figsize=(8,4.6))
    for p in [0,1]:
        trace=read_json(root/'artifacts'/'runs'/f'MLP_p{int(p*100):03d}_s17'/'training_trace.json')
        ax.plot(np.arange(1,len(trace['loss_curve'])+1),trace['loss_curve'],label=f'MLP p={p}, seed=17')
    ax.set(xlabel='Training epoch',ylabel='Training objective',title='MLP optimization traces (one predeclared seed)');ax.legend();ax.grid(alpha=.2)
    save_figure(fig,output/'04_mlp_loss.png')

    clean=summary[summary.p==0].set_index('model');random=summary[summary.p==1].set_index('model')
    table_rows=[]
    for model in ['LR','RF','MLP']:
        table_rows.append({'model':model,'selected_params':json.dumps(selected[model]['params']),
                           'validation_tuning_f1':selected[model]['validation_macro_f1'],
                           'clean_test_f1':clean.loc[model,'test_macro_f1_mean'],
                           'clean_test_f1_sd':clean.loc[model,'test_macro_f1_std'],
                           'random_test_f1':random.loc[model,'test_macro_f1_mean'],
                           'random_train_fit_accuracy':random.loc[model,'train_acc_noisy_mean'],
                           'clean_mean_fit_seconds':clean.loc[model,'fit_seconds_mean']})
    overview=pd.DataFrame(table_rows);overview.to_csv(output/'overview.csv',index=False)
    total_fit=float(df.fit_seconds.sum()+pd.read_csv(root/'results'/'tuning.csv').fit_seconds.sum())
    warning_runs=df[df.warning_count>0]
    paragraphs=[
      '本阶段已完成独立环境、数据处理、训练集 EDA、10 次干净标签调参、45 次主实验与冻结后的真实标签留出测试。',
      '数据：13,611 原始行，精确去重 68 行后 13,543 行。固定 train / validation / test = 8,125 / 2,709 / 2,709；16 个原始数值特征保留。',
      '调参只使用干净验证集 Macro-F1；各模型配置在噪声主实验前冻结。五档 p 为重新抽标签的选择概率，实际标签改变率 q 另行记录。',
      '三个种子的均值和样本标准差描述标签扰动与模型随机性的联合变化；固定同一数据划分，不能当作多划分稳定性或统计显著性证据。',
    ]
    for model in ['LR','RF','MLP']:
        paragraphs.append(f'{model}：p=0 的真实测试 Macro-F1 为 {clean.loc[model,"test_macro_f1_mean"]:.4f} ± {clean.loc[model,"test_macro_f1_std"]:.4f}；p=1 时为 {random.loc[model,"test_macro_f1_mean"]:.4f}。p=1 对扰动训练标签的准确率为 {random.loc[model,"train_acc_noisy_mean"]:.4f}。')
    mid=summary[summary.p==.75].set_index('model')
    paragraphs += [
      f'观察：在 p=0.75 条件，LR / RF / MLP 的真实测试 Macro-F1 分别为 {mid.loc["LR","test_macro_f1_mean"]:.4f} / {mid.loc["RF","test_macro_f1_mean"]:.4f} / {mid.loc["MLP","test_macro_f1_mean"]:.4f}。这支持当前配置下模型对标签扰动的反应不同，不证明某个架构普遍最优。',
      '观察：RF 在 p=1 完整拟合了随机训练标签，却没有保持干净标签测试表现；这是本实验中记忆能力和泛化表现分离的直接证据。当前 MLP 未完美拟合随机标签，必须连同容量、正则化和训练预算解释。',
      'p=0 的 LR 三次结果完全相同，SD=0；此条件标签相同且 lbfgs 为确定性优化，三个种子并没有产生三个独立划分。'
    ]
    paragraphs+= [
      f'多数类参照由干净训练集选取 {baselines["majority_class"]}，其测试 Accuracy 为 {baselines["clean_training_majority_test"]["accuracy"]:.4f}。1/7 只表示独立均匀随机预测的期望 Accuracy，并非所有随机标签模型的必然表现。',
      f'55 次研究 fit 的纯训练时间合计 {total_fit:.1f} 秒；不包含环境安装、读取、保存、推理和图表时间。{len(warning_runs)} / 45 个主 run 有收敛相关警告，详细见 CSV 和每个 meta.json。',
      '训练对扰动标签的分数与干净测试分数对应不同目标，图中的差值不命名为标准泛化差距。clean_target_gap 使用原标签训练准确率减去原标签测试准确率。',
      '结果仅限此数据集、固定 split、当前参数与训练预算。若 MLP 未完美拟合随机标签，不能由此否定原论文；其容量、优化和正则化设置不同。18 次可选容量扩展未运行。',
      '代码与模型均在本地保存；本次不包括实时推理网页部署或课程提交。'
    ]
    lines=['# ECS171 实验结果报告','']
    for p in paragraphs:lines+=[p,'']
    lines+=['## 文件入口','','- ../../results/test_metrics.csv：45 次完整指标','- ../../results/model_noise_summary.csv：15 个条件的均值与 SD','- ../../results/tuning.csv：10 个候选验证结果','- ../../config/FROZEN.json：测试前冻结证据','- ../../artifacts/runs/：每次模型、日志、训练及验证逐行预测','']
    for name in ['01_test_macro_f1.png','02_fit_and_clean_test.png','03_confusion_matrices.png','04_mlp_loss.png']:
        lines+=[f'![{name}]({name})','']
    (output/'Results_Report.md').write_text('\n'.join(lines),encoding='utf-8')
    text_html=''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
    images=''.join('<figure><img src="'+name+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>' for name,caption in [
       ('01_test_macro_f1.png','Three seeds: mean +/- sample SD, fixed split.'),
       ('02_fit_and_clean_test.png','Training and test panels use different label targets.'),
       ('03_confusion_matrices.png',f'Model selected from clean tuning validation only: {display_model}; seed 17; cell labels are counts.'),
       ('04_mlp_loss.png','Training objectives; no validation-label early stopping.')])
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECS171 Results</title><style>body{margin:0;background:#f3f5f8;color:#1b2938;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:24px auto;padding:30px;background:white;border-radius:12px}img{max-width:100%}figure{margin:30px 0}figcaption{font-size:13px;color:#657080}table{width:100%;font-size:13px;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#2866a8}.scroll{overflow:auto}</style><main><h1>ECS171 实验结果</h1><p><a href="../eda/index.html">训练集 EDA</a> · <a href="../../results/test_metrics.csv">45 次完整指标</a> · <a href="../../results/model_noise_summary.csv">条件汇总</a> · <a href="../../README.md">复现说明</a></p>'''
    document+=text_html+'<h2>三模型概览</h2><div class="scroll">'+overview.round(5).to_html(index=False)+'</div>'+images+'<h2>15 个条件</h2><div class="scroll">'+summary.round(5).to_html(index=False)+'</div></main></html>'
    (output/'index.html').write_text(document,encoding='utf-8')
    write_json(output/'report_metadata.json',{'final_evaluation_receipt_sha256':hash_file(root/'results'/'FINAL_EVALUATION.json'),
               'report_code_sha256':hash_file(Path(__file__)),
               'display_model_validation_only':display_model,'total_research_fit_seconds':total_fit,
               'main_warning_runs':len(warning_runs),'main_runs':len(df),'capacity_study_run':False})
    print('report: actual-result figures, CSV overview, Markdown and HTML generated',flush=True)
