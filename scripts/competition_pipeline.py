"""MathorCup completion pipeline. All original files are read-only inputs.

AI-assisted implementation; results and limitations are documented in the report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import warnings

if __name__ == '__main__':
    # Stable import path for joblib artifacts when this file is run as a CLI.
    sys.modules['competition_pipeline'] = sys.modules[__name__]

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/competition_completion_v1'
SEED = 42
DEFECTS = ['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal']
SUMMARY = ['esters_sum','aroma_hops_sum','acids_sum','sulfur_sum']


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper
    def fit(self,X,y=None):
        a=np.asarray(X,dtype=float)
        self.lower_bounds_=np.nanquantile(a,self.lower,axis=0)
        self.upper_bounds_=np.nanquantile(a,self.upper,axis=0)
        return self
    def transform(self,X):
        return np.clip(np.asarray(X,dtype=float),self.lower_bounds_,self.upper_bounds_)

QuantileClipper.__module__ = 'competition_pipeline'


def save(df,name):
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT/name,index=False,encoding='utf-8-sig')


def load():
    a=pd.read_csv(ROOT/'data/1.csv').set_index('beer').sort_index()
    y=pd.read_csv(ROOT/'data/4.csv').set_index('beer').sort_index()
    assert a.index.equals(y.index) and a.index.is_unique
    # Attachment4 exports barcode identifiers in low-precision scientific notation.
    # beer is the stable, unique join key; never repair identifiers by guessing.
    barcode=pd.DataFrame({'beer':a.index,'a1_beer_id':a.beer_id.to_numpy(),'a4_beer_id':y.beer_id.to_numpy()})
    barcode['exact_equal']=barcode.a1_beer_id.eq(barcode.a4_beer_id)
    barcode['relative_difference']=(barcode.a1_beer_id-barcode.a4_beer_id).abs()/barcode.a1_beer_id.abs()
    save(barcode,'barcode_precision_audit.csv')
    assert a.tasting_category_fine.equals(y.tasting_category_fine)
    x=a.drop(columns=['beer_id','tasting_category_fine'])
    y=y.drop(columns=['beer_id','tasting_category_fine'])
    sel=pd.read_csv(ROOT/'outputs/model_selection_cv_50.csv')
    y=y[sel['指标'].tolist()]
    train,test=train_test_split(x.index,test_size=.2,random_state=SEED,stratify=a.tasting_category_fine)
    return x,y,a.tasting_category_fine,sel,pd.Index(train),pd.Index(test)


def templates():
    return {
        '均值基准': DummyRegressor(strategy='mean'),
        '岭回归': Pipeline([('clip',QuantileClipper()),('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('ridge',Ridge(alpha=100))]),
        '随机森林': Pipeline([('imputer',SimpleImputer(strategy='median')),('forest',RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=5,max_features=.7,random_state=SEED,n_jobs=-1))])}


def predict(models,x):
    return pd.DataFrame({t:m.predict(x) for t,m in models.items()},index=x.index)


def chemical_evidence(row):
    target,field,origin=row['target'],row['field'],str(row['origin'])
    direct={'dms':'DMS','diacetyl':'X2.3.butanedione','X4vg':'X4VG','t2n':'X.E..2.nonenal','alcohol':'ethanol..v.v.'}
    if direct.get(target)==field:
        reason='化学实体与感官描述直接对应；感知仍受基质、阈值与混合物作用影响'
    elif target.startswith(('A_malt','F_malt')):
        reason='麦芽目标且附件标注Malt来源，具有同源解释' if origin=='Malt' else f'麦芽目标但附件来源为{origin}，应按共变代理解释，缺乏直接麦芽机制证据'
    elif target.startswith(('A_hops','F_hops')):
        reason='酒花目标且附件标注Hops来源，具有同源解释' if origin=='Hops' else f'酒花目标但附件来源为{origin}，可能由风格或共变结构产生，不建议据此添加'
    elif target.startswith(('A_esters','F_esters')):
        name=str(row['chemical_name']).lower()
        isester=any(t in name for t in ['acetate','butyrate','hexanoate','octanoate','decanoate','propionate','benzoate','formate'])
        reason='候选名称支持酯类身份，与酯香类别相符；仍需单体实验' if isester else '候选并非本附件可直接确认的酯类，排名可能反映发酵共变；不作酯香因果解释'
    elif target=='stale_hops':
        reason='Stale hops不是单一化合物；这里只是相关代理，缺少浓度阈值与机制确认'
    else:
        reason=f'附件来源记录为{origin}；数据支持关联，尚缺该感官目标的直接机制或添加实验'
    if not row['direction_agrees']:reason+='；控制风格后方向反转，证据降级'
    if row['bootstrap_frequency']<.5:reason+='；入选频率低于50%，排序较弱'
    return reason


def scoring():
    x,y,styles,sel,tr,te=load()
    reference=pd.read_csv(ROOT/'outputs/test_predictions_50.csv')
    models={}; records=[]; cuts=[]; metrics=[]
    fold=KFold(5,shuffle=True,random_state=SEED)
    for i,row in sel.iterrows():
        t,kind=row['指标'],row['候选模型']
        model=clone(templates()[kind]); model.fit(x.loc[tr],y.loc[tr,t]); models[t]=model
        testpred=model.predict(x.loc[te])
        ref=reference[reference['指标'].eq(t)].set_index('beer').loc[te,'预测值'].to_numpy()
        assert np.allclose(testpred,ref,atol=1e-8,rtol=1e-8), f'frozen-model reproduction failed: {t}'
        oof=pd.Series(index=tr,dtype=float)
        oofbase=pd.Series(index=tr,dtype=float)
        oofgrade=pd.Series(index=tr,dtype=float)
        actualgrade=pd.Series(index=tr,dtype=float)
        basegrade=pd.Series(index=tr,dtype=float)
        # Every OOF grade uses cut points fitted only on that fold's fit subset.
        for fit,valid in fold.split(tr):
            ids,vids=tr[fit],tr[valid]
            m=clone(model).fit(x.loc[ids],y.loc[ids,t])
            p=m.predict(x.loc[vids]); oof.loc[vids]=p
            mu=y.loc[ids,t].mean(); oofbase.loc[vids]=mu
            c=y.loc[ids,t].quantile(np.arange(1,7)/7).to_numpy()
            oofgrade.loc[vids]=np.searchsorted(c,p,side='right')
            actualgrade.loc[vids]=np.searchsorted(c,y.loc[vids,t],side='right')
            basegrade.loc[vids]=np.searchsorted(c,mu,side='right')
        c=y.loc[tr,t].quantile(np.arange(1,7)/7).to_numpy()
        cuts.append({'target':t,**{f'q{k}':v for k,v in enumerate(c,1)},'distinct_cutpoints':len(np.unique(c))})
        err=np.abs(oof-y.loc[tr,t])
        # Descriptive cross-fitted residual interval, not a finite-sample split-conformal guarantee.
        radius=float(np.quantile(err,.9,method='higher'))
        for group,ids,p,base in [('training_oof',tr,oof.to_numpy(),oofbase.to_numpy()),('heldout_test',te,testpred,np.repeat(y.loc[tr,t].mean(),len(te)))]:
            ag=actualgrade.to_numpy() if group=='training_oof' else np.searchsorted(c,y.loc[ids,t],side='right')
            pg=oofgrade.to_numpy() if group=='training_oof' else np.searchsorted(c,p,side='right')
            bg=basegrade.to_numpy() if group=='training_oof' else np.searchsorted(c,base,side='right')
            for j,beer in enumerate(ids):
                records.append(dict(beer=int(beer),target=t,split=group,model=kind,actual_standardized=y.at[beer,t],predicted_standardized=p[j],baseline_standardized=base[j],actual_relative_grade=int(ag[j]),predicted_relative_grade=int(pg[j]),baseline_relative_grade=int(bg[j]),lower_descriptive=p[j]-radius,upper_descriptive=p[j]+radius))
            metrics.append(dict(target=t,split=group,MAE=mean_absolute_error(y.loc[ids,t],p),baseline_MAE=mean_absolute_error(y.loc[ids,t],base),R2=r2_score(y.loc[ids,t],p),relative_grade_MAE=float(np.abs(ag-pg).mean()),relative_baseline_grade_MAE=float(np.abs(ag-bg).mean()),relative_grade_accuracy=float((ag==pg).mean()),interval_coverage=float((np.abs(y.loc[ids,t].to_numpy()-p)<=radius).mean())))
        print(f'scoring {i+1}/50 {t}: reproduced',flush=True)
    save(pd.DataFrame(records),'q1_out_of_sample_predictions.csv')
    save(pd.DataFrame(cuts),'q1_train_quantile_cutpoints.csv')
    save(pd.DataFrame(metrics),'q1_metrics.csv')
    joblib.dump({'models':models,'train':tr,'test':te,'features':x.columns.tolist(),'targets':y.columns.tolist()},OUT/'frozen_training_models.joblib')
    save(pd.DataFrame({'beer':x.index,'split':np.where(x.index.isin(tr),'training','heldout_test'),'style':styles}), 'split_manifest.csv')
    # Additional missingness is applied only to held-out inputs; model choices stay frozen.
    rng=np.random.default_rng(SEED); sensitivity=[]
    for rate in [0,.05,.1,.2]:
        for repeat in range(5):
            xx=x.loc[te].copy()
            mask=rng.random(xx.shape)<rate
            xx=xx.mask(mask)
            pp=predict(models,xx)
            sensitivity.append(dict(additional_missing_rate=rate,repeat=repeat,MAE=float(np.abs(pp-y.loc[te]).to_numpy().mean())))
    save(pd.DataFrame(sensitivity),'q1_missingness_sensitivity.csv')


def correlations():
    import networkx as nx
    x,y,styles,sel,tr,te=load()
    roles=pd.read_csv(ROOT/'outputs/feature_roles.csv')
    compounds=roles.loc[roles['暂定角色'].eq('单项化学测量候选'),'字段'].tolist()
    info=pd.read_csv(ROOT/'outputs/chemical_name_mapping.csv').set_index('field_name_R')
    names=info.loc[x.columns,'official_name'].tolist()
    official=pd.read_csv(ROOT/'data/2.csv',index_col=0).loc[names,names]
    official.index=x.columns;official.columns=x.columns
    check=[];upper=np.triu_indices(len(x.columns),1)
    for method in ['pearson','spearman']:
        calc=x.corr(method=method,min_periods=200).to_numpy()[upper]
        orig=official.to_numpy()[upper]; good=np.isfinite(calc)&np.isfinite(orig)
        check.append(dict(method=method,n=int(good.sum()),mean_abs_difference=float(np.abs(calc[good]-orig[good]).mean()),max_abs_difference=float(np.abs(calc[good]-orig[good]).max())))
    save(pd.DataFrame(check),'q2_official_correlation_type_check.csv')
    # Full compound cross-correlation, then independent bootstrap resampling of all candidates.
    z=pd.concat([x[compounds],y.add_prefix('sensory__')],axis=1)
    c=z.corr(method='spearman',min_periods=200).loc[compounds,['sensory__'+t for t in y.columns]]
    c.columns=y.columns
    counts=np.zeros(c.shape,dtype=int); boot_values=np.empty((200,len(y.columns),3))
    ranking=np.argsort(-np.abs(c.to_numpy()),axis=0,kind='stable')[:3,:]
    rng=np.random.default_rng(SEED)
    for k in range(200):
        sample=z.iloc[rng.integers(0,len(z),len(z))]
        bc=sample.corr(method='spearman',min_periods=200).loc[compounds,['sensory__'+t for t in y.columns]].to_numpy()
        for j in range(len(y.columns)):
            eligible=np.where(np.isfinite(bc[:,j]))[0]
            ids=eligible[np.argsort(-np.abs(bc[eligible,j]),kind='stable')[:3]]
            counts[ids,j]+=1
            boot_values[k,j,:]=bc[ranking[:,j],j]
        if (k+1)%25==0: print(f'bootstrap {k+1}/200',flush=True)
    evidence=[]
    for j,t in enumerate(y.columns):
        for rank,idx in enumerate(ranking[:,j],1):
            f=compounds[idx]
            data=pd.DataFrame({'x':x[f],'y':y[t],'style':styles}).dropna()
            ranks=data[['x','y']].rank();res=ranks-ranks.groupby(data['style']).transform('mean')
            adjusted=res.x.corr(res.y)
            meta=info.loc[f]
            origin=meta.get('origin_paper',np.nan);flavor=meta.get('off_flavors',np.nan)
            family='酯类' if ('ate' in str(meta.get('chemical_name','')).lower() or 'ester' in f.lower()) else '按附件字典核查'
            evidence.append(dict(target=t,rank=rank,field=f,chemical_name=meta['official_name'],rho=c.loc[f,t],n=len(data),bootstrap_frequency=counts[idx,j]/200,bootstrap_rho_low=np.nanquantile(boot_values[:,j,rank-1],.025),bootstrap_rho_high=np.nanquantile(boot_values[:,j,rank-1],.975),style_adjusted_rho=adjusted,direction_agrees=bool(np.sign(adjusted)==np.sign(c.loc[f,t])),origin=origin,unit=meta.get('unit',np.nan),off_flavors=flavor,family_hint=family,interpretation='关联候选；同源代谢和风格混杂均可能解释，不能推断单体因果'))
    evidence=pd.DataFrame(evidence)
    evidence['chemical_plausibility']=evidence.apply(chemical_evidence,axis=1)
    save(evidence,'q2_top3_all50_evidence.csv')
    save(pd.DataFrame(counts/200,index=compounds,columns=y.columns).reset_index(names='field'),'q2_all_candidate_frequencies.csv')
    # A faithful graph uses the provided matrix, not a newly estimated substitute.
    edges=[];g=nx.Graph();g.add_nodes_from(compounds)
    stable=pd.read_csv(ROOT/'outputs/network_edges_stable.csv')
    stablekeys={frozenset([r['指标1'],r['指标2']]):r['重抽样保留率'] for _,r in stable.iterrows()}
    for i,f in enumerate(compounds):
        for h in compounds[i+1:]:
            rho=official.at[f,h]
            if abs(rho)>=.6:
                retention=stablekeys.get(frozenset([f,h]),np.nan)
                edges.append(dict(left=f,right=h,rho=rho,existing_stable_retention=retention))
                g.add_edge(f,h,rho=float(rho),weight=float(abs(rho)))
    save(pd.DataFrame(edges),'q2_official_graph_edges.csv');nx.write_graphml(g,OUT/'q2_official_compound_graph.graphml')
    save(pd.DataFrame([dict(field=f,degree=g.degree(f),weighted_degree=g.degree(f,weight='weight'),component=i) for i,component in enumerate(nx.connected_components(g)) for f in component]),'q2_graph_nodes.csv')
    # Native Pearson is only a sensitivity analysis because the supplied matrix matches rank correlation.
    pear=pd.concat([x[compounds],y.add_prefix('sensory__')],axis=1).corr(method='pearson',min_periods=200)
    sens=[]
    for t in y:
        a=set(evidence.loc[evidence.target.eq(t),'field'])
        b=set(pear.loc[compounds,'sensory__'+t].abs().nlargest(3).index)
        sens.append(dict(target=t,jaccard_pearson_spearman=len(a&b)/len(a|b)))
    save(pd.DataFrame(sens),'q2_method_sensitivity.csv')


class LocalConditionalGenerator:
    """Empirical conditional kernel: draw anchor, one of k nearest same-style donors,
    and Beta(1,4) weight. Missing values propagate; imputation is distance-only.
    Native-scale interpolation is statistical, not an assertion of a physical mixing recipe.
    """
    def fit(self,x,styles):
        self.x=x.copy();self.styles=styles.reindex(x.index)
        self.imputer=SimpleImputer(strategy='median').fit(x)
        self.scaler=StandardScaler().fit(self.imputer.transform(x))
        self.z=self.scaler.transform(self.imputer.transform(x))
        self.members={s:np.where(self.styles.to_numpy()==s)[0] for s in sorted(self.styles.unique())}
        self.neighbors={}
        for s,ids in self.members.items():
            d=cdist(self.z[ids],self.z[ids]);np.fill_diagonal(d,np.inf)
            self.neighbors[s]=np.argsort(d,axis=1,kind='stable')[:,:min(5,len(ids)-1)]
        return self
    def sample(self,style,n,rng,mode='conditional_local'):
        ids=self.members[style] if mode!='unconditional_local' else np.arange(len(self.x))
        local=rng.integers(0,len(ids),n);a=ids[local]
        if mode=='conditional_resample' or len(ids)<2:
            b=a.copy();w=np.zeros(n)
        else:
            if mode=='unconditional_local':
                d=cdist(self.z[a],self.z);d[np.arange(n),a]=np.inf
                choices=np.argsort(d,axis=1,kind='stable')[:,:5]
                b=choices[np.arange(n),rng.integers(0,5,n)]
            else:
                choices=self.neighbors[style][local]
                b=ids[choices[np.arange(n),rng.integers(0,choices.shape[1],n)]]
            w=rng.beta(1,4,n)
        va=self.x.iloc[a].to_numpy();vb=self.x.iloc[b].to_numpy()
        values=va+w[:,None]*(vb-va)
        if mode=='conditional_resample': values=va.copy()
        result=pd.DataFrame(values,columns=self.x.columns)
        lineage=pd.DataFrame(dict(style=style,anchor_beer=self.x.index[a],neighbor_beer=self.x.index[b],weight=w,mode=mode))
        return result,lineage

LocalConditionalGenerator.__module__ = 'competition_pipeline'


def distribution_metrics(real,fake,train,imputer,scaler):
    r=scaler.transform(imputer.transform(real));f=scaler.transform(imputer.transform(fake));z=scaler.transform(imputer.transform(train))
    w=float(np.mean([wasserstein_distance(r[:,j],f[:,j]) for j in range(r.shape[1])]))
    # 231 dimensions are scaled by train statistics; distances are RMS per feature.
    nr=cdist(r,z)/np.sqrt(z.shape[1]);nf=cdist(f,z)/np.sqrt(z.shape[1])
    support=((fake.lt(train.min())|fake.gt(train.max())) & fake.notna()).to_numpy().sum()/max(fake.notna().to_numpy().sum(),1)
    return dict(marginal_wasserstein=w,mean_shift_RMS=float(np.sqrt(np.mean((r.mean(0)-f.mean(0))**2))),real_nearest_train=float(nr.min(1).mean()),generated_nearest_train=float(nf.min(1).mean()),duplicate_fraction=float((nf.min(1)<1e-10).mean()),outside_training_range=float(support),generated_missing_rate=float(fake.isna().to_numpy().mean()))


def generation():
    x,y,styles,sel,tr,te=load()
    bundle=joblib.load(OUT/'frozen_training_models.joblib');models=bundle['models']
    gen=LocalConditionalGenerator().fit(x.loc[tr],styles.loc[tr]);rng=np.random.default_rng(SEED)
    samples=[];lineages=[];validation=[];profiles=[]
    for style in sorted(styles.unique()):
        real=x.loc[te[styles.loc[te].eq(style).to_numpy()]]
        for mode in ['conditional_local','conditional_resample','unconditional_local']:
            fake,lineage=gen.sample(style,100,rng,mode)
            validation.append(dict(style=style,mode=mode,n_train=int((styles.loc[tr]==style).sum()),n_test=len(real),**distribution_metrics(real,fake,x.loc[tr],gen.imputer,gen.scaler)))
            if mode=='conditional_local':
                ids=[f'{style.replace("/","_")}_{i:03d}' for i in range(len(fake))]
                fake.index=ids;fake.index.name='sample_id';lineage.index=ids;lineage.index.name='sample_id'
                samples.append(fake);lineages.append(lineage)
                p=predict(models,fake);p.index.name='sample_id';profiles.append(p)
        print('generated',style,flush=True)
    fake=pd.concat(samples);lineage=pd.concat(lineages);profile=pd.concat(profiles)
    save(fake.reset_index(),'q3_generated_native_spectra.csv');save(lineage.reset_index(),'q3_generation_lineage.csv')
    save(profile.reset_index(),'q3_generated_sensory_standardized.csv');save(pd.DataFrame(validation),'q3_generation_heldout_validation.csv')
    cuts=pd.read_csv(OUT/'q1_train_quantile_cutpoints.csv').set_index('target')
    grades=pd.DataFrame({t:np.searchsorted(cuts.loc[t,[f'q{i}' for i in range(1,7)]].to_numpy(dtype=float),profile[t],side='right') for t in y.columns},index=fake.index)
    grades.index.name='sample_id';save(grades.reset_index(),'q3_generated_relative_grades.csv')
    q=profile.join(lineage[['style']]).groupby('style')
    summary=[]
    for s,g in q:
        for t in profile:
            summary.append(dict(style=s,target=t,predicted_mean=g[t].mean(),predicted_q05=g[t].quantile(.05),predicted_q95=g[t].quantile(.95),observed_training_mean=y.loc[tr[styles.loc[tr].eq(s).to_numpy()],t].mean()))
    save(pd.DataFrame(summary),'q3_style_sensory_profiles.csv')
    # Four compound thresholds are documented independently; zeros and conflicts remain explicit.
    cal=pd.read_excel(ROOT/'data/8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer')
    threshold=pd.read_excel(ROOT/'data/8.xlsx',sheet_name='Sensory thresholds');threshold['field_name_R']=threshold.field_name_R.astype('string').str.strip();threshold=threshold.set_index('field_name_R')
    audit=[]
    for f in DEFECTS:
        for beer in x.index:
            v=cal.at[beer,f];a1=x.at[beer,f];th=threshold.at[f,'threshold_median']
            unknown=pd.isna(v) or (f=='X2.3.butanedione' and pd.isna(a1) and v==0)
            conflict=f in ['X2.3.butanedione','X.E..2.nonenal'] and not (pd.isna(v) or pd.isna(a1)) and not np.isclose(v,a1,atol=1e-6,rtol=0)
            audit.append(dict(beer=beer,field=f,a1_native=a1,a8_concentration=v,unit=threshold.at[f,'unit'],threshold_median=th,threshold_mean=threshold.at[f,'threshold_mean'],unknown=unknown,source_conflict=conflict,above_median_threshold=pd.NA if unknown or pd.isna(th) or conflict else bool(v>th),interpretation='感知阈值情景，非经验证缺陷判据'))
    save(pd.DataFrame(audit),'q3_observed_defect_threshold_audit.csv')
    # Training-fitted, single validation attempt for target-style discriminability.
    from sklearn.ensemble import RandomForestClassifier
    classifier=Pipeline([('imputer',SimpleImputer(strategy='median')),('forest',RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=SEED,n_jobs=-1))]).fit(x.loc[tr],styles.loc[tr])
    from sklearn.metrics import accuracy_score,balanced_accuracy_score
    cp=classifier.predict(x.loc[te]);gp=classifier.predict(fake)
    save(pd.DataFrame([dict(heldout_style_accuracy=accuracy_score(styles.loc[te],cp),heldout_balanced_accuracy=balanced_accuracy_score(styles.loc[te],cp),generated_condition_agreement=accuracy_score(lineage['style'],gp))]),'q3_style_diagnostic.csv')
    # Correlation preservation on stable graph edges, not global mean alone.
    edges=pd.read_csv(ROOT/'outputs/network_edges_stable.csv');corr=[]
    for s in ['Tripel','Kriek','Lambic']:
        ids=lineage.index[lineage['style'].eq(s)];train=x.loc[tr[styles.loc[tr].eq(s).to_numpy()]]
        for _,e in edges.iterrows():
            l,r=e['指标1'],e['指标2']
            if train[l].nunique()>1 and train[r].nunique()>1 and fake.loc[ids,l].nunique()>1 and fake.loc[ids,r].nunique()>1:
                cr=train[l].corr(train[r],method='spearman');cg=fake.loc[ids,l].corr(fake.loc[ids,r],method='spearman')
                corr.append(dict(style=s,left=l,right=r,training_rho=cr,generated_rho=cg,absolute_difference=abs(cr-cg)))
    save(pd.DataFrame(corr),'q3_generated_network_fidelity.csv')
    transfer(gen,models,classifier,x,y,styles,tr,te)
    joblib.dump(gen,OUT/'conditional_generator.joblib')


def transfer(gen,models,classifier,x,y,styles,tr,te):
    # Constraint is stronger than any fixed monotone upper threshold: equal chemistry.
    # Stale hops is a sensory target, not a measured compound or known concentration threshold.
    sources=x.index[styles.eq('Pils/Lager')]
    target=x.loc[tr[styles.loc[tr].eq('Tripel').to_numpy()]]
    targetz=gen.scaler.transform(gen.imputer.transform(target))
    rows=[];spectra=[];sensory=[];defectlog=[];frontier=[]
    # Frozen Q1 stale_hops selection is a constant DummyRegressor, so its inequality
    # alone is vacuous. Add a prespecified nonconstant training-only guard, without
    # selecting/replacing any Q1 model from test performance.
    stale_guard=clone(templates()['随机森林']).fit(x.loc[tr],y.loc[tr,'stale_hops'])
    joblib.dump(stale_guard,OUT/'stale_hops_auxiliary_guard.joblib')
    guard_cv=[]
    for k,(fit,valid) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(tr),1):
        ids,vids=tr[fit],tr[valid]
        gm=clone(templates()['随机森林']).fit(x.loc[ids],y.loc[ids,'stale_hops'])
        gp=gm.predict(x.loc[vids]);actual=y.loc[vids,'stale_hops']
        guard_cv.append(dict(fold=k,n=len(vids),MAE=mean_absolute_error(actual,gp),baseline_MAE=mean_absolute_error(actual,np.repeat(y.loc[ids,'stale_hops'].mean(),len(vids))),R2=r2_score(actual,gp)))
    save(pd.DataFrame(guard_cv),'q3_stale_guard_training_cv.csv')
    classes=classifier.named_steps['forest'].classes_.tolist();tripidx=classes.index('Tripel')
    for beer in sources:
        src=x.loc[beer];sourcepred=predict(models,x.loc[[beer]]).iloc[0]
        source_guard=float(stale_guard.predict(x.loc[[beer]])[0])
        srcz=gen.scaler.transform(gen.imputer.transform(x.loc[[beer]]))[0]
        source_distance=float(cdist(srcz[None,:],targetz).min()/np.sqrt(x.shape[1]))
        candidates=[];meta=[]
        for donor in target.index:
            # Retain at least 25% of source native coordinates; avoid donor replacement.
            for alpha in [.25,.5,.75]:
                v=src+alpha*(target.loc[donor]-src)
                # Never fill missing source entries, including any unknown defect.
                v.loc[src.isna()]=np.nan
                v.loc[DEFECTS]=src.loc[DEFECTS]
                candidates.append(v);meta.append((donor,alpha))
        # Always retain identity candidate; if no improvement is feasible, report failure.
        candidates.append(src);meta.append((beer,0.))
        cand=pd.DataFrame(candidates).reset_index(drop=True)
        preds=predict(models,cand)
        zz=gen.scaler.transform(gen.imputer.transform(cand))
        distances=cdist(zz,targetz).min(1)/np.sqrt(x.shape[1])
        movement=np.sqrt(np.mean((zz-srcz)**2,axis=1))
        stale_ok=preds['stale_hops'].to_numpy()<=sourcepred['stale_hops']+1e-12
        guarded_stale=stale_guard.predict(cand)
        stale_ok &= guarded_stale<=source_guard+1e-12
        defect_ok=np.all(np.isclose(cand[DEFECTS].to_numpy(),np.tile(src[DEFECTS].to_numpy(),(len(cand),1)),atol=0,rtol=0,equal_nan=True),axis=1)
        objectives=distances+.1*movement
        feasible=stale_ok&defect_ok
        objectives[~feasible]=np.inf
        chosen=int(np.argmin(objectives));v=cand.iloc[chosen];p=preds.iloc[chosen]
        prob=classifier.predict_proba(cand)[:,tripidx]
        gain=source_distance-distances[chosen]
        for j,(donor,alpha) in enumerate(meta):
            frontier.append(dict(source_beer=beer,donor_beer=int(donor),alpha=alpha,target_distance=distances[j],movement=movement[j],stale_predicted=preds.iloc[j]['stale_hops'],auxiliary_stale_guard=guarded_stale[j],feasible=bool(feasible[j]),chosen=j==chosen))
        donor,alpha=meta[chosen]
        rows.append(dict(source_beer=beer,source_split='training' if beer in tr else 'heldout_test',target_style='Tripel',target_donor_beer=int(donor),alpha=alpha,source_target_distance=source_distance,transferred_target_distance=distances[chosen],distance_improvement=gain,relative_distance_improvement=gain/source_distance if source_distance else 0,movement=movement[chosen],source_stale_predicted=sourcepred['stale_hops'],transferred_stale_predicted=p['stale_hops'],source_stale_auxiliary_guard=source_guard,transferred_stale_auxiliary_guard=guarded_stale[chosen],defect_native_exactly_preserved=bool(defect_ok[chosen]),source_trip_probability=float(classifier.predict_proba(x.loc[[beer]])[0,tripidx]),transferred_trip_probability=float(prob[chosen]),status='feasible_improvement' if alpha>0 and gain>1e-9 else 'no_supported_improvement'))
        spectra.append({'source_beer':beer,**v.to_dict()})
        for t in y:
            sensory.append(dict(source_beer=beer,target=t,before=sourcepred[t],after=p[t],delta=p[t]-sourcepred[t]))
        for f in DEFECTS:
            known=not pd.isna(src[f])
            defectlog.append(dict(source_beer=beer,field=f,before_native=src[f],after_native=v[f],known=known,guarantee='native value unchanged; concentration non-worsening only under common monotone inverse' if known else 'unknown retained, no safety claim'))
        print('transfer',beer,rows[-1]['status'],flush=True)
    save(pd.DataFrame(rows),'q3_transfer_summary.csv');save(pd.DataFrame(spectra),'q3_transferred_native_spectra.csv')
    save(pd.DataFrame(sensory),'q3_transfer_sensory.csv');save(pd.DataFrame(defectlog),'q3_transfer_defect_constraints.csv');save(pd.DataFrame(frontier),'q3_transfer_search_frontier.csv')


def inventory():
    original={}
    for folder in ['data','notebooks','outputs']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or OUT in p.parents or p.name.startswith(('06_','07_','08_')):continue
            original[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    path=OUT/'original_sha256.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));assert all(original.get(k)==v for k,v in old.items()),'original file modified'
    else: path.write_text(json.dumps(original,ensure_ascii=False,indent=2),encoding='utf-8')
    versions={m:__import__(m).__version__ for m in ['numpy','pandas','scipy','sklearn','joblib','nbformat','nbclient','networkx']}
    (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':versions},indent=2),encoding='utf-8')
    print('Original file hash check:',len(original),'files intact',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['scoring','correlations','generation','all'],default='all');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);inventory()
    for stage,fn in [('scoring',scoring),('correlations',correlations),('generation',generation)]:
        if args.stage in ['all',stage]: fn()
    inventory()

if __name__=='__main__': main()
