"""Bounded, preregistered comparisons. Existing data/results are read-only."""
from pathlib import Path
import os,sys,json,hashlib,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import joblib

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'scripts/competition_pipeline.py').is_file())
V3=Path(os.environ.get('MATHORCUP_BASE',str(ROOT/'results' if (ROOT/'results').is_dir() else ROOT/'outputs/competition_candidate_v3'))).resolve()
sys.path.insert(0,str(ROOT/'scripts'))
os.environ['MATHORCUP_OUTPUT']=str(HERE)
from competition_pipeline import templates,DEFECTS
SUMMARY=['esters_sum','aroma_hops_sum','acids_sum','sulfur_sum','kcalperc']
def save(d,n):d.to_csv(HERE/n,index=False,encoding='utf-8-sig')
def dump(d,n):(HERE/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
fits=pd.read_csv(V3/'q1_preprocessing_fit_evidence.csv')
fitrows=fits.loc[fits.target.eq('A_malt_all')].set_index('fold')
TRAIN=pd.Index(json.loads(fitrows.loc[0,'fit_beers']))
TEST=set(json.loads(fitrows.loc[0,'valid_beers']))
rawx=pd.read_csv(ROOT/'data/1.csv').set_index('beer')
rawy=pd.read_csv(ROOT/'data/4.csv').set_index('beer').drop(columns=['beer_id','tasting_category_fine'])
X=rawx.loc[TRAIN].drop(columns=['beer_id','tasting_category_fine'])
Y=rawy.loc[TRAIN];styles=rawx.loc[TRAIN,'tasting_category_fine']
assert not set(TRAIN)&TEST and len(TRAIN)==200

def calibration():
    path=HERE/'sources/raw_sensory_anonymized.csv'
    assert hashlib.md5(path.read_bytes()).hexdigest()=='a8f911d8c0061e096abf3b087678e164'
    raw=pd.read_csv(path,encoding='latin1').rename(columns={'A_ester_fruity':'A_esters_fruity','F_ester_fruity':'F_esters_fruity'})
    cols=list(Y.columns);assert raw[cols].ge(0).all().all() and raw[cols].le(6).all().all()
    means=raw.groupby('taster')[cols].transform('mean');sd=raw.groupby('taster')[cols].transform('std')
    z=(raw[cols]-means)/sd
    reconstructed=z.groupby(raw.beer_name).mean()
    # Identity verification only, no predictive model is fit to the held-out set.
    anchors=['A_malt_all','F_malt_all','overall']
    distance=cdist(reconstructed[anchors],rawy[anchors],metric='chebyshev')
    chosen=distance.argmin(axis=1)
    assert np.all(distance.min(1)<1e-6) and len(set(chosen))==250
    assert np.all((distance<1e-6).sum(1)==1)
    mapping=pd.DataFrame({'anonymous_beer':reconstructed.index,'beer':rawy.index[chosen],'anchor_max_error':distance.min(1),'second_candidate_error':np.sort(distance,axis=1)[:,1]})
    idmap=mapping.set_index('anonymous_beer').beer
    raw['beer']=raw.beer_name.map(idmap)
    reconstructed.index=reconstructed.index.map(idmap)
    error=(reconstructed.loc[TRAIN]-Y).abs()
    evidence=pd.DataFrame({'target':cols,'train_reconstruction_max_error':error.max().reindex(cols).values})
    evidence['eligible']=evidence.train_reconstruction_max_error.le(1e-6)
    # Independent remaining attributes verify the anchor-based identity, not nearest-profile guessing.
    assert evidence.eligible.sum()>=30
    save(mapping,'q1_identity_mapping.csv');save(evidence,'q1_reconstruction_by_dimension.csv')
    panel=raw.groupby('beer')[cols].mean()
    per_taster=raw.groupby(['beer','taster'])[cols].mean().groupby('beer').mean()
    save((panel-per_taster).abs().loc[TRAIN].agg(['mean','max']).T.reset_index(names='target'),'q1_aggregation_sensitivity.csv')
    # Original-scale labels are available, but no claim of a unique inverse after panel averaging.
    save(panel.reset_index(),'q1_observed_raw_panel_means.csv')
    pred=pd.read_csv(V3/'q1_out_of_sample_predictions.csv').query("split=='training_oof'")
    records=[];predrecords=[];params=[]
    eligible=evidence.loc[evidence.eligible,'target'].tolist()
    for t in eligible:
        for fold,r in fitrows.loc[fitrows.index>0].iterrows():
            fi=pd.Index(json.loads(r.fit_beers));va=pd.Index(json.loads(r.valid_beers))
            assert set(fi).isdisjoint(va) and set(fi)|set(va)==set(TRAIN)
            reg=LinearRegression().fit(Y.loc[fi,[t]],panel.loc[fi,t])
            assert reg.coef_[0]>0
            observed=np.clip(reg.predict(Y.loc[va,[t]]),0,6)
            p=pred.loc[pred.target.eq(t)].set_index('beer').loc[va]
            calibrated=np.clip(reg.predict(pd.DataFrame({t:p.predicted_standardized},index=va)),0,6)
            actual=panel.loc[va,t].to_numpy();baseline=np.repeat(panel.loc[fi,t].mean(),len(va))
            records.append(dict(target=t,fold=int(fold),n=len(va),calibration_MAE=float(np.abs(observed-actual).mean()),prediction_MAE=float(np.abs(calibrated-actual).mean()),baseline_MAE=float(np.abs(baseline-actual).mean()),relative_grade_vs_raw_MAE=float(np.abs(p.predicted_relative_grade.to_numpy()-actual).mean())))
            for i,b in enumerate(va):predrecords.append(dict(beer=int(b),target=t,fold=int(fold),raw_panel_mean=actual[i],predicted_raw_panel_mean=calibrated[i],baseline_raw_mean=baseline[i]))
        reg=LinearRegression().fit(Y[[t]],panel.loc[TRAIN,t])
        params.append(dict(target=t,intercept=float(reg.intercept_),slope=float(reg.coef_[0])))
    results=pd.DataFrame(records);summary=results.groupby('target').agg(calibration_MAE=('calibration_MAE','mean'),prediction_MAE=('prediction_MAE','mean'),baseline_MAE=('baseline_MAE','mean')).reset_index()
    wins=results.assign(win=results.prediction_MAE<results.baseline_MAE).groupby('target').win.sum()
    summary['better_folds']=summary.target.map(wins);summary['improvement']=1-summary.prediction_MAE/summary.baseline_MAE
    summary['adopt']=(summary.calibration_MAE<=.30)&(summary.improvement>=.10)&(summary.better_folds>=4)
    save(results,'q1_calibration_cv.csv');save(summary,'q1_calibration_decision.csv');save(pd.DataFrame(predrecords),'q1_calibrated_training_oof.csv');save(pd.DataFrame(params),'q1_calibration_parameters.csv')
    dump({'rows':len(raw),'tasters':raw.taster.nunique(),'beers':raw.beer.nunique(),'anchors':anchors,'identity':'three exact standardized fingerprints, unique bijection; anonymous numeric IDs are NOT supplied beer IDs','eligible_dimensions':eligible,'excluded_dimensions':evidence.loc[~evidence.eligible,'target'].tolist(),'adopted_dimensions':summary.loc[summary.adopt,'target'].tolist(),'calibration':'affine supervised panel mean estimate; not exact original-rating inverse','source_MD5':hashlib.md5(path.read_bytes()).hexdigest(),'source_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'test_performance_used_for_selection':False},'q1_decision.json')
    print('Q1 eligible/adopted',len(eligible),int(summary.adopt.sum()),flush=True)

def stale():
    records=[]
    for fold,r in fitrows.loc[fitrows.index>0].iterrows():
        fi=json.loads(r.fit_beers);va=json.loads(r.valid_beers)
        for label,key in [('mean','均值基准'),('ridge100','岭回归'),('frozen_forest','随机森林')]:
            model=clone(templates()[key]).fit(X.loc[fi],Y.loc[fi,'stale_hops'])
            p=model.predict(X.loc[va]);records.append(dict(fold=int(fold),method=label,MAE=float(np.abs(p-Y.loc[va,'stale_hops']).mean())))
    d=pd.DataFrame(records);save(d,'stale_cv.csv');base=d.loc[d.method.eq('mean')].set_index('fold').MAE
    rows=[]
    for name,g in d.groupby('method'):
        improvement=1-g.MAE.mean()/base.mean();wins=int((g.set_index('fold').MAE<base).sum())
        rows.append(dict(method=name,MAE=g.MAE.mean(),improvement=improvement,better_folds=wins,passes=bool(improvement>=.05 and wins>=4)))
    save(pd.DataFrame(rows),'stale_decision.csv');print('Stale comparison done',flush=True)

def transfer():
    cal=pd.read_excel(ROOT/'data/8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').reindex(index=TRAIN,columns=X.columns)
    info=pd.read_csv(V3/'chemical_name_mapping.csv').set_index('field_name_R')
    roles=pd.read_csv(V3/'feature_roles.csv');singles=set(roles.loc[roles.iloc[:,1].eq('单项化学测量候选')].iloc[:,0])
    records=[];spectra=[];foldlog=[];concentration=[]
    for seed in [42,43,44]:
        for fold,(fit,valid) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,styles),1):
            fi=X.index[fit];va=X.index[valid]
            assert set(fi).isdisjoint(va) and not (set(fi)|set(va))&TEST
            train=X.loc[fi];targets=train.loc[styles.loc[fi].eq('Tripel')];sources=X.loc[va[styles.loc[va].eq('Pils/Lager')]];held=X.loc[va[styles.loc[va].eq('Tripel')]]
            assert len(sources)>0 and len(held)>1
            im=SimpleImputer(strategy='median').fit(train);sc=StandardScaler().fit(im.transform(train))
            def z(d):return sc.transform(im.transform(d))
            tz=z(targets);hz=z(held);center=tz.mean(0)
            guard=clone(templates()['随机森林']).fit(train,Y.loc[fi,'stale_hops'])
            classifier=Pipeline([('imputer',SimpleImputer(strategy='median')),('forest',RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=2,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1))]).fit(train,styles.loc[fi])
            diagnostic_accuracy=float((classifier.predict(X.loc[va])==styles.loc[va]).mean())
            # All transform discovery uses fold-fit rows, never corresponding validation rows.
            relations={}
            for f in singles- set(DEFECTS):
                mask=cal.loc[fi,f].gt(0)&train[f].notna()
                if mask.sum()<10 or pd.isna(info.at[f,'unit']):continue
                a=train.loc[mask,f];b=cal.loc[fi].loc[mask,f]
                good=[n for n,v in [('identity',b),('log10',np.log10(b))] if np.allclose(a,v,atol=1e-6,rtol=0)]
                if len(good)==1:relations[f]=good[0]
            foldlog.append(dict(seed=seed,fold=fold,fit_beers=list(map(int,fi)),validation_beers=list(map(int,va)),source_beers=list(map(int,sources.index)),target_validation_beers=list(map(int,held.index)),concentration_relations=relations))
            method_spectra={m:[] for m in ['v3','C_concentration','N_distribution']}
            for beer,src in sources.iterrows():
                srcz=z(src.to_frame().T)[0];beforeguard=guard.predict(src.to_frame().T)[0]
                for method in method_spectra:
                    candidates=[];meta=[]
                    for donor,d in targets.iterrows():
                        for alpha in [.25,.5,.75]:
                            if method=='C_concentration':
                                v=src.copy()
                                for f,relation in relations.items():
                                    # Validation source chemistry is an input, not used to fit transforms.
                                    b=cal.at[beer,f];e=cal.at[donor,f]
                                    if not(np.isfinite(b) and np.isfinite(e) and b>0 and e>0):continue
                                    nativeb=np.log10(b) if relation=='log10' else b
                                    nativee=np.log10(e) if relation=='log10' else e
                                    if not(np.isclose(nativeb,src[f],atol=1e-6,rtol=0) and np.isclose(nativee,d[f],atol=1e-6,rtol=0)):continue
                                    mixed=(1-alpha)*b+alpha*e;v[f]=np.log10(mixed) if relation=='log10' else mixed
                            else:
                                v=src+alpha*(d-src)
                                if method=='N_distribution':v.loc[SUMMARY]=src[SUMMARY]
                            v.loc[src.isna()]=np.nan;v.loc[DEFECTS]=src[DEFECTS]
                            candidates.append(v);meta.append((int(donor),alpha))
                    candidates.append(src.copy());meta.append((int(beer),0.))
                    cand=pd.DataFrame(candidates,columns=X.columns)
                    zz=z(cand);movement=np.sqrt(np.mean((zz-srcz)**2,axis=1));near=cdist(zz,tz).min(1)/np.sqrt(X.shape[1])
                    ok=guard.predict(cand)<=beforeguard+1e-12
                    if method=='N_distribution':
                        ok&=(movement<=.75+1e-12)&(np.max(np.abs(zz-srcz),axis=1)<=2+1e-12)
                        objective=np.sqrt(np.mean((zz-center)**2,axis=1))+.5*movement
                    else:objective=near+.1*movement
                    objective[~ok]=np.inf;chosen=int(np.argmin(objective));v=cand.iloc[chosen]
                    assert np.allclose(v[DEFECTS],src[DEFECTS],equal_nan=True,rtol=0,atol=0)
                    assert v[src.isna()].isna().all()
                    donor,alpha=meta[chosen];newz=zz[chosen];nearheld=cdist(newz[None,:],hz).min()/np.sqrt(X.shape[1])
                    outside=((v<train.min())|(v>train.max()))&v.notna()
                    records.append(dict(seed=seed,fold=fold,beer=int(beer),method=method,donor=donor,alpha=alpha,optimized_distance=near[chosen],heldout_nearest=nearheld,movement=movement[chosen],defect_native_preserved=True,known_to_missing=int((src.notna()&v.isna()).sum()),outside_train_range=int(outside.sum()),candidate_infeasible_fraction=float((~ok[:-1]).mean()),fallback=alpha==0,diagnostic_trip=classifier.predict(v.to_frame().T)[0]=='Tripel',diagnostic_validation_accuracy=diagnostic_accuracy))
                    method_spectra[method].append(v)
                    spectra.append({'seed':seed,'fold':fold,'beer':int(beer),'method':method,**v.to_dict()})
                    for f in DEFECTS:
                        original=cal.at[beer,f];relation='log10' if f in ['DMS','X4VG'] else 'identity'
                        native=np.log10(original) if relation=='log10' and original>0 else original
                        known=bool(pd.notna(src[f]) and np.isfinite(original) and original>0 and np.isclose(src[f],native,atol=1e-6,rtol=0))
                        before=original if known else np.nan
                        after=(10**v[f] if relation=='log10' else v[f]) if known else np.nan
                        tau=info.at[f,'threshold_median'];unit=info.at[f,'unit']
                        if known:assert np.isclose(after,before,rtol=1e-8,atol=1e-8)
                        concentration.append(dict(seed=seed,fold=fold,beer=int(beer),method=method,field=f,known_trusted=known,before_concentration=before,after_concentration=after,unit=unit,transform=relation,threshold=tau,before_excess=max(before-tau,0) if known and pd.notna(tau) else np.nan,after_excess=max(after-tau,0) if known and pd.notna(tau) else np.nan,claim='same trusted concentration; threshold is reference only' if known else 'unknown/conflicted; native representation frozen only'))
            for method,vs in method_spectra.items():
                fake=pd.DataFrame(vs);gz=z(fake)
                w=np.mean([wasserstein_distance(gz[:,j],hz[:,j]) for j in range(X.shape[1])])
                for row in records:
                    if row['seed']==seed and row['fold']==fold and row['method']==method:row['heldout_wasserstein']=float(w)
            print('Transfer seed/fold',seed,fold,'sources',len(sources),'target holdout',len(held),flush=True)
    d=pd.DataFrame(records);save(d,'q3_transfer_cv_samples.csv');save(pd.DataFrame(spectra),'q3_transfer_cv_spectra.csv');save(pd.DataFrame(concentration),'q3_defect_concentration_cv.csv');dump(foldlog,'q3_fold_scope.json')
    grouped=d.groupby(['method','seed','fold']).agg(wasserstein=('heldout_wasserstein','first'),nearest=('heldout_nearest','mean'),movement=('movement','mean'),new_missing=('known_to_missing','mean'),outside=('outside_train_range','mean'),fallback=('fallback','mean'),infeasible=('candidate_infeasible_fraction','mean'),diagnostic_trip=('diagnostic_trip','mean')).reset_index()
    save(grouped,'q3_transfer_cv_folds.csv');base=grouped.loc[grouped.method.eq('v3')].set_index(['seed','fold']);rows=[]
    for method,g in grouped.groupby('method'):
        v=g.set_index(['seed','fold']);gain=1-v.wasserstein.mean()/base.wasserstein.mean();wins=int((v.wasserstein<base.wasserstein).sum())
        passed=bool(method!='v3' and gain>=.05 and wins>=7 and v.nearest.mean()<=base.nearest.mean() and v.movement.mean()<=1.05*base.movement.mean() and v.new_missing.mean()<=base.new_missing.mean() and v.outside.mean()<=base.outside.mean() and v.fallback.mean()<=base.fallback.mean()+.10)
        rows.append(dict(method=method,**v.mean(numeric_only=True).to_dict(),wasserstein_improvement=gain,better_folds=wins,adopt=passed))
    save(pd.DataFrame(rows),'q3_decision.csv')

def diagnose_v3():
    classifier=joblib.load(V3/'style_diagnostic_classifier.joblib')
    t=pd.read_csv(V3/'q3_transfer_summary.csv');t=t.loc[t.source_split.eq('training')]
    after=pd.read_csv(V3/'q3_transferred_native_spectra.csv').set_index('source_beer')
    rows=[]
    for _,r in t.iterrows():
        src=X.loc[r.source_beer];donor=X.loc[r.target_donor_beer]
        blend=src+r.alpha*(donor-src)
        for label,v in [('source',src),('donor',donor),('blend_without_defect_freeze',blend),('v3_frozen_defects',after.loc[r.source_beer,X.columns])]:
            rows.append(dict(beer=int(r.source_beer),stage=label,predicted_style=classifier.predict(v.to_frame().T)[0],tripel_probability=classifier.predict_proba(v.to_frame().T)[0,list(classifier.classes_).index('Tripel')]))
    save(pd.DataFrame(rows),'v3_training_transfer_diagnosis.csv')

if __name__=='__main__':
    started=datetime.now(timezone.utc).isoformat();begin=time.monotonic()
    protocol=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest()
    try:
        if '--transfer-only' not in sys.argv:calibration();stale()
        transfer();diagnose_v3()
        dump({'status':'passed','started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-begin,'protocol_sha256':protocol,'heldout50_used_for_predictive_selection':False},'execution.json')
    except Exception as e:
        dump({'status':'failed','started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-begin,'error':repr(e),'protocol_sha256':protocol},'execution.json');raise
