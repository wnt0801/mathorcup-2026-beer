"""Independent reconstruction of adopted calibration and bounded transfer evidence."""
from pathlib import Path
import argparse,json,hashlib,os,sys,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'scripts/competition_pipeline.py').is_file())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output).resolve();begin=time.monotonic();start=datetime.now(timezone.utc).isoformat()
    sys.path.insert(0,str(ROOT/'scripts'));os.environ['MATHORCUP_OUTPUT']=str(out)
    from competition_pipeline import templates,DEFECTS
    rawpath=HERE/'sources/raw_sensory_anonymized.csv'
    assert hashlib.md5(rawpath.read_bytes()).hexdigest()=='a8f911d8c0061e096abf3b087678e164'
    raw=pd.read_csv(rawpath,encoding='latin1').rename(columns={'A_ester_fruity':'A_esters_fruity','F_ester_fruity':'F_esters_fruity'})
    y=pd.read_csv(ROOT/'data/4.csv').set_index('beer').drop(columns=['beer_id','tasting_category_fine'])
    xx=pd.read_csv(ROOT/'data/1.csv').set_index('beer');x=xx.drop(columns=['beer_id','tasting_category_fine'])
    fits=pd.read_csv(out/'q1_preprocessing_fit_evidence.csv').query("target=='A_malt_all'").set_index('fold')
    tr=pd.Index(json.loads(fits.loc[0,'fit_beers']));te=set(json.loads(fits.loc[0,'valid_beers']))
    mapping=pd.read_csv(HERE/'q1_identity_mapping.csv');assert len(mapping)==250 and mapping.beer.is_unique and mapping.anonymous_beer.is_unique
    # Group by taster with numpy sample-SD, independently from pandas transform in experiment.
    zs=raw[list(y.columns)].astype(float).copy()
    for _,ids in raw.groupby('taster').groups.items():
        a=raw.loc[ids,y.columns].to_numpy(dtype=float);sd=a.std(0,ddof=1);sd[sd==0]=np.nan
        zs.loc[ids]=(a-a.mean(0))/sd
    means=zs.groupby(raw.beer_name).mean()
    target=y.loc[mapping.beer,['A_malt_all','F_malt_all','overall']].to_numpy()
    source=means.loc[mapping.anonymous_beer,['A_malt_all','F_malt_all','overall']].to_numpy()
    assert np.max(np.abs(target-source))<1e-6
    d=cdist(source,y[['A_malt_all','F_malt_all','overall']],metric='chebyshev');assert np.all((d<1e-6).sum(1)==1)
    raw['beer']=raw.beer_name.map(mapping.set_index('anonymous_beer').beer)
    panel=raw.groupby('beer')[list(y.columns)].mean()
    assert np.allclose(panel,pd.read_csv(HERE/'q1_observed_raw_panel_means.csv').set_index('beer').loc[panel.index,y.columns])
    original=pd.read_csv(out/'q1_out_of_sample_predictions.csv').set_index(['beer','target'])
    cv=pd.read_csv(HERE/'q1_calibrated_training_oof.csv').set_index(['beer','target'])
    params=pd.read_csv(HERE/'q1_calibration_parameters.csv').set_index('target')
    decision=pd.read_csv(HERE/'q1_calibration_decision.csv').set_index('target')
    rebuilt=[]
    for t in decision.index:
        for fold,r in fits.loc[fits.index>0].iterrows():
            fi=pd.Index(json.loads(r.fit_beers));va=pd.Index(json.loads(r.valid_beers))
            assert set(fi).isdisjoint(va) and set(fi)|set(va)==set(tr) and not (set(fi)|set(va))&te
            a,b=np.linalg.lstsq(np.column_stack([np.ones(len(fi)),y.loc[fi,t]]),panel.loc[fi,t],rcond=None)[0]
            model=np.array([original.at[(int(i),t),'predicted_standardized'] for i in va]);p=np.clip(a+b*model,0,6)
            stored=np.array([cv.at[(int(i),t),'predicted_raw_panel_mean'] for i in va]);assert np.allclose(p,stored,atol=1e-10)
            truth=panel.loc[va,t].to_numpy();base=np.repeat(panel.loc[fi,t].mean(),len(va))
            rebuilt.append(dict(target=t,fold=fold,calibration_MAE=np.abs(np.clip(a+b*y.loc[va,t],0,6)-truth).mean(),prediction_MAE=np.abs(p-truth).mean(),baseline_MAE=np.abs(base-truth).mean()))
        a,b=np.linalg.lstsq(np.column_stack([np.ones(len(tr)),y.loc[tr,t]]),panel.loc[tr,t],rcond=None)[0]
        assert np.allclose([a,b],params.loc[t,['intercept','slope']].to_numpy(float))
    q=pd.DataFrame(rebuilt)
    for t,g in q.groupby('target'):
        row=decision.loc[t];assert np.allclose(row[['calibration_MAE','prediction_MAE','baseline_MAE']].to_numpy(float),g[['calibration_MAE','prediction_MAE','baseline_MAE']].mean())
        wins=int((g.prediction_MAE<g.baseline_MAE).sum());passed=bool(g.calibration_MAE.mean()<=.3 and 1-g.prediction_MAE.mean()/g.baseline_MAE.mean()>=.1 and wins>=4)
        assert passed==bool(row.adopt)
    final=pd.read_csv(out/'q1_calibrated_predictions_partial.csv')
    accepted=set(decision.index[decision.adopt]);assert set(final.target)==accepted and len(final)==250*len(accepted)
    assert not final.duplicated(['beer','target']).any()
    for _,r in final.iterrows():
        p=cv.at[(r.beer,r.target),'predicted_raw_panel_mean'] if r['split']=='training_oof' else np.clip(params.at[r.target,'intercept']+params.at[r.target,'slope']*original.at[(r.beer,r.target),'predicted_standardized'],0,6)
        assert np.isclose(r.predicted_raw_panel_mean,p) and r.nearest_integer_grade==int(np.floor(p+.5))
    folds=json.loads((HERE/'q3_fold_scope.json').read_text());samples=pd.read_csv(HERE/'q3_transfer_cv_samples.csv');spectra=pd.read_csv(HERE/'q3_transfer_cv_spectra.csv')
    chem=pd.read_excel(ROOT/'data/8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer').reindex(columns=x.columns)
    for fold in folds:
        fi=fold['fit_beers'];va=fold['validation_beers'];assert set(fi).isdisjoint(va) and set(fi)|set(va)==set(tr) and not set(fi+va)&te
        train=x.loc[fi];im=SimpleImputer(strategy='median').fit(train);sc=StandardScaler().fit(im.transform(train))
        guard=clone(templates()['随机森林']).fit(train,y.loc[fi,'stale_hops'])
        for method in ['v3','C_concentration','N_distribution']:
            rows=samples.loc[(samples.seed==fold['seed'])&(samples.fold==fold['fold'])&(samples.method==method)]
            sp=spectra.loc[(spectra.seed==fold['seed'])&(spectra.fold==fold['fold'])&(spectra.method==method)].set_index('beer')[x.columns]
            assert set(rows.beer)==set(fold['source_beers'])==set(sp.index)
            for _,r in rows.iterrows():
                src=x.loc[r.beer];v=sp.loc[r.beer]
                assert r.donor in fi or r.alpha==0
                if r.alpha==0:expected=src.copy()
                elif method=='C_concentration':
                    expected=src.copy()
                    for field,relation in fold['concentration_relations'].items():
                        b=chem.at[r.beer,field];e=chem.at[r.donor,field]
                        if not(np.isfinite(b) and np.isfinite(e) and b>0 and e>0):continue
                        nb=np.log10(b) if relation=='log10' else b;ne=np.log10(e) if relation=='log10' else e
                        if np.isclose(nb,src[field],atol=1e-6,rtol=0) and np.isclose(ne,x.at[r.donor,field],atol=1e-6,rtol=0):
                            mixed=(1-r.alpha)*b+r.alpha*e;expected[field]=np.log10(mixed) if relation=='log10' else mixed
                else:
                    expected=src+r.alpha*(x.loc[r.donor]-src)
                    if method=='N_distribution':
                        fields=['esters_sum','aroma_hops_sum','acids_sum','sulfur_sum','kcalperc'];expected[fields]=src[fields]
                expected[src.isna()]=np.nan;expected[DEFECTS]=src[DEFECTS]
                assert np.allclose(expected,v,equal_nan=True,atol=1e-10)
                assert guard.predict(v.to_frame().T)[0]<=guard.predict(src.to_frame().T)[0]+1e-10
            zz=sc.transform(im.transform(sp));hz=sc.transform(im.transform(x.loc[fold['target_validation_beers']]))
            w=np.mean([wasserstein_distance(zz[:,j],hz[:,j]) for j in range(len(x.columns))]);assert np.allclose(rows.heldout_wasserstein,w)
            expected=cdist(zz,hz).min(1)/np.sqrt(len(x.columns));assert np.allclose(rows.set_index('beer').loc[sp.index,'heldout_nearest'],expected)
    defects=pd.read_csv(HERE/'q3_defect_concentration_cv.csv');trusted=defects[defects.known_trusted]
    assert np.allclose(trusted.before_concentration,trusted.after_concentration,atol=1e-8,rtol=1e-8)
    assert defects.loc[~defects.known_trusted,['before_concentration','after_concentration','before_excess','after_excess']].isna().all().all()
    assert defects.loc[defects.threshold.isna(),['before_excess','after_excess']].isna().all().all()
    assert not pd.read_csv(HERE/'q3_decision.csv').adopt.any()
    # PDF paper assertions focus on new scientific claims, not old run labels.
    from pypdf import PdfReader
    paper=PdfReader(out/'competition_paper.pdf');text='\n'.join(p.extract_text() for p in paper.pages)
    assert '25' in text and '0.4931' in text and '0.6466' in text and '未采纳新迁移' in text
    result={'status':'passed','started_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-begin,'calibrated_dimensions':len(accepted),'calibrated_predictions':len(final),'training_transfer_rows':len(samples),'actual_unique_pils':samples.beer.nunique(),'checks':['source MD5 and unique fingerprint identity','independent numpy per-taster standardization','raw label aggregation','foldwise affine calibration by independent least squares','adoption criteria and all reported failures','25-dimension predictions and provisional rounding','training-only fit/donor/evaluation scopes','selected spectra reconstruction','fold-fitted stale guard replay','heldout-Tripel distribution and nearest distance','trusted concentration preserved; unknowns remain unknown'],'paper_pages':len(paper.pages),'paper_sha256':hashlib.sha256((out/'competition_paper.pdf').read_bytes()).hexdigest(),'selection_heldout50_used':False,'interpretation':'Partial supervised panel-mean calibration; Q3 alternatives rejected, no new sensory safety guarantee'}
    (out/'v4_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
