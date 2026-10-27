"""Independent checks of outputs against original data and donor lineage."""
from pathlib import Path
import hashlib
import json
import sys
import joblib
import nbformat
import numpy as np
import pandas as pd
from datetime import datetime,timezone
from review_config import ROOT,OUT,FROZEN
from integrity import verify_manifest,read_manifest

def main():
    import competition_pipeline as cp
    original=read_manifest(OUT/'original_sha256.json')
    assert original==read_manifest(FROZEN/'source_sha256.json')
    integrity=verify_manifest(ROOT,original)
    x,y,s,sel,tr,te=cp.load()
    pp=pd.read_csv(OUT/'q1_out_of_sample_predictions.csv')
    assert len(pp)==250*50 and not pp.duplicated(['beer','target']).any()
    assert set(pp.beer)==set(x.index) and set(pp.target)==set(y.columns)
    assert pp.groupby('target').size().eq(250).all()
    assert pp.predicted_relative_grade.between(0,6).all()
    fits=pd.read_csv(OUT/'q1_preprocessing_fit_evidence.csv')
    assert len(fits)==300 and fits.groupby('target').size().eq(6).all()
    for _,r in fits.iterrows():
        fit=set(json.loads(r.fit_beers));valid=set(json.loads(r.valid_beers))
        assert fit.isdisjoint(valid) and fit.issubset(set(tr))
        assert len(fit)==int(r.n_fit) and int(r.n_fit)==(200 if r.fold==0 else 160)
    assert set(pp.loc[pp['split'].eq('heldout_test'),'beer'])==set(te)
    old=pd.read_csv(FROZEN/'test_predictions_50.csv')
    join=pp.query("split=='heldout_test'").merge(old,left_on=['beer','target'],right_on=['beer','指标'],validate='one_to_one')
    assert len(join)==2500 and np.allclose(join.predicted_standardized,join['预测值'],atol=1e-8,rtol=1e-8)
    models=joblib.load(OUT/'frozen_training_models.joblib')['models']
    for t in y.columns:
        current=models[t].predict(x.loc[te])
        expected=pp.query("split=='heldout_test' and target==@t").set_index('beer').loc[te,'predicted_standardized']
        assert np.allclose(current,expected,atol=1e-8,rtol=1e-8)
    # Independent target/grade reconstruction checks train-only cutpoints and fold scope.
    from sklearn.model_selection import KFold
    cuts=pd.read_csv(OUT/'q1_train_quantile_cutpoints.csv').set_index('target')
    for t in y.columns:
        rows=pp.loc[pp.target.eq(t)].set_index('beer')
        assert np.allclose(rows.loc[x.index,'actual_standardized'],y[t])
        c=y.loc[tr,t].quantile(np.arange(1,7)/7).to_numpy()
        assert np.allclose(c,cuts.loc[t,[f'q{i}' for i in range(1,7)]].to_numpy(dtype=float))
        assert np.allclose(rows.loc[te,'baseline_standardized'],y.loc[tr,t].mean())
        assert np.array_equal(rows.loc[te,'baseline_relative_grade'],np.repeat(np.searchsorted(c,y.loc[tr,t].mean(),side='right'),len(te)))
        assert np.array_equal(rows.loc[te,'predicted_relative_grade'],np.searchsorted(c,rows.loc[te,'predicted_standardized'],side='right'))
        for fit,valid in KFold(5,shuffle=True,random_state=42).split(tr):
            ids,vids=tr[fit],tr[valid];c=y.loc[ids,t].quantile(np.arange(1,7)/7).to_numpy()
            assert set(ids).isdisjoint(set(vids)) and set(ids).isdisjoint(set(te))
            assert np.allclose(rows.loc[vids,'baseline_standardized'],y.loc[ids,t].mean())
            assert np.array_equal(rows.loc[vids,'baseline_relative_grade'],np.repeat(np.searchsorted(c,y.loc[ids,t].mean(),side='right'),len(vids)))
            assert np.array_equal(rows.loc[vids,'actual_relative_grade'],np.searchsorted(c,y.loc[vids,t],side='right'))
            assert np.array_equal(rows.loc[vids,'predicted_relative_grade'],np.searchsorted(c,rows.loc[vids,'predicted_standardized'],side='right'))
    top=pd.read_csv(OUT/'q2_top3_all50_evidence.csv')
    assert len(top)==150 and top.groupby('target').size().eq(3).all()
    assert top.bootstrap_frequency.between(0,1).all()
    assert np.array_equal(top.ranking_stable_80pct,top.bootstrap_frequency.ge(.8))
    assert np.array_equal(top.candidate_class.eq('stable_association'),top.bootstrap_frequency.ge(.8)&top.direction_agrees)
    assert np.array_equal(top.candidate_class.eq('style_direction_reversal'),~top.direction_agrees)
    corr=pd.read_csv(OUT/'q2_official_correlation_type_check.csv').set_index('method')
    assert corr.at['spearman','max_abs_difference']<1e-12
    fake=pd.read_csv(OUT/'q3_generated_native_spectra.csv').set_index('sample_id')
    lineage=pd.read_csv(OUT/'q3_generation_lineage.csv').set_index('sample_id')
    assert len(fake)==2200 and fake.index.is_unique and set(fake.columns)==set(x.columns)
    assert lineage.groupby('style').size().eq(100).all()
    assert set(lineage.anchor_beer).issubset(set(tr)) and set(lineage.neighbor_beer).issubset(set(tr))
    assert all(s.loc[r.anchor_beer]==r['style']==s.loc[r.neighbor_beer] for _,r in lineage.iterrows())
    va=x.loc[lineage.anchor_beer].to_numpy();vb=x.loc[lineage.neighbor_beer].to_numpy()
    expected=va+lineage.weight.to_numpy()[:,None]*(vb-va)
    assert np.allclose(fake[x.columns],expected,equal_nan=True,atol=1e-8,rtol=1e-8)
    assert (fake[x.columns].min()>=x.loc[tr].min()-1e-8).all() and (fake[x.columns].max()<=x.loc[tr].max()+1e-8).all()
    assert np.array_equal(fake[x.columns].isna(),np.isnan(va)|np.isnan(vb))
    sensory=pd.read_csv(OUT/'q3_generated_sensory_standardized.csv').set_index('sample_id')
    generated_predictions=cp.predict(models,fake[x.columns])
    assert np.allclose(generated_predictions,sensory.loc[fake.index,y.columns],atol=1e-8,rtol=1e-8)
    # Replay fixed sampling solely to verify baseline diagnostics, never to select a method.
    generator=joblib.load(OUT/'conditional_generator.joblib');rng=np.random.default_rng(42)
    from scipy.spatial.distance import cdist
    diagnostics=pd.read_csv(OUT/'q3_style_diversity_coverage.csv').set_index(['style','mode'])
    assert len(diagnostics)==66
    for style in sorted(s.unique()):
        styletrain=x.loc[tr[s.loc[tr].eq(style).to_numpy()]]
        real=x.loc[te[s.loc[te].eq(style).to_numpy()]]
        for mode in ['conditional_local','conditional_resample','unconditional_local']:
            generated,_=generator.sample(style,100,rng,mode)
            d=diagnostics.loc[(style,mode)]
            assert int(d.n_train)==len(styletrain) and int(d.n_test)==len(real)
            vt=styletrain.var(ddof=1);vf=generated.var(ddof=1)
            eligible=vt.gt(1e-12)&vf.notna();scales=pd.Series(generator.scaler.scale_**2,index=x.columns)
            ratio=float((vf[eligible]/scales[eligible]).sum()/(vt[eligible]/scales[eligible]).sum())
            assert np.isclose(d.generated_train_total_scaled_variance_ratio,ratio,rtol=1e-10,atol=1e-10)
            zz=generator.scaler.transform(generator.imputer.transform(styletrain))
            gg=generator.scaler.transform(generator.imputer.transform(generated))
            hh=generator.scaler.transform(generator.imputer.transform(real))
            dist=cdist(zz,zz)/np.sqrt(x.shape[1]);np.fill_diagonal(dist,np.inf)
            radius=float(np.quantile(dist.min(1),.95))
            covered=((cdist(hh,gg)/np.sqrt(x.shape[1])).min(1)<=radius).mean()
            assert np.isclose(d.holdout_covered_at_train_nn95,covered)
            if len(real)<2:assert pd.isna(d.holdout_total_scaled_variance)
    summaries=pd.read_csv(OUT/'q3_summary_consistency_audit.csv')
    assert len(summaries)==5 and summaries.consistency_error.isna().all() and (~summaries.recomputed).all()
    evidence=pd.read_csv(OUT/'q3_transfer_defect_evidence.csv')
    assert len(evidence)==44 and evidence.native_preserved.all()
    rawdef=pd.read_csv(OUT/'q3_observed_defect_threshold_audit.csv')
    assert len(rawdef)==1000
    cal=pd.read_excel(ROOT/'data/8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer')
    for f in cp.DEFECTS:
        g=rawdef.loc[rawdef.field.eq(f)].set_index('beer').loc[x.index]
        assert np.array_equal(g.a1_missing,x[f].isna())
        assert np.array_equal(g.a8_missing,cal.loc[x.index,f].isna())
        if f in ['X2.3.butanedione','X.E..2.nonenal']:
            valid=x[f].notna()&cal.loc[x.index,f].notna()
            expected=valid&(~np.isclose(x[f],cal.loc[x.index,f],atol=1e-6,rtol=0))
            assert np.array_equal(g.numeric_source_conflict,expected)
    guardcv=pd.read_csv(OUT/'q3_stale_guard_training_cv.csv')
    assessment=json.loads((OUT/'q3_stale_guard_assessment.json').read_text(encoding='utf-8'))
    assert assessment['has_mean_MAE_advantage']==bool(guardcv.MAE.mean()<guardcv.baseline_MAE.mean())
    transfer=pd.read_csv(OUT/'q3_transfer_summary.csv')
    spectra=pd.read_csv(OUT/'q3_transferred_native_spectra.csv').set_index('source_beer')
    assert len(transfer)==len(spectra)==11
    guard=joblib.load(OUT/'stale_hops_auxiliary_guard.joblib')
    after=guard.predict(spectra[x.columns]);before=guard.predict(x.loc[spectra.index])
    assert np.all(after<=before+1e-12)
    for _,r in transfer.iterrows():
        src=x.loc[r.source_beer];v=spectra.loc[r.source_beer]
        assert r.alpha<=.75
        assert np.allclose(v[cp.DEFECTS],src[cp.DEFECTS],equal_nan=True,atol=0,rtol=0)
        assert v[src.isna()].isna().all()
        assert r.target_donor_beer in tr or r.alpha==0
        if r.alpha:
            donor=x.loc[r.target_donor_beer]
            z=src+r.alpha*(donor-src);z.loc[src.isna()]=np.nan;z.loc[cp.DEFECTS]=src[cp.DEFECTS]
            assert np.allclose(z,v,equal_nan=True,atol=1e-8,rtol=1e-8)
    notebooks=[]
    for p in sorted((OUT/'review_notebooks').glob('0[678]_*.ipynb')):
        n=nbformat.read(p,as_version=4);nbformat.validate(n)
        codes=[c for c in n.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in codes)
        assert not any(o.output_type=='error' for c in codes for o in c.outputs)
        notebooks.append(p.name)
    assert len(notebooks)==3
    from pypdf import PdfReader
    paper=PdfReader(OUT/'competition_paper.pdf');papertext='\n'.join(p.extract_text() for p in paper.pages)
    assert sum(len(p.images) for p in paper.pages)==5 and '![' not in papertext
    metrics=pd.read_csv(OUT/'q1_metrics.csv').query("split=='heldout_test'")
    assert f'{metrics.MAE.mean():.4f}' in papertext
    assert '模型代理' in papertext and '未验证' in papertext
    provenance=json.loads((OUT/'figures/figure_sources.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((OUT/k).read_bytes()).hexdigest()==v for k,v in provenance.items())
    validate_candidate_tables(pp,groups=None)
    result={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),'verification_root':str(ROOT),'output':str(OUT),'protected_source_files':len(original),'exact_source_files':sum(r['status']=='exact' for r in integrity),'eol_only_source_files':sum(r['status']=='eol_only' for r in integrity),'q1_predictions':len(pp),'q2_top3_records':len(top),'generated_spectra':len(fake),'transferred_spectra':len(spectra),'executed_review_notebooks':notebooks,'paper_pages':len(paper.pages),'checks':['fresh source hashes, no rebaselining','300 actual train/fold preprocessing statistic checks','frozen prediction equality','saved model round-trip','train-only relative cutpoints and OOF scopes','all50 Top3 and stability/reversal labels','Aroma/Flavour/All50 and relative-grade recomputation','official matrix type','training-only donors','style-conditional donor labels','generated donor interpolation and missing mask','training support bounds','generated sensory predictions','replayed66 diversity/coverage baseline rows','unverified summary formulas not recomputed','four-defect missing/conflict states','fixed training-only guard CV assessment','transfer reconstruction and compound/prediction deltas','exact native defect preservation incl missing','model proxy guard constraint','executed review notebooks without errors','paper/CSV numerical scope and figure-source hashes'],'limits':['Raw0–6 intensity not identified','Stale hops guard has no demonstrated advantage, model proxy only','Summary constituent identities unverified','Actual brewing/causal experiments not performed; research limitations, not added contest submission requirements','No Linux/macOS execution claim']}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

def validate_candidate_tables(predictions,groups=None):
    summaries=pd.read_csv(OUT/'q1_group_metrics.csv')
    for _,r in summaries.iterrows():
        d=predictions.loc[predictions['split'].eq(r['split'])]
        if r['group']=='Aroma':d=d.loc[d.target.str.startswith('A_')]
        if r['group']=='Flavour':d=d.loc[d.target.str.startswith('F_')]
        assert d.target.nunique()==r.n_targets and d.beer.nunique()==r.n_beers
        mae=np.abs(d.actual_standardized-d.predicted_standardized).mean();base=np.abs(d.actual_standardized-d.baseline_standardized).mean()
        assert np.isclose(mae,r.model_MAE) and np.isclose(base,r.baseline_MAE)
        assert np.isclose(100*(1-mae/base),r.improvement_pct)
        assert np.isclose(np.abs(d.actual_relative_grade-d.predicted_relative_grade).mean(),r.model_grade_MAE)
        assert np.isclose(np.abs(d.actual_relative_grade-d.baseline_relative_grade).mean(),r.baseline_grade_MAE)
        assert np.isclose((d.actual_relative_grade==d.predicted_relative_grade).mean(),r.model_grade_accuracy)
        assert np.isclose((d.actual_relative_grade==d.baseline_relative_grade).mean(),r.baseline_grade_accuracy)
    long=pd.read_csv(OUT/'q3_transfer_compound_changes.csv');source=pd.read_csv(ROOT/'data/1.csv').set_index('beer');after=pd.read_csv(OUT/'q3_transferred_native_spectra.csv').set_index('source_beer')
    assert len(long)==11*231
    for _,r in long.iterrows():
        assert np.isclose(r.before_native,source.at[r.source_beer,r.field],equal_nan=True)
        assert np.isclose(r.after_native,after.at[r.source_beer,r.field],equal_nan=True)
        assert np.isclose(r.delta_native,r.after_native-r.before_native,equal_nan=True)
    import joblib
    diagnostic=joblib.load(OUT/'style_diagnostic_classifier.joblib')
    bundle=joblib.load(OUT/'frozen_training_models.joblib');xcols=bundle['features']
    import competition_pipeline as cp
    predicted_before=cp.predict(bundle['models'],source.loc[after.index,xcols])
    predicted_after=cp.predict(bundle['models'],after[xcols])
    changes=pd.read_csv(OUT/'q3_transfer_sensory.csv')
    assert len(changes)==11*50 and not changes.duplicated(['source_beer','target']).any()
    for _,r in changes.iterrows():
        assert np.isclose(r.before,predicted_before.at[r.source_beer,r.target])
        assert np.isclose(r.after,predicted_after.at[r.source_beer,r.target])
        assert np.isclose(r.delta,r.after-r.before)
    summary=pd.read_csv(OUT/'q3_transfer_sensory_summary.csv').set_index('target')
    for t,g in changes.groupby('target'):
        assert np.allclose(summary.loc[t,['before_mean','after_mean','paired_mean_delta']].to_numpy(dtype=float),[g.before.mean(),g.after.mean(),g.delta.mean()])
    split=pd.read_csv(OUT/'split_manifest.csv')
    sd=source.loc[split.loc[split['split'].eq('training'),'beer'],xcols].std(ddof=0)
    expected=long.delta_native/long.field.map(sd).replace(0,np.nan)
    assert np.allclose(long.delta_train_SD,expected,equal_nan=True)
    edges=pd.read_csv(OUT/'q2_official_graph_edges.csv')
    edgekeys={frozenset([r.left,r.right]):r.rho for _,r in edges.iterrows()}
    top=pd.read_csv(OUT/'q2_top3_all50_evidence.csv')
    context=pd.read_csv(OUT/'q2_network_explanation.csv').set_index('target')
    for t,g in top.groupby('target'):
        fields=g.field.tolist()
        values=[edgekeys[frozenset([a,b])] for i,a in enumerate(fields) for b in fields[i+1:] if frozenset([a,b]) in edgekeys]
        assert int(context.at[t,'top3_edges_at_abs_rho06'])==len(values)
        assert np.allclose(json.loads(context.at[t,'edge_rhos']),values)
    transfer=pd.read_csv(OUT/'q3_transfer_summary.csv').set_index('source_beer')
    assert np.array_equal(diagnostic.predict(after[xcols]),transfer.loc[after.index,'after_diagnostic_style'])
    grouped=pd.read_csv(OUT/'q1_missingness_by_group.csv');all50=grouped[grouped['group'].eq('All50')].sort_values(['additional_missing_rate','repeat'])
    original=pd.read_csv(OUT/'q1_missingness_sensitivity.csv').sort_values(['additional_missing_rate','repeat'])
    assert len(grouped)==60 and np.allclose(all50.MAE,original.MAE)


if __name__=='__main__':main()
