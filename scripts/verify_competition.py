"""Independent checks of outputs against original data and donor lineage."""
from pathlib import Path
import hashlib
import json
import sys
import joblib
import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/competition_completion_v1'

def main():
    import competition_pipeline as cp
    original=json.loads((OUT/'original_sha256.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in original.items())
    x,y,s,sel,tr,te=cp.load()
    pp=pd.read_csv(OUT/'q1_out_of_sample_predictions.csv')
    assert len(pp)==250*50 and not pp.duplicated(['beer','target']).any()
    assert set(pp.beer)==set(x.index) and set(pp.target)==set(y.columns)
    assert pp.groupby('target').size().eq(250).all()
    assert pp.predicted_relative_grade.between(0,6).all()
    assert set(pp.loc[pp['split'].eq('heldout_test'),'beer'])==set(te)
    old=pd.read_csv(ROOT/'outputs/test_predictions_50.csv')
    join=pp.query("split=='heldout_test'").merge(old,left_on=['beer','target'],right_on=['beer','指标'],validate='one_to_one')
    assert len(join)==2500 and np.allclose(join.predicted_standardized,join['预测值'],atol=1e-8,rtol=1e-8)
    models=joblib.load(OUT/'frozen_training_models.joblib')['models']
    for t in y.columns:
        current=models[t].predict(x.loc[te])
        expected=pp.query("split=='heldout_test' and target==@t").set_index('beer').loc[te,'predicted_standardized']
        assert np.allclose(current,expected,atol=1e-8,rtol=1e-8)
    top=pd.read_csv(OUT/'q2_top3_all50_evidence.csv')
    assert len(top)==150 and top.groupby('target').size().eq(3).all()
    assert top.bootstrap_frequency.between(0,1).all()
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
    for p in sorted((ROOT/'notebooks').glob('0[678]_*.ipynb')):
        n=nbformat.read(p,as_version=4);nbformat.validate(n)
        codes=[c for c in n.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in codes)
        assert not any(o.output_type=='error' for c in codes for o in c.outputs)
        notebooks.append(p.name)
    assert len(notebooks)==3
    result={'status':'passed','original_files_unchanged':len(original),'q1_predictions':len(pp),'q2_top3_records':len(top),'generated_spectra':len(fake),'transferred_spectra':len(spectra),'executed_review_notebooks':notebooks,'checks':['original hashes','frozen prediction equality','saved model round-trip','all50 Top3','official matrix type','training-only donors','style-conditional donor labels','generated donor interpolation and missing mask','training support bounds','transfer reconstruction','exact defect preservation','nonconstant auxiliary stale guard constraint','executed notebooks without errors'],'limits':['No raw0–6 recovery claim','No experimental brewing or sensory safety validation','Unknown thresholds remain unknown']}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
