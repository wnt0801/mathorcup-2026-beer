"""Targeted portability/integrity regressions. Uses disposable test data only."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from integrity import sha,read_manifest,verify_manifest,normalize_manifest,relative_key


class IntegrityContracts(unittest.TestCase):
    def test_windows_keys_and_path_only_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'data').mkdir();data='啤酒\n0,1\n'.encode();(p/'data/a.csv').write_bytes(data)
            old=p/'old.json';old.write_text(json.dumps({'data\\a.csv':sha(data)}),encoding='utf-8')
            new=p/'new.json';normalize_manifest(old,new)
            self.assertEqual(read_manifest(old),read_manifest(new))
            self.assertNotIn('\\',new.read_text())
            self.assertEqual(verify_manifest(p,old)[0]['status'],'exact')

    def test_crlf_lf_is_reported_not_rebaselined(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);lf=b'x,y\n1,2\n';expected=sha(lf.replace(b'\n',b'\r\n'))
            (p/'a.csv').write_bytes(lf)
            row=verify_manifest(p,{'a.csv':expected})[0]
            self.assertEqual(row['status'],'eol_only')
            self.assertEqual(row['expected_sha256'],expected)
            self.assertEqual((p/'a.csv').read_bytes(),lf)

    def test_content_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'a.csv').write_bytes(b'x\n99\n')
            with self.assertRaisesRegex(ValueError,'CONTENT HASH MISMATCH'):
                verify_manifest(p,{'a.csv':sha(b'x\n1\n')})

    def test_binary_does_not_get_eol_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'a.pdf').write_bytes(b'pdf\n123')
            with self.assertRaises(ValueError):verify_manifest(p,{'a.pdf':sha(b'pdf\r\n123')})

    def test_invalid_paths_and_collisions(self):
        for key in ['../x','C:\\x','/tmp/x']:
            with self.assertRaises(ValueError):relative_key(key)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'m.json';p.write_text(json.dumps({'a\\b':'x','a/b':'x'}))
            with self.assertRaises(ValueError):read_manifest(p)


class AuditContracts(unittest.TestCase):
    def test_one_holdout_does_not_create_zero_variance(self):
        from review_audits import coverage_diagnostics
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        tr=pd.DataFrame({'a':[1.,2.,3.],'b':[0.,1.,2.]});h=tr.iloc[[0]];fake=tr.copy()
        im=SimpleImputer().fit(tr);sc=StandardScaler().fit(im.transform(tr))
        d=coverage_diagnostics(h,fake,tr,im,sc)
        self.assertTrue(np.isnan(d['holdout_total_scaled_variance']))
        self.assertEqual(d['holdout_variance_status'],'undefined_n_less_than_2')
        self.assertAlmostEqual(d['generated_train_total_scaled_variance_ratio'],1.)

    def test_invalid_font_gives_actionable_error(self):
        from review_config import chinese_font
        with self.assertRaisesRegex(FileNotFoundError,'--font'):
            chinese_font('/nonexistent/required-cjk-font.ttf')

    def test_western_font_is_not_a_cjk_fallback(self):
        from review_config import chinese_font
        import matplotlib
        p=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'
        with self.assertRaisesRegex(FileNotFoundError,'missing CJK'):
            chinese_font(p)


if __name__=='__main__':unittest.main(verbosity=2)
