import ast
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/ablations'))
from plan import variants,jobs,selector,EXPERIMENT
from metrics import eval_regression


class AblationPlanTests(unittest.TestCase):
    def test_catalog_and_budget(self):
        rows=variants()
        self.assertEqual(len(rows),37)
        self.assertEqual(len({r['id'] for r in rows}),37)
        self.assertEqual(sum(r['track']=='R1' for r in rows),11)
        for t in ['R2-M','R2-T']:self.assertEqual(sum(r['track']==t for r in rows),13)
        self.assertEqual(len(jobs()),111)
        self.assertEqual(sum(x['batch']=='core' for x in rows),21)

    def test_defensive_catalog_copy(self):
        rows=variants();rows[1]['changes']['uniform_routing']=False
        self.assertTrue(variants()[1]['changes']['uniform_routing'])

    def test_paired_fixed_order(self):
        queue=jobs()
        self.assertEqual([queue[i]['seed'] for i in [0,37,74]],[42,44,45])
        self.assertEqual(queue[0]['id'],'R1-00')
        self.assertEqual(queue[7]['id'],'R2-M-00')
        self.assertEqual(queue[14]['id'],'R2-T-00')
        self.assertTrue(all(x['batch']=='core' for x in queue[:21]))

    def test_reference_not_hpo_selected(self):
        for r in variants():
            if r['id'].endswith('-00'):self.assertEqual(r['changes'],{})

    def test_validation_selectors(self):
        a=dict(Acc2_nonzero=.8,Acc7=.3,MAE=.7)
        b=dict(Acc2_nonzero=.7,Acc7=.4,MAE=.6)
        self.assertGreater(selector('R1',a),selector('R1',b))
        self.assertGreater(selector('R2-M',b),selector('R2-M',a))
        self.assertGreater(selector('R2-T',b),selector('R2-T',a))

    def test_round_half_and_binary_conventions(self):
        m=eval_regression([-.5,.5,0],[0,1,-1])
        self.assertAlmostEqual(m['Acc7'],1/3)
        self.assertAlmostEqual(m['Acc2_has0'],1/3)
        self.assertAlmostEqual(m['Acc2_nonzero'],1.)

    def test_kd_and_rcd_independent_changes(self):
        for track in ['R2-M','R2-T']:
            a=next(x for x in variants() if x['id']==track+'-04')['changes']
            b=next(x for x in variants() if x['id']==track+'-05')['changes']
            both=next(x for x in variants() if x['id']==track+'-09')['changes']
            self.assertEqual(a,dict(kd_weight=0.0))
            self.assertEqual(b,dict(rcd_weight=0.0))
            self.assertEqual(both,dict(kd_weight=0.0,rcd_weight=0.0))

    def test_reconstruction_split(self):
        for track in ['R2-M','R2-T']:
            rows={r['id']:r for r in variants()}
            self.assertEqual(rows[track+'-07']['changes'],dict(text_rec_weight=0.))
            self.assertEqual(rows[track+'-08']['changes'],dict(feature_rec_weight=0.))
        text=(EXPERIMENT/'snapshot_r2/models/umrm_v2.py').read_text()
        self.assertIn('self.last_components',text)
        self.assertIn("getattr(self, 'ablation_reconstruction_weight', 1.0) * recon_loss",text)

    def test_sources_parse(self):
        for folder in [ROOT/'scripts/ablations',EXPERIMENT/'snapshot_r1/models',EXPERIMENT/'snapshot_r2/models']:
            for path in folder.glob('*.py'):ast.parse(path.read_text(),filename=str(path))


if __name__=='__main__':unittest.main()
