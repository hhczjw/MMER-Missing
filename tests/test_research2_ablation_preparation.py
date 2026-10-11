"""CPU-only planning tests; never allocate a model or interrupt R1."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r2_prep_plan',ROOT/'scripts/research2_ablations/plan.py')
plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)


class Research2PreparationTests(unittest.TestCase):
    def test_two_independent_tracks_and_budgets(self):
        self.assertEqual(len(plan.variants('R2-M')),18)
        self.assertEqual(len(plan.variants('R2-T')),15)
        self.assertEqual(len(plan.jobs()),99)
        self.assertEqual(len(plan.jobs(core_only=True)),48)
        self.assertEqual(len({j['job_id'] for j in plan.jobs()}),99)

    def test_bad_seed_not_dropped(self):
        self.assertEqual(plan.SEEDS['R2-T'],[45,44,46])
        self.assertEqual({j['seed'] for j in plan.jobs() if j['track']=='R2-T'},{44,45,46})

    def test_reference_empty_changes(self):
        for t in ['R2-M','R2-T']:self.assertEqual(plan.variants(t)[0]['changes'],{})

    def test_different_teacher_and_selection_preserved(self):
        m,t=map(plan.reference_contract,['R2-M','R2-T'])
        self.assertIn('train mode',m['teacher']);self.assertIn('eval mode',t['teacher'])
        self.assertIn('full TAV MAE',m['validation']);self.assertIn('Token10',t['validation'])
        self.assertEqual(m['drop_last'],True);self.assertEqual(m['train_batches'],160)
        self.assertEqual(t['min_epochs'],20)

    def test_warmup_not_accidentally_unified(self):
        for actual,expected in zip(plan.warmup_factors('R2-M'),[.28,.46,.64,.82,1.]):self.assertAlmostEqual(actual,expected)
        self.assertEqual(plan.warmup_factors('R2-T'),[.2,.4,.6,.8,1.])

    def test_kd_disabled_without_disabling_rcd_or_teacher(self):
        a=dict(use_kd=True,use_rcd=True,kd_weight=.3,rcd_weight=.1,uni_weight=.1,gap_weight=.05)
        b=plan.apply_loss_changes(a,{'kd_weight':0.})
        self.assertTrue(b['use_kd']);self.assertTrue(b['use_rcd'])
        self.assertEqual(b['kd_weight'],0);self.assertEqual(b['rcd_weight'],.1)
        self.assertEqual(a['kd_weight'],.3)

    def test_uni_and_gap_are_independent(self):
        a=dict(use_uni=True,uni_weight=.1,gap_weight=.05)
        b=plan.apply_loss_changes(a,{'uni_weight':0.})
        self.assertTrue(b['use_uni']);self.assertEqual(b['gap_weight'],.05)
        self.assertEqual(b['uni_weight'],0)

    def test_feature_reconstruction_not_entire_contrastive_loss(self):
        row=next(r for r in plan.variants('R2-T') if r['id']=='R2-T-08')
        self.assertEqual(row['changes'],{'feature_rec_weight':0.})
        self.assertNotIn('contrastive_loss_weight',row['changes'])

    def test_no_misleading_token_combo_head_ablation(self):
        self.assertTrue(any(r['changes'].get('combo_heads') is False for r in plan.variants('R2-M')))
        self.assertFalse(any('combo_heads' in r['changes'] for r in plan.variants('R2-T')))

    def test_sampling_and_kd_policies_not_coupled(self):
        rows={r['id']:r for r in plan.variants('R2-M')}
        self.assertEqual(rows['R2-M-16']['changes'],{'fixed_feature_rate':.6})
        self.assertEqual(rows['R2-M-17']['changes'],{'cd_kd':False})
        self.assertEqual(plan.variants('R2-T')[-1]['changes'],{'fixed_token_rate_cap':.9})

    def test_plan_is_defensive_copy(self):
        a=plan.variants('R2-T');a[4]['changes']['kd_weight']=1.
        self.assertEqual(plan.variants('R2-T')[4]['changes']['kd_weight'],0.)

    def test_preparation_cannot_launch_training(self):
        source=(ROOT/'scripts/research2_ablations/prepare.py').read_text()
        self.assertNotIn('subprocess',source);self.assertNotIn('cuda:',source)
        self.assertIn('ready_for_training=False',source)
        self.assertIn('gpu_smoke_passed=False',source)


if __name__=='__main__':unittest.main()
