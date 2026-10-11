import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from token_hpo_plan import candidates,joint,champions,finalists


def row(name,a7,a2,mae=1):
    p=dict(avg=dict(Acc7=a7,Acc2_nonzero=a2,MAE=mae),epoch=1)
    return dict(id=name,best=dict(joint=p,Acc7=p))


class TokenHPOPlan(unittest.TestCase):
    def test_fixed_bounded_plan(self):
        c=candidates()
        self.assertEqual(len(c),10)
        self.assertEqual(len({x['id'] for x in c}),10)
        self.assertEqual(c[0]['params']['dropout'],.5)
        c[0]['params']['dropout']=0
        self.assertEqual(candidates()[0]['params']['dropout'],.5)

    def test_joint_uses_both_metrics(self):
        self.assertAlmostEqual(joint(dict(Acc7=.4,Acc2_nonzero=.8)),.6)

    def test_non0_guard_prevents_acc7_only_collapse(self):
        board=[row('00_reference',.35,.75),row('collapsed',.45,.6),row('balanced',.39,.76)]
        self.assertEqual(champions(board)['Acc7_guarded']['trial'],'balanced')
        self.assertNotIn('collapsed',finalists(board))

    def test_unique_finalists(self):
        self.assertEqual(len(finalists([row('00_reference',.35,.75),row('better',.39,.77)])),2)


if __name__=='__main__':unittest.main()
