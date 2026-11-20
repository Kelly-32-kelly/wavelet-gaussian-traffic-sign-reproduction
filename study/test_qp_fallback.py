"""Independent hand-solved box-QP checks for the numerical fallback gate."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from kernel_heads import _dual_solution

class QPFallbackChecks(unittest.TestCase):
    def problem(self):
        # h=(1+epsilon)I and q=1, so min .5*alpha^2-alpha on [0,2] has alpha=1.
        eps=1e-6
        return np.eye(2),np.array([[np.sqrt(1+eps),0.]]),eps

    def test_hand_solved_optimum_has_zero_kkt_and_gap(self):
        s,r,eps=self.problem()
        with patch('kernel_heads.solvers.qp',return_value={'status':'unknown'}):
            z,diagnostic=_dual_solution(s,r,2.,eps,allow_fallback=True)
        np.testing.assert_allclose(z,[1/np.sqrt(1+eps),0.],atol=1e-8)
        self.assertLess(diagnostic['projected_gradient_inf'],1e-7)
        self.assertLess(abs(diagnostic['gap']),1e-7)

    def test_nonoptimal_candidate_is_rejected_even_if_optimizer_reports_success(self):
        s,r,eps=self.problem()
        invalid=SimpleNamespace(x=np.array([.5]),nit=1,success=True)
        with patch('kernel_heads.solvers.qp',return_value={'status':'unknown'}),patch('kernel_heads.minimize',return_value=invalid):
            with self.assertRaisesRegex(RuntimeError,'KKT/gap'):
                _dual_solution(s,r,2.,eps,allow_fallback=True)

if __name__=='__main__':unittest.main()
