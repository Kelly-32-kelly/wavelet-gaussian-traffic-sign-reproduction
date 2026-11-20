import unittest
import numpy as np
from kernel_heads import kernel, fit_twsvm, scores


class ClassifierChecks(unittest.TestCase):
    def test_kernel_symmetry_diagonal_and_psd(self):
        rng=np.random.default_rng(4)
        x=rng.uniform(size=(20,64))
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        k=kernel(x,x,'wavelet',2.)
        np.testing.assert_allclose(k,k.T,atol=1e-12)
        np.testing.assert_allclose(np.diag(k),1.,atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(k).min(),-1e-10)

    def test_twin_constraints_sign_and_easy_binary_predictions(self):
        x=np.array([[1.,0.],[0.99,0.01],[0.,1.],[0.01,0.99]])
        y=np.array([0,0,1,1])
        k=kernel(x,x,'wavelet',1.)
        model=fit_twsvm(k,y,c=10.)
        np.testing.assert_array_equal(np.argmax(scores(model,k),axis=1),y)
        pair=model['pairs'][0]
        z1,z2=pair['z1'],pair['z2']
        s=np.column_stack([k[:2],np.ones(2)])
        r=np.column_stack([k[2:],np.ones(2)])
        self.assertTrue(np.all(r@z1<0))
        self.assertTrue(np.all(s@z2>0))


if __name__=='__main__':
    unittest.main()
