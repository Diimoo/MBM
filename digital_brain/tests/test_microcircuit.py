import torch
import unittest
from digital_brain.modules.cortex import Cortex

class TestMicrocircuit(unittest.TestCase):
    def test_forward_shape(self):
        d_obs, d_z, d_act = 10, 32, 4
        cortex = Cortex(d_obs, d_z, d_act)
        x = torch.randn(2, d_obs)
        state = None
        
        z_t, pred_t, new_state = cortex.forward(x, state)
        
        self.assertEqual(z_t.shape, (2, d_z))
        self.assertEqual(pred_t.shape, (2, d_obs))
        self.assertIsInstance(new_state, tuple)
        self.assertEqual(len(new_state), 2) # (E, I)
        
    def test_ei_dynamics(self):
        # Verify that E and I activities change over multiple steps
        d_obs, d_z, d_act = 10, 32, 4
        cortex = Cortex(d_obs, d_z, d_act)
        x = torch.randn(1, d_obs)
        state = None
        
        z1, p1, s1 = cortex.forward(x, state)
        z2, p2, s2 = cortex.forward(x, s1)
        
        # Check that activities are not identical (dynamics are working)
        self.assertFalse(torch.allclose(s1[0], s2[0]))
        self.assertFalse(torch.allclose(s1[1], s2[1]))

if __name__ == '__main__':
    unittest.main()
