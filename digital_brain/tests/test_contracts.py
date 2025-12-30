# digital_brain/tests/test_contracts.py
import unittest
import torch

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.modules.cortex import Cortex
from digital_brain.modules.thalamus import Thalamus


class TestContracts(unittest.TestCase):
    def test_cortex_shapes(self):
        B = 2
        d_obs = 9
        d_z = 32
        d_act = 4

        cortex = Cortex(d_obs, d_z, d_act)

        x = torch.randn(B, d_obs)

        # cortex_state is now (E, I, trace)
        e0 = torch.zeros(B, d_z)
        i0 = torch.zeros(B, d_z)
        trace0 = torch.zeros(d_z, d_z)
        state0 = (e0, i0, trace0)

        out = cortex.forward(x, state0)

        self.assertEqual(len(out), 3) # Enforce 3-tuple return

        z_t, pred_t, new_state = out
        self.assertEqual(z_t.shape, (B, d_z))
        self.assertEqual(pred_t.shape, (B, d_obs))

        self.assertIsInstance(new_state, tuple)
        self.assertEqual(len(new_state), 3)
        self.assertEqual(new_state[0].shape, (B, d_z)) # E
        self.assertEqual(new_state[1].shape, (B, d_z)) # I
        self.assertEqual(new_state[2].shape, (d_z, d_z)) # Trace

    def test_thalamus_shapes(self):
        B = 2
        d_obs = 9
        d_sel = 4

        th = Thalamus(d_obs, d_sel)
        x = torch.randn(B, d_obs)
        selection = torch.randn(B, d_sel)

        # Minimal ModSignals stub
        class Mods:
            def __init__(self, B):
                self.ACh = torch.full((B,), 0.5)
                self.DA = torch.zeros(B)
                self.NE = torch.zeros(B)
                self.HT5 = torch.full((B,), 0.5)
                
        # In brain.py StepLog, mods are scalars if accessing .mean().item()
        # But in Thalamus.gate, it expects tensors (B,) or scalar broadcast
        # Check Thalamus implementation. Usually expects (B,) for DA/NE.
        
        gated = th.gate(x, selection, Mods(B))
        self.assertEqual(gated.shape, (B, d_obs))

    def test_brain_step_shapes(self):
        cfg = {'d_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4}
        brain = DigitalBrain(cfg)
        brain.reset(1)

        obs = Obs(x=torch.randn(1, 9))
        prev_reward = torch.tensor([[0.0]])
        prev_done = torch.tensor([[False]])

        action, log_prob, value, state, log = brain.step(obs, prev_reward, prev_done)

        self.assertEqual(action.shape, (1,))
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(value.shape, (1, 1))
        self.assertIsNotNone(state)
        self.assertEqual(len(state.cortex_state), 3) # E, I, Trace
        self.assertTrue(hasattr(log, "pred_error"))


if __name__ == "__main__":
    unittest.main()
