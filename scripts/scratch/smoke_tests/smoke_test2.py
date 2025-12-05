import unittest
import torch
import torch.nn.functional as F

from src.utils.losses import FocalLoss


class TestFocalLoss(unittest.TestCase):
    def setUp(self):
        """Set up dummy data for tests."""
        self.inputs = torch.tensor(
            [[1.5, -0.5, 0.2], [0.1, 0.2, 1.8]], requires_grad=True
        )
        self.targets = torch.tensor([0, 2])

    def manual_focal_loss(self, logits, targets, alpha, gamma):
        """Manual implementation for verification."""
        probs = F.softmax(logits, dim=1)

        loss_sum = 0
        for i in range(len(targets)):
            target = targets[i]
            p_t = probs[i, target]

            ce = -torch.log(p_t)
            modulating_factor = (1 - p_t) ** gamma

            loss = modulating_factor * ce

            if alpha is not None:
                w = alpha[target]
                loss = w * loss

            loss_sum += loss

        return loss_sum / len(targets)

    def test_forward_pass_no_alpha(self):
        """Test focal loss without alpha balancing."""
        gamma = 2.0
        criterion = FocalLoss(gamma=gamma, alpha=None, reduction="mean")

        output = criterion(self.inputs, self.targets)
        expected = self.manual_focal_loss(self.inputs, self.targets, None, gamma)

        self.assertTrue(
            torch.allclose(output, expected),
            f"Mismatch: Got {output}, expected {expected}",
        )

    def test_forward_pass_with_alpha(self):
        """Test focal loss with alpha balancing."""
        gamma = 2.0
        alpha = [0.1, 0.5, 0.9]
        criterion = FocalLoss(gamma=gamma, alpha=alpha, reduction="mean")

        output = criterion(self.inputs, self.targets)
        expected = self.manual_focal_loss(self.inputs, self.targets, alpha, gamma)

        self.assertTrue(
            torch.allclose(output, expected),
            f"Mismatch with alpha: Got {output}, expected {expected}",
        )

    def test_gradients(self):
        """Ensure backward pass works."""
        criterion = FocalLoss(gamma=2.0, reduction="mean")
        output = criterion(self.inputs, self.targets)

        output.backward()

        self.assertIsNotNone(self.inputs.grad, "Gradients were not computed!")
        self.assertNotEqual(self.inputs.grad.sum().item(), 0, "Gradients are zero!")

    def test_buffer_registration(self):
        """Test if alpha moves with the model."""
        alpha = [0.1, 0.2, 0.7]
        criterion = FocalLoss(alpha=alpha)

        self.assertEqual(criterion.alpha.dtype, torch.float32)

        criterion = criterion.double()

        self.assertEqual(
            criterion.alpha.dtype,
            torch.float64,
            "Alpha did not move/cast with the model!",
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
