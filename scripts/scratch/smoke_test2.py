import unittest
import torch
import torch.nn.functional as F

from src.utils.losses import FocalLoss

# Import your class here (assuming it's in the same file or imported)
# from your_file import FocalLoss


class TestFocalLoss(unittest.TestCase):
    def setUp(self):
        # Create some dummy data
        # Batch size = 2, Classes = 3
        self.inputs = torch.tensor(
            [[1.5, -0.5, 0.2], [0.1, 0.2, 1.8]], requires_grad=True
        )
        self.targets = torch.tensor([0, 2])  # True classes

    def manual_focal_loss(self, logits, targets, alpha, gamma):
        """
        A slow, manual implementation to verify the math
        of the optimized version.
        """
        probs = F.softmax(logits, dim=1)

        loss_sum = 0
        for i in range(len(targets)):
            target = targets[i]
            p_t = probs[i, target]

            # Standard CE part: -log(p_t)
            ce = -torch.log(p_t)

            # Focal part: (1 - p_t)^gamma
            modulating_factor = (1 - p_t) ** gamma

            loss = modulating_factor * ce

            # Alpha part
            if alpha is not None:
                w = alpha[target]
                loss = w * loss

            loss_sum += loss

        return loss_sum / len(targets)  # Mean reduction

    def test_forward_pass_no_alpha(self):
        """Test standard focal loss without alpha balancing."""
        gamma = 2.0
        criterion = FocalLoss(gamma=gamma, alpha=None, reduction="mean")

        # Optimized implementation
        output = criterion(self.inputs, self.targets)

        # Manual verification
        expected = self.manual_focal_loss(self.inputs, self.targets, None, gamma)

        self.assertTrue(
            torch.allclose(output, expected),
            f"Mismatch: Got {output}, expected {expected}",
        )

    def test_forward_pass_with_alpha(self):
        """Test focal loss WITH alpha balancing."""
        gamma = 2.0
        # Give class 0 low weight, class 2 high weight
        alpha = [0.1, 0.5, 0.9]
        criterion = FocalLoss(gamma=gamma, alpha=alpha, reduction="mean")

        output = criterion(self.inputs, self.targets)
        expected = self.manual_focal_loss(self.inputs, self.targets, alpha, gamma)

        self.assertTrue(
            torch.allclose(output, expected),
            f"Mismatch with alpha: Got {output}, expected {expected}",
        )

    def test_gradients(self):
        """Ensure backward pass works (gradients are generated)."""
        criterion = FocalLoss(gamma=2.0, reduction="mean")
        output = criterion(self.inputs, self.targets)

        output.backward()

        self.assertIsNotNone(self.inputs.grad, "Gradients were not computed!")
        self.assertNotEqual(self.inputs.grad.sum().item(), 0, "Gradients are zero!")

    def test_buffer_registration(self):
        """Test if alpha moves with the model (e.g. to double precision)."""
        alpha = [0.1, 0.2, 0.7]
        criterion = FocalLoss(alpha=alpha)

        # Default is float32
        self.assertEqual(criterion.alpha.dtype, torch.float32)

        # Move model to double (float64)
        # In a real scenario, this would be .cuda() or .to(device)
        criterion = criterion.double()

        self.assertEqual(
            criterion.alpha.dtype,
            torch.float64,
            "Alpha did not move/cast with the model!",
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
