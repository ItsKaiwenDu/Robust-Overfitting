import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train import evaluate, generate_pgd_adversarial


class TinyBatchNormClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).flatten(1))


def clone_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class EvaluationStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyBatchNormClassifier()
        self.normalizer = nn.Identity()
        self.images = torch.rand(4, 3, 8, 8)
        self.labels = torch.tensor([0, 1, 0, 1])

    def assert_state_equal(self, before, after):
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            self.assertTrue(
                torch.equal(before[name], after[name]),
                msg=f"Model state changed during evaluation: {name}",
            )

    def test_pixel_attack_preserves_train_and_eval_modes(self):
        for initial_training_mode in (True, False):
            with self.subTest(initial_training_mode=initial_training_mode):
                self.model.train(initial_training_mode)
                before = clone_state(self.model)
                generate_pgd_adversarial(
                    self.model,
                    self.normalizer,
                    self.images,
                    self.labels,
                    epsilon=8.0 / 255.0,
                    alpha=2.0 / 255.0,
                    num_steps=2,
                    device=torch.device("cpu"),
                )
                self.assertEqual(self.model.training, initial_training_mode)
                self.assert_state_equal(before, clone_state(self.model))

    def test_evaluation_preserves_mode_parameters_and_buffers(self):
        dataloader = DataLoader(
            TensorDataset(self.images, self.labels), batch_size=2, shuffle=False
        )
        for initial_training_mode in (True, False):
            with self.subTest(initial_training_mode=initial_training_mode):
                self.model.train(initial_training_mode)
                before = clone_state(self.model)
                evaluate(
                    self.model,
                    self.normalizer,
                    dataloader,
                    device=torch.device("cpu"),
                    epsilon=8.0 / 255.0,
                    alpha=2.0 / 255.0,
                    num_steps=2,
                )
                self.assertEqual(self.model.training, initial_training_mode)
                self.assert_state_equal(before, clone_state(self.model))


if __name__ == "__main__":
    unittest.main()
