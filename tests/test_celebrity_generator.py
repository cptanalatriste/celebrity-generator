from unittest import TestCase

import torch

from celebrity_generator import create_image_dataloader, scale_tensor, Discriminator
from problem_unittests import test_discriminator


class CelebrityGeneratorTest(TestCase):

    def test_create_image_dataloader(self):
        batch_size = 128
        target_image_size = 32
        data_location = '../processed_celeba_small'
        image_dataloader = create_image_dataloader(batch_size=batch_size, target_image_size=32,
                                                   data_location=data_location)

        image_samples, _ = next(iter(image_dataloader))
        self.assertEqual(len(image_samples), batch_size)

        an_image = image_samples[0]
        self.assertEqual(an_image.size(), (3, target_image_size, target_image_size))

    def test_scale_tensor(self):
        input_tensor = torch.FloatTensor([0.0, 0.5, 1.0])
        new_scale = (-1, 1)
        rescaled_tensor = scale_tensor(input_tensor, new_scale)

        new_scale_min, new_scale_max = new_scale
        self.assertEqual(rescaled_tensor[0], new_scale_min)
        self.assertEqual(rescaled_tensor[1], (new_scale_min + new_scale_max) / 2)
        self.assertEqual(rescaled_tensor[2], new_scale_max)


class TestDiscriminator(TestCase):
    def test_forward(self):
        discriminator_network = Discriminator(conv_dim=64)
        sample_input = torch.FloatTensor(1, 3, 32, 32).uniform_(-1, 1)

        output = discriminator_network.forward(sample_input)
        output = output.detach()

        self.assertEqual((1, 1), output.size())

    def test_structure(self):
        test_discriminator(Discriminator)
