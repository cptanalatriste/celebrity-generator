from unittest import TestCase

from celebrity_generator import create_image_dataloader


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
