from torchvision.transforms.v2 import GaussianBlur, GaussianNoise
import matplotlib.pyplot as plt
import numpy as np
import torch
class RandomGaussianNoise:
    def __init__(self, mean=0, sigma=0.02, clip=True):
        self.mean = mean
        self.sigma = sigma
        self.clip = clip

    def __call__(self, datapoint, **kwargs):
        sigma = np.random.uniform(self.sigma)
        for i in range(len(datapoint.frames)):
            data_type = datapoint.frames[i].data.dtype
            np_image = np.array(datapoint.frames[i].data)

            noise = np.random.normal(self.mean, sigma, np_image.shape)


            np_image = np_image + noise

            if self.clip:
                np_image = np.clip(np_image, a_min=0, a_max=1)

            datapoint.frames[i].data = torch.tensor(np.array(np_image), dtype=data_type)


        return datapoint

class RandomGaussianBlur:
    def __init__(self, kernel_size=3, sigma=(0.01, 2.0)):
        self.blur = GaussianBlur(kernel_size, sigma)

    def __call__(self, datapoint, **kwargs):
        import numpy as np
        for i in range(len(datapoint.frames)):
            datapoint.frames[i].data = self.blur(datapoint.frames[i].data)

        return datapoint