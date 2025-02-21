from torchvision.transforms.v2 import GaussianBlur, GaussianNoise
import matplotlib.pyplot as plt

class RandomGaussianNoise:
    def __init__(self, mean=0, sigma=0.1, clip=True):
        self.noise = GaussianNoise(mean, sigma, clip)

    def __call__(self, datapoint, **kwargs):
        for i in range(len(datapoint.frames)):
            datapoint.frames[i].data = self.noise(datapoint.frames[i].data)

        return datapoint

class RandomGaussianBlur:
    def __init__(self, kernel_size=3, sigma=(0.01, 2.0)):
        self.blur = GaussianBlur(kernel_size, sigma)

    def __call__(self, datapoint, **kwargs):
        for i in range(len(datapoint.frames)):
            plt.subplot(1, 2, 1)
            plt.show(datapoint.frames[i].data)
            datapoint.frames[i].data = self.blur(datapoint.frames[i].data)
            plt.subplot(1, 2, 2)
            plt.show(datapoint.frames[i].data)
            plt.savefig('example.png')
            assert 1 == 2

        return datapoint