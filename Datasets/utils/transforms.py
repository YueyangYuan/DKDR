class TwoCropsTransform:

    def __init__(self, base_transform_left, base_transform_right):
        self.base_transform_left = base_transform_left
        self.base_transform_right = base_transform_right

    def __call__(self, x):
        q = self.base_transform_left(x)
        k = self.base_transform_right(x)
        return [q, k]


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
