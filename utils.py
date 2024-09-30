from skimage import morphology


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]


def RemoveSmallObjects(mask, min_size=64):
    mask = morphology.remove_small_objects(mask, min_size)
    return mask


# import numpy as np

# multi_channel_image = np.random.randint(0, 2, size=(54, 100, 100), dtype=bool)
# print(multi_channel_image.shape)
# print(RemoveSmallObjects(multi_channel_image).shape)
