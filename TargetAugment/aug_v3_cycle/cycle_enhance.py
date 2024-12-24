import torch

from TargetAugment.aug_v3_cycle.cycle_enhance_vgg16 import cycle_enhance_vgg16

class cycle_enhance:
    def __init__(self, args):
        self.args = args
        self.adain_aug = cycle_enhance_vgg16(args)
        self.adain_de_aug = cycle_enhance_vgg16(args, de_aug=True)

    def add_style(self, content, flag, save_images=False, de_aug_save_images=False):
        aug_image = self.adain_aug.add_style(content,flag,save_images)
        de_aug_image = self.adain_de_aug.add_style(content,flag,de_aug_save_images,de_aug=True)
        return torch.concat([aug_image, de_aug_image], dim=1)

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            results.append(self.adain_aug.encoders[i](results[-1]))
        return results[1:]
