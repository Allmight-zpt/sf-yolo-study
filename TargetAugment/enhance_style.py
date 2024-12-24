import sys

import torch

from TargetAugment.aug_v3_cycle.cycle_enhance import cycle_enhance
from TargetAugment.aug_v2_atmo.atmo_enhance import atmo_enhance
from TargetAugment.enhance_vgg16 import enhance_vgg16



def get_style_images(im_data, args, adain=None):
    # TargetAugment_v5 augment core
    if args.aug_v5_ac:
        images, _ = next(adain)
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu', non_blocking=True)
        return images
    # TargetAugment_v4 cycle forward twice augment
    if args.aug_v4_cycle_forward_twice:
        cycle = adain
        if cycle is None:
            cycle = cycle_enhance(args)
        style_im_data = torch.concat([im_data, im_data], dim=1) * 0 + 1 * cycle.add_style(im_data, 0, save_images=args.save_style_samples, de_aug_save_images=args.de_aug_save_style_samples)
        # sys.exit("exit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return style_im_data
    # augment v3 cycle augment
    if args.aug_v3_cycle:
        cycle = adain
        if cycle is None:
            cycle = cycle_enhance(args)
        style_im_data = torch.concat([im_data, im_data], dim=1) * 0 + 1 * cycle.add_style(im_data, 0, save_images=args.save_style_samples, de_aug_save_images=args.de_aug_save_style_samples)
        return style_im_data
    # augment v2 atmosphere scattering
    if args.aug_v2_atmo:
        atmo = adain
        if atmo is None:
            atmo = atmo_enhance(args)
        styled_im_data = im_data * 0 + 1 * atmo.add_style(im_data, 0, save_images=args.save_style_samples)
        return styled_im_data
    # augment v1 sf-yolo
    if adain is None:
        adain = enhance_vgg16(args)
    # Apply style to the image, use save_images=True to save the images (useful for debugging)
    styled_im_data = im_data * 0 + 1 * adain.add_style(im_data, 0, save_images=args.save_style_samples)
    return styled_im_data
