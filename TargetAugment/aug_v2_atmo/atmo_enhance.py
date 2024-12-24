import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import os


class atmo_enhance:
    def __init__(self, args):
        self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        self.target_size = args.imgsz
        self.args = args
        self.step = 0
        self.haze_beta = args.haze_beta
        self.haze_brightness = args.haze_brightness

        # Clear previous logs if any
        path = os.path.join(os.path.dirname(__file__), '../..', self.args.log_dir, 'noise')
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)

    def add_style(self, content, flag, save_images=False):
        """
        Applies haze effect to the input images.
        """
        if flag == 0:
            self.step += 1
        assert len(content.size()) == 4  # Expecting batch of images

        output = []
        for i in range(content.size(0)):
            img = content[i].permute(1, 2, 0).cpu().numpy() + self.pixel_means  # De-normalize
            img = np.clip(img, 0, 255).astype(np.uint8)
            hazy_img = self.add_hazy(img, beta=self.haze_beta, brightness=self.haze_brightness)
            hazy_img = (torch.from_numpy(hazy_img).float() - self.pixel_means).permute(2, 0, 1).cuda()
            output.append(hazy_img.to(torch.float32))

        output = torch.stack(output, dim=0)

        if flag == 0:
            if self.step % 30 == 1 and save_images:
                self.show(content, content=True)
                self.show(output)

        return output.detach()

    def add_hazy(self, image, beta=0.05, brightness=0.5):
        """
        Applies haze effect to a single image.
        """
        img_f = image.astype(np.float32) / 255.0
        row, col, chs = image.shape
        size = np.sqrt(max(row, col))
        center = (row // 2, col // 2)
        y, x = np.ogrid[:row, :col]
        dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        d = -0.04 * dist + size
        td = np.exp(-beta * d)
        img_f = img_f * td[..., np.newaxis] + brightness * (1 - td[..., np.newaxis])
        hazy_img = np.clip(img_f * 255, 0, 255).astype(np.uint8)
        return hazy_img

    def show(self, feat, content=False, save=True):
        """
        Saves or displays the images for debugging purposes.
        """
        feat = torch.nan_to_num(feat)
        for i in range(feat.size(0)):
            s = feat[i].transpose(0, 1).transpose(1, 2).cpu().numpy()
            s = np.clip(s, 0, 255).astype(np.uint8)
            if save:
                path = self.args.log_dir
                if not os.path.exists(path):
                    os.makedirs(path)
                if content:
                    matplotlib.image.imsave(os.path.join(path, f'step{self.step}_real{i}.jpg'), s)
                else:
                    matplotlib.image.imsave(os.path.join(path, f'step{self.step}_{i}.jpg'), s)
            else:
                plt.imshow(s)
                plt.show()
