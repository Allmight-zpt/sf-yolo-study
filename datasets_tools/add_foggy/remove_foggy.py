import numpy as np
import cv2
import os


def remove_hazy(image, beta=0.05, brightness=0.5):
    '''
    :param image:   输入的有雾图像
    :param beta:    雾强
    :param brightness:  雾霾亮度
    :return:    去雾后的图像
    '''
    img_f = image.astype(np.float32) / 255.0
    row, col, chs = image.shape
    size = np.sqrt(max(row, col))
    center = (row // 2, col // 2)
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    d = -0.04 * dist + size
    td = np.exp(-beta * d)

    # 逆向恢复
    img_f = (img_f - brightness * (1 - td[..., np.newaxis])) / td[..., np.newaxis]
    img_f = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return img_f


def process_images_in_folder(input_folder, output_folder, beta=0.05, brightness=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        try:
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            image_clear = remove_hazy(image, beta, brightness)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image_clear)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == '__main__':
    input_folder = r'../datasets/sub/origin_foggy'  # 输入是带雾的图片
    output_folder = r'../datasets/sub/remove_foggy'  # 输出是去雾后的图片
    beta = 0.02
    brightness = 0.8

    process_images_in_folder(input_folder, output_folder, beta, brightness)
