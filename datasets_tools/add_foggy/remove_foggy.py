import numpy as np
import cv2
import os


def estimate_atmospheric_light(hazy_image):
    '''
    :param hazy_image: 输入的雾气图像
    :return: 大气光
    '''
    img_f = hazy_image.astype(np.float32) / 255.0
    bright_pixel = np.max(img_f.reshape(-1, 3), axis=0)
    return bright_pixel


def dark_channel_prior(img, window_size=15):
    '''
    :param img: 输入图像
    :param window_size: 暗通道窗口大小
    :return: 暗通道
    '''
    dark_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    return dark_channel


def estimate_transmission(hazy_image, atmospheric_light, omega=0.95):
    '''
    :param hazy_image: 输入的雾气图像
    :param atmospheric_light: 大气光
    :param omega: 暗通道权重
    :return: 透射率
    '''
    img_f = hazy_image.astype(np.float32) / 255.0
    dark_channel = dark_channel_prior(img_f / atmospheric_light, window_size=15)
    transmission = 1 - omega * dark_channel
    return transmission


def dehaze(hazy_image, atmospheric_light, transmission, t_min=0.1):
    '''
    :param hazy_image: 输入的雾气图像
    :param atmospheric_light: 大气光
    :param transmission: 透射率
    :param t_min: 最小透射率
    :return: 去雾后的清晰图像
    '''
    img_f = hazy_image.astype(np.float32) / 255.0
    transmission = np.maximum(transmission, t_min)
    clear_image = (img_f - atmospheric_light) / transmission[..., np.newaxis] + atmospheric_light
    clear_image = np.clip(clear_image, 0, 1)
    clear_image = (clear_image * 255).astype(np.uint8)
    return clear_image


def dehaze_image(hazy_image):
    '''
    :param hazy_image: 输入的雾气图像
    :return: 去雾后的清晰图像
    '''
    atmospheric_light = estimate_atmospheric_light(hazy_image)
    transmission = estimate_transmission(hazy_image, atmospheric_light)
    clear_image = dehaze(hazy_image, atmospheric_light, transmission)
    return clear_image


def process_images_in_folder(input_folder, output_folder):
    '''
    :param input_folder: 输入图像文件夹
    :param output_folder: 输出图像文件夹
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        try:
            img_path = os.path.join(input_folder, filename)
            hazy_image = cv2.imread(img_path)
            if hazy_image is None:
                print(f"Failed to load image: {img_path}")
                continue

            clear_image = dehaze_image(hazy_image)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, clear_image)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == '__main__':
    input_folder = r'../add_foggy_data'
    output_folder = r'../remove_foggy_data'

    process_images_in_folder(input_folder, output_folder)