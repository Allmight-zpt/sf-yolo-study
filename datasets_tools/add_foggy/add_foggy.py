import os
import imageio
from imgaug import augmenters as iaa


def add_foggy(image):
    """
    使用imgaug库对单张图像添加雾化效果。

    参数:
    - image: 待处理的图像（numpy数组格式）。
    - severity: 雾化程度，默认值为1，可选范围[0, 5]。
    返回:
    - 处理后的图像（numpy数组格式）。
    """
    # 创建一个Fog增强器实例，severity控制雾化强度
    fog_augmenter = iaa.Fog(seed=1)
    # 对图像应用雾化效果
    image_fog = fog_augmenter.augment_image(image)
    return image_fog


def process_images_in_folder(input_folder, output_folder, severity=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        try:
            img_path = os.path.join(input_folder, filename)
            # 确保文件是图像
            if not (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                continue

            # 使用imageio读取图像
            image = imageio.imread(img_path)

            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            # 确保图像数据类型为uint8
            if not image.dtype == 'uint8':
                image = image.astype('uint8')

            # 对图像应用雾化效果
            image_fog = add_foggy(image)
            output_path = os.path.join(output_folder, filename)

            # 使用imageio保存图像
            imageio.imwrite(output_path, image_fog)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == '__main__':
    input_folder = r'../datasets/for_show'
    output_folder = r'../datasets/test1'

    process_images_in_folder(input_folder, output_folder)
