
import cv2
import numpy as np
import os
import glob
import random
from tqdm import tqdm
import tifffile as tiff

# 设置随机种子以确保结果可复现
random.seed(42)
np.random.seed(42)

# 文件夹路径
input_folder_A = r'I:\ffc\浮点图'
input_folder_B = r'H:\20250410\shinan1_12kev_700ms_0.65um\flat'
output_folder_A = r'H:\郭静好\无监督2\train_A'
output_folder_B = r'H:\郭静好\无监督2\train_B'

# 创建输出文件夹
os.makedirs(output_folder_A, exist_ok=True)
os.makedirs(output_folder_B, exist_ok=True)

# 获取文件夹中的所有图像文件
image_files_A = glob.glob(os.path.join(input_folder_A, '*.tif'))
image_files_B = glob.glob(os.path.join(input_folder_B, '*.tiff'))

# 设置高斯噪声参数
mean = 1.0
std_dev = 0.01

# 图像尺寸
crop_width, crop_height = 1024, 1024
resize_factor = 4  # 统一的缩放因子


# 裁剪并调整图像大小的通用函数
def crop_and_resize_image(image, crop_width, crop_height, resize_factor):
    """裁剪图像并调整大小，返回处理后的图像和裁剪位置"""
    # 随机选择裁剪位置
    x_start = random.randint(0, image.shape[1] - crop_width)
    y_start = random.randint(0, image.shape[0] - crop_height)

    # 裁剪图像
    cropped_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]

    # 计算调整后的尺寸
    new_width = int(cropped_image.shape[1] / resize_factor)
    new_height = int(cropped_image.shape[0] / resize_factor)

    # 使用INTER_NEAREST插值方法，速度最快
    resized_image = cv2.resize(cropped_image, (new_width, new_height),
                               interpolation=cv2.INTER_NEAREST)

    return resized_image, (x_start, y_start)


# 高效保存图像的函数
def save_image(image, output_path, dtype=np.uint16, compress=0):
    """
    使用优化参数保存TIFF图像

    参数:
        compress: 压缩级别 (0-9), 0表示不压缩，更高值表示更高压缩率
    """
    try:
        tiff.imwrite(
            output_path,
            image,
            dtype=dtype,
            photometric='minisblack',  # 适用于灰度图像
            compression='zlib' if compress > 0 else None,
            compressionargs={'level': compress},
            contiguous=True  # 确保数据连续存储，提高读取速度
        )
        return True
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return False


# 预加载并缓存B文件夹中的图像
def load_and_cache_images(image_files, max_images=None):
    """加载并缓存图像，支持限制最大缓存数量"""
    cached_images = []
    valid_files = []

    for img_path in tqdm(image_files, desc="Caching images"):
        try:
            img = tiff.imread(img_path)

            # 验证图像是否为16位灰度图像
            if img.dtype == np.uint16 and len(img.shape) == 2:
                cached_images.append(img)
                valid_files.append(img_path)

                # 如果设置了最大缓存数量且已达到，则停止缓存
                if max_images and len(cached_images) >= max_images:
                    print(f"Reached max cache size of {max_images} images")
                    break
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")

    return cached_images, valid_files


# 处理单个图像的函数
def process_image(image_path_A, cached_images, valid_files_B):
    try:
        # 读取图像A
        image_A = tiff.imread(image_path_A)

        # 添加高斯噪声
        gaussian_noise = np.random.normal(mean, std_dev, image_A.shape)
        noisy_image_A = image_A * gaussian_noise
        noisy_image_A = np.clip(noisy_image_A, 0, np.max(image_A))

        # 随机选择B文件夹中的图像
        idx = random.randint(0, len(cached_images) - 1)
        image_B = cached_images[idx]

        # 裁剪并调整图像B的大小
        resized_image_B, (x_start, y_start) = crop_and_resize_image(
            image_B, crop_width, crop_height, resize_factor)

        # 调整带噪声图像A的大小
        new_width = resized_image_B.shape[1]
        new_height = resized_image_B.shape[0]
        resized_noisy_image_A = cv2.resize(
            noisy_image_A.astype(np.float32), (new_width, new_height),
            interpolation=cv2.INTER_NEAREST)

        # 将带噪声的图像与裁剪后的图像相乘
        result_image = resized_noisy_image_A * resized_image_B

        # 确保结果图像值在合理范围内并转换为16位
        max_val = np.iinfo(np.uint16).max
        result_image = np.clip(result_image, 0, max_val).astype(np.uint16)

        # 保存结果图像
        base_name_A = os.path.basename(image_path_A).replace('.tif', '')
        base_name = f'{base_name_A}_cropped_{x_start}_{y_start}.png'
        output_path_A = os.path.join(output_folder_A, base_name)
        output_path_B = os.path.join(output_folder_B, base_name)

        # 保存两个图像 (使用压缩级别1平衡速度和大小)
        success_A = save_image(result_image, output_path_A, compress=0)
        success_B = save_image(resized_image_B, output_path_B, compress=0)

        return success_A and success_B
    except Exception as e:
        print(f"Error processing {image_path_A}: {e}")
        return False


def main():
    # 预加载B文件夹中的图像，限制缓存数量为1000 (根据需要调整)
    cached_images, valid_files_B = load_and_cache_images(image_files_B, max_images=1000)
    print(f"Successfully cached {len(cached_images)} images from folder B")

    if not cached_images:
        print("No valid images found in folder B. Exiting.")
        return

    # 处理图像
    success_count = 0
    for image_path_A in tqdm(image_files_A, desc='Processing images'):
        if process_image(image_path_A, cached_images, valid_files_B):
            success_count += 1

    print(f"Processing complete. Successfully processed {success_count}/{len(image_files_A)} images.")


if __name__ == "__main__":
    main()