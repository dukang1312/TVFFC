import cv2
import numpy as np
import os
import glob
import random

# 输入输出文件夹路径
input_folder = r'G:\dataset\anim'
output_folder = r'I:\ffc模拟\原图'

# 获取文件夹中的所有图像文件
image_files = glob.glob(os.path.join(input_folder, '*.png'))

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 图像尺寸
image_width, image_height = 2048, 2048
crop_width, crop_height = 1536, 1024

# 随机打乱图像路径
random.shuffle(image_files)

# 一次处理7张图像，总共进行7次
for round_num in range(7):
    if len(image_files) < 7:
        break
    # 取出当前的7张图像路径
    current_files = image_files[:7]
    image_files = image_files[7:]

    # 初始化累加图像
    sum_image = np.zeros((crop_height, crop_width), dtype=np.float32)

    # 对每张图像进行处理
    for file in current_files:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)

        # 随机选择切割位置
        x_start = random.randint(0, image_width - crop_width)
        y_start = random.randint(0, image_height - crop_height)

        # 切割图像块
        crop_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]

        # 累加图像块
        sum_image += crop_image

    # 转换为16位图像
    sum_image = np.clip(sum_image, 0, 65535).astype(np.uint16)

    # 保存结果图像
    output_path = os.path.join(output_folder, f'sum_image_round_{round_num + 1}.tif')
    cv2.imwrite(output_path, sum_image)

    print(f"Round {round_num + 1} result saved to {output_path}")
