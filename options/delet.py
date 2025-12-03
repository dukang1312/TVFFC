import os
import tifffile as tiff
import numpy as np

def multiply_tiff_images(input_dir, output_dir, factor):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有 TIFF 文件
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.tiff') or file_name.lower().endswith('.tif'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            try:
                # 读取 TIFF 文件
                image = tiff.imread(input_path)

                # 检查图像是否为 16 位
                if image.dtype != np.uint16:
                    print(f"Skipping {input_path}: not a 16-bit image.")
                    continue

                # 放大像素值
                multiplied_image = np.clip(image * factor, 0, 65535).astype(np.uint16)

                # 保存放大的图像
                tiff.imwrite(output_path, multiplied_image)

                print(f"Processed {input_path} -> {output_path}")

            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

    print("Processing completed.")


import os

from skimage.metrics import structural_similarity as ssim


def calculate_ssim(input_dir, reference_image_path):
    # 读取参考图像
    reference_image = tiff.imread(reference_image_path)

    # 检查参考图像是否为16位
    if reference_image.dtype != np.uint16:
        raise ValueError(f"Reference image {reference_image_path} is not a 16-bit image.")

    # 遍历输入目录中的所有 TIFF 文件
    results = []
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.tiff') or file_name.lower().endswith('.tif'):
            input_path = os.path.join(input_dir, file_name)
            try:
                # 读取目标图像
                image = tiff.imread(input_path)

                # 检查图像是否为16位
                if image.dtype != np.uint16:
                    print(f"Skipping {input_path}: not a 16-bit image.")
                    continue

                # 计算SSIM
                ssim_value = ssim(reference_image, image, data_range=65535)
                results.append((file_name, ssim_value))
                print(f"SSIM for {input_path}: {ssim_value}")

            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

    # 打印和保存结果
    print("SSIM calculation completed.")
    for file_name, ssim_value in results:
        print(f"{file_name}: {ssim_value}")

    return results
import os


def crop_tiff_images(input_dir, output_dir, crop_start, crop_size):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有 TIFF 文件
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.tiff') or file_name.lower().endswith('.tif'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            try:
                # 读取 TIFF 文件
                image = tiff.imread(input_path)

                # 检查图像是否为16位
                if image.dtype != np.uint16:
                    print(f"Skipping {input_path}: not a 16-bit image.")
                    continue

                # 检查图像大小是否足够进行裁剪
                height, width = image.shape
                start_y, start_x = crop_start
                crop_height, crop_width = crop_size

                if start_y + crop_height > height or start_x + crop_width > width:
                    print(f"Skipping {input_path}: image size is smaller than crop area.")
                    continue

                # 裁剪图像
                cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

                # 保存裁剪后的图像
                tiff.imwrite(output_path, cropped_image)

                print(f"Cropped {input_path} -> {output_path}")

            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

    print("Cropping completed.")

# if __name__ == "__main__":
#     input_dir = r'H:\20221006dukang\fish_head\5ms'
#     output_dir = r'G:\new\datasets\cityscapes\test_A'
#     crop_start = (105, 206)  # (y, x)
#     crop_size = (1024, 1760)  # (height, width)
#     crop_tiff_images(input_dir, output_dir, crop_start, crop_size)

#
# if __name__ == "__main__":
#     input_dir = r'G:\new\results\yutou_100ms\test_latest\images\test'
#     reference_image_path = r'G:\new\results\yutou_100ms\test_latest\images\Flat_0001.tif'
#     results = calculate_ssim(input_dir, reference_image_path)
#
#     # 保存结果到文件
#     output_file = os.path.join(input_dir, '100ssim_results.txt')
#     with open(output_file, 'w') as f:
#         for file_name, ssim_value in results:
#             f.write(f" {ssim_value}\n")
#     print(f"SSIM results saved to {output_file}")

# if __name__ == "__main__":
#     input_dir = r'G:\文章\aiffc\图像测试\5ms'
#     output_dir = r'G:\文章\aiffc\图像测试\5ms20x'
#     factor = 20
#     multiply_tiff_images(input_dir, output_dir, factor)

import random
from PIL import Image, ImageDraw, ImageFont


def create_image_with_numbers():
    # 图像尺寸
    width, height = 1536, 1024

    # 创建一个新的32位浮点型图像，背景灰度值为1.0 (白色)
    img = np.ones((height, width), dtype=np.float32)

    # 准备字体
    try:
        font = ImageFont.truetype("arial.ttf", 100)  # 字体大小
    except IOError:
        font = ImageFont.load_default()

    # 数字和对应的灰度值 (0.8到0.1)
    numbers = ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2']
    gray_values = [float(num) for num in numbers]

    # 列宽度
    column_width = width // 8

    # 圆的直径略小于列宽度
    circle_diameter = column_width - 20
    circle_radius = circle_diameter // 2

    # 在每个列上写数字并绘制椭圆和圆形
    for i, (num, gray) in enumerate(zip(numbers, gray_values)):
        # 创建一个 PIL 图像对象用于绘制文本
        gray = 1-gray
        text_img = Image.new('F', (column_width, height), 1.0)  # 浮点型图像
        text_draw = ImageDraw.Draw(text_img)

        # 计算文本位置
        text_bbox = text_draw.textbbox((0, 0), num, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_position = ((column_width - text_width) / 2, (height - text_height) / 2)

        # 绘制文本，填充颜色使用浮点数灰度值
        text_draw.text(text_position, num, font=font, fill=gray)

        # 将文本图像粘贴到主图像的中间位置
        rotated_text_array = np.array(text_img)
        paste_x = column_width * i
        img[:, paste_x:paste_x + rotated_text_array.shape[1]] = rotated_text_array

        # 计算椭圆位置，绘制在数字上方
        ellipse_center_x = paste_x + column_width // 2
        ellipse_center_y = height // 4  # 图像上半部分1/4处

        # 随机椭圆大小和方向
        ellipse_width = random.randint(column_width // 2, column_width)
        ellipse_height = random.randint(column_width // 4, column_width // 2)
        angle = random.randint(0, 360)

        # 创建浮点型图像用于绘制椭圆
        ellipse_img = Image.new('F', (ellipse_width, ellipse_height), 0.0)  # 浮点型图像
        ellipse_draw = ImageDraw.Draw(ellipse_img)
        ellipse_draw.ellipse([0, 0, ellipse_width, ellipse_height], fill=gray)

        # 旋转椭圆图像
        rotated_ellipse_img = ellipse_img.rotate(angle, expand=True)
        rotated_ellipse_array = np.array(rotated_ellipse_img)

        # 确保旋转后的椭圆与主图像的灰度值正确合并
        ellipse_x = ellipse_center_x - rotated_ellipse_img.width // 2
        ellipse_y = ellipse_center_y - rotated_ellipse_img.height // 2
        for y in range(rotated_ellipse_array.shape[0]):
            for x in range(rotated_ellipse_array.shape[1]):
                if 0 <= ellipse_y + y < height and 0 <= ellipse_x + x < width:
                    # 如果不是背景，覆盖主图像的像素
                    if rotated_ellipse_array[y, x] != 0.0:
                        img[ellipse_y + y, ellipse_x + x] = rotated_ellipse_array[y, x]

        # 计算圆位置，绘制在数字下方
        circle_center_x = paste_x + column_width // 2
        circle_center_y = 3 * height // 4  # 图像下半部分的中心位置
        circle_x0 = circle_center_x - circle_radius
        circle_y0 = circle_center_y - circle_radius
        circle_x1 = circle_center_x + circle_radius
        circle_y1 = circle_center_y + circle_radius

        # 直接在主图像上绘制圆
        for y in range(circle_y0, circle_y1):
            for x in range(circle_x0, circle_x1):
                if ((x - circle_center_x) ** 2 + (y - circle_center_y) ** 2) <= circle_radius ** 2:
                    if 0 <= y < height and 0 <= x < width:
                        img[y, x] = gray

    # 保存为32位浮点型 TIFF 图像
    Image.fromarray(img).save('G:/文章/aiffc/图像测试/numbered_image_with_ellipses_and_circles-2.tiff', format='TIFF', dtype='float32')



# 运行函数创建图像
# create_image_with_numbers()

import numpy as np
import os
from PIL import Image


def create_sine_wave_images(frequency, width, height, save_dir, num_frames, total_time):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建 x 轴数据
    x = np.linspace(0, 2 * np.pi, width // 2)  # 中间部分宽度为 width // 2
    y = np.linspace(0, height, height)
    X, Y = np.meshgrid(x, y)

    # 每帧的时间间隔
    dt = total_time / num_frames / 1000  # 转换为秒

    # 生成正弦波的基础图像
    base_wave = np.sin(X * frequency)

    for frame in range(num_frames):
        # 计算当前帧的时间
        if frame < num_frames // 2:
            t = frame / (num_frames // 2)  # 前半段时间
        else:
            t = (num_frames - frame) / (num_frames // 2)  # 后半段时间，倒着移回

        # 计算当前帧的相位偏移
        phase_shift = 2 * np.pi * t  # 一个周期的移动
        Z_shifted = np.sin(X * frequency + phase_shift)+1

        # 创建完整图像，中间部分为正弦波
        full_image = np.ones((height, width), dtype=np.float32)
        full_image[:, width // 4: 3 * width // 4] = Z_shifted

        # 保存为32位浮点型 TIFF 图像
        save_path = os.path.join(save_dir, f'sine_wave_image_{frame:03d}.tiff')
        Image.fromarray(full_image).save(save_path, format='TIFF', dtype='float32')


# # 使用示例
# frequency = 5  # 频率设置为10
# width = 1024  # 图像宽度
# height = 512  # 图像高度
# num_frames = 200  # 生成的帧数，总共200帧
# total_time = 14  # 总时间为14ms
# save_dir = r'G:\文章\aiffc\图像测试\imgwave'  # 保存目录
#
# create_sine_wave_images(frequency, width, height, save_dir, num_frames, total_time)

import numpy as np
from PIL import Image
import os

def create_exposure_images(source_dir, save_dir, original_frames, original_exposure_time, new_exposure_time, total_frames):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取所有源图像文件路径并按名称排序
    source_images = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.tiff')])

    # 确保源图像数量足够
    assert len(source_images) == original_frames, "源图像数量不足"

    # 计算每张新曝光时间的图像需要的帧数
    frames_per_exposure = int((new_exposure_time / original_exposure_time) * original_frames)

    # 计算需要累加的次数
    full_cycles = frames_per_exposure // original_frames
    remaining_frames = frames_per_exposure % original_frames

    # 预累加完整的周期
    full_cycle_accumulated = np.zeros((512, 1024), dtype=np.float32)
    for img_path in source_images:
        img = np.array(Image.open(img_path), dtype=np.float32)
        full_cycle_accumulated += img

    # 预累加剩余的帧
    remaining_accumulated = np.zeros((512, 1024), dtype=np.float32)
    for i in range(remaining_frames):
        img = np.array(Image.open(source_images[i]), dtype=np.float32)
        remaining_accumulated += img

    for i in range(total_frames):
        # 初始化累积图像
        accumulated_image = full_cycle_accumulated * full_cycles + remaining_accumulated

        # 保存为32位浮点型 TIFF 图像
        save_path = os.path.join(save_dir, f'exposure_image_{i:03d}.tiff')
        Image.fromarray(accumulated_image).save(save_path, format='TIFF', dtype='float32')

# 使用示例
# source_dir = r'G:\文章\aiffc\图像测试\imgwave'  # 前面生成的图像目录
# save_dir = r'G:\文章\aiffc\图像测试\imgwave\200ms'  # 保存新的曝光时间为200ms的图像
# original_frames = 200  # 前面生成的总帧数
# original_exposure_time = 14  # 原始曝光时间14ms
# new_exposure_time = 200  # 新的曝光时间200ms
# total_frames = 100  # 生成的总帧数
#
# create_exposure_images(source_dir, save_dir, original_frames, original_exposure_time, new_exposure_time, total_frames)
# import pandas as pd
# import matplotlib.pyplot as plt
# from brokenaxes import brokenaxes
#
# # Load the data from the txt files
# file_t = 'G:\\文章\\aiffc\\oe文章图像\\oe回答\\T.txt'  # 请将 '路径/T.txt' 替换为您的 T.txt 文件的实际路径
# file_ip = 'G:\\文章\\aiffc\\oe文章图像\\oe回答\\I_p.txt'  # 请将 '路径/I_p.txt' 替换为您的 I_p.txt 文件的实际路径
#
# data_t = pd.read_csv(file_t, sep="\t", header=None, names=["Distance (pixels)", "Gray Value"])
# data_ip = pd.read_csv(file_ip, sep="\t", header=None, names=["Distance (pixels)", "Gray Value"])
#
# # Define the ranges for the broken axes
# xlims = ((0, 0.2), (0.8, data_t["Distance (pixels)"].max()))
#
# # Create the broken axes object
# fig = plt.figure(figsize=(12, 6))
# bax = brokenaxes(xlims=xlims, hspace=0.05)
#
# # Plot the data
# bax.plot(data_t["Distance (pixels)"], data_t["Gray Value"], label="T", color="blue")
# bax.plot(data_ip["Distance (pixels)"], data_ip["Gray Value"], label="I_p/G(I_p)", color="red")
#
# # Add labels and legend
# bax.set_xlabel("Distance (pixels)")
# bax.set_ylabel("Gray Value")
# bax.set_title("Comparison of T and I_p/G(I_p) Gray Values")
# bax.legend()
# bax.grid(True)
#
# # Show plot
# plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the data from the txt files
# file_t = 'G:\\文章\\aiffc\\oe文章图像\\oe回答\\T.txt'  # 请将 '路径/T.txt' 替换为您的 T.txt 文件的实际路径
# file_ip = 'G:\\文章\\aiffc\\oe文章图像\\oe回答\\I_p.txt'  # 请将 '路径/I_p.txt' 替换为您的 I_p.txt 文件的实际路径
#
# data_t = pd.read_csv(file_t, sep="\t", header=None, names=["Distance (pixels)", "Gray Value"])
# data_ip = pd.read_csv(file_ip, sep="\t", header=None, names=["Distance (pixels)", "Gray Value"])
#
# # Ensure both dataframes have the same length
# min_length = min(len(data_t), len(data_ip))
# data_t = data_t.iloc[:min_length]
# data_ip = data_ip.iloc[:min_length]
#
# # Calculate the difference, set result to zero if T's value is greater than 0.9
# data_diff = data_t.copy()
# data_diff["Gray Value"] = data_t.apply(
#     lambda row: row["Gray Value"] - data_ip.loc[row.name, "Gray Value"] if row["Gray Value"] <= 0.9 else 0,
#     axis=1
# )
#
# # Plot the data
# plt.figure(figsize=(12, 6))
# plt.plot(data_diff["Distance (pixels)"], data_diff["Gray Value"], label="T - I_p/G(I_p)", color="purple")
#
# # Add labels and legend
# plt.xlabel("Distance (pixels)")
# plt.ylabel("Gray Value Difference")
# plt.title("Difference between T and I_p/G(I_p) Gray Values (filtered)")
# plt.legend()
# plt.grid(True)
#
# # Show plot
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# 
# # Load the data from the txt files
# file_t = 'G:\\文章\\aiffc\\oe文章图像\\oe回答\\T.txt'  # 请将 '路径/T.txt' 替换为您的 T.txt 文件的实际路径
# file_ip = 'G:\\文章\\aiffc\\oe文章图像\\oe回答\\I_p.txt'  # 请将 '路径/I_p.txt' 替换为您的 I_p.txt 文件的实际路径
# 
# data_t = pd.read_csv(file_t, sep="\t", header=None, names=["Distance (pixels)", "Gray Value"])
# data_ip = pd.read_csv(file_ip, sep="\t", header=None, names=["Distance (pixels)", "Gray Value"])
# 
# # Ensure both dataframes have the same length
# min_length = min(len(data_t), len(data_ip))
# data_t = data_t.iloc[:min_length]
# data_ip = data_ip.iloc[:min_length]
# 
# # Calculate the difference, set result to zero if T's value is greater than 0.9
# data_diff = data_t.copy()
# data_diff["Gray Value"] = data_t.apply(
#     lambda row: row["Gray Value"] - data_ip.loc[row.name, "Gray Value"] if row["Gray Value"] <= 0.9 else 0,
#     axis=1
# )
# 
# # Create a figure and axis with two subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
#                                gridspec_kw={'height_ratios': [3, 1]})
# 
# # Plot the data on the main axis
# ax1.plot(data_t["Distance (pixels)"], data_t["Gray Value"], label="T", color="blue")
# ax1.plot(data_ip["Distance (pixels)"], data_ip["Gray Value"], label="I_p/G(I_p)", color="red",linestyle='-.')
# ax1.plot(data_diff["Distance (pixels)"], data_diff["Gray Value"], label="T - I_p/G(I_p)", color="purple")
# 
# ax1.set_ylabel("Gray Value")
# ax1.legend()
# ax1.grid(True)
# ax1.set_title("Comparison of T, I_p/G(I_p), and Their Difference")
# 
# # Plot the data on the zoomed-in axis
# ax2.plot(data_t["Distance (pixels)"], data_t["Gray Value"], label="T", color="blue")
# ax2.plot(data_ip["Distance (pixels)"], data_ip["Gray Value"], label="I_p/G(I_p)", color="red",linestyle='-.')
# ax2.plot(data_diff["Distance (pixels)"], data_diff["Gray Value"], label="T - I_p/G(I_p)", color="purple")
# ax2.set_xlabel("Distance (pixels)")
# ax2.set_ylabel("Gray Value")
# ax2.set_ylim(-0.04, 0.02)
# ax2.legend()
# ax2.grid(True)
# 
# # Adjust the layout to avoid overlap
# plt.tight_layout()
# plt.show()

# import cv2
# import numpy as np
# import tifffile as tiff
# import matplotlib.pyplot as plt
#
# # 读取图像并转换为浮点数图像
# image_path = r'G:\dataset\anim\test\1037_out.png'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# image_float = image.astype(np.float32) / 255.0
#
# # 计算Sobel梯度
# sobel_x = cv2.Sobel(image_float, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(image_float, cv2.CV_64F, 0, 1, ksize=3)
#
# # 计算梯度幅值
# gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
#
# # 设定白边灰度值范围
# min_val = 1.0
# max_val = 1.5
#
# # 计算梯度值与白边灰度值的线性关系
# normalized_gradient = cv2.normalize(gradient_magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
# gradient_map = min_val + normalized_gradient * (max_val - min_val)
#
# # 将梯度图与原图相乘
# output_image = image_float * gradient_map
#
# # 保存结果为32位浮点数图像
# output_path = r'G:\dataset\anim\test\1037_out_with_white_edges.tif'
# tiff.imwrite(output_path, output_image.astype(np.float32))
#
# # 显示结果
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
#
# plt.subplot(1, 3, 2)
# plt.title('Gradient Map')
# plt.imshow(gradient_map, cmap='gray')
#
# plt.subplot(1, 3, 3)
# plt.title('Image with Gradient White Edges')
# plt.imshow(output_image, cmap='gray')
#
# plt.show()


# import cv2
# import numpy as np
# import os
# import glob
# import random
# from tqdm import tqdm
# import tifffile as tiff
#
# # 输入输出文件夹路径
# input_folder = r'G:\dataset\anim'
# output_folder = r'I:\ffc模拟\原图'
#
# # 获取文件夹中的所有图像文件
# image_files = glob.glob(os.path.join(input_folder, '*.png'))
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
#
# # 图像尺寸
# image_width, image_height = 2048, 2048
# crop_width, crop_height = 1536, 1024
#
# # 总共进行7次随机打乱和处理
# for shuffle_round in range(7):
#     # 随机打乱图像路径
#     random.shuffle(image_files)
#
#     # 每次处理时初始化累加图像计数器
#     sum_image_counter = 0
#
#     # 使用tqdm显示进度
#     for i in tqdm(range(0, len(image_files), 7), desc=f'Shuffle {shuffle_round + 1}'):
#         current_files = image_files[i:i + 7]
#         if len(current_files) < 7:
#             break
#
#         # 初始化累加图像
#         sum_image = np.zeros((crop_height, crop_width), dtype=np.float32)
#
#         # 记录当前处理的文件名
#         file_names = []
#
#         # 对每张图像进行处理
#         for file in current_files:
#             file_names.append(os.path.basename(file))
#             image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#             image = image.astype(np.float32)
#
#             # 随机选择切割位置
#             x_start = random.randint(0, image_width - crop_width)
#             y_start = random.randint(0, image_height - crop_height)
#
#             # 切割图像块
#             crop_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]
#
#             # 累加图像块
#             sum_image += crop_image
#
#         # 转换为16位图像
#         sum_image = np.clip(sum_image, 0, 65535).astype(np.uint16)
#
#         # 保存结果图像
#         sum_image_counter += 1
#         file_names_str = "_".join([os.path.splitext(name)[0] for name in file_names])
#         output_path = os.path.join(output_folder,
#                                    f'sum_image_shuffle_{shuffle_round + 1}_group_{sum_image_counter}_{file_names_str}.tif')
#         tiff.imwrite(output_path, sum_image)

        # print(f"Shuffle {shuffle_round + 1}, Group {sum_image_counter} result saved to {output_path}")


# import cv2
# import numpy as np
# import os
# import glob
# import random
# from tqdm import tqdm
# import tifffile as tiff
#
# # 输入输出文件夹路径
# input_folder = r'G:\dataset\anim'
# output_folder = r'I:\ffc模拟\原图'
# float_output_folder = r'I:\ffc模拟\浮点图'
#
# # 获取文件夹中的所有图像文件
# image_files = glob.glob(os.path.join(input_folder, '*.png'))
#
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
# os.makedirs(float_output_folder, exist_ok=True)
#
# # 图像尺寸
# image_width, image_height = 2048, 2048
# crop_width, crop_height = 1536, 1024
#
# # 处理图像时的最大值
# max_val = 7 * 255
#
# # 总共进行7次随机打乱和处理
# for shuffle_round in range(7):
#     # 随机打乱图像路径
#     random.shuffle(image_files)
#
#     # 每次处理时初始化累加图像计数器
#     sum_image_counter = 0
#
#     # 使用tqdm显示进度
#     for i in tqdm(range(0, len(image_files), 7), desc=f'Shuffle {shuffle_round + 1}'):
#         current_files = image_files[i:i + 7]
#         if len(current_files) < 7:
#             break
#
#         # 初始化累加图像
#         sum_image = np.zeros((crop_height, crop_width), dtype=np.float32)
#
#         # 记录当前处理的文件名
#         file_names = []
#
#         # 对每张图像进行处理
#         for file in current_files:
#             file_names.append(os.path.basename(file))
#             image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#             image = image.astype(np.float32)
#
#             # 随机选择切割位置
#             x_start = random.randint(0, image_width - crop_width)
#             y_start = random.randint(0, image_height - crop_height)
#
#             # 切割图像块
#             crop_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]
#
#             # 累加图像块
#             sum_image += crop_image
#
#         # 转换为16位图像并保存
#         sum_image_16bit = np.clip(sum_image, 0, max_val).astype(np.uint16)
#         sum_image_counter += 1
#         file_names_str = "_".join([os.path.splitext(name)[0] for name in file_names])
#         output_path = os.path.join(output_folder,
#                                    f'sum_image_shuffle_{shuffle_round + 1}_group_{sum_image_counter}_{file_names_str}.tif')
#         # tiff.imwrite(output_path, sum_image_16bit)
#
#         # 对16位图像进行梯度处理并保存为浮点图像
#         sum_image_normalized = sum_image / max_val  # 将图像归一化到0到1之间
#
#         # 计算Sobel梯度
#         sobel_x = cv2.Sobel(sum_image_normalized, cv2.CV_64F, 1, 0, ksize=3)
#         sobel_y = cv2.Sobel(sum_image_normalized, cv2.CV_64F, 0, 1, ksize=3)
#
#         # 计算梯度幅值
#         gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
#
#         # 设定白边灰度值范围
#         gradient_min_val = 1.0
#         gradient_max_val = 2.0
#
#         # 计算梯度值与白边灰度值的线性关系
#         normalized_gradient = cv2.normalize(gradient_magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#         gradient_map = gradient_min_val + normalized_gradient * (gradient_max_val - gradient_min_val)
#
#         # 将原图像与梯度图相乘
#         float_output_image = sum_image_normalized * gradient_map
#
#         # 保存为32位浮点数图像
#         float_output_path = os.path.join(float_output_folder,
#                                          f'float_image_shuffle_{shuffle_round + 1}_group_{sum_image_counter}_{file_names_str}.tif')
#         tiff.imwrite(float_output_path, float_output_image.astype(np.float32))

# import cv2
# import numpy as np
# import tifffile as tiff
# import matplotlib.pyplot as plt
#
# # 读取图像
# image_path = r'I:\ffc模拟\整\float_image_shuffle_1_group_19_2659014_out_2207128_out_5049_out_2877107_out_1824141_out_2261071_out_2007051_out.tif'
# image = tiff.imread(image_path)
#
# # 设置高斯噪声参数
# mean = 1.0
# std_dev = 0.003  # 可以调整这个值来改变方差
#
# # 生成高斯噪声
# gaussian_noise = np.random.normal(mean, std_dev, image.shape)
#
# # 将噪声添加到图像
# noisy_image = image * gaussian_noise
#
# # 确保图像值在合理范围内
# # noisy_image = np.clip(noisy_image, 0, np.max(image))
#
# # 保存带噪声的图像
# output_path = r'I:\ffc模拟\整\noisy_image.tif'
# tiff.imwrite(output_path, noisy_image.astype(np.float32))
#
# # 显示原图像和带噪声的图像
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
#
# plt.subplot(1, 2, 2)
# plt.title('Noisy Image')
# plt.imshow(noisy_image, cmap='gray')
#
# plt.show()
#
# print(f"Noisy image saved to {output_path}")


# import cv2
# import numpy as np
# import os
# import glob
# import random
# from tqdm import tqdm
# import tifffile as tiff
#
# # 文件夹路径
# input_folder_A = r'I:\ffc模拟\浮点图'
# input_folder_B = r'J:\20240130DK\flat'
# output_folder_A = r'I:\ffc模拟\train_A'
# output_folder_B = r'I:\ffc模拟\train_B'
#
# # 创建输出文件夹
# os.makedirs(output_folder_A, exist_ok=True)
# os.makedirs(output_folder_B, exist_ok=True)
#
# # 获取文件夹中的所有图像文件
# image_files_A = glob.glob(os.path.join(input_folder_A, '*.tif'))
# image_files_B = glob.glob(os.path.join(input_folder_B, '*.png'))
#
# # 设置高斯噪声参数
# mean = 1.0
# std_dev = 0.003
#
# # 图像尺寸
# crop_width, crop_height = 1536, 1024
#
# # 处理A文件夹中的图像并添加高斯噪声
# for image_path_A in tqdm(image_files_A, desc='Processing images from A'):
#     # 读取图像
#     image_A = tiff.imread(image_path_A)
#
#     # 生成高斯噪声
#     # gaussian_noise = np.random.normal(mean, std_dev, image_A.shape)
#     gaussian_noise = 0
#
#     # 将噪声添加到图像
#     noisy_image_A = image_A * gaussian_noise
#
#     # 确保图像值在合理范围内
#     noisy_image_A = np.clip(noisy_image_A, 0, np.max(image_A))
#
#     # 随机选择B文件夹中的图像
#     image_path_B = random.choice(image_files_B)
#     image_B = cv2.imread(image_path_B, cv2.IMREAD_UNCHANGED)  # 读取16位灰度图像
#
#     # 确保B图像为16位灰度图像
#     if image_B is None or image_B.dtype != np.uint16:
#         print(f"Skipping {image_path_B} as it is not a 16-bit gray image.")
#         continue
#
#     # 随机选择裁剪位置
#     x_start = random.randint(0, image_B.shape[1] - crop_width)
#     y_start = random.randint(0, image_B.shape[0] - crop_height)
#
#     # 裁剪B文件夹中的图像
#     cropped_image_B = image_B[y_start:y_start + crop_height, x_start:x_start + crop_width]
#
#     # 保存裁剪后的图像为TIFF格式
#     base_name_A = os.path.basename(image_path_A).replace('.tif', '')
#     base_name_B = f'{base_name_A}_cropped_{x_start}_{y_start}.tif'
#     output_path_B = os.path.join(output_folder_B, base_name_B)
#     tiff.imwrite(output_path_B, cropped_image_B)
#
#     # 将带噪声的图像与裁剪后的图像相乘
#     result_image = noisy_image_A * cropped_image_B
#
#     # 确保结果图像值在合理范围内
#     result_image = np.clip(result_image, 0, np.max(result_image)).astype(np.uint16)
#
#     # 保存结果图像为16位TIFF格式
#     output_path_A = os.path.join(output_folder_A, base_name_B)
#     tiff.imwrite(output_path_A, result_image)
#
#     # print(f"Processed: {base_name_B}")
#
# print("Processing complete.")

import cv2
import numpy as np
import os
import glob
import random
from tqdm import tqdm
import tifffile as tiff

# 文件夹路径
input_folder_A = r'I:\ffc\浮点图'
input_folder_B = r'H:\20250410\honghuacujiangcao2_12kev_700ms_0.65um\flat'
output_folder_A = r'H:\郭静好\无监督\train_A'
output_folder_B = r'H:\郭静好\无监督\train_B'

# 创建输出文件夹

os.makedirs(output_folder_A, exist_ok=True)
os.makedirs(output_folder_B, exist_ok=True)

# 获取文件夹中的所有图像文件
image_files_A = glob.glob(os.path.join(input_folder_A, '*.tif'))
image_files_B = glob.glob(os.path.join(input_folder_B, '*.tiff'))

# 设置高斯噪声参数
mean = 1.0
std_dev = 0.011

# 图像尺寸
crop_width, crop_height = 1024, 1024

# 处理A文件夹中的图像并添加高斯噪声
for image_path_A in tqdm(image_files_A, desc='Processing images from A'):
    # 读取图像
    image_A = tiff.imread(image_path_A)

    # 生成高斯噪声
    gaussian_noise = np.random.normal(mean, std_dev, image_A.shape)


    # 将噪声添加到图像
    noisy_image_A = image_A * gaussian_noise

    # 确保图像值在合理范围内
    noisy_image_A = np.clip(noisy_image_A, 0, np.max(image_A))

    # 随机选择B文件夹中的图像
    image_path_B = random.choice(image_files_B)
    # image_B = cv2.imread(image_path_B, cv2.IMREAD_UNCHANGED)  # 读取16位灰度图像
    image_B = Image.open(image_path_B)
    image_B = np.array(image_B)

    # 确保B图像为16位灰度图像
    if image_B is None or image_B.dtype != np.uint16:
        print(f"Skipping {image_path_B} as it is not a 16-bit gray image.")
        continue

    # 随机选择裁剪位置

    x_start = random.randint(0, image_B.shape[1] - crop_width)
    y_start = random.randint(0, image_B.shape[0] - crop_height)

    # 裁剪B文件夹中的图像
    cropped_image_B = image_B[y_start:y_start + crop_height, x_start:x_start + crop_width]

    # 缩小图像尺寸为原来的四分之一
    new_width = int(cropped_image_B.shape[1] / 4)
    new_height = int(cropped_image_B.shape[0] / 4)
    resized_image_B = cv2.resize(cropped_image_B, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 缩小noisy_image_A的尺寸为原来的四分之一
    # new_width_A = int(noisy_image_A.shape[1] / 4)
    # new_height_A = int(noisy_image_A.shape[0] / 4)
    new_width_A = int(cropped_image_B.shape[1] / 4)
    new_height_A = int(cropped_image_B.shape[0] / 4)
    resized_noisy_image_A = cv2.resize(noisy_image_A.astype(np.float32), (new_width_A, new_height_A), interpolation=cv2.INTER_AREA)

    # 保存裁剪后的图像为TIFF格式
    base_name_A = os.path.basename(image_path_A).replace('.tif', '')
    base_name_B = f'{base_name_A}_cropped_{x_start}_{y_start}.png'
    output_path_B = os.path.join(output_folder_B, base_name_B)

    # tiff.imwrite(output_path_B, resized_image_B)

    # 将带噪声的图像与裁剪后的图像相乘
    result_image = resized_noisy_image_A * resized_image_B

    # 确保结果图像值在合理范围内
    result_image = np.clip(result_image, 0, np.max(result_image)).astype(np.uint16)

    # 保存结果图像为16位TIFF格式
    output_path_A = os.path.join(output_folder_A, base_name_B)
    img_I= Image.fromarray(result_image)
    img_I.save(output_path_A)
    img_I2= Image.fromarray(resized_image_B)
    img_I2.save(output_path_B)
    # print(f"Processed: {base_name_B}")

print("Processing complete.")

# import os
# import cv2


#
# def add_gaussian_noise(image, mean=1, variance=0.001):
#     """
#     给 16 位图像添加高斯噪声，噪声与原图相乘
#     :param image: 输入的 16 位图像
#     :param mean: 高斯噪声的均值
#     :param variance: 高斯噪声的方差
#     :return: 添加噪声后的 16 位图像
#     """
#     sigma = np.sqrt(variance)
#     gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
#     noisy_image = image.astype(np.float32) * gaussian
#     noisy_image = np.clip(noisy_image, 0, 65535).astype(np.uint16)
#     return noisy_image
#
#
# def process_images(input_folder_path, output_folder_path):
#     """
#     处理指定文件夹中的所有 16 位 PNG 图像
#     :param input_folder_path: 输入文件夹路径
#     :param output_folder_path: 输出文件夹路径
#     """
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)
#
#     for filename in os.listdir(input_folder_path):
#         if filename.endswith('.png'):
#             file_path = os.path.join(input_folder_path, filename)
#             # 以 16 位灰度模式读取图像
#             image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
#             if image is not None:
#                 noisy_image = add_gaussian_noise(image)
#                 # 保存添加噪声后的 16 位图像
#                 output_path = os.path.join(output_folder_path, f'{filename}')
#                 cv2.imwrite(output_path, noisy_image)
#                 print(f'已处理并保存: {output_path}')
#             else:
#                 print(f'无法读取图像: {file_path}')
#
#
# if __name__ == "__main__":
#     input_folder_path = r'G:\latent_diffusion\Palette-Image-to-Image-Diffusion-Models-main\datasets\yumeiren\img\train_A1'
#     output_folder_path = r'G:\latent_diffusion\Palette-Image-to-Image-Diffusion-Models-main\datasets\yumeiren\img\train_A2'
#     process_images(input_folder_path, output_folder_path)
#

