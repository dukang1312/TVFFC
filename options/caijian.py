import imageio
import os
from PIL import Image
import tqdm
import numpy as np
import tifffile as tif
def convert_tif_to_png(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in tqdm.tqdm(os.listdir(input_folder),desc="png convert:"):
        if filename.endswith('.tiff'):
            # 构建输入文件的完整路径
            input_path = os.path.join(input_folder, filename)
            try:
                # 使用 imageio 读取 tiff 图像
                img = imageio.imread(input_path)
                # 构建输出文件的完整路径，将扩展名改为 .png
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)
                # 使用 imageio 保存为 16 位的 png 格式
                tif.imwrite(output_path, img)
                # print(f"已将 {input_path} 转换为 {output_path}")
            except Exception as e:
                print(f"处理 {input_path} 时出错: {e}")

def resize_png_images(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in tqdm.tqdm(os.listdir(input_folder),desc="resize："):
        if filename.endswith('.png'):
            # 构建输入文件的完整路径
            input_path = os.path.join(input_folder, filename)
            try:
                # 打开 PNG 图像
                with Image.open(input_path) as img:
                    # 计算新的尺寸，横纵像素值都变为原来的一半
                    new_width = img.width //4
                    new_height = img.height //4
                    new_size = (new_width, new_height)

                    # 调整图像大小
                    resized_img = img.resize(new_size)

                    # 构建输出文件的完整路径
                    output_path = os.path.join(output_folder, filename)
                    image = np.array(resized_img, dtype=np.uint16)
                    image = image[128:128+256,128:128+256]
                    tif.imwrite(output_path,image)

                    # 保存调整大小后的图像
                    # resized_img.save(output_path, 'PNG')
                    # print(f"已将 {input_path} 缩小为 {output_path}")
            except Exception as e:
                print(f"处理 {input_path} 时出错: {e}")
if __name__ == "__main__":
    input_folder = r'H:\20250410\honghuacujiangcao2_12kev_700ms_0.65um'
    output_folder = r'H:\20250410\honghuacujiangcao2_12kev_700ms_0.65um\png'
    INPUT1 = r"H:\20250410\honghuacujiangcao2_12kev_700ms_0.65um\png_resize"
    convert_tif_to_png(input_folder,output_folder)
    resize_png_images(output_folder, INPUT1)

