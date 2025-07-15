import os
from PIL import Image


def find_layer_images(suf):
    # 获取当前目录所有 *_layer.png 的图片，并排序
    return sorted([f for f in os.listdir('.') if f.endswith(suf + '.png')])


def concatenate_images_vertically(image_files, output_file):
    if not image_files:
        print("没有找到任何 *_layer.png 图片。")
        return

    images = []
    for f in image_files:
        if 'c1355' in f or 'c432' in f or 'c499' in f or 'combined' in f:
            continue
        print(f)
        images.append(Image.open(f))

    # 获取单张图片的尺寸，假设宽度相同
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    width = widths[0]

    # 创建拼接后的新图像
    combined_image = Image.new('RGB', (width, total_height))

    # 粘贴每张图
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # 保存合成图
    combined_image.save(output_file + '.png')
    print(f"已保存拼接图为：{output_file}.png")

    images = []
    for f in image_files:
        if 'c432' in f or 'c499' in f:
            images.append(Image.open(f))

    # 获取单张图片的尺寸，假设宽度相同
    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    width = widths[0]

    # 创建拼接后的新图像
    combined_image = Image.new('RGB', (width, total_height))

    # 粘贴每张图
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # 保存合成图
    combined_image.save(output_file + '_c432_c499' + '.png')
    print(f"已保存拼接图为：{output_file}_c432_c499.png")


if __name__ == '__main__':
    files = find_layer_images('_layer')
    concatenate_images_vertically(files, 'combined_layer')
    files = find_layer_images('_backward')
    concatenate_images_vertically(files, 'combined_backward')
