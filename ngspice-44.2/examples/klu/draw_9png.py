import os
from PIL import Image


def draw_9(suf):
    # 1. 获取所有 *_devC.png 文件并排序
    all_images = sorted(
        [f for f in os.listdir('.') if f.endswith('_' + suf + '.png')])

    # 2. 取排序后的最后 9 张图像
    selected_images = all_images[-9:]

    # 3. 图像尺寸（假设都是 900x900）
    img_width, img_height = 900, 900
    grid_size = 3  # 3x3 九宫格

    # 4. 创建一个新的空白图像（900 x 900）
    output_img = Image.new('RGB',
                           (img_width * grid_size, img_height * grid_size))

    # 5. 按九宫格布局粘贴图像
    for idx, filename in enumerate(selected_images):
        row = idx // grid_size
        col = idx % grid_size
        img = Image.open(filename)
        output_img.paste(img, (col * img_width, row * img_height))


# 6. 保存输出图像
    output_img.save(suf + '_grid_output.png')
    print(f"九宫格图像已保存为 {suf}_grid_output.png")


def draw_10(suf):
    # 1. 获取所有 *_devC.png 文件并排序
    all_images = sorted(
        [f for f in os.listdir('.') if f.endswith('_' + suf + '.png')])

    # 2. 取排序后的最后 10 张图像
    selected_images = all_images[-10:]

    # 3. 图像尺寸（假设都是 900x900）
    img_width, img_height = 900, 900
    (x, y) = (5, 2)

    # 4. 创建一个新的空白图像（4500 x 1800）
    output_img = Image.new('RGB', (img_width * x, img_height * y))

    # 5. 按九宫格布局粘贴图像
    for idx, filename in enumerate(selected_images):
        row = idx // x
        col = idx % x
        img = Image.open(filename)
        output_img.paste(img, (col * img_width, row * img_height))


# 6. 保存输出图像
    output_img.save(suf + '_grid10.png')
    print(f"已保存为 {suf}_grid10.png")

draw_10("devC")
draw_10("devM")
draw_10("spm")
draw_10("centerdeg")
draw_10("centernodeinring")
draw_10("centersimplering")
"""
draw_9("devC")
draw_9("devM")
draw_9("spm")
draw_9("centerdeg")
draw_9("centernodeinring")
draw_9("centersimplering")
"""
