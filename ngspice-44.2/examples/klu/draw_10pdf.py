import fitz  # PyMuPDF
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
import os

# 你的 PDF 文件名列表（按你想要的顺序）
pdf_filenames = [
    "c1355_devC_1e-15.pdf", "c1908_devC_1e-15.pdf", "c2670_devC_1e-15.pdf",
    "c3540_devC_1e-15.pdf", "c5315_devC_1e-15.pdf", "c6288_devC_1e-15.pdf",
    "c7552_devC_1e-15.pdf", "c880_devC_1e-15.pdf", "c432_devC_1e-15.pdf",
    "c499_devC_1e-15.pdf"
]
pdf_filenames = sorted(pdf_filenames)

# 每行排列数量
columns = 5
rows = (len(pdf_filenames) + columns - 1) // columns

# 读取第一个 PDF 图像尺寸（假设全部图像大小相同）
doc_sample = fitz.open(pdf_filenames[0])
rect = doc_sample[0].rect
width, height = rect.width, rect.height

# 计算最终大图的尺寸
final_width = width * columns
final_height = height * rows

# 创建新 PDF 文档用于拼接
output_doc = fitz.open()

# 创建空白页
new_page = output_doc.new_page(width=final_width, height=final_height)

# 插入每一个 PDF 图
for index, filename in enumerate(pdf_filenames):
    row = index // columns
    col = index % columns

    x0 = col * width
    y0 = row * height

    # 加载并获取图页
    src_doc = fitz.open(filename)
    src_page = src_doc[0]

    # 将原始 PDF 页作为图像插入到新页面上
    new_page.show_pdf_page(
        fitz.Rect(x0, y0, x0 + width, y0 + height),  # 放置位置
        src_doc,  # 来源文档
        0  # 来源页码
    )

# 保存输出
output_doc.save("combined_grid_devC.pdf")
print("拼接完成，保存为 combined_grid_devC.pdf")
