from PIL import Image
from fpdf import FPDF
import os

# 画像ファイルのパスを取得
image_folder = "path/to/your/png_folder"  # PNGファイルが保存されているフォルダ
output_pdf = "output.pdf"  # 出力するPDFファイル名

# PNGファイルをリストアップ
png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
png_files.sort()  # 名前順にソート（必要に応じて変更）

# PDF作成
pdf = FPDF()
for png_file in png_files:
    img_path = os.path.join(image_folder, png_file)
    img = Image.open(img_path)

    # 画像サイズの取得
    width, height = img.size
    pdf.add_page()
    pdf.image(
        img_path, x=0, y=0, w=210, h=297
    )  # A4サイズに合わせる例（変更可能）

# PDF保存
pdf.output(output_pdf)
print(f"PDFが作成されました: {output_pdf}")
