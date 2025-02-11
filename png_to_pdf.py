from PIL import Image
from fpdf import FPDF
import os

targets = [
    'dGs',
    'Ebd',
    'log10(lifetime)',
    'logD',
    'logP',
    'logS',
    'pKaA',
    'pKaB',
    'RI',
    'Tb',
    'Tm',
]

# PDF作成
pdf = FPDF()
for target in targets:
    # 画像ファイルのパスを取得
    img_path = f"/Users/sshunsuke/Downloads/rmse_{target}.png"  # PNGファイルが保存されているフォルダ
    output_pdf = f"/Users/sshunsuke/Downloads/rmse_{target}.pdf"  # 出力するPDFファイル名

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
