import easyocr
from PIL import Image
import numpy as np
import io

# EasyOCR 모델 로딩 (서버 시작 시 1회)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

async def run_ocr(file):
    """이미지 파일에서 텍스트만 추출하는 간단한 OCR 함수"""

    # 파일 읽기
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # RGBA → RGB 변환
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')

    img_array = np.array(image)

    # EasyOCR 실행
    result = reader.readtext(img_array, detail=1)

    # 텍스트만 추출
    texts = []
    for item in result:
        try:
            _, text, conf = item
            if text and text.strip():
                texts.append(text.strip())
        except:
            continue

    # 문장 리스트 또는 raw 텍스트 반환
    raw_text = " ".join(texts)
    return raw_text
