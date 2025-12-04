import easyocr
from PIL import Image, ImageOps, ImageEnhance # ImageOps 필수
import numpy as np
import io

reader = easyocr.Reader(['ko','en'], gpu=False)

async def run_ocr(file):
    print("[OCR] run_ocr called")

    contents = await file.read()
    print(f"[OCR] file size: {len(contents)} bytes")

    try:
        image = Image.open(io.BytesIO(contents))
        
        # [필수 1] EXIF 회전 정보 반영 (이게 없으면 폰 사진은 90% 실패합니다)
        image = ImageOps.exif_transpose(image)
        
        print(f"[OCR] image loaded & oriented: mode={image.mode}, size={image.size}")
    except Exception as e:
        print("[OCR] failed to load image:", e)
        return {
            "success": False,
            "textBoxes": None,
            "sentences": None,
            "raw_text": None,
            "confidence": 0,
            "imageWidth": 0,
            "imageHeight": 0,
            "message": "Image load failed"
        }

    # 이미지 모드 통일 (RGBA 등 -> RGB)
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
        
    # [수정] MedianFilter 삭제함 (글자를 뭉개버리는 주범)
    # image = image.filter(ImageFilter.MedianFilter(size=3))  <-- 절대 금지

    # [필수 2] 리사이징
    MAX_SIZE = 1200 
    width, height = image.size
    long_side = max(width, height)

    if long_side > MAX_SIZE:
        scale = MAX_SIZE / long_side
        new_width = int(width * scale)
        new_height = int(height * scale)

        print(f"[OCR] Resizing from {width}x{height} -> {new_width}x{new_height}")
        # [수정] BILINEAR -> LANCZOS (글자 선명도 유지에 필수)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        width, height = new_width, new_height

    # 흑백 변환
    image = image.convert("L")
    
    # 선명도 조절 (너무 과하면 노이즈가 생기니 1.5 정도로 낮춤)
    sharpener = ImageEnhance.Sharpness(image)
    image = sharpener.enhance(1.5)

    img_array = np.array(image)
    print(f"[OCR] numpy array shape: {img_array.shape}")

    try:
        # detail=1 로 상세 정보 획득
        result = reader.readtext(img_array, detail=1)
        print(f"[OCR] result length: {len(result)}")
    except Exception as e:
        print("[OCR] easyocr readtext error:", e)
        return {
            "success": False,
            "textBoxes": None,
            "sentences": None,
            "raw_text": None,
            "confidence": 0,
            "imageWidth": width,
            "imageHeight": height,
            "message": f"EasyOCR failed: {str(e)}"
        }

    sentences = []
    confidences = []
    text_boxes = []

    for item in result:
        try:
            box, text, conf = item
            if text.strip():
                sentences.append(text.strip())
                confidences.append(float(conf))

                box_clean = [[float(x), float(y)] for x, y in box]

                text_boxes.append({
                    "box": box_clean,
                    "text": text,
                    "conf": float(conf)
                })
        except:
            continue

    raw_text = " ".join(sentences)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "success": True,
        "textBoxes": text_boxes,
        "sentences": sentences,
        "raw_text": raw_text,
        "confidence": avg_conf,
        "imageWidth": width,
        "imageHeight": height,
        "message": None
    }