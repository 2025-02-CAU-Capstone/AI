import easyocr
from PIL import Image
import numpy as np
import io

reader = easyocr.Reader(['ko','en'], gpu=False)

async def run_ocr(file):
    print("[OCR] run_ocr called")

    contents = await file.read()
    print(f"[OCR] file size: {len(contents)} bytes")

    try:
        image = Image.open(io.BytesIO(contents))
        print(f"[OCR] image loaded: mode={image.mode}, size={image.size}")
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

    width, height = image.size

    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
        print("[OCR] Converted to RGB")
    
    #image resize
    MAX_SIZE = 1600
    long_side = max(width, height)

    if long_side > MAX_SIZE:
        scale = MAX_SIZE / long_side
        new_width = int(width * scale)
        new_height = int(height * scale)

        print(f"[OCR] Resizing image from {width}x{height} → {new_width}x{new_height}")
        image = image.resize((new_width, new_height))

        #image size update
        width, height = new_width, new_height

    #converted to gray-scale
    image = image.convert("L")

    img_array = np.array(image)

    print(f"[OCR] numpy array shape: {img_array.shape}")

    try:
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
