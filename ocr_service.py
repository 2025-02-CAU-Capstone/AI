import easyocr
from PIL import Image
import numpy as np
import io

reader = easyocr.Reader(['ko','en'], gpu=False)

async def run_ocr(file):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    width, height = image.size

    if image.mode in ('RGBA','LA','P'):
        image = image.convert('RGB')

    img_array = np.array(image)
    result = reader.readtext(img_array, detail=1)

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
