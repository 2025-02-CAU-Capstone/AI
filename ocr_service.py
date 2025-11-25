from fastapi import UploadFile, File
from pydantic import BaseModel

class OCRResponse(BaseModel):
    success: bool
    textBoxes: list | None
    sentences: list | None
    raw_text: str | None
    confidence: float
    imageWidth: int
    imageHeight: int
    message: str | None


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        width, height = image.size

        if image.mode in ('RGBA', 'LA', 'P'):
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
                    confidences.append(conf)
                    text_boxes.append({"box": box, "text": text, "conf": conf})
            except:
                continue

        raw_text = " ".join(sentences)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResponse(
            success=True,
            textBoxes=text_boxes,
            sentences=sentences,
            raw_text=raw_text,
            confidence=avg_conf,
            imageWidth=width,
            imageHeight=height,
            message=None
        )

    except Exception as e:
        return OCRResponse(
            success=False,
            textBoxes=None,
            sentences=None,
            raw_text=None,
            confidence=0,
            imageWidth=0,
            imageHeight=0,
            message=str(e)
        )