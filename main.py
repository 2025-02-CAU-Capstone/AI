from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from search import get_start_timestamp
import subprocess
from ocr_service import run_ocr
from ocr_post_process import post_ocr_processing

app = FastAPI()


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    ocr_raw = await run_ocr(file)
    if not ocr_raw["success"]:
        return ocr_raw
    
    processed = post_ocr_processing(ocr_raw["textBoxes"])

    return {
        "success": True,
        "imageWidth": ocr_raw["imageWidth"],
        "imageHeight": ocr_raw["imageHeight"],
        "raw_text": ocr_raw["raw_text"],
        "avg_conf": ocr_raw["confidence"],
        "processed_groups": processed 
    }

# "processed_groups" part will contain the final JSON where OpenAI API created. It looks like following:
"""
    {
    "choose_id": 1,
    "group_position": [[...], [...], ...],
    "merged_text": "..."
    }
"""

class SearchRequest(BaseModel):
    question: str


@app.post("/search")
def search_endpoint(req: SearchRequest):
    res = get_start_timestamp(req.question)
    return res


@app.post("/rebuild-embeddings")
def rebuild_embeddings(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_embedding_script)
    return {"status": "queued"}

def run_embedding_script():
    subprocess.run(["python3", "create_embedding.py"], check=True)
