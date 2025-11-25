from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from search import get_start_timestamp
import subprocess
from ocr_service import run_ocr

app = FastAPI()


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    text = await run_ocr(file)
    return text


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
