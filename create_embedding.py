"""
임베딩 생성 + 서버 업로드 버전 (백엔드 엔드포인트 구조에 맞춤)
- transcripts는 서버에서 직접 GET
- embeddings는 numpy로 생성
- metadata(JSON) / embeddings(NPY)를 2번 POST
"""

import json
import numpy as np
import io
import requests
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------
TRANSCRIPT_API = "https://13-209-30-220.nip.io/api/transcripts"
UPLOAD_META_API = "https://13-209-30-220.nip.io/api/embeddings/upload"
UPLOAD_NPY_API = "https://13-209-30-220.nip.io/api/embeddings/upload-npy"

model = SentenceTransformer('jhgan/ko-sroberta-multitask')


# ----------------------------
# 1. Transcript GET
# ----------------------------
def load_transcripts_from_server():
    r = requests.get(TRANSCRIPT_API, timeout=20)
    r.raise_for_status()
    data = r.json()

    timestamps = [x["startTime"] for x in data]
    texts = [x["content"] for x in data]
    lecture_ids = [x["lectureId"] for x in data]
    chapter_ids = [x["chapterId"] for x in data]

    return timestamps, texts, lecture_ids, chapter_ids


# ----------------------------
# 2. Embedding 생성
# ----------------------------
def create_embeddings(timestamps, texts, lecture_ids, chapter_ids):

    print(f"총 문장 수: {len(texts)}")
    print("임베딩 생성 중…")

    emb = model.encode(texts, convert_to_tensor=False)
    emb = np.array(emb)

    print("임베딩 완료.")

    # metadata (JSON)
    metadata = {
        "timestamp": timestamps,
        "text": texts,
        "lectureId": lecture_ids,
        "chapterId": chapter_ids,
        "vector_dim": emb.shape[1],
        "count": emb.shape[0]
    }

    # numpy npy 메모리 파일
    emb_file = io.BytesIO()
    np.save(emb_file, emb)
    emb_file.seek(0)

    return metadata, emb_file


# ----------------------------
# 3. 서버 업로드
# ----------------------------
def upload_to_server(metadata, emb_file):

    print("1) metadata 업로드 중…")
    r1 = requests.post(UPLOAD_META_API, json=metadata, timeout=60)
    r1.raise_for_status()

    print("2) npy 업로드 중…")
    r2 = requests.post(
        UPLOAD_NPY_API,
        data=emb_file.read(), 
        headers={"Content-Type": "application/octet-stream"},
        timeout=60
    )
    r2.raise_for_status()

    print("업로드 성공!")


# ----------------------------
# MAIN
# ----------------------------
def main():
    timestamps, texts, lecture_ids, chapter_ids = load_transcripts_from_server()
    metadata, emb_file = create_embeddings(timestamps, texts, lecture_ids, chapter_ids)
    upload_to_server(metadata, emb_file)


if __name__ == "__main__":
    main()
