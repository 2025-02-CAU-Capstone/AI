"""
Smart Search Version
- 외부 입력으로 embedding 파일 로드
- 문제를 직접 입력받아 검색
- Smart Start Find:
    1) 패턴 기반 Boost (키워드, 개념어)
    2) 유사도 드롭 감지 (Δ0.20 + abs < 0.45)
"""
# import sys
# import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests
import base64
from io import BytesIO

DROP_THRESHOLD = 0.20
ABS_SIM_THRESHOLD = 0.45
CONCEPT_KEYWORDS = ["이론", "개념", "정의", "모형", "분류", "특징"]

EMB_URL = "https://13-209-30-220.nip.io/api/embeddings/latest"

# 1. 임베딩 로드
def load_embedding_file():
    # with open(path, "rb") as f:
    #     data = pickle.load(f)

    r = requests.get(EMB_URL, timeout=20)
    r.raise_for_status()
    data = r.json()

    emb_bytes = base64.b64decode(data["embeddingsB64"])
    embeddings = np.load(BytesIO(emb_bytes))

    return (
        data["timestamp"],
        data["text"],
        embeddings,
        data["lectureId"],
        data["chapterId"]
    )


# 2. Boost
def apply_pattern_boost(problem, sentence, score):
    problem_words = set(problem.replace(",", " ").split())
    sentence_words = set(sentence.replace(",", " ").split())

    if len(problem_words & sentence_words) > 0:
        score += 0.05

    if any(k in sentence for k in CONCEPT_KEYWORDS):
        score += 0.03

    return score


# 3. Smart search → top1 peak만 필요
def smart_search(problem, timestamps, sentences, embeddings, model):
    q_emb = model.encode(problem, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()

    boosted = []
    for idx, sen in enumerate(sentences):
        s = apply_pattern_boost(problem, sen, sims[idx])
        boosted.append((idx, s))

    boosted.sort(key=lambda x: x[1], reverse=True)

    top_idx, top_score = boosted[0]
    return top_idx, sims


# 4. Smart Start 탐색 (peak 이전)
def find_start_timestamp(peak_idx, sims):
    start_idx = peak_idx

    for i in range(peak_idx - 1, -1, -1):
        # peak로 향하는 방향은 유사도가 증가하는 방향이어야 함
        if sims[i] <= sims[i + 1]:  # 증가하지 않으면 stop
            break
        start_idx = i

    return start_idx


def get_start_timestamp(problem):
    timestamps, sentences, embeddings, lecture_ids, chapter_ids = load_embedding_file()

    model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    peak_idx, sims = smart_search(problem, timestamps, sentences, embeddings, model)
    start_idx = find_start_timestamp(peak_idx, sims)

    return {
        "startTimestamp": timestamps[start_idx],
        "peakTimestamp": timestamps[peak_idx],
        "lectureId": lecture_ids[peak_idx],
        "chapterId": chapter_ids[peak_idx],
        "sentence": sentences[peak_idx]
    }

# MAIN
def main():
    while True:
        problem = input("\n문제 입력 (q: 종료) >> ").strip()

        if problem.lower() == "q":
            print("\n종료합니다.")
            break

        result = get_start_timestamp(problem)

        print("\n=== 결과 ===")
        print(f"Lecture: {result['lectureId']}")
        print(f"Chapter: {result['chapterId']}")
        print(f"Start Timestamp: {result['startTimestamp']}")
        print(f"Peak Timestamp: {result['peakTimestamp']}")
        print(f"Sentence: {result['sentence']}")


if __name__ == "__main__":
    main()
