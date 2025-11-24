import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

logger.info("Loading search model...")
search_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
logger.info("Search model loaded")