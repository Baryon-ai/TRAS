"""
🧠 TRAS BERT 모듈
고속이고 신뢰 가능한 BERT 기반 정부 인재 추천 시스템

강의 내용을 바탕으로 구현된 실제 운영 시스템
- 토큰화: 효율적인 한국어 토큰화
- 임베딩: 문맥적 벡터 표현
- 어텐션: 중요 정보 집중 분석
- 최적화: 고속 추론과 신뢰성 향상
"""

__version__ = "1.0.0"
__author__ = "TRAS AI Team"

from .tokenizer import KoreanTokenizer
from .embedding import ContextualEmbedding
from .attention import MultiHeadAttention
from .bert_model import FastBERT, ReliableBERT
from .optimizer import BERTOptimizer
from .evaluator import TrustScoreCalculator

__all__ = [
    "KoreanTokenizer",
    "ContextualEmbedding", 
    "MultiHeadAttention",
    "FastBERT",
    "ReliableBERT",
    "BERTOptimizer",
    "TrustScoreCalculator"
] 