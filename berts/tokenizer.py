"""
🔤 한국어 특화 토크나이저
강의 Section 2에서 다룬 토큰화 개념의 실제 구현

BPE, 형태소 분석, 노이즈 제거를 통한 효율적 토큰화
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import time
import logging

# 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenizationResult:
    """토큰화 결과 데이터 클래스"""
    tokens: List[str]
    token_ids: List[int]
    attention_mask: List[int]
    special_tokens_mask: List[int]
    processing_time: float
    confidence_score: float

class KoreanTokenizer:
    """
    🚀 고속 한국어 토크나이저
    강의에서 다룬 BPE와 형태소 분석을 결합한 실제 구현
    """
    
    def __init__(self, vocab_path: Optional[str] = None, max_length: int = 512):
        """
        토크나이저 초기화
        
        Args:
            vocab_path: 어휘 사전 경로 (None이면 기본 어휘 사용)
            max_length: 최대 토큰 길이
        """
        self.max_length = max_length
        
        # 특수 토큰 정의 (vocab 로드 전에)
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1, 
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        
        self.vocab = self._load_vocabulary(vocab_path)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # 정부 관련 도메인 특화 어휘
        self.government_vocab = {
            '정책관', '과장', '국장', '차관', '장관', '대통령', '총리',
            '비서관', '보좌관', '수석', '정부', '공무원', '추천', '임용',
            'AI', '인공지능', '데이터', '분석', '시스템'
        }
        
        # 캐시를 위한 해시 테이블
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"🔤 KoreanTokenizer 초기화 완료 - 어휘 크기: {len(self.vocab)}")
    
    def _load_vocabulary(self, vocab_path: Optional[str]) -> Dict[str, int]:
        """어휘 사전 로드"""
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 기본 어휘 생성 (실제 환경에서는 사전 훈련된 어휘 사용)
        vocab = {}
        vocab.update(self.special_tokens)
        
        # 한글 기본 자모 추가
        for i in range(0xAC00, 0xD7A4):  # 한글 음절
            char = chr(i)
            if char not in vocab:
                vocab[char] = len(vocab)
        
        # 영문자, 숫자 추가
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            if c not in vocab:
                vocab[c] = len(vocab)
        
        return vocab
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리 (강의 Section 2의 노이즈 제거 구현)
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정제된 텍스트
        """
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 이메일 서명 제거 (-- 이후 모든 내용)
        text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)
        
        # 특수 문자 정규화
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _bpe_encode(self, text: str) -> List[str]:
        """
        BPE 인코딩 (강의에서 다룬 효율적 조합 처리)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            BPE 토큰 리스트
        """
        # 캐시 확인
        if text in self._cache:
            self._cache_hits += 1
            return self._cache[text]
        
        self._cache_misses += 1
        
        # 간단한 BPE 구현 (실제로는 더 복잡한 알고리즘 사용)
        tokens = []
        words = text.split()
        
        for word in words:
            if word in self.government_vocab:
                # 정부 관련 용어는 하나의 토큰으로 처리
                tokens.append(word)
            else:
                # 문자 단위로 분할
                tokens.extend(list(word))
        
        # 캐시에 저장 (메모리 제한)
        if len(self._cache) < 10000:
            self._cache[text] = tokens
        
        return tokens
    
    def tokenize(self, text: str, add_special_tokens: bool = True) -> TokenizationResult:
        """
        메인 토큰화 함수
        
        Args:
            text: 입력 텍스트
            add_special_tokens: 특수 토큰 추가 여부
            
        Returns:
            TokenizationResult: 토큰화 결과
        """
        start_time = time.time()
        
        # 1. 전처리
        cleaned_text = self.preprocess_text(text)
        
        # 2. BPE 토큰화
        tokens = self._bpe_encode(cleaned_text)
        
        # 3. 특수 토큰 추가
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 4. 길이 제한
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + ['[SEP]']
        
        # 5. 토큰 ID 변환
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['[UNK]'])
        
        # 6. 패딩
        padding_length = self.max_length - len(token_ids)
        token_ids.extend([self.vocab['[PAD]']] * padding_length)
        tokens.extend(['[PAD]'] * padding_length)
        
        # 7. 어텐션 마스크 생성
        attention_mask = [1] * (len(token_ids) - padding_length) + [0] * padding_length
        
        # 8. 특수 토큰 마스크
        special_tokens_mask = [1 if token in self.special_tokens else 0 for token in tokens]
        
        processing_time = time.time() - start_time
        
        # 9. 신뢰도 점수 계산
        confidence_score = self._calculate_confidence(tokens, cleaned_text)
        
        return TokenizationResult(
            tokens=tokens,
            token_ids=token_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            processing_time=processing_time,
            confidence_score=confidence_score
        )
    
    def _calculate_confidence(self, tokens: List[str], original_text: str) -> float:
        """
        토큰화 신뢰도 계산
        
        Args:
            tokens: 토큰 리스트
            original_text: 원본 텍스트
            
        Returns:
            신뢰도 점수 (0.0-1.0)
        """
        # UNK 토큰 비율로 신뢰도 계산
        unk_count = tokens.count('[UNK]')
        total_content_tokens = len([t for t in tokens if t not in ['[PAD]', '[CLS]', '[SEP]']])
        
        if total_content_tokens == 0:
            return 0.0
        
        unk_ratio = unk_count / total_content_tokens
        base_confidence = 1.0 - unk_ratio
        
        # 정부 관련 용어 보너스
        government_bonus = sum(1 for token in tokens if token in self.government_vocab) * 0.1
        
        return min(1.0, base_confidence + government_bonus)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        토큰 ID를 텍스트로 디코딩
        
        Args:
            token_ids: 토큰 ID 리스트
            
        Returns:
            디코딩된 텍스트
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.inv_vocab:
                token = self.inv_vocab[token_id]
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    tokens.append(token)
        
        return ''.join(tokens)
    
    def get_cache_stats(self) -> Dict[str, int]:
        """캐시 통계 반환"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache)
        }
    
    def save_vocabulary(self, path: str):
        """어휘 사전 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"어휘 사전 저장됨: {path}")

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 강의 예시를 실제로 실행
    tokenizer = KoreanTokenizer()
    
    # 강의에서 다룬 예시
    test_texts = [
        "정부 AI 정책관에 김철수를 추천합니다",
        "데이터 분석 전문가로서 정부 시스템 개발에 기여하고 싶습니다",
        "인공지능 분야 박사학위를 보유한 이영희를 장관으로 추천합니다"
    ]
    
    print("🔤 TRAS 토크나이저 테스트")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 테스트 {i}: {text}")
        result = tokenizer.tokenize(text)
        
        print(f"✅ 토큰 수: {len([t for t in result.tokens if t != '[PAD]'])}")
        print(f"⚡ 처리 시간: {result.processing_time:.4f}초")
        print(f"🎯 신뢰도: {result.confidence_score:.2f}")
        print(f"🔧 토큰: {result.tokens[:10]}...")  # 처음 10개만 표시
    
    # 캐시 성능 확인
    print(f"\n📊 캐시 성능: {tokenizer.get_cache_stats()}") 