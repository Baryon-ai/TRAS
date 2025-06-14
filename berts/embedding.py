"""
📊 문맥적 임베딩 모듈
강의 Section 2에서 다룬 "의미의 지도" 개념의 실제 구현

Word2Vec부터 문맥적 임베딩까지, 기하학적 직관을 코드로 구현
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math
import time
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """임베딩 결과 데이터 클래스"""
    embeddings: torch.Tensor
    attention_weights: Optional[torch.Tensor]
    similarity_scores: Optional[Dict[str, float]]
    processing_time: float
    geometric_properties: Dict[str, float]

class PositionalEncoding(nn.Module):
    """
    🌊 위치 인코딩
    강의에서 다룬 "단어의 순서" 정보를 기하학적으로 표현
    """
    
    def __init__(self, d_model: int, max_length: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # 위치 인코딩 매트릭스 생성
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        # 기하학적 직관: 사인/코사인 함수로 순환하는 패턴 생성
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """위치 정보를 임베딩에 추가"""
        return x + self.pe[:, :x.size(1)]

class ContextualEmbedding(nn.Module):
    """
    🎭 문맥적 임베딩 모듈
    강의에서 다룬 "카멜레온 단어" 개념의 실제 구현
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 768, 
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 기본 토큰 임베딩 (강의의 "의미의 지도" 기본 좌표)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm (수치 안정성)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 정부 관련 도메인 가중치 (도메인 특화)
        self.domain_weight = nn.Linear(d_model, d_model)
        
        # 임베딩 초기화 (Xavier/Glorot 초기화)
        self._init_embeddings()
        
        logger.info(f"📊 ContextualEmbedding 초기화 완료 - 차원: {d_model}")
    
    def _init_embeddings(self):
        """임베딩 가중치 초기화 (기하학적 최적화)"""
        # Xavier 초기화로 벡터들이 구 표면에 고르게 분포
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.domain_weight.weight)
        
        # 패딩 토큰은 영벡터로 초기화
        with torch.no_grad():
            self.token_embedding.weight[0].fill_(0)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> EmbeddingResult:
        """
        순전파 (문맥적 임베딩 생성)
        
        Args:
            input_ids: 토큰 ID 텐서 [batch_size, seq_length]
            attention_mask: 어텐션 마스크 [batch_size, seq_length]
            
        Returns:
            EmbeddingResult: 임베딩 결과
        """
        start_time = time.time()
        batch_size, seq_length = input_ids.shape
        
        # 1. 기본 토큰 임베딩
        token_embeds = self.token_embedding(input_ids)  # [B, L, D]
        
        # 2. 위치 인코딩 추가
        embeddings = self.positional_encoding(token_embeds)
        
        # 3. 정규화 및 드롭아웃
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 4. 도메인 특화 가중치 적용
        domain_weighted = self.domain_weight(embeddings)
        embeddings = embeddings + 0.1 * domain_weighted  # 잔차 연결
        
        # 5. 어텐션 마스크 적용
        if attention_mask is not None:
            # 패딩 토큰의 임베딩을 0으로 마스킹
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings)
            embeddings = embeddings * mask_expanded.float()
        
        processing_time = time.time() - start_time
        
        # 6. 기하학적 속성 계산
        geometric_props = self._calculate_geometric_properties(embeddings)
        
        return EmbeddingResult(
            embeddings=embeddings,
            attention_weights=None,  # 나중에 어텐션 모듈에서 계산
            similarity_scores=None,
            processing_time=processing_time,
            geometric_properties=geometric_props
        )
    
    def _calculate_geometric_properties(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """
        임베딩의 기하학적 속성 계산
        강의에서 다룬 벡터 공간의 수학적 특성
        """
        with torch.no_grad():
            # 벡터의 크기 (norm) 통계
            norms = torch.norm(embeddings, dim=-1)
            
            # 코사인 유사도 분포 (벡터 간 각도)
            flat_embeds = embeddings.view(-1, embeddings.size(-1))
            sample_size = min(1000, flat_embeds.size(0))
            sample_indices = torch.randperm(flat_embeds.size(0))[:sample_size]
            sample_embeds = flat_embeds[sample_indices]
            
            # 평균 코사인 유사도 계산
            if sample_size > 1:
                similarities = torch.mm(
                    sample_embeds, 
                    sample_embeds.transpose(0, 1)
                ) / (torch.norm(sample_embeds, dim=1, keepdim=True) * 
                     torch.norm(sample_embeds, dim=1).unsqueeze(0))
                
                # 자기 자신과의 유사도 제외
                mask = torch.eye(sample_size, dtype=torch.bool)
                similarities = similarities[~mask]
                avg_similarity = similarities.mean().item()
            else:
                avg_similarity = 0.0
            
            return {
                'avg_norm': norms.mean().item(),
                'std_norm': norms.std().item(),
                'max_norm': norms.max().item(),
                'min_norm': norms.min().item(),
                'avg_cosine_similarity': avg_similarity,
                'embedding_dimension': embeddings.size(-1)
            }
    
    def get_word_embedding(self, word_id: int) -> torch.Tensor:
        """단일 단어의 임베딩 벡터 반환"""
        return self.token_embedding.weight[word_id]
    
    def similarity_search(
        self, 
        query_embedding: torch.Tensor, 
        candidate_embeddings: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        유사도 기반 검색 (강의의 "의미의 지도"에서 가까운 점 찾기)
        
        Args:
            query_embedding: 쿼리 임베딩
            candidate_embeddings: 후보 임베딩들
            top_k: 반환할 상위 k개
            
        Returns:
            (인덱스, 유사도) 튜플 리스트
        """
        # 코사인 유사도 계산 (기하학적 각도 기반)
        query_norm = query_embedding / torch.norm(query_embedding)
        candidate_norms = candidate_embeddings / torch.norm(
            candidate_embeddings, dim=-1, keepdim=True
        )
        
        similarities = torch.matmul(query_norm, candidate_norms.transpose(-2, -1))
        
        # 상위 k개 선택
        top_values, top_indices = torch.topk(similarities, k=top_k)
        
        return [(idx.item(), score.item()) for idx, score in zip(top_indices, top_values)]

class GeometricAnalyzer:
    """
    📐 기하학적 분석기
    강의에서 다룬 벡터 공간의 기하학적 성질 분석
    """
    
    @staticmethod
    def vector_arithmetic(
        embeddings: Dict[str, torch.Tensor], 
        operation: str
    ) -> torch.Tensor:
        """
        벡터 연산 (king - man + woman = queen 스타일)
        
        Args:
            embeddings: 단어별 임베딩 딕셔너리
            operation: 연산 문자열 (예: "김철수 - 남성 + 정책관")
            
        Returns:
            연산 결과 벡터
        """
        # 간단한 파서 구현 (실제로는 더 정교한 파싱 필요)
        tokens = operation.replace('+', ' + ').replace('-', ' - ').split()
        
        result = None
        current_op = '+'
        
        for token in tokens:
            if token in ['+', '-']:
                current_op = token
            elif token in embeddings:
                vector = embeddings[token]
                if result is None:
                    result = vector.clone()
                elif current_op == '+':
                    result += vector
                elif current_op == '-':
                    result -= vector
        
        return result if result is not None else torch.zeros_like(next(iter(embeddings.values())))
    
    @staticmethod
    def analyze_cluster(embeddings: torch.Tensor, labels: List[str]) -> Dict[str, any]:
        """
        클러스터 분석 (의미적으로 유사한 단어들의 기하학적 배치)
        
        Args:
            embeddings: 임베딩 텐서 [num_words, embedding_dim]
            labels: 단어 라벨 리스트
            
        Returns:
            클러스터 분석 결과
        """
        # 차원 축소를 위한 PCA (시각화용)
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        
        embeddings_np = embeddings.detach().numpy()
        
        # PCA로 2D 투영
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_np)
        
        # K-means 클러스터링
        n_clusters = min(5, len(labels))
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        # 클러스터별 단어 그룹화
        clusters = {}
        for i, (word, cluster_id) in enumerate(zip(labels, cluster_labels)):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(word)
        
        return {
            'clusters': clusters,
            'embeddings_2d': embeddings_2d,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    @staticmethod
    def semantic_direction(
        embeddings: Dict[str, torch.Tensor], 
        positive_examples: List[str], 
        negative_examples: List[str]
    ) -> torch.Tensor:
        """
        의미적 방향 벡터 계산
        예: 성별 방향 = (남성 단어들 평균) - (여성 단어들 평균)
        
        Args:
            embeddings: 단어별 임베딩 딕셔너리
            positive_examples: 양의 예시 단어들
            negative_examples: 음의 예시 단어들
            
        Returns:
            의미적 방향 벡터
        """
        pos_vectors = torch.stack([embeddings[word] for word in positive_examples if word in embeddings])
        neg_vectors = torch.stack([embeddings[word] for word in negative_examples if word in embeddings])
        
        pos_mean = pos_vectors.mean(dim=0)
        neg_mean = neg_vectors.mean(dim=0)
        
        return pos_mean - neg_mean

# 실제 테스트 및 사용 예시
if __name__ == "__main__":
    print("📊 TRAS 임베딩 모듈 테스트")
    print("=" * 50)
    
    # 테스트 설정
    vocab_size = 1000
    d_model = 256
    batch_size = 2
    seq_length = 10
    
    # 모델 생성
    embedding_model = ContextualEmbedding(vocab_size, d_model)
    
    # 테스트 데이터
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    print(f"📝 입력 형태: {input_ids.shape}")
    
    # 임베딩 생성
    result = embedding_model(input_ids, attention_mask)
    
    print(f"✅ 출력 형태: {result.embeddings.shape}")
    print(f"⚡ 처리 시간: {result.processing_time:.4f}초")
    print(f"📐 기하학적 속성:")
    for key, value in result.geometric_properties.items():
        print(f"   {key}: {value:.4f}")
    
    # 유사도 검색 테스트  
    query = result.embeddings[0, 0]  # 첫 번째 배치의 첫 번째 토큰
    candidates = result.embeddings[0, 1:]  # 나머지 토큰들
    
    similar_indices = embedding_model.similarity_search(query, candidates, top_k=3)
    print(f"🔍 유사도 검색 결과: {similar_indices}")
    
    print("\n�� 임베딩 모듈 테스트 완료!") 