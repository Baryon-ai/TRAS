"""
👁️ 어텐션 메커니즘
강의 Section 2에서 다룬 "AI의 집중력 모델링" 실제 구현

조명 시스템과 단어들의 대화를 수학적으로 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List
import time
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AttentionResult:
    """어텐션 결과 데이터 클래스"""
    output: torch.Tensor
    attention_weights: torch.Tensor
    processing_time: float
    attention_stats: Dict[str, float]
    head_importance: Optional[torch.Tensor] = None

class ScaledDotProductAttention(nn.Module):
    """
    ⚡ 스케일드 닷 프로덕트 어텐션
    강의에서 다룬 기본 어텐션 메커니즘의 수학적 구현
    """
    
    def __init__(self, temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        어텐션 계산 (강의의 "조명 시스템" 구현)
        
        Args:
            query: 쿼리 텐서 [B, L_q, D] - "무엇을 찾고 있는가?"
            key: 키 텐서 [B, L_k, D] - "각 위치에 무엇이 있는가?"
            value: 값 텐서 [B, L_v, D] - "실제 정보는 무엇인가?"
            mask: 마스크 텐서 [B, L_q, L_k] - "어디를 보면 안 되는가?"
            
        Returns:
            output: 어텐션 적용된 출력 [B, L_q, D]
            attention: 어텐션 가중치 [B, L_q, L_k]
        """
        batch_size, len_q, d_k = query.size()
        len_k = key.size(1)
        
        # 1. 유사도 계산 (Q · K^T) - 강의의 "손전등으로 방 비추기"
        scores = torch.matmul(query, key.transpose(-2, -1))  # [B, L_q, L_k]
        
        # 2. 스케일링 (수치 안정성을 위해 √d_k로 나누기)
        scores = scores / math.sqrt(d_k)
        
        # 3. 마스킹 적용 (패딩이나 미래 토큰 가리기)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. 소프트맥스로 확률 분포 변환 - "조명의 밝기 분배"
        attention_weights = F.softmax(scores, dim=-1)  # [B, L_q, L_k]
        attention_weights = self.dropout(attention_weights)
        
        # 5. 가중 평균으로 최종 출력 - "밝게 비춰진 보물들 수집"
        output = torch.matmul(attention_weights, value)  # [B, L_q, D]
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    🎭 멀티헤드 어텐션
    강의에서 다룬 "여러 전문가 패널" 개념의 실제 구현
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 선형 변환 레이어들
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> AttentionResult:
        """멀티헤드 어텐션 계산"""
        start_time = time.time()
        
        batch_size, seq_len = query.size(0), query.size(1)
        residual = query
        
        # 1. 선형 변환 및 헤드 분할
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 어텐션 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                mask = mask.expand(batch_size, 1, seq_len, seq_len)  # [batch_size, 1, seq_len, seq_len]
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            # 모든 헤드로 확장
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 3. 가중 평균 계산
        output = torch.matmul(attention_weights, V)
        
        # 4. 헤드 결합
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        output = self.dropout(output)
        
        # 5. 잔차 연결 및 정규화
        output = self.layer_norm(output + residual)
        
        processing_time = time.time() - start_time
        
        # 7. 통계 계산
        attention_stats = self._calculate_attention_stats(attention_weights)
        
        # 8. 헤드 중요도 계산
        head_importance = self._calculate_head_importance(attention_weights)
        
        return AttentionResult(
            output=output,
            attention_weights=attention_weights,
            processing_time=processing_time,
            attention_stats=attention_stats,
            head_importance=head_importance
        )
    
    def _calculate_attention_stats(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        어텐션 통계 계산 (강의의 "집중도 분석")
        
        Args:
            attention_weights: [B, H, L, L]
            
        Returns:
            통계 딕셔너리
        """
        with torch.no_grad():
            # 엔트로피 계산 (집중도 측정)
            entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8), 
                dim=-1
            ).mean()
            
            # 최대 어텐션 값 (가장 강한 집중)
            max_attention = attention_weights.max()
            
            # 평균 어텐션 값
            mean_attention = attention_weights.mean()
            
            # 대각선 어텐션 (자기 자신에 대한 집중)
            diagonal_attention = torch.diagonal(attention_weights, dim1=-2, dim2=-1).mean()
            
            # 어텐션 분산 (집중의 일관성)
            attention_variance = attention_weights.var()
            
            return {
                'entropy': entropy.item(),
                'max_attention': max_attention.item(),
                'mean_attention': mean_attention.item(),
                'diagonal_attention': diagonal_attention.item(),
                'attention_variance': attention_variance.item()
            }
    
    def _calculate_head_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        각 어텐션 헤드의 중요도 계산
        
        Args:
            attention_weights: [B, H, L, L]
            
        Returns:
            헤드별 중요도 점수 [H]
        """
        with torch.no_grad():
            # 각 헤드의 어텐션 분포 다양성으로 중요도 측정
            # 더 다양한 패턴을 보이는 헤드가 더 중요
            head_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8),
                dim=(-2, -1)
            )  # [B, H]
            
            # 배치 전체 평균
            head_importance = head_entropy.mean(dim=0)  # [H]
            
            return head_importance

class SelfAttention(nn.Module):
    """
    🗣️ 셀프 어텐션
    강의에서 다룬 "단어들 간의 대화" 구현
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> AttentionResult:
        """
        셀프 어텐션 계산 (Query, Key, Value가 모두 같은 입력)
        
        Args:
            hidden_states: 입력 히든 상태 [B, L, D]
            attention_mask: 어텐션 마스크 [B, L]
            
        Returns:
            AttentionResult: 어텐션 결과
        """
        # 어텐션 마스크를 4D로 확장
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.size()
            # [B, L] -> [B, L, L] (각 쿼리 위치에서 키 위치로의 마스크)
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        
        # 셀프 어텐션: Q, K, V 모두 같은 입력 사용
        return self.multi_head_attention(
            query=hidden_states,
            key=hidden_states, 
            value=hidden_states,
            mask=attention_mask
        )

class GovernmentPositionAttention(nn.Module):
    """
    🏛️ 정부 직책 특화 어텐션
    TRAS 시스템을 위한 도메인 특화 어텐션 메커니즘
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12):
        super().__init__()
        self.self_attention = SelfAttention(d_model, num_heads)
        
        # 정부 관련 키워드 가중치
        self.position_keywords = {
            '정책관': 1.5, '과장': 1.3, '국장': 1.4, '차관': 1.6, '장관': 1.7,
            '대통령': 2.0, '총리': 1.8, '비서관': 1.2, '보좌관': 1.1, '수석': 1.3,
            'AI': 1.4, '인공지능': 1.4, '데이터': 1.2, '분석': 1.1, '추천': 1.3
        }
        
        # 직책 추출을 위한 분류 헤드
        self.position_classifier = nn.Linear(d_model, len(self.position_keywords))
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_labels: Optional[List[List[str]]] = None
    ) -> Tuple[AttentionResult, torch.Tensor]:
        """
        정부 직책 특화 어텐션 계산
        
        Args:
            hidden_states: 입력 히든 상태 [B, L, D]
            attention_mask: 어텐션 마스크 [B, L]
            token_labels: 토큰별 라벨 (정부 키워드 식별용)
            
        Returns:
            attention_result: 어텐션 결과
            position_scores: 직책 관련도 점수 [B, L, num_positions]
        """
        # 1. 기본 셀프 어텐션
        attention_result = self.self_attention(hidden_states, attention_mask)
        
        # 2. 정부 직책 관련도 계산
        position_scores = self.position_classifier(attention_result.output)
        
        # 3. 도메인 특화 가중치 적용
        if token_labels is not None:
            domain_weights = self._calculate_domain_weights(token_labels)
            # 가중치를 어텐션에 반영
            weighted_attention = attention_result.attention_weights * domain_weights
            attention_result.attention_weights = weighted_attention
        
        return attention_result, position_scores
    
    def _calculate_domain_weights(self, token_labels: List[List[str]]) -> torch.Tensor:
        """도메인 특화 가중치 계산"""
        batch_size = len(token_labels)
        max_seq_len = max(len(labels) for labels in token_labels)
        
        weights = torch.ones(batch_size, max_seq_len, max_seq_len)
        
        for batch_idx, labels in enumerate(token_labels):
            for i, label_i in enumerate(labels):
                for j, label_j in enumerate(labels):
                    # 정부 키워드 간의 가중치 부여
                    weight_i = self.position_keywords.get(label_i, 1.0)
                    weight_j = self.position_keywords.get(label_j, 1.0)
                    weights[batch_idx, i, j] = math.sqrt(weight_i * weight_j)
        
        return weights

class AttentionVisualizer:
    """
    👀 어텐션 시각화 도구
    강의에서 다룬 어텐션 패턴을 시각적으로 분석
    """
    
    @staticmethod
    def analyze_attention_pattern(
        attention_weights: torch.Tensor, 
        tokens: List[str]
    ) -> Dict[str, any]:
        """
        어텐션 패턴 분석
        
        Args:
            attention_weights: [H, L, L] (단일 샘플)
            tokens: 토큰 리스트
            
        Returns:
            분석 결과 딕셔너리
        """
        num_heads, seq_len, _ = attention_weights.shape
        
        # 1. 헤드별 어텐션 패턴 분석
        head_patterns = {}
        for head_idx in range(num_heads):
            head_attention = attention_weights[head_idx].numpy()
            
            # 대각선 집중도 (자기 참조)
            diagonal_focus = np.diag(head_attention).mean()
            
            # 장거리 의존성 (먼 토큰 간 어텐션)
            long_range_dep = 0
            for i in range(seq_len):
                for j in range(seq_len):
                    if abs(i - j) > seq_len // 2:
                        long_range_dep += head_attention[i, j]
            long_range_dep /= (seq_len * seq_len / 4)
            
            head_patterns[f'head_{head_idx}'] = {
                'diagonal_focus': diagonal_focus,
                'long_range_dependency': long_range_dep,
                'max_attention': head_attention.max(),
                'entropy': -np.sum(head_attention * np.log(head_attention + 1e-8))
            }
        
        # 2. 토큰별 중요도 계산
        token_importance = attention_weights.mean(dim=0).sum(dim=0).numpy()
        
        # 3. 어텐션 그래프 구성 (강한 연결 관계)
        attention_graph = {}
        threshold = 0.1  # 어텐션 임계값
        
        for i, token_i in enumerate(tokens):
            connections = []
            for j, token_j in enumerate(tokens):
                avg_attention = attention_weights[:, i, j].mean().item()
                if avg_attention > threshold and i != j:
                    connections.append((token_j, avg_attention))
            
            # 상위 3개 연결만 유지
            connections.sort(key=lambda x: x[1], reverse=True)
            attention_graph[token_i] = connections[:3]
        
        return {
            'head_patterns': head_patterns,
            'token_importance': list(zip(tokens, token_importance)),
            'attention_graph': attention_graph,
            'global_stats': {
                'avg_entropy': np.mean([p['entropy'] for p in head_patterns.values()]),
                'avg_diagonal_focus': np.mean([p['diagonal_focus'] for p in head_patterns.values()]),
                'most_important_token': tokens[np.argmax(token_importance)]
            }
        }

# 테스트 및 사용 예시
if __name__ == "__main__":
    print("👁️ TRAS 어텐션 모듈 테스트")
    print("=" * 50)
    
    # 테스트 설정
    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 8
    
    # 모델 생성
    attention_model = MultiHeadAttention(d_model, num_heads)
    
    # 테스트 데이터 생성
    test_input = torch.randn(batch_size, seq_len, d_model)
    test_mask = torch.ones(batch_size, seq_len, seq_len)
    
    print(f"📝 입력 형태: {test_input.shape}")
    
    # 어텐션 계산
    result = attention_model(test_input, test_input, test_input, test_mask)
    
    print(f"✅ 출력 형태: {result.output.shape}")
    print(f"👁️ 어텐션 가중치 형태: {result.attention_weights.shape}")
    print(f"⚡ 처리 시간: {result.processing_time:.4f}초")
    
    print(f"\n📊 어텐션 통계:")
    for key, value in result.attention_stats.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\n🎭 헤드별 중요도:")
    for i, importance in enumerate(result.head_importance):
        print(f"   Head {i}: {importance:.4f}")
    
    # 정부 특화 어텐션 테스트
    print(f"\n🏛️ 정부 특화 어텐션 테스트")
    gov_attention = GovernmentPositionAttention(d_model, num_heads)
    
    gov_result, position_scores = gov_attention(test_input)
    print(f"✅ 정부 어텐션 출력: {gov_result.output.shape}")
    print(f"🎯 직책 점수 형태: {position_scores.shape}")
    
    print("\n�� 어텐션 모듈 테스트 완료!") 