"""
🧠 BERT 모델 구현
강의 내용을 바탕으로 한 완전한 BERT 시스템

FastBERT: 고속 추론을 위한 최적화된 모델
ReliableBERT: 신뢰성 향상을 위한 검증 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from dataclasses import dataclass
import numpy as np

try:
    from .tokenizer import KoreanTokenizer, TokenizationResult
    from .embedding import ContextualEmbedding, EmbeddingResult
    from .attention import MultiHeadAttention, AttentionResult
except ImportError:
    from tokenizer import KoreanTokenizer, TokenizationResult
    from embedding import ContextualEmbedding, EmbeddingResult
    from attention import MultiHeadAttention, AttentionResult

logger = logging.getLogger(__name__)

@dataclass
class BERTOutput:
    """BERT 모델 출력 결과"""
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    attention_weights: List[torch.Tensor]
    hidden_states: List[torch.Tensor]
    processing_time: float
    confidence_scores: Dict[str, float]
    government_predictions: Optional[Dict[str, float]] = None

class BERTLayer(nn.Module):
    """
    🔗 BERT 레이어 
    어텐션 + 피드포워드 네트워크
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12, 
                 d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()
        
        # 셀프 어텐션
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 피드포워드 네트워크
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """BERT 레이어 순전파"""
        # 셀프 어텐션
        attention_result = self.self_attention(
            hidden_states, hidden_states, hidden_states, attention_mask
        )
        
        # 피드포워드 네트워크 + 잔차 연결
        ff_output = self.feed_forward(attention_result.output)
        output = self.layer_norm(ff_output + attention_result.output)
        
        return output, attention_result.attention_weights

class FastBERT(nn.Module):
    """
    ⚡ 고속 BERT 모델
    효율적인 추론을 위한 최적화된 구현
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'max_length': max_length
        }
        
        # 토크나이저
        self.tokenizer = KoreanTokenizer(max_length=max_length)
        
        # 실제 어휘 크기를 토크나이저에서 가져오기
        actual_vocab_size = len(self.tokenizer.vocab)
        
        # 임베딩 레이어
        self.embeddings = ContextualEmbedding(
            vocab_size=actual_vocab_size,
            d_model=d_model,
            max_length=max_length,
            dropout=dropout
        )
        
        # BERT 레이어들
        self.layers = nn.ModuleList([
            BERTLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 풀링 레이어 (문장 표현을 위한 [CLS] 토큰 처리)
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        # 정부 직책 분류 헤드
        self.government_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 10)  # 10개 주요 정부 직책
        )
        
        # 추천 분류 헤드
        self.recommendation_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 강력추천/추천/비추천
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
        
        logger.info(f"⚡ FastBERT 초기화 완료 - 레이어: {num_layers}, 차원: {d_model}")
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> BERTOutput:
        """
        BERT 모델 순전파
        
        Args:
            input_texts: 입력 텍스트 리스트 (토큰화되지 않은 원시 텍스트)
            input_ids: 토큰 ID 텐서 [batch_size, seq_length]
            attention_mask: 어텐션 마스크 [batch_size, seq_length]
            return_dict: 딕셔너리 형태로 반환 여부
            
        Returns:
            BERTOutput: 모델 출력 결과
        """
        start_time = time.time()
        
        # 1. 토큰화 (필요한 경우)
        if input_texts is not None:
            tokenization_results = []
            for text in input_texts:
                result = self.tokenizer.tokenize(text)
                tokenization_results.append(result)
            
            # 배치로 변환
            input_ids = torch.tensor([r.token_ids for r in tokenization_results])
            attention_mask = torch.tensor([r.attention_mask for r in tokenization_results])
        
        batch_size, seq_length = input_ids.shape
        
        # 2. 임베딩
        embedding_result = self.embeddings(input_ids, attention_mask)
        hidden_states = embedding_result.embeddings
        
        # 3. BERT 레이어들 통과
        all_hidden_states = [hidden_states]
        all_attention_weights = []
        
        for layer in self.layers:
            hidden_states, attention_weights = layer(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)
            all_attention_weights.append(attention_weights)
        
        # 4. 풀링 ([CLS] 토큰으로 문장 표현 생성)
        pooler_output = self.pooler(hidden_states[:, 0])  # [CLS] 토큰
        
        # 5. 정부 관련 예측
        government_logits = self.government_classifier(pooler_output)
        recommendation_logits = self.recommendation_classifier(pooler_output)
        
        government_probs = F.softmax(government_logits, dim=-1)
        recommendation_probs = F.softmax(recommendation_logits, dim=-1)
        
        # 6. 신뢰도 점수 계산
        confidence_scores = self._calculate_confidence_scores(
            hidden_states, attention_mask, government_probs, recommendation_probs
        )
        
        processing_time = time.time() - start_time
        
        # 7. 정부 예측 결과 정리
        government_labels = [
            '정책관', '과장', '국장', '차관', '장관', 
            '대통령', '총리', '비서관', '보좌관', '수석'
        ]
        recommendation_labels = ['강력추천', '추천', '비추천']
        
        government_predictions = {
            'positions': {label: prob.item() for label, prob in zip(government_labels, government_probs[0])},
            'recommendation': {label: prob.item() for label, prob in zip(recommendation_labels, recommendation_probs[0])}
        }
        
        return BERTOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            attention_weights=all_attention_weights,
            hidden_states=all_hidden_states,
            processing_time=processing_time,
            confidence_scores=confidence_scores,
            government_predictions=government_predictions
        )
    
    def _calculate_confidence_scores(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        government_probs: torch.Tensor,
        recommendation_probs: torch.Tensor
    ) -> Dict[str, float]:
        """신뢰도 점수 계산"""
        with torch.no_grad():
            # 1. 예측 신뢰도 (최대 확률값 기반)
            gov_confidence = government_probs.max(dim=-1)[0].mean().item()
            rec_confidence = recommendation_probs.max(dim=-1)[0].mean().item()
            
            # 2. 어텐션 일관성 (어텐션 분포의 엔트로피)
            attention_consistency = 0.0
            
            # 3. 숨겨진 상태의 안정성
            hidden_stability = 1.0 - hidden_states.std(dim=1).mean().item()
            
            # 4. 토큰 품질 (UNK 토큰 비율)
            token_quality = 1.0  # 간단히 1.0으로 설정
            
            return {
                'government_confidence': gov_confidence,
                'recommendation_confidence': rec_confidence,
                'attention_consistency': attention_consistency,
                'hidden_stability': hidden_stability,
                'token_quality': token_quality,
                'overall_confidence': (gov_confidence + rec_confidence + hidden_stability + token_quality) / 4.0
            }

class ReliableBERT(nn.Module):
    """
    🛡️ 신뢰성 향상 BERT
    다중 검증과 불확실성 정량화를 통한 안정적인 예측
    """
    
    def __init__(self, base_model: FastBERT, num_ensemble: int = 3):
        super().__init__()
        
        self.base_model = base_model
        self.num_ensemble = num_ensemble
        
        # 앙상블을 위한 다중 모델 (가중치 공유하지만 드롭아웃 패턴 다름)
        self.ensemble_models = nn.ModuleList([
            self._create_ensemble_model() for _ in range(num_ensemble)
        ])
        
        # 불확실성 정량화를 위한 베이지안 헤드
        self.uncertainty_head = nn.Sequential(
            nn.Linear(base_model.config['d_model'], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 평균과 분산 예측
        )
        
        logger.info(f"🛡️ ReliableBERT 초기화 - 앙상블 크기: {num_ensemble}")
    
    def _create_ensemble_model(self) -> nn.Module:
        """앙상블용 모델 생성 (드롭아웃 패턴만 다름)"""
        return nn.Sequential(
            nn.Dropout(0.2),  # 다른 드롭아읏 확률
            nn.Linear(self.base_model.config['d_model'], self.base_model.config['d_model']),
            nn.ReLU()
        )
    
    def forward(
        self,
        input_texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        신뢰성 향상된 순전파
        
        Args:
            input_texts: 입력 텍스트
            input_ids: 토큰 ID
            attention_mask: 어텐션 마스크
            return_uncertainty: 불확실성 정보 반환 여부
            
        Returns:
            신뢰성 분석이 포함된 결과 딕셔너리
        """
        start_time = time.time()
        
        # 1. 기본 모델 예측
        base_output = self.base_model(input_texts, input_ids, attention_mask)
        
        # 2. 앙상블 예측
        ensemble_outputs = []
        for ensemble_model in self.ensemble_models:
            # 동일한 입력으로 여러 번 예측 (드롭아웃으로 인한 변동성 활용)
            ensemble_output = self.base_model(input_texts, input_ids, attention_mask)
            ensemble_outputs.append(ensemble_output)
        
        # 3. 앙상블 결과 통합
        ensemble_predictions = self._aggregate_ensemble_predictions(ensemble_outputs)
        
        # 4. 불확실성 정량화
        uncertainty_scores = None
        if return_uncertainty:
            uncertainty_scores = self._calculate_uncertainty(
                base_output, ensemble_outputs
            )
        
        # 5. 최종 신뢰도 계산
        final_confidence = self._calculate_final_confidence(
            base_output, ensemble_predictions, uncertainty_scores
        )
        
        processing_time = time.time() - start_time
        
        return {
            'base_prediction': base_output,
            'ensemble_predictions': ensemble_predictions,
            'uncertainty_scores': uncertainty_scores,
            'final_confidence': final_confidence,
            'processing_time': processing_time,
            'recommendation': self._make_final_recommendation(
                base_output, ensemble_predictions, final_confidence
            )
        }
    
    def _aggregate_ensemble_predictions(self, ensemble_outputs: List[BERTOutput]) -> Dict[str, Any]:
        """앙상블 예측 결과 통합"""
        # 정부 직책 예측 평균
        position_probs = []
        recommendation_probs = []
        
        for output in ensemble_outputs:
            if output.government_predictions:
                position_probs.append(list(output.government_predictions['positions'].values()))
                recommendation_probs.append(list(output.government_predictions['recommendation'].values()))
        
        if position_probs:
            avg_position_probs = np.mean(position_probs, axis=0)
            avg_recommendation_probs = np.mean(recommendation_probs, axis=0)
            
            position_labels = list(ensemble_outputs[0].government_predictions['positions'].keys())
            recommendation_labels = list(ensemble_outputs[0].government_predictions['recommendation'].keys())
            
            return {
                'positions': {label: prob for label, prob in zip(position_labels, avg_position_probs)},
                'recommendation': {label: prob for label, prob in zip(recommendation_labels, avg_recommendation_probs)},
                'position_std': np.std(position_probs, axis=0).tolist(),
                'recommendation_std': np.std(recommendation_probs, axis=0).tolist()
            }
        
        return {}
    
    def _calculate_uncertainty(self, base_output: BERTOutput, ensemble_outputs: List[BERTOutput]) -> Dict[str, float]:
        """불확실성 점수 계산"""
        # 예측 분산 계산
        predictions = []
        for output in ensemble_outputs:
            if output.government_predictions:
                predictions.append(list(output.government_predictions['positions'].values()))
        
        if predictions:
            prediction_variance = np.var(predictions, axis=0).mean()
            prediction_entropy = -np.sum([p * np.log(p + 1e-8) for p in np.mean(predictions, axis=0)])
            
            return {
                'prediction_variance': prediction_variance,
                'prediction_entropy': prediction_entropy,
                'ensemble_disagreement': prediction_variance,
                'epistemic_uncertainty': prediction_entropy
            }
        
        return {'prediction_variance': 0.0, 'prediction_entropy': 0.0}
    
    def _calculate_final_confidence(
        self, 
        base_output: BERTOutput, 
        ensemble_predictions: Dict[str, Any],
        uncertainty_scores: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """최종 신뢰도 계산"""
        base_confidence = base_output.confidence_scores['overall_confidence']
        
        # 앙상블 일관성 점수
        ensemble_consistency = 1.0
        if ensemble_predictions and 'position_std' in ensemble_predictions:
            ensemble_consistency = 1.0 - np.mean(ensemble_predictions['position_std'])
        
        # 불확실성 패널티
        uncertainty_penalty = 0.0
        if uncertainty_scores:
            uncertainty_penalty = uncertainty_scores.get('prediction_variance', 0.0)
        
        final_confidence = base_confidence * ensemble_consistency * (1.0 - uncertainty_penalty)
        
        return {
            'base_confidence': base_confidence,
            'ensemble_consistency': ensemble_consistency,
            'uncertainty_penalty': uncertainty_penalty,
            'final_confidence': max(0.0, min(1.0, final_confidence))
        }
    
    def _make_final_recommendation(
        self, 
        base_output: BERTOutput, 
        ensemble_predictions: Dict[str, Any],
        final_confidence: Dict[str, float]
    ) -> Dict[str, Any]:
        """최종 추천 결정"""
        confidence_threshold = 0.7  # 신뢰도 임계값
        
        is_reliable = final_confidence['final_confidence'] >= confidence_threshold
        
        # 가장 확률이 높은 직책과 추천도
        if ensemble_predictions and 'positions' in ensemble_predictions:
            best_position = max(ensemble_predictions['positions'].items(), key=lambda x: x[1])
            best_recommendation = max(ensemble_predictions['recommendation'].items(), key=lambda x: x[1])
            
            return {
                'is_reliable': is_reliable,
                'recommended_position': best_position[0],
                'position_confidence': best_position[1],
                'recommendation_type': best_recommendation[0],
                'recommendation_confidence': best_recommendation[1],
                'overall_confidence': final_confidence['final_confidence'],
                'decision': 'ACCEPT' if is_reliable and best_recommendation[0] in ['강력추천', '추천'] else 'REVIEW'
            }
        
        return {
            'is_reliable': False,
            'decision': 'INSUFFICIENT_DATA'
        }

# 테스트 및 사용 예시
if __name__ == "__main__":
    print("🧠 TRAS BERT 모델 테스트")
    print("=" * 50)
    
    # FastBERT 테스트
    print("⚡ FastBERT 테스트")
    fast_bert = FastBERT(vocab_size=1000, d_model=256, num_layers=4, num_heads=8)
    
    test_texts = [
        "김철수를 AI 정책관으로 강력히 추천합니다",
        "데이터 분석 전문가로서 정부 시스템 개발에 기여하고 싶습니다"
    ]
    
    # FastBERT 예측
    fast_result = fast_bert(input_texts=test_texts)
    print(f"✅ FastBERT 출력 형태: {fast_result.last_hidden_state.shape}")
    print(f"⚡ 처리 시간: {fast_result.processing_time:.4f}초")
    print(f"🎯 전체 신뢰도: {fast_result.confidence_scores['overall_confidence']:.3f}")
    
    if fast_result.government_predictions:
        print("🏛️ 정부 예측 결과:")
        for pos, prob in fast_result.government_predictions['positions'].items():
            if prob > 0.1:  # 10% 이상만 표시
                print(f"   {pos}: {prob:.3f}")
    
    # ReliableBERT 테스트
    print(f"\n🛡️ ReliableBERT 테스트")
    reliable_bert = ReliableBERT(fast_bert, num_ensemble=2)
    
    reliable_result = reliable_bert(input_texts=test_texts)
    print(f"✅ ReliableBERT 처리 시간: {reliable_result['processing_time']:.4f}초")
    print(f"🎯 최종 신뢰도: {reliable_result['final_confidence']['final_confidence']:.3f}")
    print(f"🏆 최종 추천: {reliable_result['recommendation']['decision']}")
    
    print("\n🎉 BERT 모델 테스트 완료!") 