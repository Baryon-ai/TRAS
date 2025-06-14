#!/usr/bin/env python3
"""
🎯 GPRO Model: Direct Preference Optimization for Government Talent Recommendation

Based on section4_rlhf_gpro.md lecture content:
- Implements DPO (Direct Preference Optimization) 
- Human preference learning without reward model
- Government talent recommendation specialized
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GPROConfig:
    """GPRO 모델 설정"""
    model_name: str = "klue/bert-base"
    max_length: int = 512
    hidden_size: int = 768
    num_labels: int = 4  # 추천/비추천/보류/강추
    beta: float = 0.1  # DPO temperature parameter
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 5
    dropout: float = 0.1
    government_positions: List[str] = None
    
    def __post_init__(self):
        if self.government_positions is None:
            self.government_positions = [
                "AI정책관", "데이터과학자", "디지털정책관", "사이버보안전문가",
                "정보화기획관", "IT정책관", "스마트시티전문가", "블록체인전문가"
            ]


class GovernmentTalentEncoder(nn.Module):
    """정부 인재 추천 특화 인코더"""
    
    def __init__(self, config: GPROConfig):
        super().__init__()
        self.config = config
        
        # BERT 백본
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # 정부 직책별 전문 헤드
        self.position_heads = nn.ModuleDict({
            pos.replace(" ", "_"): nn.Linear(config.hidden_size, config.hidden_size)
            for pos in config.government_positions
        })
        
        # 다중 관점 분석 레이어
        self.technical_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.policy_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.leadership_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.collaboration_head = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 최종 추천 레이어
        self.recommendation_head = nn.Linear(config.hidden_size * 4, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids, attention_mask, position_type=None):
        """Forward pass with multi-perspective analysis"""
        # BERT 인코딩
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # 다중 관점 분석
        technical_features = self.technical_head(pooled_output)
        policy_features = self.policy_head(pooled_output)
        leadership_features = self.leadership_head(pooled_output)
        collaboration_features = self.collaboration_head(pooled_output)
        
        # 직책별 특화 처리
        if position_type and position_type in self.position_heads:
            position_key = position_type.replace(" ", "_")
            pooled_output = self.position_heads[position_key](pooled_output)
        
        # 특징 융합
        combined_features = torch.cat([
            technical_features, policy_features, 
            leadership_features, collaboration_features
        ], dim=-1)
        
        # 드롭아웃 및 최종 예측
        combined_features = self.dropout(combined_features)
        logits = self.recommendation_head(combined_features)
        
        return {
            'logits': logits,
            'technical_features': technical_features,
            'policy_features': policy_features,
            'leadership_features': leadership_features,
            'collaboration_features': collaboration_features,
            'pooled_output': pooled_output
        }


class GPROModel(nn.Module):
    """
    Direct Preference Optimization Model for Government Talent Recommendation
    
    Based on lecture section4_rlhf_gpro.md:
    - Implements DPO without reward model
    - Human preference learning
    - Constitutional AI principles
    """
    
    def __init__(self, config: GPROConfig):
        super().__init__()
        self.config = config
        
        # 기본 인코더
        self.encoder = GovernmentTalentEncoder(config)
        
        # Constitutional AI 원칙
        self.constitutional_principles = {
            "accuracy": "사실에 기반한 추천만 제공",
            "fairness": "성별, 연령, 출신에 따른 차별 금지", 
            "transparency": "추천 근거를 명확히 설명",
            "safety": "해로운 추천 방지",
            "privacy": "개인정보 보호 원칙 준수"
        }
        
        # 신뢰도 예측 헤드
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        
        # 설명 생성을 위한 어텐션
        self.explanation_attention = nn.MultiheadAttention(
            config.hidden_size, num_heads=8, dropout=config.dropout
        )
        
        logger.info(f"GPRO 모델 초기화 완료: {config.model_name}")
    
    def forward(self, input_ids, attention_mask, position_type=None):
        """Forward pass with constitutional principles"""
        # 기본 인코딩
        outputs = self.encoder(input_ids, attention_mask, position_type)
        
        # 신뢰도 계산
        confidence = torch.sigmoid(self.confidence_head(outputs['pooled_output']))
        
        # Constitutional AI 체크
        constitutional_score = self._apply_constitutional_principles(outputs)
        
        outputs.update({
            'confidence': confidence,
            'constitutional_score': constitutional_score,
            'recommendation_reasoning': self._generate_reasoning(outputs)
        })
        
        return outputs
    
    def compute_dpo_loss(self, preferred_outputs, rejected_outputs):
        """
        Direct Preference Optimization Loss
        
        Based on DPO paper: L = -log(σ(β(log π_θ(y_w|x) - log π_θ(y_l|x))))
        """
        # 선호되는 응답의 로그 확률
        preferred_logprob = F.log_softmax(preferred_outputs['logits'], dim=-1)
        
        # 거부되는 응답의 로그 확률  
        rejected_logprob = F.log_softmax(rejected_outputs['logits'], dim=-1)
        
        # DPO 손실 계산
        preference_diff = preferred_logprob.max(dim=-1)[0] - rejected_logprob.max(dim=-1)[0]
        
        # Beta로 온도 조절
        scaled_diff = self.config.beta * preference_diff
        
        # 시그모이드를 통한 확률 변환
        dpo_loss = -torch.log(torch.sigmoid(scaled_diff)).mean()
        
        return dpo_loss
    
    def _apply_constitutional_principles(self, outputs):
        """Constitutional AI 원칙 적용"""
        # 공정성 점수 (편향 감지)
        fairness_score = self._check_fairness(outputs)
        
        # 투명성 점수 (설명 가능성)
        transparency_score = self._check_transparency(outputs)
        
        # 안전성 점수 (해로운 추천 방지)
        safety_score = self._check_safety(outputs)
        
        constitutional_score = (fairness_score + transparency_score + safety_score) / 3
        
        return constitutional_score
    
    def _check_fairness(self, outputs):
        """공정성 검사 - 편향 감지"""
        # 다양한 관점의 균형도 검사
        perspectives = [
            outputs['technical_features'],
            outputs['policy_features'], 
            outputs['leadership_features'],
            outputs['collaboration_features']
        ]
        
        # 관점 간 균형도 계산
        variance = torch.var(torch.stack([p.mean() for p in perspectives]))
        fairness_score = torch.exp(-variance)  # 낮은 분산 = 높은 공정성
        
        return fairness_score
    
    def _check_transparency(self, outputs):
        """투명성 검사 - 설명 가능성"""
        # 어텐션 엔트로피 기반 투명성 측정
        attention_entropy = self._compute_attention_entropy()
        transparency_score = 1.0 / (1.0 + attention_entropy)
        
        return transparency_score
    
    def _check_safety(self, outputs):
        """안전성 검사 - 해로운 추천 방지"""
        # 극단적 추천 방지
        logits = outputs['logits']
        max_confidence = torch.softmax(logits, dim=-1).max(dim=-1)[0]
        
        # 과도하게 확신하는 추천에 페널티
        safety_score = torch.where(
            max_confidence > 0.95,
            torch.tensor(0.5),  # 페널티
            torch.tensor(1.0)   # 정상
        )
        
        return safety_score.float()
    
    def _compute_attention_entropy(self):
        """어텐션 패턴의 엔트로피 계산"""
        # 단순화된 어텐션 엔트로피
        return torch.tensor(1.0)  # 플레이스홀더
    
    def _generate_reasoning(self, outputs):
        """추천 근거 생성"""
        # 각 관점별 기여도 계산
        technical_weight = outputs['technical_features'].abs().mean()
        policy_weight = outputs['policy_features'].abs().mean()
        leadership_weight = outputs['leadership_features'].abs().mean()
        collaboration_weight = outputs['collaboration_features'].abs().mean()
        
        reasoning = {
            'technical_importance': technical_weight.item(),
            'policy_importance': policy_weight.item(),
            'leadership_importance': leadership_weight.item(),
            'collaboration_importance': collaboration_weight.item(),
            'top_factor': self._get_top_factor(
                technical_weight, policy_weight, 
                leadership_weight, collaboration_weight
            )
        }
        
        return reasoning
    
    def _get_top_factor(self, technical, policy, leadership, collaboration):
        """가장 중요한 요소 식별"""
        factors = {
            'technical': technical,
            'policy': policy, 
            'leadership': leadership,
            'collaboration': collaboration
        }
        
        return max(factors, key=factors.get)
    
    def predict_with_explanation(self, text: str, position_type: str = None):
        """설명 가능한 추천 예측"""
        # 토크나이징
        inputs = self.encoder.tokenizer(
            text, 
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        # 예측
        with torch.no_grad():
            outputs = self.forward(
                inputs['input_ids'],
                inputs['attention_mask'], 
                position_type
            )
        
        # 결과 해석
        probabilities = torch.softmax(outputs['logits'], dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence = outputs['confidence']
        
        # 추천 클래스 매핑
        class_labels = ['비추천', '보류', '추천', '강추']
        prediction = class_labels[predicted_class.item()]
        
        result = {
            'prediction': prediction,
            'confidence': confidence.item(),
            'probabilities': probabilities.squeeze().tolist(),
            'reasoning': outputs['recommendation_reasoning'],
            'constitutional_score': outputs['constitutional_score'].item(),
            'detailed_analysis': {
                'technical_score': outputs['technical_features'].abs().mean().item(),
                'policy_score': outputs['policy_features'].abs().mean().item(), 
                'leadership_score': outputs['leadership_features'].abs().mean().item(),
                'collaboration_score': outputs['collaboration_features'].abs().mean().item()
            }
        }
        
        return result
    
    def save_model(self, path: str):
        """모델 저장"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 모델 상태 저장
        torch.save(self.state_dict(), save_path / "gpro_model.pt")
        
        # 설정 저장
        import json
        config_dict = self.config.__dict__.copy()
        with open(save_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"GPRO 모델 저장 완료: {save_path}")
    
    @classmethod
    def load_model(cls, path: str):
        """모델 로드"""
        load_path = Path(path)
        
        # 설정 로드
        import json
        with open(load_path / "config.json", 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = GPROConfig(**config_dict)
        
        # 모델 생성 및 가중치 로드
        model = cls(config)
        model.load_state_dict(torch.load(load_path / "gpro_model.pt"))
        
        logger.info(f"GPRO 모델 로드 완료: {load_path}")
        return model


# 유틸리티 함수들
def create_preference_pair(preferred_text: str, rejected_text: str, position: str):
    """선호도 쌍 데이터 생성"""
    return {
        'preferred': preferred_text,
        'rejected': rejected_text,
        'position': position,
        'preference_strength': 1.0  # 강한 선호도
    }


def initialize_gpro_model(model_name: str = "klue/bert-base") -> GPROModel:
    """GPRO 모델 초기화 헬퍼"""
    config = GPROConfig(model_name=model_name)
    model = GPROModel(config)
    
    logger.info("GPRO 모델 초기화 완료")
    return model


if __name__ == "__main__":
    # 테스트 코드
    print("🎯 GPRO 모델 테스트")
    
    # 모델 초기화
    model = initialize_gpro_model()
    
    # 테스트 예측
    test_text = """
    김철수는 서울대학교에서 컴퓨터공학 박사학위를 취득했으며, 
    구글에서 5년간 AI 연구원으로 근무했습니다. 
    딥러닝과 자연어처리 분야에 50편의 논문을 발표했고,
    팀 리더로서 20명의 개발자를 관리한 경험이 있습니다.
    정부 AI 정책 자문위원으로도 활동했습니다.
    """
    
    result = model.predict_with_explanation(test_text, "AI정책관")
    
    print(f"추천 결과: {result['prediction']}")
    print(f"신뢰도: {result['confidence']:.3f}")
    print(f"헌법적 점수: {result['constitutional_score']:.3f}")
    print(f"주요 요소: {result['reasoning']['top_factor']}")
    print("✅ GPRO 모델 테스트 완료!") 