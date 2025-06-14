#!/usr/bin/env python3
"""
👥 Human Feedback Simulator: Government Expert Feedback Simulation

Simulates realistic human expert feedback patterns for GPRO training.
"""

import random
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeedbackData:
    """인간 피드백 데이터 클래스"""
    expert_id: str
    expert_type: str
    candidate_info: str
    position: str
    preferred_response: str
    rejected_response: str
    preference_strength: float
    reasoning: str
    confidence: float
    timestamp: str
    bias_factors: Dict[str, float]


class HumanFeedbackSimulator:
    """인간 전문가 피드백 시뮬레이터"""
    
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 전문가 프로필들
        self.expert_profiles = {
            "hr_specialist_1": {
                "name": "박인사",
                "expertise": "인사관리",
                "experience_years": 15,
                "preferences": {
                    "fairness_weight": 0.9,
                    "process_weight": 0.8,
                    "diversity_weight": 0.7,
                    "consistency_weight": 0.85
                },
                "bias_tendencies": {
                    "experience_bias": 0.2,
                    "education_bias": 0.15,
                    "stability_bias": 0.25
                }
            },
            "tech_expert_1": {
                "name": "이기술",
                "expertise": "기술평가",
                "experience_years": 12,
                "preferences": {
                    "technical_weight": 0.95,
                    "innovation_weight": 0.8,
                    "problem_solving_weight": 0.85,
                    "learning_weight": 0.75
                },
                "bias_tendencies": {
                    "tech_company_bias": 0.3,
                    "publication_bias": 0.4,
                    "trend_bias": 0.2
                }
            }
        }
        
        logger.info("인간 피드백 시뮬레이터 초기화 완료")
    
    def generate_expert_feedback(
        self, 
        candidate_info: str,
        position: str,
        ai_recommendation: Dict,
        expert_id: str = None
    ) -> FeedbackData:
        """전문가 피드백 생성"""
        if expert_id is None:
            expert_id = random.choice(list(self.expert_profiles.keys()))
        
        expert = self.expert_profiles[expert_id]
        
        # 후보자 특성 분석
        candidate_features = self._analyze_candidate(candidate_info)
        
        # 전문가 평가
        expert_evaluation = self._evaluate_from_expert_perspective(
            candidate_features, position, expert, ai_recommendation
        )
        
        # 선호도 쌍 생성
        preferred, rejected = self._generate_preference_pair(
            expert_evaluation, ai_recommendation, expert
        )
        
        return FeedbackData(
            expert_id=expert_id,
            expert_type=expert["expertise"],
            candidate_info=candidate_info,
            position=position,
            preferred_response=preferred,
            rejected_response=rejected,
            preference_strength=expert_evaluation["preference_strength"],
            reasoning=expert_evaluation["reasoning"],
            confidence=expert_evaluation["confidence"],
            timestamp=datetime.now().isoformat(),
            bias_factors=expert_evaluation["bias_factors"]
        )
    
    def _analyze_candidate(self, candidate_info: str) -> Dict:
        """후보자 정보 분석"""
        features = {
            "education_level": 0.5,
            "experience_years": 0.5,
            "technical_skills": 0.5,
            "leadership_exp": 0.5,
            "government_exp": 0.5,
            "publication_count": 0.5
        }
        
        text = candidate_info.lower()
        
        # 교육 수준
        if "박사" in text:
            features["education_level"] = 0.9
        elif "석사" in text:
            features["education_level"] = 0.7
        
        # 기술 키워드
        tech_keywords = ["ai", "머신러닝", "딥러닝", "데이터"]
        tech_count = sum(1 for keyword in tech_keywords if keyword in text)
        features["technical_skills"] = min(tech_count / len(tech_keywords), 1.0)
        
        # 리더십 경험
        if any(word in text for word in ["팀장", "리더", "관리"]):
            features["leadership_exp"] = 0.8
        
        # 정부 경험
        if any(word in text for word in ["정부", "자문", "위원회"]):
            features["government_exp"] = 0.7
        
        return features
    
    def _evaluate_from_expert_perspective(
        self, 
        candidate_features: Dict,
        position: str,
        expert: Dict,
        ai_recommendation: Dict
    ) -> Dict:
        """전문가 관점 평가"""
        # 간단한 점수 계산
        expert_score = sum(candidate_features.values()) / len(candidate_features)
        ai_score = ai_recommendation.get("confidence", 0.5)
        
        score_diff = abs(expert_score - ai_score)
        preference_strength = min(score_diff * 2.0, 1.0)
        
        confidence = min(expert["experience_years"] / 20.0, 0.95)
        
        reasoning = f"전문가 관점에서 종합 점수 {expert_score:.2f}로 평가"
        
        return {
            "expert_score": expert_score,
            "ai_score": ai_score,
            "preference_strength": preference_strength,
            "confidence": confidence,
            "reasoning": reasoning,
            "bias_factors": expert["bias_tendencies"]
        }
    
    def _generate_preference_pair(
        self, 
        expert_evaluation: Dict,
        ai_recommendation: Dict,
        expert: Dict
    ) -> Tuple[str, str]:
        """선호도 쌍 생성"""
        expert_score = expert_evaluation["expert_score"]
        
        if expert_score >= 0.7:
            expert_recommendation = "추천"
        elif expert_score >= 0.5:
            expert_recommendation = "보류"
        else:
            expert_recommendation = "비추천"
        
        preferred = f"이 후보자를 {expert_recommendation}합니다. {expert_evaluation['reasoning']}"
        rejected = f"이 후보자에 대한 다른 평가는 부적절합니다."
        
        return preferred, rejected


def create_sample_candidates() -> List[Dict]:
    """샘플 후보자 생성"""
    return [
        {
            "info": "김AI는 서울대 박사로 구글에서 5년 근무했습니다.",
            "position": "AI정책관"
        }
    ]


if __name__ == "__main__":
    print("👥 Human Feedback Simulator 테스트")
    
    simulator = HumanFeedbackSimulator()
    
    test_candidate = "김철수는 서울대 박사로 구글에서 AI 연구를 했습니다."
    test_ai_recommendation = {
        'prediction': '추천',
        'confidence': 0.85
    }
    
    feedback = simulator.generate_expert_feedback(
        test_candidate, "AI정책관", test_ai_recommendation
    )
    
    print(f"전문가: {feedback.expert_id}")
    print(f"선호도: {feedback.preference_strength:.3f}")
    print("✅ 테스트 완료!") 