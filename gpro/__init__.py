"""
🎯 GPRO (Direct Preference Optimization) 연구 모듈
정부 인재 추천을 위한 인간 피드백 기반 AI 시스템

이 모듈은 section4_rlhf_gpro.md 강의 내용을 바탕으로
실제 동작하는 RLHF/GPRO 시스템을 구현합니다.

주요 구성요소:
- GPROModel: 직접 선호도 최적화 모델
- OpenAIValidator: OpenAI API 기반 검증자
- HumanFeedbackSimulator: 인간 피드백 시뮬레이션
- PreferenceOptimizer: 선호도 최적화 엔진
- ComparisonStudy: BERT vs GPRO 성능 비교
"""

from .gpro_model import GPROModel, GPROConfig
from .openai_validator import OpenAIValidator, ValidationResult
from .human_feedback import HumanFeedbackSimulator, FeedbackData
from .preference_optimizer import PreferenceOptimizer, OptimizationConfig
from .comparison_study import ModelComparison, PerformanceMetrics

__version__ = "1.0.0"
__author__ = "BarionLabs"

__all__ = [
    # Core Models
    "GPROModel",
    "GPROConfig", 
    
    # Validation System
    "OpenAIValidator",
    "ValidationResult",
    
    # Human Feedback
    "HumanFeedbackSimulator", 
    "FeedbackData",
    
    # Optimization
    "PreferenceOptimizer",
    "OptimizationConfig",
    
    # Comparison & Evaluation
    "ModelComparison",
    "PerformanceMetrics"
]

# 연구 프로젝트 메타데이터
RESEARCH_INFO = {
    "project_name": "TRAS-GPRO: Government Talent Recommendation with Human Feedback",
    "base_lecture": "section4_rlhf_gpro.md",
    "objective": "Build better-than-BERT classifier using human preference optimization",
    "approach": "Direct Preference Optimization (DPO) with OpenAI validation",
    "expected_improvements": [
        "Better alignment with human values",
        "More interpretable recommendations", 
        "Higher expert satisfaction",
        "Robust preference learning"
    ]
} 