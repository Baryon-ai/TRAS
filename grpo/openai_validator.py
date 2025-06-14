#!/usr/bin/env python3
"""
🤖 OpenAI Validator: Expert-level validation using GPT-4

Simulates human expert feedback for government talent recommendations
using OpenAI API as a sophisticated validation system.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI library not installed. Run: pip install openai")

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """검증 결과 데이터 클래스"""
    overall_score: float  # 0-10 점수
    recommendation: str   # 추천/보류/비추천/강추
    reasoning: str       # 상세 근거
    strengths: List[str] # 강점들
    weaknesses: List[str] # 약점들
    improvement_suggestions: List[str] # 개선 제안
    confidence: float    # 검증자 신뢰도
    bias_check: Dict[str, float] # 편향 검사 결과
    timestamp: str       # 검증 시각


class OpenAIValidator:
    """
    OpenAI GPT-4 기반 전문가 검증 시스템
    
    정부 인재 추천의 품질을 전문가 수준으로 평가하고
    인간 피드백을 시뮬레이션합니다.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        OpenAI 검증자 초기화
        
        Args:
            api_key: OpenAI API 키 (환경변수 OPENAI_API_KEY에서 자동 로드)
            model: 사용할 모델 (gpt-4, gpt-3.5-turbo 등)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        # API 키 설정
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.model = model
        
        # 전문가 프로필들
        self.expert_profiles = {
            "hr_specialist": {
                "name": "인사 전문가",
                "expertise": "인사 정책, 채용 전략, 조직 관리",
                "perspective": "체계적이고 공정한 인사 관리 관점"
            },
            "domain_expert": {
                "name": "분야 전문가", 
                "expertise": "해당 직책의 전문 기술과 경험",
                "perspective": "기술적 역량과 실무 적합성 중심"
            },
            "policy_expert": {
                "name": "정책 전문가",
                "expertise": "정부 정책, 공공 서비스, 거버넌스",
                "perspective": "정책 이해도와 공공성 중심"
            },
            "leadership_expert": {
                "name": "리더십 전문가",
                "expertise": "조직 리더십, 팀 관리, 의사소통",
                "perspective": "리더십 역량과 조직 기여도 중심"
            }
        }
        
        logger.info(f"OpenAI 검증자 초기화 완료: {model}")
    
    def validate_recommendation(
        self, 
        candidate_info: str, 
        position: str,
        ai_recommendation: Dict,
        expert_type: str = "comprehensive"
    ) -> ValidationResult:
        """
        AI 추천 결과를 전문가 관점에서 검증
        
        Args:
            candidate_info: 후보자 정보
            position: 지원 직책
            ai_recommendation: AI 추천 결과
            expert_type: 전문가 유형 또는 'comprehensive'
            
        Returns:
            ValidationResult: 검증 결과
        """
        try:
            # 검증 프롬프트 생성
            validation_prompt = self._create_validation_prompt(
                candidate_info, position, ai_recommendation, expert_type
            )
            
            # OpenAI API 호출
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(expert_type)},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.3,  # 일관성을 위해 낮은 온도
                max_tokens=1500
            )
            
            # 응답 파싱
            result = self._parse_validation_response(response.choices[0].message.content)
            
            logger.info(f"검증 완료: {position} - {result.overall_score:.1f}점")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI 검증 실패: {e}")
            # 폴백 결과 반환
            return self._create_fallback_result()
    
    def _get_system_prompt(self, expert_type: str) -> str:
        """전문가 유형에 따른 시스템 프롬프트"""
        if expert_type == "comprehensive":
            return """
            당신은 정부 인재 채용 분야의 최고 전문가입니다. 
            인사 관리, 기술 평가, 정책 이해, 리더십 모든 분야에 정통합니다.
            
            다음 원칙을 반드시 지켜주세요:
            1. 객관적이고 공정한 평가
            2. 구체적인 근거 제시
            3. 편향 없는 판단
            4. 건설적인 피드백
            5. 정부 조직의 특성 고려
            
            평가 결과를 JSON 형식으로 정확히 제공해주세요.
            """
        
        profile = self.expert_profiles.get(expert_type, self.expert_profiles["hr_specialist"])
        return f"""
        당신은 {profile['name']}입니다.
        전문 분야: {profile['expertise']}
        평가 관점: {profile['perspective']}
        
        정부 인재 추천을 평가할 때 다음을 중시합니다:
        1. 객관적 자격 요건 충족도
        2. 해당 직책의 특수성
        3. 정부 조직 적합성
        4. 장기적 기여 가능성
        
        평가 결과를 JSON 형식으로 제공해주세요.
        """
    
    def _create_validation_prompt(
        self, 
        candidate_info: str, 
        position: str,
        ai_recommendation: Dict,
        expert_type: str
    ) -> str:
        """검증용 프롬프트 생성"""
        return f"""
        다음 정부 인재 추천 건을 전문가 관점에서 평가해주세요.

        **지원 직책**: {position}

        **후보자 정보**:
        {candidate_info}

        **AI 시스템 추천 결과**:
        - 추천 등급: {ai_recommendation.get('prediction', 'N/A')}
        - 신뢰도: {ai_recommendation.get('confidence', 0):.3f}
        - 주요 요소: {ai_recommendation.get('reasoning', {}).get('top_factor', 'N/A')}
        - 기술 점수: {ai_recommendation.get('detailed_analysis', {}).get('technical_score', 0):.3f}
        - 정책 점수: {ai_recommendation.get('detailed_analysis', {}).get('policy_score', 0):.3f}
        - 리더십 점수: {ai_recommendation.get('detailed_analysis', {}).get('leadership_score', 0):.3f}
        - 협업 점수: {ai_recommendation.get('detailed_analysis', {}).get('collaboration_score', 0):.3f}

        **평가 요청사항**:
        1. AI 추천의 적절성을 0-10점으로 평가
        2. 추천/보류/비추천/강추 중 최종 의견
        3. 상세한 평가 근거
        4. 후보자의 주요 강점 3가지
        5. 주요 약점 및 우려사항 3가지
        6. 구체적인 개선 제안사항
        7. 성별/연령/출신 등 편향 요소 검사
        8. 평가 신뢰도 (0-1)

        다음 JSON 형식으로 정확히 응답해주세요:
        {{
            "overall_score": 8.5,
            "recommendation": "추천",
            "reasoning": "상세한 평가 근거...",
            "strengths": ["강점1", "강점2", "강점3"],
            "weaknesses": ["약점1", "약점2", "약점3"],
            "improvement_suggestions": ["제안1", "제안2", "제안3"],
            "confidence": 0.85,
            "bias_check": {{
                "gender_bias": 0.1,
                "age_bias": 0.05,
                "regional_bias": 0.0,
                "educational_bias": 0.15
            }}
        }}
        """
    
    def _parse_validation_response(self, response_text: str) -> ValidationResult:
        """OpenAI 응답을 ValidationResult로 파싱"""
        try:
            # JSON 부분 추출
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("JSON 형식을 찾을 수 없습니다")
            
            json_text = response_text[start_idx:end_idx]
            result_dict = json.loads(json_text)
            
            # ValidationResult 객체 생성
            return ValidationResult(
                overall_score=float(result_dict.get('overall_score', 5.0)),
                recommendation=result_dict.get('recommendation', '보류'),
                reasoning=result_dict.get('reasoning', '평가 근거 없음'),
                strengths=result_dict.get('strengths', []),
                weaknesses=result_dict.get('weaknesses', []),
                improvement_suggestions=result_dict.get('improvement_suggestions', []),
                confidence=float(result_dict.get('confidence', 0.5)),
                bias_check=result_dict.get('bias_check', {}),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            return self._create_fallback_result()
    
    def _create_fallback_result(self) -> ValidationResult:
        """파싱 실패 시 폴백 결과"""
        return ValidationResult(
            overall_score=5.0,
            recommendation="보류",
            reasoning="검증 시스템 오류로 인한 기본 평가",
            strengths=["평가 필요"],
            weaknesses=["평가 필요"],
            improvement_suggestions=["상세 검토 필요"],
            confidence=0.3,
            bias_check={},
            timestamp=datetime.now().isoformat()
        )
    
    def multi_expert_validation(
        self, 
        candidate_info: str, 
        position: str,
        ai_recommendation: Dict
    ) -> Dict[str, ValidationResult]:
        """다중 전문가 검증"""
        results = {}
        
        for expert_type in ["hr_specialist", "domain_expert", "policy_expert", "leadership_expert"]:
            try:
                result = self.validate_recommendation(
                    candidate_info, position, ai_recommendation, expert_type
                )
                results[expert_type] = result
                
                # API 호출 제한 고려 (1초 대기)
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"전문가 {expert_type} 검증 실패: {e}")
                results[expert_type] = self._create_fallback_result()
        
        return results
    
    def generate_consensus(self, multi_expert_results: Dict[str, ValidationResult]) -> ValidationResult:
        """다중 전문가 의견을 종합하여 합의 결과 생성"""
        if not multi_expert_results:
            return self._create_fallback_result()
        
        # 점수 평균
        scores = [result.overall_score for result in multi_expert_results.values()]
        avg_score = sum(scores) / len(scores)
        
        # 신뢰도 평균
        confidences = [result.confidence for result in multi_expert_results.values()]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 추천 의견 집계
        recommendations = [result.recommendation for result in multi_expert_results.values()]
        recommendation_counts = {}
        for rec in recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        consensus_recommendation = max(recommendation_counts, key=recommendation_counts.get)
        
        # 강점/약점 통합
        all_strengths = []
        all_weaknesses = []
        all_suggestions = []
        
        for result in multi_expert_results.values():
            all_strengths.extend(result.strengths)
            all_weaknesses.extend(result.weaknesses)
            all_suggestions.extend(result.improvement_suggestions)
        
        # 중복 제거 및 상위 3개 선택
        unique_strengths = list(set(all_strengths))[:3]
        unique_weaknesses = list(set(all_weaknesses))[:3]
        unique_suggestions = list(set(all_suggestions))[:3]
        
        # 편향 검사 평균
        bias_scores = {}
        for result in multi_expert_results.values():
            for bias_type, score in result.bias_check.items():
                if bias_type not in bias_scores:
                    bias_scores[bias_type] = []
                bias_scores[bias_type].append(score)
        
        avg_bias_check = {
            bias_type: sum(scores) / len(scores)
            for bias_type, scores in bias_scores.items()
        }
        
        return ValidationResult(
            overall_score=avg_score,
            recommendation=consensus_recommendation,
            reasoning=f"다중 전문가 합의 결과 (전문가 {len(multi_expert_results)}명)",
            strengths=unique_strengths,
            weaknesses=unique_weaknesses,
            improvement_suggestions=unique_suggestions,
            confidence=avg_confidence,
            bias_check=avg_bias_check,
            timestamp=datetime.now().isoformat()
        )
    
    def create_training_feedback(
        self, 
        candidate_info: str, 
        position: str,
        correct_recommendation: str,
        ai_prediction: str
    ) -> Dict:
        """GPRO 학습용 피드백 데이터 생성"""
        # 올바른 추천과 AI 예측을 비교하여 선호도 쌍 생성
        if correct_recommendation == ai_prediction:
            # 정답인 경우 - 긍정적 피드백
            preferred = f"후보자를 {correct_recommendation}합니다. AI 분석이 정확합니다."
            rejected = f"후보자를 다른 등급으로 추천하는 것은 부적절합니다."
        else:
            # 오답인 경우 - 교정 피드백
            preferred = f"후보자를 {correct_recommendation}해야 합니다."
            rejected = f"AI가 제안한 {ai_prediction} 추천은 부적절합니다."
        
        return {
            'prompt': f"직책: {position}\n후보자: {candidate_info}",
            'preferred': preferred,
            'rejected': rejected,
            'preference_strength': 1.0 if correct_recommendation != ai_prediction else 0.8
        }


# 편의 함수들
def validate_with_openai(
    candidate_info: str,
    position: str, 
    ai_recommendation: Dict,
    api_key: Optional[str] = None
) -> ValidationResult:
    """간편한 OpenAI 검증 함수"""
    validator = OpenAIValidator(api_key=api_key)
    return validator.validate_recommendation(candidate_info, position, ai_recommendation)


def multi_expert_validate(
    candidate_info: str,
    position: str,
    ai_recommendation: Dict,
    api_key: Optional[str] = None
) -> ValidationResult:
    """다중 전문가 검증 및 합의 생성"""
    validator = OpenAIValidator(api_key=api_key)
    multi_results = validator.multi_expert_validation(candidate_info, position, ai_recommendation)
    return validator.generate_consensus(multi_results)


if __name__ == "__main__":
    # 테스트 코드
    print("🤖 OpenAI Validator 테스트")
    
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY 환경변수를 설정해주세요")
        exit(1)
    
    # 테스트 데이터
    test_candidate = """
    김철수는 서울대학교 컴퓨터공학과 박사 학위를 보유하고 있으며,
    구글에서 5년간 AI 연구원으로 근무했습니다.
    자연어처리 분야에서 50편의 논문을 발표했고,
    정부 AI 자문위원회에서 2년간 활동했습니다.
    팀 리더 경험이 있으며 다국적 팀을 관리한 경험이 있습니다.
    """
    
    test_ai_result = {
        'prediction': '추천',
        'confidence': 0.85,
        'reasoning': {'top_factor': 'technical'},
        'detailed_analysis': {
            'technical_score': 0.92,
            'policy_score': 0.71,
            'leadership_score': 0.78,
            'collaboration_score': 0.83
        }
    }
    
    try:
        # 검증 실행
        validator = OpenAIValidator(api_key)
        result = validator.validate_recommendation(
            test_candidate, "AI정책관", test_ai_result
        )
        
        print(f"검증 점수: {result.overall_score:.1f}/10")
        print(f"최종 추천: {result.recommendation}")
        print(f"주요 강점: {', '.join(result.strengths[:2])}")
        print("✅ OpenAI Validator 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print("API 키를 확인하거나 네트워크 연결을 점검해주세요.") 