"""
🎯 TRAS 전문가 에이전트들
=========================

Section 6 강의의 "전문가 에이전트 팀" 구현

정부 인재 추천을 위한 다양한 전문가 에이전트들:
- AITechnicalAgent: AI/ML 기술 전문가
- PolicyExpertAgent: 정책 전문가  
- LeadershipAgent: 리더십 평가 전문가
- BiasDetectionAgent: 편향 검사 전문가
- MasterCoordinatorAgent: 마스터 코디네이터
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import random


class BaseAgent:
    """기본 에이전트 클래스"""
    
    def __init__(self, agent_id: str, name: str, specialty: str):
        self.agent_id = agent_id
        self.name = name
        self.specialty = specialty
        self.experience_years = 10
        self.confidence_base = 0.8
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """기본 분석 메소드"""
        return {
            "agent_id": self.agent_id,
            "analysis": f"{self.specialty} 분석 완료",
            "confidence": self.confidence_base
        }


class AITechnicalAgent(BaseAgent):
    """AI/ML 기술 전문가 에이전트"""
    
    def __init__(self):
        super().__init__(
            agent_id="ai_tech_expert",
            name="AI 기술 전문가",
            specialty="AI/ML 기술 평가"
        )
        self.focus_areas = ["딥러닝", "NLP", "컴퓨터비전", "MLOps"]
        self.experience_years = 15
        self.confidence_base = 0.9
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """AI 기술 역량 분석"""
        candidate_background = data.get("background", "")
        target_position = data.get("target_position", "")
        
        # AI 관련 키워드 분석
        ai_keywords = ["AI", "ML", "딥러닝", "머신러닝", "데이터사이언스", "Python", "TensorFlow"]
        ai_score = sum(1 for keyword in ai_keywords if keyword.lower() in candidate_background.lower())
        
        # 기술 점수 계산 (0-10)
        technical_score = min(10, (ai_score / len(ai_keywords)) * 10 + random.uniform(0, 2))
        
        # 연구 경험 평가
        research_indicators = ["박사", "연구소", "논문", "특허", "프로젝트"]
        research_score = sum(1 for indicator in research_indicators if indicator in candidate_background)
        
        # 최종 평가
        overall_assessment = (technical_score + research_score * 2) / 3
        
        recommendation = self._generate_technical_recommendation(technical_score, research_score)
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "technical_score": round(technical_score, 1),
            "research_experience": research_score,
            "overall_assessment": round(overall_assessment, 1),
            "recommendation": recommendation,
            "strengths": self._identify_strengths(candidate_background),
            "improvements": self._identify_improvements(candidate_background),
            "confidence": min(0.95, self.confidence_base + technical_score/20),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_technical_recommendation(self, tech_score: float, research_score: int) -> str:
        """기술적 추천 생성"""
        if tech_score >= 8 and research_score >= 3:
            return "강력 추천 - AI 분야 세계적 수준"
        elif tech_score >= 6 and research_score >= 2:
            return "추천 - AI 기술 역량 우수"
        elif tech_score >= 4:
            return "조건부 추천 - 추가 기술 교육 필요"
        else:
            return "비추천 - 기술 역량 부족"
    
    def _identify_strengths(self, background: str) -> List[str]:
        """강점 식별"""
        strengths = []
        if "박사" in background:
            strengths.append("고급 학위 보유")
        if "연구" in background:
            strengths.append("연구 경험 풍부")
        if any(keyword in background.lower() for keyword in ["ai", "ml"]):
            strengths.append("AI/ML 전문성")
        return strengths
    
    def _identify_improvements(self, background: str) -> List[str]:
        """개선점 식별"""
        improvements = []
        if "정책" not in background:
            improvements.append("정책 이해도 향상 필요")
        if "관리" not in background and "리더" not in background:
            improvements.append("리더십 경험 보완 필요")
        return improvements


class PolicyExpertAgent(BaseAgent):
    """정책 전문가 에이전트"""
    
    def __init__(self):
        super().__init__(
            agent_id="policy_expert",
            name="정책 전문가", 
            specialty="정부 정책 분석"
        )
        self.government_experience = 10
        self.focus_areas = ["디지털전환", "규제", "혁신정책", "공공서비스"]
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """정책 이해도 및 정부 업무 적합성 분석"""
        candidate_background = data.get("background", "")
        target_position = data.get("target_position", "")
        
        # 정책 관련 키워드 분석
        policy_keywords = ["정책", "정부", "공공", "행정", "법규", "규제", "제도"]
        policy_score = sum(1 for keyword in policy_keywords if keyword in candidate_background)
        
        # 정부 경험 평가
        gov_experience = ["공무원", "정부", "부처", "청", "위원회", "자문"]
        gov_score = sum(1 for exp in gov_experience if exp in candidate_background)
        
        # 정책 점수 계산
        policy_understanding = min(10, (policy_score + gov_score) * 1.5 + random.uniform(0, 2))
        
        recommendation = self._generate_policy_recommendation(policy_understanding, gov_score)
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "policy_score": round(policy_understanding, 1),
            "government_experience": gov_score,
            "public_service_mindset": self._assess_public_service_mindset(candidate_background),
            "recommendation": recommendation,
            "policy_strengths": self._identify_policy_strengths(candidate_background),
            "development_needs": self._identify_development_needs(candidate_background),
            "confidence": min(0.9, 0.7 + policy_understanding/15),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_policy_recommendation(self, policy_score: float, gov_exp: int) -> str:
        """정책 관련 추천 생성"""
        if policy_score >= 7 and gov_exp >= 2:
            return "정책 이해도 우수 - 정부 업무 적합"
        elif policy_score >= 5:
            return "기본적 정책 이해 - 추가 교육으로 보완 가능"
        else:
            return "정책 경험 부족 - 집중적 정책 교육 필요"
    
    def _assess_public_service_mindset(self, background: str) -> str:
        """공공 서비스 마인드 평가"""
        public_indicators = ["봉사", "공익", "사회", "국민", "공공"]
        score = sum(1 for indicator in public_indicators if indicator in background)
        
        if score >= 3:
            return "우수"
        elif score >= 1:
            return "보통"
        else:
            return "개발 필요"
    
    def _identify_policy_strengths(self, background: str) -> List[str]:
        strengths = []
        if "정책" in background:
            strengths.append("정책 분야 경험")
        if "공공" in background:
            strengths.append("공공 부문 이해")
        return strengths
    
    def _identify_development_needs(self, background: str) -> List[str]:
        needs = []
        if "기술" not in background:
            needs.append("기술 이해도 향상")
        if "데이터" not in background:
            needs.append("데이터 기반 정책 역량")
        return needs


class LeadershipAgent(BaseAgent):
    """리더십 평가 전문가"""
    
    def __init__(self):
        super().__init__(
            agent_id="leadership_expert",
            name="리더십 전문가",
            specialty="리더십 및 관리 능력 평가"
        )
        self.management_experience = 18
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """리더십 및 관리 능력 분석"""
        candidate_background = data.get("background", "")
        
        # 리더십 키워드 분석
        leadership_keywords = ["리더", "관리", "팀장", "부장", "책임자", "프로젝트매니저"]
        leadership_score = sum(1 for keyword in leadership_keywords if keyword in candidate_background)
        
        # 협업 지표
        collaboration_keywords = ["협업", "팀워크", "소통", "조율", "협력"]
        collaboration_score = sum(1 for keyword in collaboration_keywords if keyword in candidate_background)
        
        # 성과 관리 지표
        performance_keywords = ["성과", "목표", "KPI", "결과", "달성"]
        performance_score = sum(1 for keyword in performance_keywords if keyword in candidate_background)
        
        # 종합 리더십 점수
        total_leadership = (leadership_score * 3 + collaboration_score * 2 + performance_score * 2) / 7
        final_score = min(10, total_leadership * 2 + random.uniform(0, 1))
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "leadership_score": round(final_score, 1),
            "team_management": leadership_score,
            "collaboration": collaboration_score,
            "performance_management": performance_score,
            "leadership_style": self._assess_leadership_style(candidate_background),
            "recommendation": self._generate_leadership_recommendation(final_score),
            "development_areas": self._identify_leadership_development(candidate_background),
            "confidence": min(0.88, 0.75 + final_score/20),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _assess_leadership_style(self, background: str) -> str:
        """리더십 스타일 평가"""
        if "혁신" in background or "창의" in background:
            return "혁신형 리더십"
        elif "협업" in background or "소통" in background:
            return "협력형 리더십"
        elif "성과" in background or "목표" in background:
            return "성과지향형 리더십"
        else:
            return "전통형 리더십"
    
    def _generate_leadership_recommendation(self, score: float) -> str:
        """리더십 추천 생성"""
        if score >= 8:
            return "뛰어난 리더십 - 고위직 적합"
        elif score >= 6:
            return "우수한 리더십 - 중간관리직 적합"
        elif score >= 4:
            return "기본적 리더십 - 리더십 개발 권장"
        else:
            return "리더십 개발 필요 - 집중 교육 권장"
    
    def _identify_leadership_development(self, background: str) -> List[str]:
        """리더십 개발 영역 식별"""
        areas = []
        if "소통" not in background:
            areas.append("커뮤니케이션 스킬")
        if "변화" not in background and "혁신" not in background:
            areas.append("변화 관리 능력")
        if "전략" not in background:
            areas.append("전략적 사고")
        return areas


class BiasDetectionAgent(BaseAgent):
    """편향 검사 전문가"""
    
    def __init__(self):
        super().__init__(
            agent_id="bias_detector",
            name="편향 검사 전문가",
            specialty="편향 감지 및 공정성 검토"
        )
        self.ethics_training = True
        self.focus_areas = ["성별편향", "지역편향", "학벌편향", "연령편향"]
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """편향 검사 및 공정성 분석"""
        candidate_info = data.get("candidate_info", {})
        evaluation_results = data.get("evaluation_results", {})
        
        # 편향 위험도 분석
        bias_risks = self._detect_potential_biases(candidate_info, evaluation_results)
        
        # 공정성 점수 계산
        fairness_score = self._calculate_fairness_score(bias_risks)
        
        # 다양성 평가
        diversity_assessment = self._assess_diversity_impact(candidate_info)
        
        return {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "bias_risks": bias_risks,
            "fairness_score": fairness_score,
            "diversity_impact": diversity_assessment,
            "recommendations": self._generate_fairness_recommendations(bias_risks),
            "audit_passed": fairness_score >= 7.0,
            "confidence": 0.92,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _detect_potential_biases(self, candidate_info: Dict, evaluation_results: Dict) -> Dict[str, float]:
        """잠재적 편향 감지"""
        biases = {
            "gender_bias": self._check_gender_bias(candidate_info),
            "education_bias": self._check_education_bias(candidate_info),
            "regional_bias": self._check_regional_bias(candidate_info),
            "age_bias": self._check_age_bias(candidate_info)
        }
        return biases
    
    def _check_gender_bias(self, info: Dict) -> float:
        """성별 편향 검사"""
        # 실제 구현에서는 더 정교한 편향 검사 로직
        return random.uniform(0.1, 0.3)  # 낮은 편향 위험
    
    def _check_education_bias(self, info: Dict) -> float:
        """학벌 편향 검사"""
        return random.uniform(0.1, 0.4)
    
    def _check_regional_bias(self, info: Dict) -> float:
        """지역 편향 검사"""
        return random.uniform(0.0, 0.2)
    
    def _check_age_bias(self, info: Dict) -> float:
        """연령 편향 검사"""
        return random.uniform(0.1, 0.3)
    
    def _calculate_fairness_score(self, bias_risks: Dict[str, float]) -> float:
        """공정성 점수 계산"""
        avg_bias = sum(bias_risks.values()) / len(bias_risks)
        fairness_score = 10 - (avg_bias * 20)  # 편향이 낮을수록 높은 점수
        return max(0, min(10, fairness_score))
    
    def _assess_diversity_impact(self, info: Dict) -> str:
        """다양성 영향 평가"""
        diversity_factors = ["성별", "지역", "연령", "배경"]
        # 실제로는 더 복잡한 다양성 평가
        return "다양성 증진에 기여" if random.random() > 0.3 else "다양성 영향 중립"
    
    def _generate_fairness_recommendations(self, bias_risks: Dict[str, float]) -> List[str]:
        """공정성 개선 권장사항"""
        recommendations = []
        
        for bias_type, risk_level in bias_risks.items():
            if risk_level > 0.3:
                recommendations.append(f"{bias_type} 위험 - 추가 검토 필요")
        
        if not recommendations:
            recommendations.append("편향 위험 낮음 - 공정한 평가")
        
        return recommendations


class MasterCoordinatorAgent(BaseAgent):
    """마스터 코디네이터 - 모든 에이전트의 결과를 종합"""
    
    def __init__(self):
        super().__init__(
            agent_id="master_coordinator",
            name="마스터 코디네이터",
            specialty="의견 통합 및 최종 결정"
        )
        self.decision_framework = "다면평가 + 합의도출"
    
    def synthesize_decision(
        self,
        specialist_analyses: Dict[str, Any],
        peer_reviews: Dict[str, Any],
        bias_check: Dict[str, Any],
        position_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """모든 전문가 의견을 종합하여 최종 결정"""
        
        # 각 에이전트 결과 수집
        tech_result = specialist_analyses.get("ai_tech_expert", {})
        policy_result = specialist_analyses.get("policy_expert", {})
        leadership_result = specialist_analyses.get("leadership_expert", {})
        
        # 가중치 계산 (신뢰도 기반)
        weights = self._calculate_weights(specialist_analyses)
        
        # 종합 점수 계산
        final_scores = self._calculate_final_scores(specialist_analyses, weights)
        
        # 최종 추천 등급
        overall_score = sum(final_scores.values()) / len(final_scores)
        recommendation_level = self._determine_recommendation_level(overall_score)
        
        # 합의 수준 평가
        consensus_level = self._assess_consensus_level(specialist_analyses)
        
        return {
            "agent_id": self.agent_id,
            "final_decision": {
                "recommendation_level": recommendation_level,
                "overall_score": round(overall_score, 2),
                "detailed_scores": final_scores,
                "consensus_level": consensus_level,
                "bias_audit_passed": bias_check.get("audit_passed", False)
            },
            "decision_rationale": self._generate_decision_rationale(
                specialist_analyses, final_scores, consensus_level
            ),
            "next_steps": self._recommend_next_steps(recommendation_level, bias_check),
            "confidence": min(0.95, 0.8 + consensus_level * 0.15),
            "decision_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_weights(self, analyses: Dict[str, Any]) -> Dict[str, float]:
        """에이전트별 가중치 계산"""
        weights = {}
        total_confidence = 0
        
        for agent_id, analysis in analyses.items():
            confidence = analysis.get("confidence", 0.5)
            weights[agent_id] = confidence
            total_confidence += confidence
        
        # 정규화
        if total_confidence > 0:
            for agent_id in weights:
                weights[agent_id] /= total_confidence
        
        return weights
    
    def _calculate_final_scores(self, analyses: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, float]:
        """최종 점수 계산"""
        criteria = ["technical", "policy", "leadership"]
        final_scores = {}
        
        for criterion in criteria:
            weighted_sum = 0
            total_weight = 0
            
            for agent_id, analysis in analyses.items():
                score_key = f"{criterion}_score"
                if score_key in analysis:
                    weight = weights.get(agent_id, 0.33)
                    weighted_sum += analysis[score_key] * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_scores[criterion] = weighted_sum / total_weight
            else:
                final_scores[criterion] = 5.0  # 기본값
        
        return final_scores
    
    def _determine_recommendation_level(self, overall_score: float) -> str:
        """추천 등급 결정"""
        if overall_score >= 8.5:
            return "강력 추천"
        elif overall_score >= 7.0:
            return "추천"
        elif overall_score >= 5.5:
            return "조건부 추천"
        elif overall_score >= 4.0:
            return "보류"
        else:
            return "비추천"
    
    def _assess_consensus_level(self, analyses: Dict[str, Any]) -> float:
        """합의 수준 평가"""
        confidences = [analysis.get("confidence", 0.5) for analysis in analyses.values()]
        
        if len(confidences) < 2:
            return 1.0
        
        import statistics
        mean_conf = statistics.mean(confidences)
        std_conf = statistics.stdev(confidences) if len(confidences) > 1 else 0
        
        # 표준편차가 낮을수록 높은 합의
        consensus = max(0, 1 - (std_conf / mean_conf) if mean_conf > 0 else 0)
        return min(1.0, consensus)
    
    def _generate_decision_rationale(
        self, 
        analyses: Dict[str, Any], 
        scores: Dict[str, float], 
        consensus: float
    ) -> str:
        """결정 근거 생성"""
        rationale_parts = []
        
        # 각 영역별 평가 요약
        for criterion, score in scores.items():
            if score >= 8:
                rationale_parts.append(f"{criterion} 영역 우수 ({score:.1f})")
            elif score >= 6:
                rationale_parts.append(f"{criterion} 영역 양호 ({score:.1f})")
            else:
                rationale_parts.append(f"{criterion} 영역 보완 필요 ({score:.1f})")
        
        # 합의 수준 언급
        if consensus >= 0.8:
            rationale_parts.append("전문가 간 높은 합의")
        elif consensus >= 0.6:
            rationale_parts.append("전문가 간 보통 수준 합의")
        else:
            rationale_parts.append("전문가 간 의견 차이 존재")
        
        return " | ".join(rationale_parts)
    
    def _recommend_next_steps(self, recommendation: str, bias_check: Dict[str, Any]) -> List[str]:
        """다음 단계 권장사항"""
        steps = []
        
        if recommendation == "강력 추천":
            steps.append("즉시 면접 진행")
            steps.append("최우선 후보로 검토")
        elif recommendation == "추천":
            steps.append("면접 일정 조정")
            steps.append("추가 서류 검토")
        elif recommendation == "조건부 추천":
            steps.append("보완 자료 요청")
            steps.append("추가 검증 절차")
        else:
            steps.append("재검토 필요")
            steps.append("다른 후보 우선 고려")
        
        if not bias_check.get("audit_passed", False):
            steps.append("편향성 재검토 필요")
        
        return steps


# 사용 예시
if __name__ == "__main__":
    print("🎯 TRAS 전문가 에이전트 테스트")
    
    # 테스트 데이터
    test_data = {
        "candidate_name": "김철수",
        "background": "AI 박사, 연구소 3년, 프로젝트 리더 경험",
        "target_position": "AI정책관"
    }
    
    # 각 에이전트 테스트
    agents = [
        AITechnicalAgent(),
        PolicyExpertAgent(),
        LeadershipAgent(),
        BiasDetectionAgent()
    ]
    
    results = {}
    for agent in agents:
        result = agent.analyze(test_data)
        results[agent.agent_id] = result
        print(f"✅ {agent.name}: {result.get('recommendation', 'N/A')}")
    
    # 마스터 코디네이터 최종 결정
    coordinator = MasterCoordinatorAgent()
    final_decision = coordinator.synthesize_decision(
        specialist_analyses=results,
        peer_reviews={},
        bias_check=results.get("bias_detector", {}),
        position_requirements={}
    )
    
    print(f"\n🎯 최종 결정: {final_decision['final_decision']['recommendation_level']}")
    print(f"📊 종합 점수: {final_decision['final_decision']['overall_score']}")
    print(f"🤝 합의 수준: {final_decision['final_decision']['consensus_level']:.2f}") 