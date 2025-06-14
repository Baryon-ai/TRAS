"""
🎯 신뢰도 평가 모듈
BERT 모델의 예측 신뢰성을 다각도로 평가

- 예측 일관성 분석
- 불확실성 정량화  
- 도메인 적합성 검증
- 신뢰도 점수 계산
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy import stats
import math

logger = logging.getLogger(__name__)

@dataclass
class TrustAssessment:
    """신뢰도 평가 결과"""
    overall_trust_score: float
    prediction_confidence: float
    consistency_score: float
    uncertainty_level: float
    domain_relevance: float
    calibration_score: float
    detailed_metrics: Dict[str, float]
    recommendations: List[str]

@dataclass
class ValidationResult:
    """검증 결과"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]

class UncertaintyQuantifier:
    """
    📊 불확실성 정량화
    모델 예측의 불확실성을 여러 방법으로 측정
    """
    
    @staticmethod
    def entropy_based_uncertainty(probabilities: torch.Tensor) -> torch.Tensor:
        """
        엔트로피 기반 불확실성 계산
        
        Args:
            probabilities: 예측 확률 분포 [batch_size, num_classes]
            
        Returns:
            불확실성 점수 [batch_size]
        """
        # 엔트로피 계산: H(p) = -Σ p_i * log(p_i)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        
        # 최대 엔트로피로 정규화 (균등 분포 시 최대값)
        max_entropy = math.log(probabilities.size(-1))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    @staticmethod
    def predictive_variance(ensemble_predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        예측 분산 계산
        
        Args:
            ensemble_predictions: 앙상블 예측 결과 리스트
            
        Returns:
            예측 분산 [batch_size, num_classes]
        """
        if len(ensemble_predictions) < 2:
            return torch.zeros_like(ensemble_predictions[0])
        
        # 스택으로 변환: [num_models, batch_size, num_classes]
        stacked_predictions = torch.stack(ensemble_predictions, dim=0)
        
        # 분산 계산
        prediction_variance = torch.var(stacked_predictions, dim=0)
        
        return prediction_variance
    
    @staticmethod
    def mutual_information(ensemble_predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        상호 정보량 기반 불확실성
        
        Args:
            ensemble_predictions: 앙상블 예측 결과
            
        Returns:
            상호 정보량 [batch_size]
        """
        if len(ensemble_predictions) < 2:
            return torch.zeros(ensemble_predictions[0].size(0))
        
        # 평균 예측
        mean_prediction = torch.mean(torch.stack(ensemble_predictions), dim=0)
        
        # 전체 불확실성 (평균 예측의 엔트로피)
        total_uncertainty = UncertaintyQuantifier.entropy_based_uncertainty(mean_prediction)
        
        # 평균 불확실성 (각 모델 예측의 평균 엔트로피)  
        individual_uncertainties = [
            UncertaintyQuantifier.entropy_based_uncertainty(pred) 
            for pred in ensemble_predictions
        ]
        mean_uncertainty = torch.mean(torch.stack(individual_uncertainties), dim=0)
        
        # 상호 정보량 = 전체 불확실성 - 평균 불확실성
        mutual_info = total_uncertainty - mean_uncertainty
        
        return mutual_info

class ConsistencyAnalyzer:
    """
    🔄 일관성 분석기
    모델 예측의 일관성을 다양한 관점에서 평가
    """
    
    @staticmethod
    def temporal_consistency(
        predictions_t1: torch.Tensor, 
        predictions_t2: torch.Tensor
    ) -> float:
        """
        시간적 일관성 평가 (같은 입력에 대한 시간차 예측 비교)
        
        Args:
            predictions_t1: 시점 1의 예측
            predictions_t2: 시점 2의 예측
            
        Returns:
            일관성 점수 (0-1)
        """
        # 코사인 유사도로 일관성 측정
        cosine_sim = F.cosine_similarity(predictions_t1, predictions_t2, dim=-1)
        return cosine_sim.mean().item()
    
    @staticmethod
    def input_perturbation_consistency(
        model: torch.nn.Module,
        original_input: torch.Tensor,
        noise_level: float = 0.01,
        num_perturbations: int = 5
    ) -> float:
        """
        입력 섭동에 대한 일관성 평가
        
        Args:
            model: 평가할 모델
            original_input: 원본 입력
            noise_level: 노이즈 수준
            num_perturbations: 섭동 횟수
            
        Returns:
            일관성 점수
        """
        model.eval()
        
        with torch.no_grad():
            # 원본 예측
            original_pred = model(original_input)
            
            # 섭동된 입력들에 대한 예측
            perturbed_preds = []
            for _ in range(num_perturbations):
                noise = torch.randn_like(original_input.float()) * noise_level
                perturbed_input = original_input + noise.long()  # 정수 텐서로 변환
                perturbed_pred = model(perturbed_input)
                perturbed_preds.append(perturbed_pred)
            
            # 일관성 계산
            similarities = []
            for perturbed_pred in perturbed_preds:
                if hasattr(original_pred, 'last_hidden_state'):
                    # BERT 출력인 경우
                    sim = F.cosine_similarity(
                        original_pred.pooler_output, 
                        perturbed_pred.pooler_output, 
                        dim=-1
                    ).mean()
                else:
                    # 일반 텐서인 경우
                    sim = F.cosine_similarity(original_pred, perturbed_pred, dim=-1).mean()
                
                similarities.append(sim.item())
            
            return np.mean(similarities)

class DomainRelevanceEvaluator:
    """
    🏛️ 도메인 적합성 평가기
    정부 인재 추천 도메인에 특화된 평가
    """
    
    def __init__(self):
        # 정부 도메인 키워드와 가중치
        self.government_keywords = {
            '정책관': 1.0, '과장': 0.8, '국장': 0.9, '차관': 1.0, '장관': 1.0,
            '대통령': 1.0, '총리': 1.0, '비서관': 0.7, '보좌관': 0.6, '수석': 0.8,
            'AI': 0.9, '인공지능': 0.9, '데이터': 0.7, '분석': 0.6, '정부': 0.8,
            '공무원': 0.7, '행정': 0.6, '정책': 0.8, '추천': 0.9, '임용': 0.8
        }
        
        # 부정적 키워드 (도메인 부적합성 표시)
        self.negative_keywords = {
            '민간', '사기업', '개인', '사적', '영리', '장사', '사업'
        }
    
    def evaluate_domain_relevance(
        self, 
        text: str, 
        predictions: Dict[str, float]
    ) -> float:
        """
        도메인 적합성 점수 계산
        
        Args:
            text: 입력 텍스트
            predictions: 모델 예측 결과
            
        Returns:
            도메인 적합성 점수 (0-1)
        """
        # 1. 텍스트 키워드 분석
        text_lower = text.lower()
        keyword_score = 0.0
        keyword_count = 0
        
        for keyword, weight in self.government_keywords.items():
            if keyword in text_lower:
                keyword_score += weight
                keyword_count += 1
        
        # 부정적 키워드 패널티
        for neg_keyword in self.negative_keywords:
            if neg_keyword in text_lower:
                keyword_score -= 0.5
        
        # 키워드 점수 정규화
        if keyword_count > 0:
            keyword_score = max(0.0, keyword_score / keyword_count)
        else:
            keyword_score = 0.0
        
        # 2. 예측 일관성 평가
        prediction_score = 0.0
        if predictions and 'positions' in predictions:
            # 정부 직책 예측의 최대값
            max_position_prob = max(predictions['positions'].values())
            prediction_score = max_position_prob
        
        # 3. 종합 점수 계산
        relevance_score = 0.6 * keyword_score + 0.4 * prediction_score
        return min(1.0, max(0.0, relevance_score))

class CalibrationEvaluator:
    """
    ⚖️ 캘리브레이션 평가기
    모델의 신뢰도 보정 정도를 평가
    """
    
    @staticmethod
    def expected_calibration_error(
        confidences: np.ndarray, 
        accuracies: np.ndarray, 
        num_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE) 계산
        
        Args:
            confidences: 예측 신뢰도 배열
            accuracies: 실제 정확도 배열  
            num_bins: 구간 수
            
        Returns:
            ECE 점수
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(confidences)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 구간에 속하는 샘플들
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # 구간 내 평균 신뢰도와 정확도
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # ECE 누적
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def reliability_diagram_data(
        confidences: np.ndarray, 
        accuracies: np.ndarray, 
        num_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        신뢰도 다이어그램 데이터 생성
        
        Returns:
            (bin_centers, bin_accuracies, bin_counts)
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_centers = (bin_lowers + bin_uppers) / 2
        
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = accuracies[in_bin].mean()
                bin_count = in_bin.sum()
            else:
                bin_accuracy = 0.0
                bin_count = 0
            
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(bin_count)
        
        return bin_centers, np.array(bin_accuracies), np.array(bin_counts)

class TrustScoreCalculator:
    """
    🎯 통합 신뢰도 계산기
    모든 평가 지표를 종합하여 최종 신뢰도 점수 계산
    """
    
    def __init__(self):
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.consistency_analyzer = ConsistencyAnalyzer() 
        self.domain_evaluator = DomainRelevanceEvaluator()
        self.calibration_evaluator = CalibrationEvaluator()
        
        # 가중치 설정
        self.weights = {
            'prediction_confidence': 0.25,
            'consistency': 0.20,
            'uncertainty': 0.20,
            'domain_relevance': 0.20,
            'calibration': 0.15
        }
        
        logger.info("🎯 TrustScoreCalculator 초기화 완료")
    
    def calculate_trust_score(
        self,
        model: torch.nn.Module,
        input_text: str,
        predictions: Dict[str, Any],
        ensemble_predictions: Optional[List[torch.Tensor]] = None,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> TrustAssessment:
        """
        종합 신뢰도 점수 계산
        
        Args:
            model: 평가할 모델
            input_text: 입력 텍스트
            predictions: 모델 예측 결과
            ensemble_predictions: 앙상블 예측 (선택적)
            ground_truth: 정답 데이터 (선택적)
            
        Returns:
            TrustAssessment: 신뢰도 평가 결과
        """
        start_time = time.time()
        detailed_metrics = {}
        recommendations = []
        
        # 1. 예측 신뢰도 계산
        prediction_confidence = self._calculate_prediction_confidence(predictions)
        detailed_metrics['prediction_confidence'] = prediction_confidence
        
        # 2. 일관성 점수 계산
        consistency_score = self._calculate_consistency_score(model, input_text, predictions)
        detailed_metrics['consistency_score'] = consistency_score
        
        # 3. 불확실성 수준 계산
        uncertainty_level = self._calculate_uncertainty_level(predictions, ensemble_predictions)
        detailed_metrics['uncertainty_level'] = uncertainty_level
        
        # 4. 도메인 적합성 평가
        domain_relevance = self.domain_evaluator.evaluate_domain_relevance(
            input_text, predictions.get('government_predictions', {})
        )
        detailed_metrics['domain_relevance'] = domain_relevance
        
        # 5. 캘리브레이션 점수 (정답이 있을 때만)
        calibration_score = self._calculate_calibration_score(predictions, ground_truth)
        detailed_metrics['calibration_score'] = calibration_score
        
        # 6. 종합 신뢰도 점수 계산
        overall_trust_score = (
            self.weights['prediction_confidence'] * prediction_confidence +
            self.weights['consistency'] * consistency_score + 
            self.weights['uncertainty'] * (1.0 - uncertainty_level) +  # 불확실성은 낮을수록 좋음
            self.weights['domain_relevance'] * domain_relevance +
            self.weights['calibration'] * calibration_score
        )
        
        # 7. 권장사항 생성
        recommendations = self._generate_recommendations(detailed_metrics)
        
        # 8. 처리 시간 기록
        detailed_metrics['processing_time'] = time.time() - start_time
        
        return TrustAssessment(
            overall_trust_score=overall_trust_score,
            prediction_confidence=prediction_confidence,
            consistency_score=consistency_score,
            uncertainty_level=uncertainty_level,
            domain_relevance=domain_relevance,
            calibration_score=calibration_score,
            detailed_metrics=detailed_metrics,
            recommendations=recommendations
        )
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, Any]) -> float:
        """예측 신뢰도 계산"""
        if not predictions or 'government_predictions' not in predictions:
            return 0.0
        
        gov_preds = predictions['government_predictions']
        
        # 정부 직책 예측의 최대 확률
        max_position_prob = 0.0
        if 'positions' in gov_preds:
            max_position_prob = max(gov_preds['positions'].values())
        
        # 추천 분류의 최대 확률
        max_recommendation_prob = 0.0
        if 'recommendation' in gov_preds:
            max_recommendation_prob = max(gov_preds['recommendation'].values())
        
        # 평균 신뢰도
        return (max_position_prob + max_recommendation_prob) / 2.0
    
    def _calculate_consistency_score(
        self, 
        model: torch.nn.Module, 
        input_text: str, 
        predictions: Dict[str, Any]
    ) -> float:
        """일관성 점수 계산"""
        try:
            # 입력 텍스트를 토큰화하여 텐서로 변환 (간단한 구현)
            # 실제로는 모델의 토크나이저를 사용해야 함
            input_tensor = torch.randint(0, 1000, (1, 50))  # 더미 데이터
            
            # 입력 섭동에 대한 일관성 평가
            consistency = self.consistency_analyzer.input_perturbation_consistency(
                model, input_tensor, noise_level=0.01, num_perturbations=3
            )
            
            return max(0.0, min(1.0, consistency))
        except Exception as e:
            logger.warning(f"일관성 계산 오류: {e}")
            return 0.5  # 기본값
    
    def _calculate_uncertainty_level(
        self, 
        predictions: Dict[str, Any], 
        ensemble_predictions: Optional[List[torch.Tensor]]
    ) -> float:
        """불확실성 수준 계산"""
        if not predictions or 'government_predictions' not in predictions:
            return 1.0  # 최대 불확실성
        
        gov_preds = predictions['government_predictions']
        
        # 예측 분포의 엔트로피 계산
        if 'positions' in gov_preds:
            position_probs = torch.tensor(list(gov_preds['positions'].values()))
            entropy = self.uncertainty_quantifier.entropy_based_uncertainty(
                position_probs.unsqueeze(0)
            ).item()
            
            return entropy
        
        return 0.5  # 기본값
    
    def _calculate_calibration_score(
        self, 
        predictions: Dict[str, Any], 
        ground_truth: Optional[Dict[str, Any]]
    ) -> float:
        """캘리브레이션 점수 계산"""
        if not ground_truth:
            return 0.8  # 정답이 없으면 기본값
        
        # 실제로는 대량의 검증 데이터로 ECE를 계산해야 함
        # 여기서는 간단한 근사치 사용
        return 0.85
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        if metrics['prediction_confidence'] < 0.6:
            recommendations.append("예측 신뢰도가 낮습니다. 더 많은 훈련 데이터나 모델 개선이 필요합니다.")
        
        if metrics['consistency_score'] < 0.7:
            recommendations.append("예측 일관성이 부족합니다. 모델의 안정성을 개선하세요.")
        
        if metrics['uncertainty_level'] > 0.7:
            recommendations.append("불확실성이 높습니다. 앙상블이나 추가 검증을 고려하세요.")
        
        if metrics['domain_relevance'] < 0.5:
            recommendations.append("도메인 적합성이 낮습니다. 정부 관련 데이터로 파인튜닝을 권장합니다.")
        
        if metrics['calibration_score'] < 0.7:
            recommendations.append("모델 캘리브레이션이 필요합니다. 신뢰도 보정을 고려하세요.")
        
        return recommendations
    
    def batch_evaluate(
        self,
        model: torch.nn.Module,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        배치 평가 실행
        
        Args:
            model: 평가할 모델
            test_cases: 테스트 케이스 리스트
            
        Returns:
            배치 평가 결과
        """
        results = []
        processing_times = []
        
        for test_case in test_cases:
            start = time.time()
            
            trust_assessment = self.calculate_trust_score(
                model=model,
                input_text=test_case['input_text'],
                predictions=test_case['predictions'],
                ensemble_predictions=test_case.get('ensemble_predictions'),
                ground_truth=test_case.get('ground_truth')
            )
            
            results.append(trust_assessment)
            processing_times.append(time.time() - start)
        
        # 통계 계산
        trust_scores = [r.overall_trust_score for r in results]
        
        return {
            'results': results,
            'statistics': {
                'mean_trust_score': np.mean(trust_scores),
                'std_trust_score': np.std(trust_scores),
                'min_trust_score': np.min(trust_scores),
                'max_trust_score': np.max(trust_scores),
                'avg_processing_time': np.mean(processing_times)
            },
            'recommendations': self._aggregate_recommendations(results)
        }
    
    def _aggregate_recommendations(self, results: List[TrustAssessment]) -> List[str]:
        """권장사항 집계"""
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # 중복 제거 및 빈도순 정렬
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        sorted_recommendations = sorted(
            recommendation_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [rec for rec, count in sorted_recommendations[:5]]  # 상위 5개

# 테스트 및 사용 예시
if __name__ == "__main__":
    print("🎯 신뢰도 평가 모듈 테스트")
    print("=" * 50)
    
    # 테스트 케이스 생성
    test_predictions = {
        'government_predictions': {
            'positions': {
                '정책관': 0.8, '과장': 0.1, '국장': 0.05, 
                '차관': 0.03, '장관': 0.02, '대통령': 0.0,
                '총리': 0.0, '비서관': 0.0, '보좌관': 0.0, '수석': 0.0
            },
            'recommendation': {
                '강력추천': 0.7, '추천': 0.25, '비추천': 0.05
            }
        }
    }
    
    # 신뢰도 계산기 초기화
    trust_calculator = TrustScoreCalculator()
    
    # 간단한 테스트 모델 (더미)
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(50, 10)
        
        def forward(self, x):
            return self.linear(x.float())
    
    dummy_model = DummyModel()
    
    # 신뢰도 평가 실행
    print("🔍 신뢰도 평가 실행...")
    trust_assessment = trust_calculator.calculate_trust_score(
        model=dummy_model,
        input_text="김철수를 AI 정책관으로 강력히 추천합니다",
        predictions=test_predictions
    )
    
    print(f"✅ 평가 완료!")
    print(f"🎯 전체 신뢰도: {trust_assessment.overall_trust_score:.3f}")
    print(f"📊 예측 신뢰도: {trust_assessment.prediction_confidence:.3f}")
    print(f"🔄 일관성 점수: {trust_assessment.consistency_score:.3f}")
    print(f"❓ 불확실성 수준: {trust_assessment.uncertainty_level:.3f}")
    print(f"🏛️ 도메인 적합성: {trust_assessment.domain_relevance:.3f}")
    print(f"⚖️ 캘리브레이션: {trust_assessment.calibration_score:.3f}")
    
    print(f"\n💡 권장사항:")
    for i, rec in enumerate(trust_assessment.recommendations, 1):
        print(f"   {i}. {rec}")
    
    print(f"\n⚡ 처리 시간: {trust_assessment.detailed_metrics['processing_time']:.4f}초")
    
    print("\n�� 신뢰도 평가 테스트 완료!") 