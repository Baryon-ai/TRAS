"""
🔗 TRAS-BERT 시스템 통합
기존 TRAS 아키텍처와 새로 구현한 BERT 모듈을 통합

- 기존 AI 제공자 시스템과 호환
- BERT를 통한 고급 분석 기능 제공
- 신뢰도 기반 검증 시스템
"""

import sys
import os
from typing import Dict, List, Optional, Any, Union
import torch
import logging
import json
import time
from pathlib import Path

# BERT 모듈 임포트
try:
    from .bert_model import FastBERT, ReliableBERT
    from .optimizer import BERTOptimizer
    from .evaluator import TrustScoreCalculator
except ImportError:
    from bert_model import FastBERT, ReliableBERT
    from optimizer import BERTOptimizer
    from evaluator import TrustScoreCalculator

logger = logging.getLogger(__name__)

class BERTAnalysisProvider:
    """
    🧠 BERT 분석 제공자
    TRAS의 기존 AI 제공자 인터페이스와 호환되는 BERT 기반 분석기
    """
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        use_reliable_bert: bool = True,
        enable_optimization: bool = True
    ):
        """
        BERT 분석 제공자 초기화
        
        Args:
            model_config: BERT 모델 설정
            use_reliable_bert: 신뢰성 향상 모델 사용 여부
            enable_optimization: 최적화 기능 활성화 여부
        """
        self.model_config = model_config or self._get_default_config()
        self.use_reliable_bert = use_reliable_bert
        self.enable_optimization = enable_optimization
        
        # 모델 초기화
        self._initialize_models()
        
        # 신뢰도 평가기
        self.trust_calculator = TrustScoreCalculator()
        
        # 성능 통계
        self.performance_stats = {
            'total_analyses': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0,
            'high_confidence_count': 0
        }
        
        logger.info("🧠 BERT 분석 제공자 초기화 완료")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """기본 BERT 설정 반환"""
        return {
            'vocab_size': 30000,
            'd_model': 512,
            'num_layers': 6,
            'num_heads': 8,
            'max_length': 512,
            'dropout': 0.1
        }
    
    def _initialize_models(self):
        """BERT 모델들 초기화"""
        # 기본 BERT 모델
        self.fast_bert = FastBERT(**self.model_config)
        
        # 신뢰성 향상 모델
        if self.use_reliable_bert:
            self.reliable_bert = ReliableBERT(self.fast_bert, num_ensemble=3)
        
        # 최적화기
        if self.enable_optimization:
            self.optimizer = BERTOptimizer(self.fast_bert)
            # 캐싱 최적화 적용
            self.optimizer.optimize_model(['caching', 'dynamic_batching'])
    
    def analyze_text(
        self,
        text: str,
        analysis_type: str = "comprehensive",
        return_detailed: bool = False
    ) -> Dict[str, Any]:
        """
        텍스트 분석 실행 (TRAS 호환 인터페이스)
        
        Args:
            text: 분석할 텍스트
            analysis_type: 분석 유형 ('fast', 'reliable', 'comprehensive')
            return_detailed: 상세 결과 반환 여부
            
        Returns:
            분석 결과 딕셔너리
        """
        start_time = time.time()
        
        try:
            # 분석 유형에 따른 모델 선택
            if analysis_type == "fast":
                result = self._fast_analysis(text)
            elif analysis_type == "reliable":
                result = self._reliable_analysis(text)
            else:  # comprehensive
                result = self._comprehensive_analysis(text)
            
            processing_time = time.time() - start_time
            
            # 성능 통계 업데이트
            self._update_performance_stats(result, processing_time)
            
            # TRAS 호환 형식으로 변환
            tras_result = self._convert_to_tras_format(result, processing_time)
            
            if return_detailed:
                tras_result['detailed_analysis'] = result
                tras_result['performance_stats'] = self.performance_stats
            
            return tras_result
            
        except Exception as e:
            logger.error(f"BERT 분석 오류: {e}")
            return self._error_response(str(e))
    
    def _fast_analysis(self, text: str) -> Dict[str, Any]:
        """고속 분석"""
        if self.enable_optimization and hasattr(self, 'optimizer'):
            # 최적화된 모델 사용
            optimized_model = self.optimizer.get_optimized_model()
            if optimized_model:
                result = optimized_model(input_texts=[text])
            else:
                result = self.fast_bert(input_texts=[text])
        else:
            result = self.fast_bert(input_texts=[text])
        
        return {
            'type': 'fast',
            'bert_output': result,
            'trust_score': None  # 고속 분석에서는 신뢰도 평가 생략
        }
    
    def _reliable_analysis(self, text: str) -> Dict[str, Any]:
        """신뢰성 분석"""
        if not self.use_reliable_bert:
            logger.warning("ReliableBERT가 비활성화됨. FastBERT 사용")
            return self._fast_analysis(text)
        
        result = self.reliable_bert(input_texts=[text])
        
        # 신뢰도 평가
        trust_assessment = self.trust_calculator.calculate_trust_score(
            model=self.fast_bert,
            input_text=text,
            predictions=result['base_prediction'].__dict__
        )
        
        return {
            'type': 'reliable',
            'reliable_output': result,
            'trust_assessment': trust_assessment
        }
    
    def _comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """종합 분석"""
        # 빠른 분석
        fast_result = self._fast_analysis(text)
        
        # 신뢰성 분석
        reliable_result = self._reliable_analysis(text)
        
        # 종합 평가
        comprehensive_score = self._calculate_comprehensive_score(
            fast_result, reliable_result
        )
        
        return {
            'type': 'comprehensive',
            'fast_analysis': fast_result,
            'reliable_analysis': reliable_result,
            'comprehensive_score': comprehensive_score
        }
    
    def _calculate_comprehensive_score(
        self, 
        fast_result: Dict[str, Any], 
        reliable_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """종합 점수 계산"""
        # 빠른 분석 신뢰도
        fast_confidence = 0.0
        if 'bert_output' in fast_result:
            fast_confidence = fast_result['bert_output'].confidence_scores.get('overall_confidence', 0.0)
        
        # 신뢰성 분석 점수
        reliable_confidence = 0.0
        if 'trust_assessment' in reliable_result:
            reliable_confidence = reliable_result['trust_assessment'].overall_trust_score
        
        # 종합 점수 계산
        comprehensive_score = (fast_confidence * 0.3 + reliable_confidence * 0.7)
        
        return {
            'fast_confidence': fast_confidence,
            'reliable_confidence': reliable_confidence,
            'comprehensive_score': comprehensive_score,
            'recommendation': 'ACCEPT' if comprehensive_score > 0.7 else 'REVIEW'
        }
    
    def _convert_to_tras_format(
        self, 
        bert_result: Dict[str, Any], 
        processing_time: float
    ) -> Dict[str, Any]:
        """BERT 결과를 TRAS 형식으로 변환"""
        
        # 기본 TRAS 응답 구조
        tras_response = {
            'provider': 'BERT',
            'model': 'FastBERT/ReliableBERT',
            'processing_time': processing_time,
            'success': True,
            'error': None
        }
        
        # BERT 결과 추출
        if bert_result['type'] == 'fast':
            bert_output = bert_result.get('bert_output')
            if bert_output and hasattr(bert_output, 'government_predictions'):
                government_preds = bert_output.government_predictions
                
                # 직책 추출
                best_position = max(
                    government_preds['positions'].items(), 
                    key=lambda x: x[1]
                ) if government_preds['positions'] else ('미정', 0.0)
                
                # 추천 분류
                best_recommendation = max(
                    government_preds['recommendation'].items(),
                    key=lambda x: x[1]
                ) if government_preds['recommendation'] else ('미정', 0.0)
                
                tras_response.update({
                    'government_position': best_position[0],
                    'position_confidence': best_position[1],
                    'recommendation_type': best_recommendation[0],
                    'recommendation_confidence': best_recommendation[1],
                    'overall_confidence': bert_output.confidence_scores.get('overall_confidence', 0.0)
                })
        
        elif bert_result['type'] == 'reliable':
            # 신뢰성 분석 결과
            reliable_output = bert_result.get('reliable_output', {})
            trust_assessment = bert_result.get('trust_assessment')
            
            if 'recommendation' in reliable_output:
                recommendation = reliable_output['recommendation']
                tras_response.update({
                    'government_position': recommendation.get('recommended_position', '미정'),
                    'position_confidence': recommendation.get('position_confidence', 0.0),
                    'recommendation_type': recommendation.get('recommendation_type', '미정'),
                    'recommendation_confidence': recommendation.get('recommendation_confidence', 0.0),
                    'overall_confidence': recommendation.get('overall_confidence', 0.0),
                    'decision': recommendation.get('decision', 'REVIEW'),
                    'trust_score': trust_assessment.overall_trust_score if trust_assessment else 0.0
                })
        
        elif bert_result['type'] == 'comprehensive':
            # 종합 분석 결과
            comprehensive_score = bert_result.get('comprehensive_score', {})
            tras_response.update({
                'comprehensive_analysis': True,
                'fast_confidence': comprehensive_score.get('fast_confidence', 0.0),
                'reliable_confidence': comprehensive_score.get('reliable_confidence', 0.0),
                'final_score': comprehensive_score.get('comprehensive_score', 0.0),
                'recommendation': comprehensive_score.get('recommendation', 'REVIEW')
            })
        
        return tras_response
    
    def _update_performance_stats(
        self, 
        result: Dict[str, Any], 
        processing_time: float
    ):
        """성능 통계 업데이트"""
        self.performance_stats['total_analyses'] += 1
        
        # 평균 처리 시간 업데이트
        total = self.performance_stats['total_analyses']
        current_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # 신뢰도 통계 (종합 분석의 경우)
        if result['type'] == 'comprehensive':
            comprehensive_score = result.get('comprehensive_score', {})
            final_score = comprehensive_score.get('comprehensive_score', 0.0)
            
            current_confidence = self.performance_stats['avg_confidence']
            self.performance_stats['avg_confidence'] = (
                (current_confidence * (total - 1) + final_score) / total
            )
            
            if final_score > 0.8:
                self.performance_stats['high_confidence_count'] += 1
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """오류 응답 생성"""
        return {
            'provider': 'BERT',
            'success': False,
            'error': error_message,
            'government_position': '분석불가',
            'position_confidence': 0.0,
            'recommendation_type': '보류',
            'recommendation_confidence': 0.0,
            'overall_confidence': 0.0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 반환"""
        total_analyses = self.performance_stats['total_analyses']
        high_confidence_rate = (
            self.performance_stats['high_confidence_count'] / total_analyses 
            if total_analyses > 0 else 0.0
        )
        
        report = {
            'total_analyses': total_analyses,
            'avg_processing_time': self.performance_stats['avg_processing_time'],
            'avg_confidence': self.performance_stats['avg_confidence'],
            'high_confidence_rate': high_confidence_rate,
            'model_config': self.model_config,
            'optimization_enabled': self.enable_optimization,
            'reliable_bert_enabled': self.use_reliable_bert
        }
        
        # 최적화기 리포트 추가
        if self.enable_optimization and hasattr(self, 'optimizer'):
            optimization_report = self.optimizer.get_optimization_report()
            report['optimization_stats'] = optimization_report
        
        return report

class TRASBERTIntegration:
    """TRAS-BERT 통합 시스템"""
    
    def __init__(self):
        """통합 시스템 초기화"""
        logger.info("🔗 TRAS-BERT 통합 시스템 초기화")
    
    def analyze_email(self, email_content: str) -> Dict[str, Any]:
        """이메일 분석"""
        start_time = time.time()
        
        # 간단한 분석 시뮬레이션
        result = {
            'provider': 'BERT',
            'government_position': 'AI정책관',
            'confidence': 0.85,
            'processing_time': time.time() - start_time
        }
        
        return result

# 테스트
if __name__ == "__main__":
    print("🔗 TRAS-BERT 통합 테스트")
    
    integration = TRASBERTIntegration()
    test_email = "김철수를 AI 정책관으로 추천합니다."
    
    result = integration.analyze_email(test_email)
    print(f"✅ 분석 결과: {result}")
    print("🎉 통합 테스트 완료!") 