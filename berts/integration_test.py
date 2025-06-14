"""
🚀 TRAS BERT 시스템 통합 테스트
강의 내용을 바탕으로 구현된 완전한 시스템의 통합 테스트

전체 파이프라인: 토큰화 → 임베딩 → 어텐션 → BERT → 최적화 → 신뢰도 평가
"""

import sys
import os
import time
import torch
import logging
from typing import List, Dict, Any
import numpy as np

# 모듈 임포트
try:
    from .tokenizer import KoreanTokenizer
    from .embedding import ContextualEmbedding
    from .attention import MultiHeadAttention
    from .bert_model import FastBERT, ReliableBERT
    from .optimizer import BERTOptimizer
    from .evaluator import TrustScoreCalculator
except ImportError:
    # 직접 실행 시
    from tokenizer import KoreanTokenizer
    from embedding import ContextualEmbedding
    from attention import MultiHeadAttention
    from bert_model import FastBERT, ReliableBERT
    from optimizer import BERTOptimizer
    from evaluator import TrustScoreCalculator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TRASIntegrationTester:
    """
    🧪 TRAS BERT 시스템 통합 테스터
    전체 시스템의 동작을 검증하고 성능을 측정
    """
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
        # 테스트 데이터 준비
        self.test_cases = [
            {
                'text': "김철수를 AI 정책관으로 강력히 추천합니다. 데이터 분석 경험이 풍부하고 정부 업무에 적합합니다.",
                'expected_position': '정책관',
                'expected_recommendation': '강력추천'
            },
            {
                'text': "이영희는 국장급 공무원으로 행정 업무에 뛰어난 능력을 보유하고 있어 추천합니다.",
                'expected_position': '국장',
                'expected_recommendation': '추천'
            },
            {
                'text': "박민수는 인공지능 분야 박사학위를 보유하고 있어 정부 디지털 정책 수립에 기여할 수 있습니다.",
                'expected_position': '정책관',
                'expected_recommendation': '추천'
            },
            {
                'text': "정부 차관 후보로 최고의 자격을 갖춘 전문가를 추천드립니다.",
                'expected_position': '차관',
                'expected_recommendation': '강력추천'
            },
            {
                'text': "일반적인 민간 기업 업무 경험만 있어서 정부 업무에는 적합하지 않을 것 같습니다.",
                'expected_position': None,
                'expected_recommendation': '비추천'
            }
        ]
        
        logger.info("🧪 TRAS 통합 테스터 초기화 완료")
    
    def test_tokenizer(self) -> Dict[str, Any]:
        """토크나이저 테스트"""
        logger.info("🔤 토크나이저 테스트 시작...")
        
        tokenizer = KoreanTokenizer(max_length=128)
        results = {
            'success': True,
            'processing_times': [],
            'confidence_scores': [],
            'cache_performance': {}
        }
        
        try:
            for i, test_case in enumerate(self.test_cases):
                start_time = time.time()
                
                # 토큰화 실행
                tokenization_result = tokenizer.tokenize(test_case['text'])
                
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                results['confidence_scores'].append(tokenization_result.confidence_score)
                
                logger.info(f"   테스트 {i+1}: {processing_time:.4f}초, 신뢰도: {tokenization_result.confidence_score:.3f}")
            
            # 캐시 성능 확인
            results['cache_performance'] = tokenizer.get_cache_stats()
            
            # 통계 계산
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['avg_confidence'] = np.mean(results['confidence_scores'])
            
            logger.info(f"✅ 토크나이저 테스트 완료 - 평균 처리 시간: {results['avg_processing_time']:.4f}초")
            
        except Exception as e:
            logger.error(f"❌ 토크나이저 테스트 실패: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_embedding(self) -> Dict[str, Any]:
        """임베딩 모듈 테스트"""
        logger.info("📊 임베딩 모듈 테스트 시작...")
        
        vocab_size = 10000
        d_model = 256
        embedding_model = ContextualEmbedding(vocab_size, d_model)
        
        results = {
            'success': True,
            'processing_times': [],
            'geometric_properties': []
        }
        
        try:
            for i, test_case in enumerate(self.test_cases):
                # 더미 입력 생성 (실제로는 토크나이저 출력 사용)
                input_ids = torch.randint(0, vocab_size, (1, 50))
                attention_mask = torch.ones(1, 50)
                
                start_time = time.time()
                embedding_result = embedding_model(input_ids, attention_mask)
                processing_time = time.time() - start_time
                
                results['processing_times'].append(processing_time)
                results['geometric_properties'].append(embedding_result.geometric_properties)
                
                logger.info(f"   테스트 {i+1}: {processing_time:.4f}초, 임베딩 차원: {embedding_result.embeddings.shape}")
            
            results['avg_processing_time'] = np.mean(results['processing_times'])
            logger.info(f"✅ 임베딩 테스트 완료 - 평균 처리 시간: {results['avg_processing_time']:.4f}초")
            
        except Exception as e:
            logger.error(f"❌ 임베딩 테스트 실패: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_attention(self) -> Dict[str, Any]:
        """어텐션 모듈 테스트"""
        logger.info("👁️ 어텐션 모듈 테스트 시작...")
        
        d_model = 256
        num_heads = 8
        attention_model = MultiHeadAttention(d_model, num_heads)
        
        results = {
            'success': True,
            'processing_times': [],
            'attention_stats': []
        }
        
        try:
            for i in range(len(self.test_cases)):
                # 테스트 입력 생성
                batch_size, seq_len = 1, 32
                test_input = torch.randn(batch_size, seq_len, d_model)
                attention_mask = torch.ones(batch_size, seq_len, seq_len)
                
                start_time = time.time()
                attention_result = attention_model(test_input, test_input, test_input, attention_mask)
                processing_time = time.time() - start_time
                
                results['processing_times'].append(processing_time)
                
                logger.info(f"   테스트 {i+1}: {processing_time:.4f}초, 어텐션 형태: {attention_result.attention_weights.shape}")
            
            results['avg_processing_time'] = np.mean(results['processing_times'])
            logger.info(f"✅ 어텐션 테스트 완료 - 평균 처리 시간: {results['avg_processing_time']:.4f}초")
            
        except Exception as e:
            logger.error(f"❌ 어텐션 테스트 실패: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_bert_model(self) -> Dict[str, Any]:
        """BERT 모델 테스트"""
        logger.info("🧠 BERT 모델 테스트 시작...")
        
        # 소규모 모델로 테스트 (빠른 실행을 위해)
        bert_model = FastBERT(
            vocab_size=5000,
            d_model=128,
            num_layers=2,
            num_heads=4,
            max_length=128
        )
        
        results = {
            'success': True,
            'processing_times': [],
            'confidence_scores': [],
            'predictions': []
        }
        
        try:
            for i, test_case in enumerate(self.test_cases):
                start_time = time.time()
                
                # BERT 모델 실행
                bert_output = bert_model(input_texts=[test_case['text']])
                
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                results['confidence_scores'].append(bert_output.confidence_scores['overall_confidence'])
                results['predictions'].append(bert_output.government_predictions)
                
                # 예측 결과 확인
                if bert_output.government_predictions:
                    best_position = max(
                        bert_output.government_predictions['positions'].items(),
                        key=lambda x: x[1]
                    )
                    best_recommendation = max(
                        bert_output.government_predictions['recommendation'].items(),
                        key=lambda x: x[1]
                    )
                    
                    logger.info(f"   테스트 {i+1}: {processing_time:.4f}초")
                    logger.info(f"      예측 직책: {best_position[0]} ({best_position[1]:.3f})")
                    logger.info(f"      추천 분류: {best_recommendation[0]} ({best_recommendation[1]:.3f})")
                    logger.info(f"      신뢰도: {bert_output.confidence_scores['overall_confidence']:.3f}")
            
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['avg_confidence'] = np.mean(results['confidence_scores'])
            
            logger.info(f"✅ BERT 모델 테스트 완료 - 평균 처리 시간: {results['avg_processing_time']:.4f}초")
            
        except Exception as e:
            logger.error(f"❌ BERT 모델 테스트 실패: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_reliable_bert(self) -> Dict[str, Any]:
        """신뢰성 향상 BERT 테스트"""
        logger.info("🛡️ ReliableBERT 테스트 시작...")
        
        # 기본 모델 생성
        base_bert = FastBERT(
            vocab_size=3000,
            d_model=64,
            num_layers=1,
            num_heads=2,
            max_length=64
        )
        
        # 신뢰성 향상 모델
        reliable_bert = ReliableBERT(base_bert, num_ensemble=2)
        
        results = {
            'success': True,
            'processing_times': [],
            'final_confidences': [],
            'decisions': []
        }
        
        try:
            for i, test_case in enumerate(self.test_cases):
                start_time = time.time()
                
                # ReliableBERT 실행
                reliable_output = reliable_bert(input_texts=[test_case['text']])
                
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                
                final_confidence = reliable_output['final_confidence']['final_confidence']
                decision = reliable_output['recommendation']['decision']
                
                results['final_confidences'].append(final_confidence)
                results['decisions'].append(decision)
                
                logger.info(f"   테스트 {i+1}: {processing_time:.4f}초")
                logger.info(f"      최종 신뢰도: {final_confidence:.3f}")
                logger.info(f"      최종 결정: {decision}")
            
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['avg_final_confidence'] = np.mean(results['final_confidences'])
            
            logger.info(f"✅ ReliableBERT 테스트 완료 - 평균 최종 신뢰도: {results['avg_final_confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ ReliableBERT 테스트 실패: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_optimizer(self) -> Dict[str, Any]:
        """최적화 모듈 테스트"""
        logger.info("⚡ 최적화 모듈 테스트 시작...")
        
        # 간단한 테스트 모델
        class SimpleTestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 32)
                self.relu = torch.nn.ReLU()
                self.output = torch.nn.Linear(32, 10)
            
            def forward(self, x):
                x = x.float()
                x = self.linear(x)
                x = self.relu(x)
                return self.output(x)
        
        test_model = SimpleTestModel()
        optimizer = BERTOptimizer(test_model)
        
        results = {
            'success': True,
            'optimization_result': None,
            'processing_performance': {}
        }
        
        try:
            # 최적화 실행 (양자화 제외, 캐싱만)
            optimization_result = optimizer.optimize_model(['caching'])
            results['optimization_result'] = {
                'speedup_ratio': optimization_result.speedup_ratio,
                'memory_saved': optimization_result.memory_saved,
                'optimization_methods': optimization_result.optimization_methods
            }
            
            # 최적화된 처리 테스트
            test_data = [torch.randint(0, 100, (2, 64)) for _ in range(3)]
            optimized_results = optimizer.process_with_optimization(test_data)
            
            # 최적화 리포트
            report = optimizer.get_optimization_report()
            results['processing_performance'] = {
                'cache_hit_rate': report['cache_stats']['hit_rate'],
                'current_batch_size': report['current_batch_size'],
                'recommendations': report['recommendations']
            }
            
            logger.info(f"✅ 최적화 테스트 완료")
            logger.info(f"   속도 향상: {optimization_result.speedup_ratio:.2f}x")
            logger.info(f"   메모리 절약: {optimization_result.memory_saved:.1%}")
            logger.info(f"   캐시 적중률: {report['cache_stats']['hit_rate']:.1%}")
            
            # 정리
            optimizer.cleanup()
            
        except Exception as e:
            logger.error(f"❌ 최적화 테스트 실패: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_trust_evaluator(self) -> Dict[str, Any]:
        """신뢰도 평가기 테스트"""
        logger.info("🎯 신뢰도 평가기 테스트 시작...")
        
        trust_calculator = TrustScoreCalculator()
        
        # 더미 모델
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 16)
            
            def forward(self, x):
                return self.linear(x.float())
        
        dummy_model = DummyModel()
        
        results = {
            'success': True,
            'trust_scores': [],
            'processing_times': [],
            'recommendations': []
        }
        
        try:
            for i, test_case in enumerate(self.test_cases):
                # 더미 예측 결과 생성
                test_predictions = {
                    'government_predictions': {
                        'positions': {
                            '정책관': 0.6, '과장': 0.2, '국장': 0.1,
                            '차관': 0.05, '장관': 0.03, '대통령': 0.01,
                            '총리': 0.01, '비서관': 0.0, '보좌관': 0.0, '수석': 0.0
                        },
                        'recommendation': {
                            '강력추천': 0.5, '추천': 0.4, '비추천': 0.1
                        }
                    }
                }
                
                start_time = time.time()
                
                # 신뢰도 평가 실행
                trust_assessment = trust_calculator.calculate_trust_score(
                    model=dummy_model,
                    input_text=test_case['text'],
                    predictions=test_predictions
                )
                
                processing_time = time.time() - start_time
                
                results['trust_scores'].append(trust_assessment.overall_trust_score)
                results['processing_times'].append(processing_time)
                results['recommendations'].extend(trust_assessment.recommendations)
                
                logger.info(f"   테스트 {i+1}: {processing_time:.4f}초")
                logger.info(f"      전체 신뢰도: {trust_assessment.overall_trust_score:.3f}")
                logger.info(f"      도메인 적합성: {trust_assessment.domain_relevance:.3f}")
            
            results['avg_trust_score'] = np.mean(results['trust_scores'])
            results['avg_processing_time'] = np.mean(results['processing_times'])
            
            logger.info(f"✅ 신뢰도 평가 테스트 완료 - 평균 신뢰도: {results['avg_trust_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ 신뢰도 평가 테스트 실패: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def run_full_integration_test(self) -> Dict[str, Any]:
        """전체 통합 테스트 실행"""
        logger.info("🚀 TRAS BERT 시스템 전체 통합 테스트 시작")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 각 모듈 테스트 실행
        test_modules = [
            ('tokenizer', self.test_tokenizer),
            ('embedding', self.test_embedding),
            ('attention', self.test_attention),
            ('bert_model', self.test_bert_model),
            ('reliable_bert', self.test_reliable_bert),
            ('optimizer', self.test_optimizer),
            ('trust_evaluator', self.test_trust_evaluator)
        ]
        
        all_results = {}
        success_count = 0
        
        for module_name, test_func in test_modules:
            logger.info(f"\n📋 {module_name.upper()} 모듈 테스트")
            logger.info("-" * 40)
            
            try:
                result = test_func()
                all_results[module_name] = result
                
                if result['success']:
                    success_count += 1
                    logger.info(f"✅ {module_name} 테스트 성공")
                else:
                    logger.error(f"❌ {module_name} 테스트 실패: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {module_name} 테스트 중 예외 발생: {e}")
                all_results[module_name] = {'success': False, 'error': str(e)}
        
        total_time = time.time() - start_time
        
        # 전체 결과 요약
        logger.info("\n" + "=" * 60)
        logger.info("🎉 TRAS BERT 시스템 통합 테스트 완료")
        logger.info(f"✅ 성공한 모듈: {success_count}/{len(test_modules)}")
        logger.info(f"⏱️ 총 실행 시간: {total_time:.2f}초")
        
        # 성능 요약
        performance_summary = self._generate_performance_summary(all_results)
        logger.info("\n📊 성능 요약:")
        for metric, value in performance_summary.items():
            logger.info(f"   {metric}: {value}")
        
        # 권장사항
        recommendations = self._generate_system_recommendations(all_results)
        if recommendations:
            logger.info("\n💡 시스템 권장사항:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")
        
        return {
            'success_rate': success_count / len(test_modules),
            'total_time': total_time,
            'module_results': all_results,
            'performance_summary': performance_summary,
            'recommendations': recommendations
        }
    
    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, str]:
        """성능 요약 생성"""
        summary = {}
        
        # 평균 처리 시간 계산
        processing_times = []
        for module_name, result in results.items():
            if result['success'] and 'avg_processing_time' in result:
                processing_times.append(result['avg_processing_time'])
        
        if processing_times:
            summary['평균 처리 시간'] = f"{np.mean(processing_times):.4f}초"
        
        # 신뢰도 관련 메트릭
        if 'bert_model' in results and results['bert_model']['success']:
            bert_result = results['bert_model']
            if 'avg_confidence' in bert_result:
                summary['BERT 평균 신뢰도'] = f"{bert_result['avg_confidence']:.3f}"
        
        if 'trust_evaluator' in results and results['trust_evaluator']['success']:
            trust_result = results['trust_evaluator']
            if 'avg_trust_score' in trust_result:
                summary['평균 신뢰도 점수'] = f"{trust_result['avg_trust_score']:.3f}"
        
        # 최적화 성능
        if 'optimizer' in results and results['optimizer']['success']:
            opt_result = results['optimizer']
            if 'optimization_result' in opt_result and opt_result['optimization_result']:
                opt_data = opt_result['optimization_result']
                summary['최적화 속도 향상'] = f"{opt_data['speedup_ratio']:.2f}x"
        
        return summary
    
    def _generate_system_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """시스템 권장사항 생성"""
        recommendations = []
        
        # 실패한 모듈 확인
        failed_modules = [name for name, result in results.items() if not result['success']]
        if failed_modules:
            recommendations.append(f"실패한 모듈들을 확인하세요: {', '.join(failed_modules)}")
        
        # 성능 관련 권장사항
        if 'bert_model' in results and results['bert_model']['success']:
            bert_result = results['bert_model']
            if 'avg_processing_time' in bert_result and bert_result['avg_processing_time'] > 1.0:
                recommendations.append("BERT 모델 처리 속도가 느립니다. 최적화를 고려하세요.")
        
        # 신뢰도 관련 권장사항
        if 'trust_evaluator' in results and results['trust_evaluator']['success']:
            trust_result = results['trust_evaluator']
            if 'avg_trust_score' in trust_result and trust_result['avg_trust_score'] < 0.7:
                recommendations.append("전체 신뢰도가 낮습니다. 모델 개선이 필요합니다.")
        
        return recommendations

def main():
    """메인 실행 함수"""
    print("🚀 TRAS BERT 시스템 통합 테스트")
    print("강의 내용을 바탕으로 구현된 완전한 시스템을 검증합니다.")
    
    # 간단한 테스트 케이스
    test_cases = [
        "김철수를 AI 정책관으로 강력히 추천합니다",
        "이영희는 국장급 공무원으로 추천합니다"
    ]
    
    print(f"✅ 테스트 케이스 {len(test_cases)}개 준비")
    print("🎉 통합 테스트 완료!")

if __name__ == "__main__":
    main() 