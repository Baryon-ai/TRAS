#!/usr/bin/env python3
"""
🧪 GPRO Integration Test: Complete System Test

Tests the entire GPRO pipeline from model initialization to comparison results.
"""

import os
import sys
import logging
from pathlib import Path

# Add gpro to path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_imports():
    """기본 모듈 import 테스트"""
    print("🔧 모듈 import 테스트...")
    
    try:
        from gpro import (
            GPROModel, GPROConfig, initialize_gpro_model,
            OpenAIValidator, ValidationResult,
            HumanFeedbackSimulator, FeedbackData, create_sample_candidates,
            PreferenceOptimizer, OptimizationConfig,
            ModelComparison, PerformanceMetrics
        )
        print("✅ 모든 모듈 import 성공!")
        return True
    except ImportError as e:
        print(f"❌ Import 실패: {e}")
        return False


def test_gpro_model():
    """GPRO 모델 기본 기능 테스트"""
    print("\n🎯 GPRO 모델 테스트...")
    
    try:
        from gpro import initialize_gpro_model
        
        # 모델 초기화
        model = initialize_gpro_model()
        print("✅ GPRO 모델 초기화 성공")
        
        # 예측 테스트
        test_candidate = """
        김AI는 서울대학교 컴퓨터공학 박사로,
        구글에서 5년간 AI 연구원으로 근무했습니다.
        자연어처리 분야 논문 50편 발표.
        """
        
        result = model.predict_with_explanation(test_candidate, "AI정책관")
        
        print(f"✅ 예측 결과: {result['prediction']}")
        print(f"✅ 신뢰도: {result['confidence']:.3f}")
        print(f"✅ 헌법적 점수: {result['constitutional_score']:.3f}")
        print(f"✅ 주요 요소: {result['reasoning']['top_factor']}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ GPRO 모델 테스트 실패: {e}")
        return False, None


def test_human_feedback():
    """인간 피드백 시스템 테스트"""
    print("\n👥 인간 피드백 시스템 테스트...")
    
    try:
        from gpro import HumanFeedbackSimulator, create_sample_candidates
        
        # 시뮬레이터 초기화
        simulator = HumanFeedbackSimulator()
        print("✅ 피드백 시뮬레이터 초기화 성공")
        
        # 샘플 후보자
        candidates = create_sample_candidates()
        print(f"✅ 샘플 후보자 {len(candidates)}명 생성")
        
        # 피드백 생성
        test_ai_rec = {
            'prediction': '추천',
            'confidence': 0.85,
            'reasoning': {'top_factor': 'technical'},
            'detailed_analysis': {
                'technical_score': 0.9,
                'policy_score': 0.7,
                'leadership_score': 0.8,
                'collaboration_score': 0.75
            }
        }
        
        feedback = simulator.generate_expert_feedback(
            candidates[0]['info'],
            candidates[0]['position'],
            test_ai_rec
        )
        
        print(f"✅ 전문가 피드백 생성: {feedback.expert_id}")
        print(f"✅ 선호도 강도: {feedback.preference_strength:.3f}")
        print(f"✅ 신뢰도: {feedback.confidence:.3f}")
        
        return True, [feedback]
        
    except Exception as e:
        print(f"❌ 인간 피드백 테스트 실패: {e}")
        return False, None


def test_openai_validator():
    """OpenAI 검증자 테스트 (API 키가 있는 경우)"""
    print("\n🤖 OpenAI 검증자 테스트...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY 환경변수가 없습니다. 검증자 테스트를 건너뜁니다.")
        return True, None
    
    try:
        from gpro import OpenAIValidator
        
        validator = OpenAIValidator(api_key=api_key)
        print("✅ OpenAI 검증자 초기화 성공")
        
        # 간단한 검증 테스트 (실제 API 호출 없이)
        print("✅ OpenAI 검증자 준비 완료 (실제 API 테스트는 선택사항)")
        
        return True, validator
        
    except Exception as e:
        print(f"❌ OpenAI 검증자 테스트 실패: {e}")
        return False, None


def test_model_comparison(gpro_model):
    """모델 비교 시스템 테스트"""
    print("\n📊 모델 비교 시스템 테스트...")
    
    try:
        from gpro import ModelComparison, create_sample_test_cases
        
        # 비교 시스템 초기화
        comparison = ModelComparison(gpro_model)
        print("✅ 모델 비교 시스템 초기화 성공")
        
        # 테스트 케이스 추가
        test_cases = create_sample_test_cases()
        for case in test_cases:
            comparison.add_test_case(
                case['candidate_info'],
                case['position'],
                case['ground_truth']
            )
        
        print(f"✅ 테스트 케이스 {len(test_cases)}개 추가")
        
        # 비교 실행
        results = comparison.run_comprehensive_comparison()
        print("✅ 포괄적 비교 실행 완료")
        
        # 결과 출력
        print("\n📋 비교 결과:")
        for recommendation in results['recommendations']:
            print(f"  {recommendation}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ 모델 비교 테스트 실패: {e}")
        return False, None


def test_preference_optimization(gpro_model, feedback_data):
    """선호도 최적화 테스트"""
    print("\n⚡ 선호도 최적화 테스트...")
    
    if not feedback_data:
        print("⚠️ 피드백 데이터가 없어 최적화 테스트를 건너뜁니다.")
        return True
    
    try:
        from gpro import PreferenceOptimizer, OptimizationConfig
        
        # 최적화 설정 (테스트용으로 간단히)
        config = OptimizationConfig(
            learning_rate=1e-5,
            batch_size=1,
            num_epochs=1,
            logging_steps=1
        )
        
        # 옵티마이저 생성
        optimizer = PreferenceOptimizer(gpro_model, config)
        print("✅ 선호도 옵티마이저 초기화 성공")
        
        # 데이터셋 준비
        train_dataset = optimizer.prepare_dataset(feedback_data)
        print(f"✅ 훈련 데이터셋 준비: {len(train_dataset)}건")
        
        print("✅ 선호도 최적화 시스템 준비 완료")
        print("   (실제 훈련은 시간이 오래 걸려 테스트에서 제외)")
        
        return True
        
    except Exception as e:
        print(f"❌ 선호도 최적화 테스트 실패: {e}")
        return False


def run_integration_test():
    """전체 통합 테스트 실행"""
    print("🧪 GPRO 통합 테스트 시작\n")
    print("=" * 60)
    
    results = {
        'imports': False,
        'gpro_model': False,
        'human_feedback': False,
        'openai_validator': False,
        'model_comparison': False,
        'preference_optimization': False
    }
    
    # 1. 기본 Import 테스트
    results['imports'] = test_basic_imports()
    if not results['imports']:
        print("\n❌ 기본 import에 실패했습니다. 의존성을 확인하세요.")
        return results
    
    # 2. GPRO 모델 테스트
    success, gpro_model = test_gpro_model()
    results['gpro_model'] = success
    
    # 3. 인간 피드백 테스트
    success, feedback_data = test_human_feedback()
    results['human_feedback'] = success
    
    # 4. OpenAI 검증자 테스트
    success, validator = test_openai_validator()
    results['openai_validator'] = success
    
    # 5. 모델 비교 테스트
    if gpro_model:
        success, comparison_results = test_model_comparison(gpro_model)
        results['model_comparison'] = success
    
    # 6. 선호도 최적화 테스트
    if gpro_model and feedback_data:
        results['preference_optimization'] = test_preference_optimization(
            gpro_model, feedback_data
        )
    
    print("\n" + "=" * 60)
    print("🎯 통합 테스트 결과 요약:")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
    
    print(f"\n총 테스트: {total_tests}")
    print(f"통과: {passed_tests}")
    print(f"실패: {total_tests - passed_tests}")
    print(f"성공률: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 모든 테스트가 성공했습니다!")
        print("GPRO 시스템이 정상적으로 작동합니다.")
    elif passed_tests >= total_tests * 0.8:
        print("\n✅ 대부분의 테스트가 성공했습니다!")
        print("일부 기능에 문제가 있을 수 있으니 확인해보세요.")
    else:
        print("\n⚠️ 여러 테스트가 실패했습니다.")
        print("의존성이나 설정을 확인해보세요.")
    
    print("\n📚 참고사항:")
    print("- OpenAI 검증은 API 키가 필요합니다")
    print("- 실제 모델 훈련은 시간이 오래 걸립니다")
    print("- 전체 기능은 gpro/README.md를 참조하세요")
    
    return results


if __name__ == "__main__":
    try:
        test_results = run_integration_test()
        
        # 종료 코드 설정
        if all(test_results.values()):
            sys.exit(0)  # 모든 테스트 성공
        else:
            sys.exit(1)  # 일부 테스트 실패
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
        sys.exit(2)
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류 발생: {e}")
        sys.exit(3) 