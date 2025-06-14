# 🎯 GPRO Research Project
## "인간 피드백으로 더 똑똑한 AI 만들기"

> **section4_rlhf_gpro.md** 강의 내용을 실제 동작하는 시스템으로 구현한 연구 프로젝트

---

## 📋 프로젝트 개요

이 프로젝트는 **RLHF/GPRO (Direct Preference Optimization)** 방법론을 정부 인재 추천 시스템에 적용하여 기존 BERT 모델보다 우수한 성능을 달성하는 것을 목표로 합니다.

### 🎯 핵심 가설
**"인간 피드백 기반 학습을 통해 AI가 단순한 패턴 매칭을 넘어서 인간의 가치와 정렬된 더 지능적인 추천을 할 수 있다"**

### 🚀 기대 효과
- **더 높은 전문가 만족도**: 인간 가치와 정렬된 추천
- **향상된 설명 가능성**: 근거가 명확한 추천 시스템
- **편향 감소**: Constitutional AI 원칙 적용
- **지속적 개선**: 피드백 기반 모델 업데이트

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GPRO Model    │    │ Human Feedback  │    │ OpenAI Validator│
│                 │    │   Simulator     │    │                 │
│ • BERT Backbone │◄──►│ • Expert Profs  │◄──►│ • GPT-4 Based   │
│ • Multi-Head    │    │ • Bias Modeling │    │ • Multi-Expert  │
│ • Constitutional│    │ • Preference    │    │ • Consensus     │
│   AI Principles │    │   Generation    │    │   Building      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Preference      │    │ Model Comparison│    │ Training Loop   │
│ Optimizer       │    │                 │    │                 │
│ • DPO Algorithm │    │ • BERT vs GPRO  │    │ • Feedback      │
│ • Constitutional│    │ • Metrics       │    │   Collection    │
│   Loss          │    │ • Visualization │    │ • Model Update  │
│ • Checkpointing │    │ • Reporting     │    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📦 모듈 구성

### 1. **`gpro_model.py`** - 핵심 GPRO 모델
```python
from gpro import GPROModel, initialize_gpro_model

# GPRO 모델 초기화
model = initialize_gpro_model()

# 설명 가능한 예측
result = model.predict_with_explanation(
    "김철수는 서울대 AI 박사로 구글에서 5년 근무...",
    "AI정책관"
)

print(f"추천: {result['prediction']}")
print(f"신뢰도: {result['confidence']:.3f}")
print(f"주요 요소: {result['reasoning']['top_factor']}")
```

**핵심 특징:**
- 🎯 **다중 관점 분석**: 기술/정책/리더십/협업 4개 관점
- 🏛️ **Constitutional AI**: 공정성, 투명성, 안전성 원칙 내장
- 📊 **직책별 특화**: 정부 직책별 맞춤형 평가
- 🔍 **설명 가능성**: 추천 근거 자동 생성

### 2. **`openai_validator.py`** - GPT-4 기반 검증 시스템
```python
from gpro import OpenAIValidator

validator = OpenAIValidator(api_key="your-openai-key")

# 단일 전문가 검증
result = validator.validate_recommendation(
    candidate_info, position, ai_recommendation
)

# 다중 전문가 합의
consensus = validator.multi_expert_validation(
    candidate_info, position, ai_recommendation
)
```

**검증자 프로필:**
- 👨‍💼 **인사 전문가**: 공정성과 절차 중시
- 👨‍💻 **기술 전문가**: 기술 역량과 혁신성 평가
- 👨‍🏛️ **정책 전문가**: 정부 업무 적합성 판단
- 👨‍🎓 **리더십 전문가**: 리더십과 소통 능력 평가

### 3. **`human_feedback.py`** - 인간 피드백 시뮬레이터
```python
from gpro import HumanFeedbackSimulator

simulator = HumanFeedbackSimulator()

# 전문가 피드백 생성
feedback = simulator.generate_expert_feedback(
    candidate_info, position, ai_recommendation
)

print(f"전문가: {feedback.expert_id}")
print(f"선호도 강도: {feedback.preference_strength:.3f}")
```

**시뮬레이션 특징:**
- 🎭 **실제적 편향 모델링**: 경험편향, 학력편향, 안정성편향
- 📊 **선호도 강도 계산**: AI와 전문가 의견 차이 기반
- 🔄 **다양한 전문가 프로필**: 서로 다른 평가 기준

### 4. **`preference_optimizer.py`** - DPO 최적화 엔진
```python
from gpro import PreferenceOptimizer, OptimizationConfig, train_gpro_with_feedback

# 설정
config = OptimizationConfig(
    learning_rate=1e-5,
    batch_size=8,
    beta=0.1  # DPO temperature
)

# 피드백 데이터로 학습
model, history = train_gpro_with_feedback(
    model, feedback_data, config
)
```

**DPO 핵심 공식:**
```
L = -log(σ(β * (log π(y_preferred|x) - log π(y_rejected|x))))
```

### 5. **`comparison_study.py`** - BERT vs GPRO 성능 비교
```python
from gpro import ModelComparison

comparison = ModelComparison(gpro_model)

# 테스트 케이스 추가
comparison.add_test_case(candidate_info, position, ground_truth)

# 포괄적 비교 실행
results = comparison.run_comprehensive_comparison()

# 결과 시각화
comparison.visualize_comparison(results, "comparison_report.png")
```

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 의존성 설치
pip install torch transformers openai numpy matplotlib seaborn

# OpenAI API 키 설정
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 기본 사용법
```python
#!/usr/bin/env python3

from gpro import (
    initialize_gpro_model,
    HumanFeedbackSimulator,
    OpenAIValidator,
    ModelComparison,
    create_sample_candidates
)

# 1. GPRO 모델 초기화
print("🎯 GPRO 모델 초기화...")
model = initialize_gpro_model()

# 2. 테스트 후보자
candidate = """
김AI는 서울대학교 컴퓨터공학 박사로, 
구글에서 5년간 AI 연구원으로 근무했습니다.
자연어처리 분야 논문 50편, 정부 AI 자문위원 경험.
"""

# 3. GPRO 예측
print("\n🔮 GPRO 예측 실행...")
result = model.predict_with_explanation(candidate, "AI정책관")

print(f"추천 결과: {result['prediction']}")
print(f"신뢰도: {result['confidence']:.3f}")
print(f"헌법적 점수: {result['constitutional_score']:.3f}")
print(f"주요 요소: {result['reasoning']['top_factor']}")

# 4. OpenAI 검증 (API 키가 있는 경우)
if os.getenv("OPENAI_API_KEY"):
    print("\n🤖 OpenAI 검증 실행...")
    validator = OpenAIValidator()
    validation = validator.validate_recommendation(
        candidate, "AI정책관", result
    )
    print(f"검증 점수: {validation.overall_score:.1f}/10")
    print(f"전문가 추천: {validation.recommendation}")

# 5. 성능 비교
print("\n📊 BERT vs GPRO 비교...")
comparison = ModelComparison(model)
comparison.add_test_case(candidate, "AI정책관", "추천")

compare_results = comparison.run_comprehensive_comparison()
print("비교 결과:")
for rec in compare_results['recommendations']:
    print(f"  {rec}")

print("\n✅ 빠른 시작 완료!")
```

### 3. 고급 사용법 - 전체 학습 파이프라인
```python
#!/usr/bin/env python3
"""
완전한 GPRO 학습 파이프라인 예제
"""

from gpro import *

# 1. 피드백 데이터 생성
print("👥 인간 피드백 데이터 생성...")
simulator = HumanFeedbackSimulator()
candidates = create_sample_candidates()

feedback_data = []
for i in range(50):  # 50개 피드백 생성
    candidate = candidates[i % len(candidates)]
    
    # 가상 AI 추천
    ai_rec = model.predict_with_explanation(
        candidate['info'], candidate['position']
    )
    
    # 전문가 피드백
    feedback = simulator.generate_expert_feedback(
        candidate['info'], candidate['position'], ai_rec
    )
    feedback_data.append(feedback)

print(f"피드백 데이터 {len(feedback_data)}건 생성 완료")

# 2. GPRO 모델 학습
print("\n⚡ GPRO 모델 학습...")
config = OptimizationConfig(
    learning_rate=1e-5,
    batch_size=4,
    num_epochs=2,
    beta=0.1
)

trained_model, history = train_gpro_with_feedback(
    model, feedback_data, config, validation_split=0.2
)

print("학습 완료!")
print(f"최종 손실: {history['train_loss'][-1]:.4f}")

# 3. 성능 평가
print("\n📊 성능 평가...")
comparison = ModelComparison(trained_model)

# 테스트 케이스 추가
test_cases = [
    ("김박사는 MIT 졸업 후 애플에서 10년 근무", "AI정책관"),
    ("이학사는 지방대 졸업 후 중소기업 2년", "데이터과학자"),
    ("박정책은 행정고시 출신 15년 공무원", "디지털정책관")
]

for candidate, position in test_cases:
    comparison.add_test_case(candidate, position)

results = comparison.run_comprehensive_comparison()

# 4. 결과 리포트
print("\n📋 최종 결과:")
for recommendation in results['recommendations']:
    print(f"  {recommendation}")

# 개선율 출력
improvements = results['comparison_metrics']['improvements']
print(f"\n📈 핵심 개선율:")
print(f"  정확도: +{improvements['accuracy_improvement']:.1f}%")
print(f"  전문가 만족도: +{improvements['expert_satisfaction_improvement']:.1f}%")
print(f"  설명 품질: +{improvements['explanation_quality_gain']*100:.1f}%")

print("\n🎉 전체 파이프라인 완료!")
```

---

## 📊 예상 성능 개선

| 지표 | 기존 BERT | GPRO | 개선율 |
|------|-----------|------|--------|
| **전문가 만족도** | 72.5% | **89.1%** | +22.9% |
| **설명 품질** | 40% | **90%** | +125% |
| **편향 감소** | 35% | **15%** | -57% |
| **헌법 준수** | 60% | **85%** | +42% |
| **신뢰도** | 75% | **87%** | +16% |

---

## 🔬 연구 질문과 가설 검증

### 🤔 핵심 연구 질문
1. **"RLHF/GPRO가 정말 BERT보다 나은가?"**
2. **"인간 피드백이 AI 성능 향상에 기여하는가?"**
3. **"Constitutional AI가 편향을 줄이는가?"**
4. **"전문가 만족도가 실제로 높아지는가?"**

### 🧪 실험 설계
```python
# 실험 1: BERT vs GPRO 직접 비교
experiment_1 = ModelComparison(gpro_model)
experiment_1.run_comprehensive_comparison()

# 실험 2: 피드백 양에 따른 성능 변화
for feedback_size in [10, 50, 100, 500]:
    model_variant = train_with_different_feedback_size(feedback_size)
    evaluate_performance(model_variant)

# 실험 3: Constitutional AI 효과 측정
constitutional_enabled = GPROModel(config_with_constitutional=True)
constitutional_disabled = GPROModel(config_with_constitutional=False)
compare_bias_levels(constitutional_enabled, constitutional_disabled)
```

### 📈 성과 측정 지표
- **정량적 지표**: 정확도, 정밀도, 재현율, F1 점수
- **정성적 지표**: 전문가 만족도, 설명 품질, 편향 수준
- **효율성 지표**: 추론 시간, 메모리 사용량, 학습 시간
- **신뢰성 지표**: 일관성, 헌법 준수도, 안전성

---

## 🎯 실전 활용 시나리오

### 시나리오 1: 정부 인사담당자
```python
# 실제 채용 상황에서 GPRO 활용
candidate_pool = load_candidates_from_database()

for candidate in candidate_pool:
    # GPRO 추천
    recommendation = model.predict_with_explanation(
        candidate.resume, candidate.applied_position
    )
    
    # 전문가 검증
    validation = validator.validate_recommendation(
        candidate.resume, candidate.applied_position, recommendation
    )
    
    # 최종 의사결정 지원
    final_decision = make_decision(recommendation, validation)
    
    print(f"{candidate.name}: {final_decision}")
```

### 시나리오 2: AI 연구자
```python
# 새로운 피드백 데이터로 모델 개선
new_feedback = collect_real_expert_feedback()

# 지속적 학습
improved_model = continuous_learning(
    current_model, new_feedback
)

# A/B 테스트
ab_test_results = run_ab_test(
    current_model, improved_model, test_cases
)
```

### 시나리오 3: 정책 결정자
```python
# 정책적 편향 분석
bias_analysis = analyze_systematic_bias(
    model, demographic_groups
)

# 공정성 개선 방안
fairness_improvements = suggest_fairness_improvements(
    bias_analysis
)
```

---

## 🚧 제한사항 및 고려사항

### ⚠️ 현재 제한사항
1. **데이터 의존성**: 고품질 인간 피드백 데이터 필요
2. **계산 비용**: BERT 대비 추론 시간 1.5-2배
3. **복잡성**: 구현 및 유지보수 복잡도 증가
4. **검증 필요**: 실제 환경에서의 추가 검증 필요

### 🔮 향후 개선 방향
1. **효율성 최적화**: 모델 경량화 및 추론 속도 개선
2. **다국어 지원**: 영어 및 기타 언어 지원
3. **도메인 확장**: 다른 분야로의 적용 확대
4. **실시간 학습**: 온라인 피드백 기반 실시간 모델 업데이트

---

## 📚 참고 자료

### 🎓 핵심 논문
- **InstructGPT**: "Training language models to follow instructions with human feedback"
- **Constitutional AI**: "Constitutional AI: Harmlessness from AI Feedback"
- **DPO**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"

### 🔗 관련 프로젝트
- **OpenAI Alignment**: https://openai.com/alignment
- **Anthropic Constitutional AI**: https://www.anthropic.com/constitutional-ai
- **Hugging Face TRL**: https://github.com/huggingface/trl

### 📖 추천 자료
- section4_rlhf_gpro.md 강의 슬라이드
- TRAS 프로젝트 문서
- Constitutional AI 가이드라인

---

## 🤝 기여하기

### 🛠️ 개발 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd gpro

# 개발 의존성 설치
pip install -r requirements-dev.txt

# 테스트 실행
python -m pytest tests/

# 코드 품질 검사
flake8 gpro/
black gpro/
```

### 📝 기여 가이드라인
1. **이슈 확인**: 기여하기 전 관련 이슈 확인
2. **브랜치 생성**: feature/your-feature-name
3. **테스트 작성**: 새 기능에 대한 테스트 추가
4. **문서 업데이트**: README 및 코드 주석 업데이트
5. **Pull Request**: 상세한 설명과 함께 PR 생성

---

## 📞 문의 및 지원

### 💬 문의 채널
- **이메일**: admin@barion.ai
- **이슈 트래커**: GitHub Issues
- **토론**: GitHub Discussions

### 🆘 자주 묻는 질문

**Q: OpenAI API 없이도 사용 가능한가요?**
A: 네, 기본 GPRO 모델은 OpenAI 없이 동작합니다. 검증 기능만 API가 필요합니다.

**Q: 실제 정부 프로젝트에 바로 적용 가능한가요?**
A: 이것은 연구 프로토타입입니다. 실제 적용을 위해서는 추가 검증과 보안 검토가 필요합니다.

**Q: 다른 도메인에도 적용할 수 있나요?**
A: 기본 구조는 재사용 가능하지만, 도메인별 전문가 프로필과 평가 기준을 새로 정의해야 합니다.

---

## 📄 라이선스

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.

---

## 🎉 마치며

이 프로젝트는 **"AI가 인간을 대체하는 것이 아니라, 인간과 협력하여 더 나은 결정을 내리는 도구"**라는 철학을 구현한 것입니다.

RLHF/GPRO 방법론을 통해 AI가 단순한 패턴 매칭을 넘어서 인간의 가치와 정렬된 지능적인 판단을 할 수 있음을 보여주고자 합니다.

**함께 더 좋은 AI를 만들어 나가요!** 🚀

---

*© 2025 BarionLabs. Built with ❤️ for better AI-human collaboration.* 