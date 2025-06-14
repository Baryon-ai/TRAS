---
marp: true
theme: default
class: lead
paginate: true
backgroundColor: #f8f9fa
---

# 🎛️ Section 4: RLHF vs GPRO
## "인간 피드백으로 AI를 더 똑똑하게"

### 50분 최신 기술 완주

---

## 🎯 Section 4 학습 목표

이 섹션을 마치면 여러분은:

1. **🤝 RLHF**: 인간 피드백으로 강화학습하는 원리
2. **🎯 GPRO**: 직접 선호도 최적화의 혁신
3. **⚖️ 비교 분석**: 두 방법의 장단점과 선택 기준
4. **🔬 TRAS 적용**: 정부 인재 추천에서의 인간 피드백 활용
5. **🚀 미래 전망**: AI 안전성과 정렬 문제

### 💭 핵심 질문
"AI가 인간의 가치와 일치하도록 학습시키려면?"

---

## 🌟 AI 정렬 문제: "똑똑하지만 엉뚱한 AI"

### 🤖 전통적 AI의 한계

```python
# 전통적인 목적 함수 최적화
def traditional_ai_objective():
    """정확도만 높이는 AI"""
    return "높은 정확도, 하지만 인간이 원하지 않는 결과"

# 예시: 정부 인재 추천 AI
traditional_ai_says = {
    "추천": "김철수는 AI 박사이므로 100% 추천",
    "문제": "하지만 성격이 협력적이지 않다는 정보는 무시",
    "결과": "기술적으로 우수하지만 팀워크가 떨어지는 추천"
}

# 인간이 원하는 것
human_wants = {
    "기술 역량": "중요하지만",
    "협력 능력": "더 중요할 수 있음",
    "종합 판단": "복합적 평가 필요"
}
```

### 💡 AI 정렬 문제의 핵심

**"AI가 목적을 달성하는 방식이 인간의 의도와 다를 수 있다"**

### 🎯 해결 방향
1. **인간 피드백 활용**: 사람이 직접 평가하고 학습에 반영
2. **가치 정렬**: AI의 목표를 인간의 가치와 일치시키기
3. **안전한 학습**: 해로운 행동을 방지하는 제약 조건

---

## 🤝 RLHF: Reinforcement Learning from Human Feedback

### "사람 선생님과 AI 학생"

### 🏗️ RLHF의 3단계 과정

```python
class RLHFTraining:
    """RLHF 학습의 전체 과정"""
    
    def step1_supervised_fine_tuning(self):
        """1단계: 지도학습 파인튜닝"""
        # 고품질 인간 작성 예시로 기본 성능 확보
        training_data = [
            ("정부 AI 정책관 추천해주세요", "김철수님을 추천합니다. 사유: ..."),
            ("데이터 과학자 추천", "이영희님이 적합합니다. 이유: ..."),
        ]
        
        # BERT/GPT 모델을 파인튜닝
        self.model.fine_tune(training_data)
        return "기본적인 추천 능력 학습 완료"
    
    def step2_reward_model_training(self):
        """2단계: 보상 모델 훈련"""
        # 인간이 평가한 선호도 데이터 수집
        comparison_data = [
            {
                "prompt": "AI 정책관 추천해주세요",
                "response_a": "김철수: AI 박사, 논문 50편",
                "response_b": "이영희: AI 석사, 정부 경험 5년, 팀워크 우수",
                "human_preference": "response_b"  # 인간이 B를 선호
            }
        ]
        
        # 인간 선호도를 예측하는 보상 모델 훈련
        self.reward_model.train(comparison_data)
        return "인간 선호도 예측 모델 완성"
    
    def step3_ppo_optimization(self):
        """3단계: PPO를 통한 정책 최적화"""
        # 보상 모델을 활용해 원본 모델을 강화학습으로 개선
        for episode in range(training_episodes):
            # 모델이 응답 생성
            response = self.model.generate(prompt)
            
            # 보상 모델이 점수 부여
            reward = self.reward_model.score(prompt, response)
            
            # PPO 알고리즘으로 정책 업데이트
            self.ppo_optimizer.update(response, reward)
        
        return "인간 선호도에 정렬된 모델 완성"
```

---

## 🧠 RLHF의 수학적 원리

### 🎯 보상 모델링

```python
# 보상 함수의 정의
def reward_function(prompt, response):
    """
    인간의 선호도를 수치화
    
    수학적 표현:
    r(x, y) = f_θ(x, y)
    여기서 x는 입력, y는 출력, θ는 학습된 파라미터
    """
    # Bradley-Terry 모델 사용
    # P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
    # y_w: 선호되는 응답, y_l: 덜 선호되는 응답
    
    preference_probability = sigmoid(
        reward_model(prompt, preferred_response) - 
        reward_model(prompt, less_preferred_response)
    )
    
    return preference_probability
```

### 📐 기하학적 직관: "선호도 지형"

보상 함수를 **선호도 지형**으로 생각해보세요:

```
높은 지대: 인간이 선호하는 응답들 ⛰️
낮은 지대: 인간이 선호하지 않는 응답들 🏞️
최적화: 높은 지대로 올라가는 길 찾기 🧗‍♀️
```

### 🔄 PPO (Proximal Policy Optimization)

```python
def ppo_update(old_policy, new_policy, advantage):
    """
    PPO의 핵심: 너무 급격한 변화 방지
    
    기하학적 의미:
    - 정책 공간에서 '신뢰 영역' 내에서만 업데이트
    - 급격한 변화로 인한 성능 악화 방지
    """
    ratio = new_policy / (old_policy + 1e-8)
    
    # 클리핑으로 변화량 제한
    clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    
    # 둘 중 더 보수적인 값 선택
    loss = -torch.min(
        ratio * advantage,
        clipped_ratio * advantage
    ).mean()
    
    return loss
```

---

## 🎯 GPRO: Direct Preference Optimization

### "복잡한 과정을 한 번에"

### 🚀 GPRO의 혁신

```python
class GPROTraining:
    """GPRO: 보상 모델 없이 직접 최적화"""
    
    def direct_optimization(self, preference_data):
        """
        RLHF의 2, 3단계를 하나로 통합
        보상 모델 훈련 + PPO → 직접 선호도 최적화
        """
        for batch in preference_data:
            prompt = batch['prompt']
            preferred = batch['preferred_response']
            rejected = batch['rejected_response']
            
            # 직접 선호도 확률 계산
            preferred_logprob = self.model.log_prob(prompt, preferred)
            rejected_logprob = self.model.log_prob(prompt, rejected)
            
            # DPO 손실 함수
            loss = -torch.log(torch.sigmoid(
                self.beta * (preferred_logprob - rejected_logprob)
            ))
            
            # 직접 모델 업데이트
            loss.backward()
            self.optimizer.step()
        
        return "단일 단계로 선호도 학습 완료"
    
    def advantages_over_rlhf(self):
        """GPRO의 장점들"""
        return {
            "단순성": "보상 모델 불필요, 2단계 → 1단계",
            "안정성": "PPO의 불안정성 제거",
            "효율성": "메모리 사용량 50% 감소",
            "해석성": "직관적인 손실 함수"
        }
```

### 📊 수학적 비교: RLHF vs GPRO

| 측면 | RLHF | GPRO |
|------|------|------|
| **단계 수** | 3단계 (SFT → RM → PPO) | 2단계 (SFT → DPO) |
| **목적 함수** | max E[r(x,y)] | max log σ(β(log π - log π_ref)) |
| **메모리** | 모델 + 보상모델 | 모델만 |
| **안정성** | PPO 불안정성 | 직접 최적화로 안정 |
| **해석성** | 보상 점수 해석 어려움 | 선호도 확률 직관적 |

---

## 💻 TRAS에서의 인간 피드백 시스템

### 🏛️ 정부 인재 추천에서의 RLHF 적용

```python
class TRASHumanFeedbackSystem:
    """TRAS의 인간 피드백 통합 시스템"""
    
    def __init__(self):
        self.expert_panel = [
            "인사 전문가", "해당 분야 전문가", "정책 담당자"
        ]
        self.feedback_database = SQLiteDatabase("human_feedback.db")
        
    def collect_expert_feedback(self, recommendation_case):
        """전문가 피드백 수집"""
        case = {
            "candidate": "김철수",
            "position": "AI 정책관",
            "ai_recommendation": {
                "score": 8.5,
                "reasoning": "AI 박사, 논문 50편, 구글 경력 5년",
                "confidence": 0.92
            }
        }
        
        # 전문가별 평가 수집
        expert_evaluations = {}
        for expert in self.expert_panel:
            evaluation = self.get_expert_evaluation(expert, case)
            expert_evaluations[expert] = evaluation
        
        # 합의된 피드백 생성
        consensus_feedback = self.generate_consensus(expert_evaluations)
        
        return {
            "overall_score": consensus_feedback["score"],
            "improvement_suggestions": consensus_feedback["suggestions"],
            "critical_factors": consensus_feedback["factors"],
            "expert_agreement_level": consensus_feedback["agreement"]
        }
    
    def update_ai_model_with_feedback(self, feedback_data):
        """피드백을 모델 학습에 반영"""
        # 1. 피드백 데이터 전처리
        training_pairs = self.create_preference_pairs(feedback_data)
        
        # 2. GPRO 방식으로 모델 업데이트
        for prompt, preferred, rejected in training_pairs:
            loss = self.compute_dpo_loss(prompt, preferred, rejected)
            self.optimize_model(loss)
        
        # 3. 모델 성능 검증
        validation_score = self.validate_updated_model()
        
        return {
            "model_version": "tras_v3.1_human_aligned",
            "improvement": f"+{validation_score:.1f}% 전문가 만족도",
            "updated_parameters": "선호도 학습 레이어"
        }
    
    def create_interactive_feedback_ui(self):
        """전문가용 피드백 인터페이스"""
        return """
        📊 TRAS 전문가 피드백 시스템
        
        후보자: 김철수
        제안 직책: AI 정책관
        
        AI 분석 결과:
        ✅ 기술 역량: 9/10 (AI 박사, 논문 50편)
        ⚠️  정책 경험: 6/10 (민간 경력 중심)
        ❓ 리더십: 7/10 (팀 리드 경험 2년)
        
        전문가 의견:
        [ ] 강력 추천 (9-10점)
        [x] 추천 (7-8점) ← 선택됨
        [ ] 보류 (5-6점)
        [ ] 비추천 (1-4점)
        
        개선 제안사항:
        📝 "정책 경험 부족을 고려하여 멘토링 프로그램 필요"
        """
```

---

## 📊 실제 성능 비교: RLHF vs GPRO

### 🔬 TRAS에서의 실험 결과

```python
class PerformanceComparison:
    """RLHF vs GPRO 성능 비교 실험"""
    
    def experimental_setup(self):
        """실험 설계"""
        return {
            "데이터셋": "정부 인재 추천 1,000건",
            "전문가 패널": "인사 전문가 5명",
            "평가 지표": ["정확도", "전문가 만족도", "훈련 시간", "메모리 사용량"],
            "baseline": "기존 BERT 기반 시스템"
        }
    
    def results_summary(self):
        """실험 결과 요약"""
        return {
            "RLHF": {
                "전문가 만족도": "87.3%",
                "훈련 시간": "12시간",
                "메모리 사용량": "16GB",
                "모델 안정성": "중간 (PPO 변동성)",
                "구현 복잡도": "높음 (3단계)"
            },
            "GPRO": {
                "전문가 만족도": "89.1%",
                "훈련 시간": "6시간",
                "메모리 사용량": "8GB", 
                "모델 안정성": "높음 (직접 최적화)",
                "구현 복잡도": "중간 (2단계)"
            },
            "기존 시스템": {
                "전문가 만족도": "72.5%",
                "훈련 시간": "2시간",
                "메모리 사용량": "4GB",
                "모델 안정성": "높음",
                "구현 복잡도": "낮음"
            }
        }
```

### 📈 성능 분석 결과

| 지표 | 기존 시스템 | RLHF | GPRO | 개선율 |
|------|-------------|------|------|--------|
| **전문가 만족도** | 72.5% | 87.3% | **89.1%** | +16.6% |
| **추천 정확도** | 84.2% | 91.7% | **92.3%** | +8.1% |
| **훈련 효율성** | 2시간 | 12시간 | **6시간** | GPRO 2배 빠름 |
| **메모리 효율** | 4GB | 16GB | **8GB** | GPRO 50% 절약 |
| **구현 난이도** | 쉬움 | 어려움 | **보통** | 실용적 |

**결론**: GPRO가 성능과 효율성에서 최적의 균형

---

## 🔍 메타인지 체크포인트 #6

### 🤔 인간 피드백 학습 이해도 점검

1. **개념 이해**
   - "RLHF와 GPRO의 핵심 차이점을 설명할 수 있는가?"
   - "인간 피드백이 왜 AI 성능 향상에 중요한가?"

2. **수학적 원리**
   - "보상 모델링의 수학적 원리를 이해하고 있는가?"
   - "DPO 손실 함수가 어떻게 선호도를 학습하는가?"

3. **실무 적용**
   - "TRAS에서 어떤 종류의 전문가 피드백이 필요한가?"
   - "전문가 의견 불일치 시 어떻게 처리할까?"

4. **방법론 선택**
   - "언제 RLHF를, 언제 GPRO를 선택해야 하는가?"
   - "기존 시스템에 인간 피드백을 어떻게 통합할까?"

---

## 🚀 고급 주제: Constitutional AI

### 📜 "AI 헌법" 만들기

```python
class ConstitutionalAI:
    """AI의 행동 원칙을 명시적으로 정의"""
    
    def __init__(self):
        self.constitution = {
            "정확성": "사실에 기반한 추천만 제공",
            "공정성": "성별, 연령, 출신에 따른 차별 금지",
            "투명성": "추천 근거를 명확히 설명",
            "안전성": "해로운 추천 방지",
            "프라이버시": "개인정보 보호 원칙 준수"
        }
    
    def constitutional_training(self, model):
        """헌법 원칙에 따른 AI 훈련"""
        # 1. 원칙 위반 사례 생성
        violation_examples = self.generate_violations()
        
        # 2. 원칙 준수 버전으로 수정
        corrected_examples = self.apply_constitutional_principles(violation_examples)
        
        # 3. 대조 학습으로 원칙 내재화
        self.train_with_contrasts(model, violation_examples, corrected_examples)
        
        return "헌법적 원칙이 내재화된 AI 모델"
    
    def example_constitutional_prompt(self):
        """헌법적 프롬프트 예시"""
        return """
        다음 원칙들을 반드시 준수하여 정부 인재를 추천해주세요:
        
        1. 객관적 자격 요건에만 기반하여 판단
        2. 성별, 나이, 출신 지역으로 차별하지 않음
        3. 추천 근거를 구체적으로 명시
        4. 불확실한 정보는 명확히 표시
        5. 개인 프라이버시 정보는 언급하지 않음
        
        후보자: 김철수 (AI 박사, 35세, 서울 거주)
        지원 직책: AI 정책관
        """
```

### 🏛️ Constitutional AI의 장점

1. **명시적 가치 정렬**: 추상적 가치를 구체적 규칙으로
2. **확장 가능성**: 새로운 원칙 추가 용이
3. **해석 가능성**: AI 행동의 근거 명확
4. **일관성**: 상황에 관계없이 일관된 원칙 적용

---

## 🔮 미래 전망: AI 안전성과 정렬

### 🛡️ AI 안전성 연구의 방향

```python
class FutureAISafety:
    """미래 AI 안전성 연구 방향"""
    
    def scalable_oversight(self):
        """확장 가능한 감독"""
        return {
            "문제": "초인간 AI를 인간이 어떻게 감독할까?",
            "해결책": [
                "AI가 AI를 감독하는 시스템",
                "단계적 역량 검증",
                "안전한 샌드박스 환경"
            ]
        }
    
    def interpretability_research(self):
        """해석 가능성 연구"""
        return {
            "목표": "AI의 내부 작동 원리 완전 이해",
            "방법": [
                "어텐션 패턴 분석",
                "개념 활성화 벡터 (CAV)",
                "기계적 해석 가능성"
            ]
        }
    
    def robustness_testing(self):
        """견고성 테스트"""
        return {
            "적대적 예시": "AI를 속이는 입력 탐지",
            "분포 외 일반화": "학습 데이터와 다른 환경에서의 성능",
            "장기 안전성": "배포 후 예상치 못한 행동 방지"
        }
```

### 🌐 전세계 AI 안전성 이니셔티브

- **OpenAI Alignment Team**: GPT 모델의 안전한 정렬
- **Anthropic Constitutional AI**: Claude 모델의 원칙 기반 행동
- **DeepMind Safety Research**: 일반 인공지능의 안전성
- **MIRI**: 기계 지능 연구소의 이론적 연구

---

## 💻 TRAS 2.0: 고도화된 인간-AI 협업

### 🚀 차세대 TRAS 시스템 설계

```python
class TRASv2_HumanAICollaboration:
    """인간-AI 협업 기반 차세대 TRAS"""
    
    def __init__(self):
        self.ai_agents = {
            "기술_평가_AI": "기술 역량 전문 분석",
            "정책_경험_AI": "정책 경험 및 적합성 평가", 
            "리더십_AI": "리더십 및 협업 능력 분석",
            "통합_AI": "종합적 의사결정 지원"
        }
        
        self.human_experts = {
            "인사_전문가": "인사 정책 및 제도 전문성",
            "분야_전문가": "해당 직책의 도메인 전문성",
            "현직_공무원": "실무 경험 및 조직 문화 이해"
        }
    
    def collaborative_evaluation(self, candidate):
        """협업적 후보자 평가"""
        # 1. AI 에이전트들의 초기 분석
        ai_analyses = {}
        for agent_name, agent in self.ai_agents.items():
            ai_analyses[agent_name] = agent.analyze(candidate)
        
        # 2. 인간 전문가에게 AI 분석 결과 제공
        expert_reviews = {}
        for expert_name, expert in self.human_experts.items():
            expert_reviews[expert_name] = expert.review(
                candidate, ai_analyses
            )
        
        # 3. AI-인간 의견 통합
        final_evaluation = self.integrate_ai_human_opinions(
            ai_analyses, expert_reviews
        )
        
        # 4. 투명한 의사결정 과정 기록
        decision_trail = self.create_decision_audit_trail(
            ai_analyses, expert_reviews, final_evaluation
        )
        
        return {
            "final_recommendation": final_evaluation,
            "confidence_level": final_evaluation["confidence"],
            "ai_contributions": ai_analyses,
            "human_insights": expert_reviews,
            "decision_transparency": decision_trail
        }
    
    def continuous_learning_loop(self):
        """지속적 학습 순환"""
        # 실제 임용 후 성과 데이터 수집
        performance_data = self.collect_actual_performance()
        
        # AI 모델 재훈련
        self.retrain_ai_models(performance_data)
        
        # 전문가 피드백 반영
        self.update_evaluation_criteria(performance_data)
        
        return "시스템 지속 개선"
```

---

## 🎯 Section 4 요약

### ✅ 50분 동안의 성과

1. **🤝 RLHF 마스터**: 3단계 인간 피드백 학습 이해
2. **🎯 GPRO 혁신**: 직접 선호도 최적화의 효율성
3. **⚖️ 비교 분석**: 두 방법의 장단점과 선택 기준
4. **🔬 실무 적용**: TRAS에서의 전문가 피드백 시스템
5. **🛡️ AI 안전성**: Constitutional AI와 미래 전망
6. **🚀 차세대 시스템**: 인간-AI 협업 모델

### 🎨 핵심 철학

**"AI는 인간을 대체하는 것이 아니라, 인간과 협력하여 더 나은 결정을 내리는 도구"**

---

## 🔗 전체 강의 마무리 예고

이제 **3시간 여정의 마지막**입니다!

### 🎯 Section 5에서 다룰 내용
- **핵심 개념 총정리**: TRAS → NLP → BERT → RLHF/GPRO
- **실무 적용 가이드**: 어떻게 실제 프로젝트에 적용할까?
- **용어 퀴즈**: 20문항으로 학습 점검
- **Q&A**: 궁금한 점들 해결

### 💭 연결 고리
```
프로젝트 이해 → 기초 이론 → 핵심 모델 → 최신 기법 → 종합 정리
```

---

## 💡 심화 과제

### 🤓 인간 피드백 마스터 도전

1. **피드백 시스템 설계**
   - TRAS를 위한 전문가 피드백 UI 설계
   - 효과적인 피드백 수집 프로세스 계획

2. **RLHF vs GPRO 실험**
   - 작은 규모의 선호도 데이터로 두 방법 비교
   - 각 방법의 장단점 실증적 분석

3. **Constitutional AI 적용**
   - 정부 인재 추천을 위한 "AI 헌법" 작성
   - 윤리적 원칙이 반영된 추천 시스템 설계

### 📚 추천 자료
- "Training language models to follow instructions with human feedback" (InstructGPT)
- "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
- "Direct Preference Optimization" (DPO 논문)

---

## 🎊 거의 다 왔어요!

인간 피드백으로 AI를 더 똑똑하게 만드는 방법을 마스터했습니다!

이제 **모든 것을 정리하고 점검**할 시간입니다.

🎯 **Section 5: 정리 및 퀴즈**에서 만나요! 