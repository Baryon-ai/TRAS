---
marp: true
theme: default
class: lead
paginate: true
backgroundColor: #f8f9fa
---

# 🚀 AI 기반 인재 추천 시스템으로 배우는
## 자연어처리와 BERT, 그리고 RLHF vs GPRO

### TRAS 오픈소스 프로젝트를 중심으로

**3시간 특강**
*2025년 6월 14일*

---

## 👨‍🏫 강사 소개

- **AI 연구팀**
- **TRAS (Talent Recommendation Analysis System) 개발자**
- **자연어처리 및 대화형 AI 전문가**

### 📧 연락처
- GitHub: [Baryon-ai/TRAS](https://github.com/Baryon-ai/TRAS)
- Email: ai-team@example.com

---

## 🎯 학습 목표

이 특강을 통해 여러분은 다음을 이해하게 됩니다:

1. **🏗️ 실무 AI 시스템**: TRAS 오픈소스 프로젝트의 구조와 설계 철학
2. **📝 자연어처리 기초**: 토큰화부터 어텐션 메커니즘까지
3. **🧠 BERT 모델**: Transformer 아키텍처와 사전훈련의 원리
4. **🎛️ 고급 최적화**: RLHF와 GPRO의 차이점과 응용

---

## ⏰ 특강 일정 (180분)

| 시간 | 주제 | 내용 |
|------|------|------|
| **10분** | 🏗️ TRAS 소개 | 오픈소스 프로젝트 개요 |
| **50분** | 📝 자연어처리 기초 | 토큰화, 임베딩, 어텐션 |
| **60분** | 🧠 BERT 깊이 이해 | 구조, 사전훈련, 파인튜닝 |
| **50분** | 🎛️ RLHF vs GPRO | 인간피드백 vs 직접최적화 |
| **10분** | 🎯 정리 및 퀴즈 | 핵심 개념 점검 |

---

## 🛠️ 실습 환경

### 필요한 도구들
```bash
# TRAS 프로젝트 클론
git clone https://github.com/Baryon-ai/TRAS.git
cd TRAS

# UV로 환경 구축
uv sync --extra ai

# 주요 라이브러리
# - transformers (BERT 모델)
# - torch (딥러닝 프레임워크)
# - pandas (데이터 처리)
# - ollama (로컬 LLM)
```

---

## 🌟 왜 TRAS로 시작하는가?

### 📊 실무적 관점
- **정부 인재 추천**: 실제 도메인 문제
- **멀티모달 분석**: 이메일 + 소셜미디어
- **AI 통합**: Ollama, OpenAI, Claude 지원

### 🎓 교육적 관점
- **완전한 파이프라인**: 데이터 → 전처리 → 모델 → 결과
- **확장 가능한 구조**: 새로운 AI 모델 쉽게 추가
- **현대적 개발**: UV, Python 3.8+, 타입 힌트

---

## 🧭 메타인지 체크포인트

각 섹션마다 다음을 확인해보세요:

### 🤔 이해도 점검
- "이 개념을 동료에게 설명할 수 있는가?"
- "실제 코드에서 어떻게 구현되는지 알고 있는가?"
- "왜 이 방법을 선택했는지 이유를 알고 있는가?"

### 💡 연결성 사고
- "이전 개념과 어떻게 연결되는가?"
- "실무에서 어떻게 활용할 수 있는가?"
- "한계점과 개선 방향은 무엇인가?"

---

## 📚 학습 철학: "코끼리 만지기"

### 🐘 전체 → 부분 → 통합

1. **🌍 Big Picture**: TRAS 시스템 전체 보기
2. **🔍 Deep Dive**: 각 구성요소 자세히 탐구
3. **🔗 Integration**: 모든 것이 어떻게 연결되는지 이해

### 💭 비유적 학습
- **수식**: 기하학적 직관으로 이해
- **코드**: 요약과 핵심 포인트
- **이론**: 일상적 비유로 설명

---

## 🎪 Section 1: TRAS 프로젝트 소개

### 10분 개요
- 프로젝트 구조와 설계 철학
- AI 기반 인재 추천의 실무적 의미
- 코드 아키텍처 살펴보기

---

## 📝 Section 2: 자연어처리 기초

### 50분 심화
- 토큰화: 언어를 컴퓨터가 이해하는 방식
- 임베딩: 단어를 벡터 공간에 매핑하기
- 어텐션: "집중"의 수학적 모델링
- 실습: TRAS의 텍스트 전처리 파이프라인

---

## 🧠 Section 3: BERT 깊이 이해

### 60분 완전정복
- Transformer 아키텍처: 어텐션만으로 모든 것을
- BERT의 혁신: 양방향 인코더 표현
- 사전훈련과 파인튜닝: 전이학습의 실제
- 실습: TRAS에서 BERT 모델 사용하기

---

## 🎛️ Section 4: RLHF vs GPRO

### 50분 최신 기술
- RLHF: 인간 피드백으로 학습하기
- GPRO: 직접 선호도 최적화
- 실무 적용: TRAS에서의 AI 품질 향상
- 실습: 피드백 시스템 구현하기

---

## 🎯 Section 5: 정리 및 퀴즈

### 10분 마무리
- 핵심 개념 요약
- 용어 퀴즈 (20문항)
- 추가 학습 자료 안내
- Q&A

---

## 📖 참고 자료

### 📚 추천 도서
- "Natural Language Processing with Python" (NLTK Book)
- "Attention Is All You Need" (Original Transformer Paper)
- "BERT: Pre-training of Deep Bidirectional Transformers"

### 🌐 온라인 리소스
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Papers With Code](https://paperswithcode.com/)
- [Distill.pub](https://distill.pub/) - 시각적 설명

---

## ⚡ 실습 준비

### 지금 바로 시작해보기

```bash
# 1. 프로젝트 클론
git clone https://github.com/Baryon-ai/TRAS.git

# 2. 의존성 설치
cd TRAS
uv sync --extra ai

# 3. 샘플 실행
uv run python main.py
```

### 🔍 미리 살펴볼 파일들
- `main.py`: 메인 실행 파일
- `email_analyzer.py`: 이메일 분석 모듈
- `scripts/`: 버전 관리 및 릴리스 도구

---

## 🚀 시작합니다!

### 준비되셨나요?

다음 슬라이드부터 본격적인 여행이 시작됩니다.
**TRAS 오픈소스 프로젝트**와 함께 
**자연어처리의 세계**로 떠나보겠습니다!

---

<!-- 여기서부터 각 섹션별 슬라이드가 이어집니다 --> 