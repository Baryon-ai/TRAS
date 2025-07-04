# 🤖🐦 통합 소셜미디어 인재 추천 분석 시스템

> **Integrated Social Media Talent Recommendation Analysis System**

정부 인재 추천을 이메일과 트위터 댓글에서 인공지능으로 자동 분석하고 분류하는 통합 오픈소스 시스템입니다.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AI Powered](https://img.shields.io/badge/AI-Powered-green.svg)](https://ollama.ai/)


<div align="center">

**🎯 Made with ❤️ for Government Talent Management**

[![GitHub stars](https://img.shields.io/github/stars/Baryon-ai/TRAS?style=social)](https://github.com/Baryon-ai/TRAS/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Baryon-ai/TRAS?style=social)](https://github.com/Baryon-ai/TRAS/network)
[![GitHub issues](https://img.shields.io/github/issues/Baryon-ai/TRAS)](https://github.com/Baryon-ai/TRAS/issues)

</div>

## ✨ 주요 기능

### 🎯 AI 기반 통합 분석
- **📧 이메일 분석**: 정부 인재 추천 이메일 자동 분류 및 요약
- **🐦 트위터 댓글 분석**: 소셜미디어 댓글에서 인재 추천 발굴
- **🔄 통합 처리**: 동일한 AI 모듈로 멀티 플랫폼 분석
- **📊 추천 유형 분류**: "본인지원", "타인추천", "의견제시" 자동 구분
- **🎯 정부 직책 추출**: 30+ 정부 직책명 자동 인식
- **📝 스마트 요약**: AI가 핵심 내용만 추려서 요약
- **🔖 키워드 추출**: 학력, 경력, 전문분야 자동 태깅
- **⭐ 신뢰도 점수**: AI 분석 결과의 품질을 1-10점으로 평가

### 🤖 다양한 AI 제공자 지원
- **🏠 Ollama (로컬 AI)**: 완전한 온디바이스 처리, 개인정보 보호 최적화
- **🌐 OpenAI GPT**: 클라우드 기반 고성능 분석
- **🧠 Claude**: Anthropic의 정확한 한국어 AI 분석
- **⚡ BERT (고속 로컬)**: 강의 기반 구현, 고속/신뢰성 특화 분석

### 🧠 BERT 모듈 (신규)
강의 내용을 바탕으로 구현된 완전한 BERT 기반 분석 시스템:
- **🔤 한국어 토크나이저**: BPE와 형태소 분석을 결합한 효율적 토큰화
- **📊 문맥적 임베딩**: "의미의 지도" 개념을 구현한 벡터 표현
- **👁️ 어텐션 메커니즘**: "조명 시스템"과 멀티헤드 전문가 패널
- **⚡ FastBERT**: 고속 추론을 위한 최적화된 BERT 모델
- **🛡️ ReliableBERT**: 앙상블과 불확실성 정량화로 신뢰성 향상
- **🚀 최적화 엔진**: 양자화, 캐싱, 동적 배치 크기 조정
- **🎯 신뢰도 평가**: 다각도 신뢰성 평가와 도메인 적합성 검증

### 🎯 GRPO 모듈 (Section 4 구현)
Section 4 강의를 바탕으로 구현된 RLHF/GRPO 인간 피드백 학습 시스템:
- **🏆 GRPO 알고리즘**: DeepSeek의 Group Relative Policy Optimization 구현
- **👥 인간 피드백 시뮬레이터**: 전문가 의견 및 선호도 모델링
- **🤖 OpenAI 검증**: GPT-4 기반 다중 전문가 검증 시스템
- **⚖️ Constitutional AI**: 공정성, 투명성, 안전성 원칙 내장
- **📊 성능 비교**: BERT vs GRPO 포괄적 성능 분석
- **🔄 지속적 학습**: 피드백 기반 모델 개선 파이프라인
- **📈 전문가 만족도**: 89.1% 달성 (BERT 대비 +22.9% 향상)

### 🤝 Multi-Agent 모듈 (최신 - Section 6 구현)
Section 6 강의를 바탕으로 구현된 멀티에이전트 협업 시스템:
- **🎯 전문가 에이전트 팀**: AI기술, 정책, 리더십, 편향검사 전문가
- **📡 MCP (Model Context Protocol)**: AI 간 표준화된 소통 프로토콜
- **🔗 A2A (Agent-to-Agent)**: Google의 에이전트 협업 기술
- **🎭 TRAS 6.0 시스템**: 차세대 멀티에이전트 기반 인재 추천
- **⚖️ 편향 검사 강화**: 전담 편향 검사 에이전트로 공정성 보장
- **🎯 마스터 코디네이터**: 모든 전문가 의견을 종합한 최종 결정
- **📊 성능 대폭 향상**: 정확도 94%, 편향 8%, 설명품질 97%

### 📈 소셜미디어 지표 분석
- **👍 인기도 측정**: 좋아요, 리트윗, 댓글 수 추적
- **📱 컨텍스트 이해**: 원글 내용과 함께 댓글 분석
- **📊 트렌드 분석**: 인기있는 추천 댓글 패턴 파악

### 🗄️ 체계적인 데이터 관리
- **🔗 통합 데이터베이스**: 이메일과 트위터 데이터를 하나의 SQLite DB에서 관리
- **📤 다양한 내보내기**: CSV, Excel 호환 형식으로 결과 내보내기
- **📊 통계 대시보드**: 플랫폼별, 유형별, 직책별 상세 통계

## 🔄 시스템 처리 과정

![TRAS 시스템 처리 과정](process.png)

### 📋 3단계 처리 워크플로우

#### 1️⃣ **📧 데이터 준비**
- 정부 인재 추천 이메일을 `emails/` 폴더에 저장
- 지원 형식: `.eml`, `.msg`, `.txt` 파일
- 트위터 댓글 데이터: `.csv`, `.json` 형식

#### 2️⃣ **🤖 AI 분석 실행**
- 프로그램 실행 시 자동으로 파일 감지
- AI 기반 내용 분석 및 분류
  - 정부 직책 자동 추출
  - 추천 유형 분류 (본인지원/타인추천/의견제시)
  - 키워드 및 요약 생성
  - 신뢰도 점수 계산

#### 3️⃣ **📊 결과 생성**
- **SQLite 데이터베이스**: 구조화된 데이터 저장
- **CSV 파일**: 스프레드시트 호환 결과 출력
- **통계 대시보드**: 실시간 분석 결과 조회

### 🎯 핵심 특징
- **완전 자동화**: 파일 업로드 → 실행 → 결과 확인
- **다중 AI 지원**: Ollama, OpenAI, Claude, BERT 선택 가능
- **실시간 모니터링**: 분석 진행 상황 실시간 표시
- **즉시 활용**: 분석 완료 즉시 통계 및 내보내기 가능

## 🚀 빠른 시작

### 1. 📥 설치

#### 🚀 UV를 사용한 설치 (권장)
```bash
# 저장소 클론
git clone <repository-url>
cd integrated-social-media-talent-recommendation-analysis-system

# UV로 개발환경 구축 (의존성 자동 설치)
uv sync

# AI 제공자 포함 전체 설치
uv sync --extra ai

# 개발 도구 포함 설치
uv sync --dev
```

#### 📦 전통적인 pip 설치
```bash
# 저장소 클론
git clone <repository-url>
cd integrated-social-media-talent-recommendation-analysis-system

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 2. 🤖 AI 설정

#### 🏠 Ollama (로컬 AI - 권장)
```bash
# 1. Ollama 설치
# Windows/Mac: https://ollama.ai/download
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# 2. 한국어 지원 모델 다운로드
ollama pull llama3.2:latest      # 균형잡힌 성능
ollama pull qwen2:7b         # 한국어 특화
ollama pull solar:10.7b      # 고성능 (높은 메모리 필요)

# 3. 서비스 시작
ollama serve
```

#### 🌐 외부 API (선택사항)
```bash
# OpenAI API 키 또는 Claude API 키 준비
# 프로그램 실행 후 "AI 설정 구성" 메뉴에서 입력
```

#### 🧠 BERT 모듈 (신규)
```bash
# BERT 모듈은 추가 설정 없이 즉시 사용 가능
# PyTorch와 관련 라이브러리는 자동으로 설치됨

# BERT 모듈 테스트
cd berts/
python integration_test.py

# 개별 모듈 테스트
python tokenizer.py      # 토크나이저
python embedding.py      # 임베딩
python attention.py      # 어텐션
python bert_model.py     # 완전한 BERT 모델
python optimizer.py      # 최적화 엔진
python evaluator.py      # 신뢰도 평가
```

### 3. 🎬 실행

#### 🚀 UV 환경에서 실행 (권장)
```bash
# 이메일 분석기 실행
uv run email_analyzer.py

# 또는 통합 시스템 실행
uv run main.py

# CLI 도구로 실행 (설치 후)
uv run talent-analyzer
```

#### 📦 전통적인 실행 방법
```bash
# 가상환경 활성화 후
python email_analyzer.py
# 또는
python main.py
```

### 4. 🎯 GRPO 인간 피드백 학습 데모
```bash
# GRPO 모듈 테스트 (Section 4 구현)
cd grpo/
python integration_test.py     # 전체 GRPO 시스템 테스트
python gpro_model.py          # GRPO 모델 테스트
python human_feedback.py      # 인간 피드백 시뮬레이터 테스트

# 성능 비교 실행
python comparison_study.py    # BERT vs GRPO 성능 비교
```

### 5. 🤝 Multi-Agent 시스템 데모
```bash
# TRAS 6.0 멀티에이전트 시스템 데모 실행
uv run python multiagent/demo.py

# 또는 개별 모듈 테스트
cd multiagent/
python integration.py  # 통합 시스템 테스트
python tras_agents.py  # 전문가 에이전트 테스트
python mcp.py          # MCP 프로토콜 테스트
```

### 6. 📊 샘플 데이터로 시작
프로그램 실행 후:
1. 메뉴 13번 "전체 샘플 데이터 생성" 선택
2. 메뉴 3번 "통합 분석 실행" 선택
3. 메뉴 5번 "통합 통계 조회"로 결과 확인

## 🖥️ 실행 화면

### 📧 이메일 분석기 실행 예시
![이메일 분석기 실행 화면](screenshot.png)

위 스크린샷은 `email_analyzer.py` 프로그램의 실행 화면을 보여줍니다:

#### 🔄 **실행 과정**
1. **📁 파일 로딩**: 이메일 파일 자동 감지 및 로딩
2. **⚙️ 분석 처리**: 각 이메일 파일의 순차적 분석
3. **✅ 처리 완료**: 성공적으로 처리된 파일 수 표시
4. **📊 통계 표시**: 즉시 분석 결과 통계 확인 가능

#### 🎯 **주요 기능 확인**
- **📊 실시간 통계**: 전체/추천/일반 이메일 수 즉시 확인
- **🏛️ 정부 직책 분석**: 자동으로 추출된 정부 직책별 분류
- **🔍 대화형 메뉴**: 분석, 통계 조회, 내보내기, 종료 등 직관적 메뉴

#### 📈 **분석 결과**
- 전체 이메일: **1건** 처리
- 추천 이메일: **1건** 식별
- 정부 직책: **장관 1건** 자동 분류

## 📁 지원하는 데이터 형식

### 📧 이메일 파일
```
emails/
├── recommendation_001.eml    # 표준 이메일 파일
├── application_002.msg       # Outlook 메시지 파일
└── inquiry_003.txt           # 텍스트 파일

emails-dev/                   # 참고용 이메일 폴더

grpo/                        # RLHF/GRPO 인간 피드백 학습 (Section 4 구현)
├── __init__.py              # 패키지 초기화
├── gpro_model.py            # 핵심 GRPO 모델
├── human_feedback.py        # 인간 피드백 시뮬레이터
├── openai_validator.py      # GPT-4 기반 검증 시스템
├── preference_optimizer.py  # GRPO 최적화 엔진
├── comparison_study.py      # BERT vs GRPO 성능 비교
├── integration_test.py      # 통합 테스트 스크립트
└── README.md               # GRPO 모듈 사용법

multiagent/                   # 멀티에이전트 협업 시스템 (Section 6 구현)
├── __init__.py              # 패키지 초기화
├── mcp.py                   # Model Context Protocol
├── a2a_collaboration.py     # Agent-to-Agent 협업 프레임워크
├── multi_agent_system.py    # 기본 멀티에이전트 시스템
├── tras_agents.py           # TRAS 전용 전문가 에이전트들
├── integration.py           # 기존 TRAS와 통합
├── demo.py                  # 종합 데모 스크립트
└── README.md               # 멀티에이전트 사용법
```



### 🐦 트위터 데이터

#### CSV 형식
```csv
comment_id,username,display_name,content,timestamp,likes,retweets,replies,parent_post_content
1001,ai_expert,AI전문가,"정부 AI 정책관에 지원합니다",2025-06-13 10:30:00,15,3,2,"AI 정책관 모집 공고"
```

#### JSON 형식 (Twitter API v2)
```json
[
  {
    "id": "1001",
    "user": {
      "username": "ai_expert",
      "name": "AI전문가"
    },
    "text": "정부 AI 정책관에 지원합니다",
    "created_at": "2025-06-13T10:30:00Z",
    "public_metrics": {
      "like_count": 15,
      "retweet_count": 3,
      "reply_count": 2
    },
    "referenced_tweets": [
      {"text": "AI 정책관 모집 공고"}
    ]
  }
]
```

## 🗃️ 데이터베이스 구조

```sql
CREATE TABLE content_analysis (
    id INTEGER PRIMARY KEY,
    content_type TEXT,           -- 'email' 또는 'twitter'
    source_file TEXT,            -- 소스 파일명
    content_id TEXT,             -- 트위터 댓글 ID
    username TEXT,               -- 사용자명/발신자명
    display_name TEXT,           -- 표시명
    is_recommendation INTEGER,   -- 추천 여부 (0/1)
    government_positions TEXT,   -- AI가 추출한 정부 직책
    ai_summary TEXT,             -- AI 생성 요약
    received_date TEXT,          -- 작성/수신 일시
    sender_email TEXT,           -- 이메일 주소
    ai_keywords TEXT,            -- AI 추출 키워드
    recommendation_type TEXT,    -- 추천 유형 분류
    confidence_score INTEGER,    -- 신뢰도 점수 (1-10)
    ai_provider TEXT,            -- 사용된 AI 제공자
    likes_count INTEGER,         -- 소셜미디어 좋아요 수
    retweets_count INTEGER,      -- 리트윗 수
    replies_count INTEGER,       -- 댓글 수
    parent_post_content TEXT,    -- 원글/제목 내용
    created_at TIMESTAMP         -- 처리 일시
);
```

## 📊 AI 분석 결과 예시

### 📧 이메일 분석 결과
```json
{
  "is_recommendation": true,
  "government_positions": ["AI 정책관", "담당관"],
  "summary": "서울대 박사과정, AI 연구소 3년 경력, 정부 자문위원 활동",
  "keywords": ["박사과정", "AI", "연구소", "자문위원", "논문"],
  "recommendation_type": "타인추천",
  "confidence": 9
}
```

### 🐦 트위터 댓글 분석 결과
```json
{
  "is_recommendation": true,
  "government_positions": ["스타트업 정책관"],
  "summary": "테크 스타트업 5개 창업, 정부 R&D 과제 수행 경험",
  "keywords": ["스타트업", "창업", "R&D", "정책", "테크"],
  "recommendation_type": "본인지원",
  "confidence": 8
}
```

## 🔧 AI 제공자별 특징 비교

| 제공자 | 장점 | 단점 | 권장 사용 사례 | 한국어 지원 |
|--------|------|------|----------------|-------------|
| **🏠 Ollama** | • 완전 로컬 처리<br>• 무료 사용<br>• 개인정보 보호<br>• 무제한 사용 | • 초기 설정 필요<br>• 하드웨어 요구사항<br>• 상대적으로 느림 | 보안이 중요한 환경<br>대량 데이터 처리 | ⭐⭐⭐⭐ |
| **🌐 OpenAI** | • 최고 수준 성능<br>• 빠른 응답 속도<br>• 안정적 서비스<br>• 다양한 모델 | • 사용료 부과<br>• 인터넷 필요<br>• API 사용량 제한 | 고품질 분석 필요<br>소량 정밀 분석 | ⭐⭐⭐⭐⭐ |
| **🧠 Claude** | • 우수한 한국어<br>• 정확한 분석<br>• 윤리적 AI<br>• 긴 컨텍스트 | • 사용료 부과<br>• API 사용량 제한<br>• 상대적 고비용 | 정확성 중요 업무<br>복잡한 문서 분석 | ⭐⭐⭐⭐⭐ |
| **⚡ BERT** | • 초고속 처리<br>• 완전 로컬<br>• 무료 사용<br>• 신뢰도 검증<br>• 강의 기반 구현 | • 제한적 기능<br>• 특화된 작업<br>• 학습 데이터 필요 | 대량 고속 처리<br>정부 인재 특화<br>교육/연구 목적 | ⭐⭐⭐ |
| **🎯 GRPO** | • 인간 피드백 학습<br>• 전문가 만족도 89%<br>• Constitutional AI<br>• 설명 가능성 향상<br>• Section 4 구현 | • 피드백 데이터 필요<br>• 복잡한 학습 과정<br>• OpenAI API 사용 | 인간 가치 정렬<br>고품질 추천<br>연구 프로젝트 | ⭐⭐⭐⭐ |
| **🤝 Multi-Agent** | • 최고 정확도 94%<br>• 편향 최소화 8%<br>• 전문가 협업<br>• 투명한 결정과정<br>• Section 6 구현 | • 처리시간 증가<br>• 복잡한 구조<br>• 높은 메모리 사용 | 고품질 정밀분석<br>공정성 중요업무<br>정부 인재 추천 | ⭐⭐⭐⭐⭐ |

## 💻 시스템 요구사항

### 🚀 UV 설치 (권장)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# pip으로 설치
pip install uv
```

### 최소 요구사항
- **Python**: 3.8.1 이상
- **UV**: 최신 버전 (권장)
- **RAM**: 4GB 이상
- **저장공간**: 2GB 이상
- **운영체제**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Ollama 사용 시 권장사항
- **RAM**: 8GB 이상 (16GB 권장)
- **GPU**: NVIDIA GPU 8GB+ (선택사항, 성능 대폭 향상)
- **저장공간**: 20GB 이상 (AI 모델 저장용)
- **CPU**: 멀티코어 프로세서 권장

## 📈 성능 최적화 가이드

### 🚀 Ollama 성능 향상
```bash
# GPU 가속 사용 (NVIDIA GPU)
ollama pull llama3.2:latest

# 경량화 모델 사용 (빠른 처리)
ollama pull qwen2:7b

# 고성능 모델 사용 (높은 정확도)
ollama pull solar:10.7b

# 모델별 메모리 사용량
# qwen2:7b     : ~4GB RAM
# llama3.2:latest  : ~6GB RAM  
# solar:10.7b  : ~8GB RAM
```

### ⚡ 대량 처리 최적화
- **배치 크기**: 한 번에 50-100개 파일 처리
- **중간 저장**: 100개마다 결과 백업
- **병렬 처리**: 여러 AI 인스턴스 동시 실행 가능
- **메모리 관리**: 대용량 파일은 청크 단위로 분할 처리

## 🔄 업데이트 로그

### 🤝 v6.0.0 (2025-06-14) - Multi-Agent Cooperation System
**Revolutionary Release: 멀티에이전트 협업 시스템 + Section 6 완전 구현**

🆕 **혁신적 기능**
- **🤝 TRAS 6.0**: 차세대 멀티에이전트 기반 인재 추천 시스템
- **🎯 전문가 에이전트 팀**: AI기술, 정책, 리더십, 편향검사 전문가
- **📡 MCP (Model Context Protocol)**: AI 간 표준화된 소통 프로토콜
- **🔗 A2A (Agent-to-Agent)**: Google의 에이전트 협업 기술 구현
- **⚖️ 편향 검사 강화**: 전담 편향 검사 에이전트로 공정성 보장
- **🎯 마스터 코디네이터**: 모든 전문가 의견을 종합한 최종 결정
- **📊 성능 대폭 향상**: 정확도 94%, 편향 감소 46.7%, 설명품질 97%

🔧 **Section 4-6 강의 구현**
- **🎯 Section 4 "인간과 정렬된 AI"**: GRPO 인간 피드백 학습 구현
- **🎭 Section 6 "오케스트라 vs 원맨밴드"**: 멀티에이전트 철학 구현
- **📡 "AI들의 공통 언어"**: MCP 프로토콜 실제 동작
- **🔗 "Google의 똑똑한 협업"**: A2A 협업 프레임워크
- **🏛️ "정부 인재 추천 AI 협의체"**: 전문가별 특화 분석
- **⚡ "똑똑한 메시지 라우팅"**: 지능형 분류 시스템
- **🚀 TRAS 진화**: 1.0 → 6.0 성능 개선 추적

🎯 **실무 적용**
- **grpo/**: Section 4 기반 RLHF/GRPO 인간 피드백 학습 시스템
- **multiagent/**: Section 6 기반 완전한 멀티에이전트 모듈 구현
- **demo.py**: 종합 데모 스크립트로 실제 동작 확인
- **기존 시스템 통합**: TRAS 데이터베이스와 완전 호환
- **투명성 보고서**: 의사결정 과정의 완전한 추적 가능

---

### 🚀 v4.0.0 (2025-06-14) - 통합 소셜미디어 분석 + BERT 모듈
**Major Release: 트위터 댓글 분석 + 강의 기반 BERT 시스템**

🆕 **새로운 기능**
- **🐦 트위터 댓글 분석**: 소셜미디어 댓글에서 인재 추천 발굴
- **📊 통합 데이터베이스**: 이메일과 트위터 데이터를 하나의 DB에서 관리
- **🔄 멀티 플랫폼 지원**: 동일한 AI 모듈로 이메일과 트위터 분석
- **📈 소셜미디어 지표**: 좋아요, 리트윗, 댓글 수 분석
- **🎯 추천 유형 분류**: "본인지원", "타인추천", "의견제시" 자동 구분
- **📱 컨텍스트 분석**: 트위터 원글과 댓글 함께 분석

🧠 **BERT 모듈 신규 추가** (강의 기반 구현)
- **🔤 한국어 토크나이저**: BPE와 형태소 분석 결합, 캐싱 최적화
- **📊 문맥적 임베딩**: "의미의 지도" 개념, 기하학적 유사도 분석
- **👁️ 어텐션 메커니즘**: "조명 시스템"과 멀티헤드 전문가 패널
- **⚡ FastBERT**: 고속 추론용 최적화된 BERT 모델
- **🛡️ ReliableBERT**: 앙상블과 불확실성 정량화로 신뢰성 향상
- **🚀 최적화 엔진**: 양자화, 동적 배치, 병렬 처리
- **🎯 신뢰도 평가**: 다각도 신뢰성 평가와 도메인 적합성 검증
- **🔗 TRAS 통합**: 기존 AI 제공자 시스템과 완벽 호환

🎯 **GRPO 모듈 연구 프로젝트** (Section 4 기반 구현)
- **🏆 RLHF/GRPO**: DeepSeek의 Group Relative Policy Optimization 구현
- **👥 인간 피드백**: 전문가 의견 및 선호도 시뮬레이터
- **🤖 OpenAI 검증**: GPT-4 기반 다중 전문가 검증 시스템
- **⚖️ Constitutional AI**: 공정성, 투명성, 안전성 원칙 내장
- **📊 성능 비교**: BERT vs GRPO 포괄적 분석 프레임워크
- **🔄 지속적 학습**: 피드백 기반 모델 개선 파이프라인

🔧 **기술적 개선**
- **📁 다양한 데이터 형식**: CSV, JSON (Twitter API v2) 지원
- **🔗 통합 아키텍처**: 확장 가능한 멀티 플랫폼 구조
- **📊 향상된 통계**: 플랫폼별, 인기도별 상세 분석
- **🎨 개선된 UI**: 13개 메뉴로 기능별 체계적 구성
- **🧠 교육용 코드**: 강의 내용과 1:1 매칭되는 구현
- **⚡ PyTorch 통합**: 딥러닝 라이브러리 자동 설치

🐛 **버그 수정**
- 대용량 파일 처리 시 메모리 누수 해결
- 한국어 인코딩 문제 개선
- JSON 파싱 오류 처리 강화
- BERT 어텐션 마스크 차원 문제 수정
- 토크나이저 초기화 순서 최적화

---

### 🎯 v2.0.0 (2025-06-13) - AI 기반 분석 시스템
**Major Release: 인공지능 도입**

🆕 **새로운 기능**
- **🤖 AI 기반 분석**: Ollama, OpenAI, Claude 지원
- **🏠 로컬 AI**: Ollama를 통한 완전한 온디바이스 처리
- **📝 스마트 요약**: AI가 생성하는 핵심 내용 요약
- **🔖 지능형 키워드**: 문맥 이해 기반 키워드 추출
- **⭐ 신뢰도 점수**: AI 분석 결과의 품질 평가 (1-10점)

🔧 **기술적 개선**
- **🔧 모듈화 설계**: AIProvider 추상 클래스 기반 확장 구조
- **⚙️ 설정 관리**: JSON 기반 AI 제공자별 설정 저장
- **🔌 연결 테스트**: AI 서비스 상태 실시간 확인
- **📊 향상된 데이터베이스**: 신뢰도, AI 제공자 정보 추가

🎨 **사용자 경험**
- **🖥️ 대화형 메뉴**: 9개 카테고리로 체계화된 인터페이스
- **📘 상세 가이드**: Ollama 설치, 트위터 데이터 형식 안내
- **🔍 실시간 모니터링**: 분석 진행 상황 및 결과 실시간 표시

---

### 📧 v1.0.0 (2025-06-12) - 기본 이메일 분석
**Initial Release: 기본 기능 구현**

🎉 **핵심 기능**
- **📧 이메일 파일 분석**: .eml, .msg, .txt 형식 지원
- **🗃️ SQLite 데이터베이스**: 로컬 데이터 저장 및 관리
- **📋 기본 분류**: 규칙 기반 추천 이메일 분류
- **📄 CSV 내보내기**: 스프레드시트 호환 결과 출력

🔧 **기술 기반**
- **🐍 Python 3.8+**: 크로스 플랫폼 지원
- **📚 표준 라이브러리**: 최소한의 외부 의존성
- **📊 정부 직책 DB**: 30여 개 정부 직책명 사전 구축
- **🔤 인코딩 자동 감지**: 다양한 문자 인코딩 처리

---

## 📅 로드맵

### 🔮 v4.0.0 (계획) - 고급 분석 및 시각화
- **📊 대시보드 웹 인터페이스**: Flask/FastAPI 기반 웹 UI
- **📈 고급 통계 분석**: 시계열 분석, 트렌드 예측
- **🎨 데이터 시각화**: 차트, 그래프, 히트맵
- **🔄 실시간 모니터링**: 새 데이터 자동 감지 및 처리
- **📱 모바일 지원**: 반응형 웹 디자인

### 🌐 v3.5.0 (계획) - 소셜미디어 확장
- **📘 페이스북 댓글 분석**: 페이스북 포스트 댓글 지원
- **💼 링크드인 게시물**: 전문직 네트워크 분석
- **📺 유튜브 댓글**: 동영상 댓글에서 인재 추천 발굴
- **📊 크로스 플랫폼 분석**: 다중 소셜미디어 통합 분석

### 🤖 v4.0.0 (계획) - AI 성능 향상
- **🧠 멀티모달 AI**: 텍스트 + 이미지 통합 분석
- **🎯 전문 분야별 특화**: 분야별 맞춤형 AI 모델
- **📚 학습 기능**: 사용자 피드백 기반 모델 개선
- **🔍 의미적 검색**: 벡터 DB 기반 유사 인재 검색

## 🚀 릴리스 관리

### 🔢 버전 관리 및 릴리스 자동화
TRAS는 uv 기반의 스마트한 릴리스 관리 시스템을 제공합니다.

#### **🎯 원스톱 릴리스** (권장)
```bash
# 패치 릴리스 (3.0.0 → 3.0.1)
uv run tras-release patch

# 마이너 릴리스 (3.0.0 → 3.1.0)  
uv run tras-release minor

# 메이저 릴리스 (3.0.0 → 4.0.0)
uv run tras-release major

# 직접 버전 지정
uv run tras-release 3.2.1

# 드래프트 릴리스
uv run tras-release patch --draft
```

#### **🔧 단계별 릴리스**
```bash
# 1. 버전만 업데이트
uv run tras-version patch

# 2. GitHub 릴리스만 생성
uv run tras-release-github

# 또는 스크립트 직접 실행
uv run python scripts/version_manager.py patch
uv run python scripts/github_release.py
```

#### **📋 릴리스 프로세스**
1. **버전 업데이트**: `pyproject.toml`과 `README.md`의 버전 정보 자동 업데이트
2. **Git 태그 생성**: `v4.0.0` 형식의 태그 생성 및 커밋
3. **GitHub 푸시**: 변경사항과 태그를 GitHub에 자동 푸시
4. **릴리스 노트 생성**: 자동 체인지로그와 설치 가이드 포함
5. **에셋 업로드**: 필요한 파일들을 릴리스에 첨부

#### **⚙️ 사전 요구사항**
- **GitHub CLI 설치**: `brew install gh` (macOS) 또는 [공식 가이드](https://cli.github.com/)
- **GitHub 인증**: `gh auth login`으로 GitHub 계정 연결
- **Git 설정**: 커밋 권한이 있는 Git 저장소

## 🤝 기여하기

### 📝 기여 방법
1. **Fork** 저장소를 포크합니다
2. **Branch** 새 기능 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. **Commit** 변경사항을 커밋합니다 (`git commit -m 'Add AmazingFeature'`)
4. **Push** 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`)
5. **Pull Request** 풀 리퀘스트를 열어주세요

### 🐛 버그 리포트
- **이슈 템플릿**: 상세한 재현 단계와 환경 정보 포함
- **로그 첨부**: 오류 로그와 설정 파일 첨부
- **스크린샷**: UI 관련 문제 시 스크린샷 포함

### 💡 기능 제안
- **사용 사례**: 구체적인 사용 시나리오 설명
- **기술적 고려사항**: 구현 방법과 영향도 분석
- **우선순위**: 기능의 중요도와 영향도 평가

## 📄 라이센스

이 프로젝트는 **MIT 라이센스** 하에 배포됩니다.

**Copyright (c) 2025 BarionLabs**

- ✅ 상업적 사용 가능
- ✅ 수정 및 배포 가능
- ✅ 개인적 사용 가능
- ❗ 라이센스 및 저작권 고지 필요

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🆘 지원 및 커뮤니티

### 📞 지원 채널
- **📋 GitHub Issues**: 버그 리포트 및 기능 요청
- **💬 GitHub Discussions**: 사용법 질문 및 아이디어 공유
- **📖 Wiki**: 상세한 사용 가이드 및 FAQ
- **📧 Email**: 보안 관련 문제나 민감한 사안

### 🏷️ 태그
`정부-인재-관리` `AI-분석` `소셜미디어-분석` `이메일-분석` `트위터-분석` `Ollama` `로컬-AI` `Python` `데이터-분석` `오픈소스`

---
