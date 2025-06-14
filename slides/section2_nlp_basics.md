---
marp: true
theme: default
class: lead
paginate: true
backgroundColor: #f8f9fa
---

# 📝 Section 2: 자연어처리 기초
## "컴퓨터가 언어를 이해하는 방법"

### 50분 완전 정복

---

## 🎯 Section 2 학습 목표

이 섹션을 마치면 여러분은:

1. **🔤 토큰화**: 문장을 컴퓨터가 처리할 수 있는 단위로 나누기
2. **📊 임베딩**: 단어를 숫자 벡터로 변환하는 마법
3. **👁️ 어텐션**: AI가 "집중"하는 방법의 수학적 모델
4. **🔄 실습**: TRAS에서 이 모든 것이 어떻게 작동하는지

### 💭 큰 그림 미리보기
```
문장 → 토큰화 → 임베딩 → 어텐션 → 이해
"안녕하세요" → ["안녕", "하세요"] → [벡터들] → [가중치들] → 의미
```

---

## 🌍 자연어처리란? "디지털 바벨탑 건설하기"

### 🏗️ 근본적인 문제

```python
# 인간의 언어 vs 컴퓨터의 언어
human_language = "김철수를 AI 정책관으로 추천합니다."
computer_language = [0.2, 0.8, 0.1, -0.3, 0.9, ...]  # 숫자 벡터

# 자연어처리의 목표
def nlp_magic(human_language):
    return computer_language  # 어떻게?!
```

### 💡 비유: 번역가의 역할
- **입력**: 인간의 복잡하고 모호한 언어
- **출력**: 컴퓨터가 이해할 수 있는 정확한 수치
- **과정**: 단계별 변환을 통한 의미 보존

### 🎯 TRAS에서의 적용
```
"정부 AI 정책관에 김철수를 추천합니다" 
→ AI가 이해 가능한 형태로 변환
→ 추천 여부, 직책, 신뢰도 점수 추출
```

---

## 🔤 1단계: 토큰화 (Tokenization)

### "문장을 레고 블록으로 나누기"

### 🧩 기본 개념

```python
# 간단한 토큰화 (공백 기준)
text = "정부 AI 정책관에 김철수를 추천합니다"
simple_tokens = text.split()
print(simple_tokens)
# ['정부', 'AI', '정책관에', '김철수를', '추천합니다']

# 문제점: '정책관에' → '정책관' + '에' 로 나누어야 함
```

### 🔍 한국어 토큰화의 도전

```python
# 형태소 분석 필요
from konlpy.tag import Okt

okt = Okt()
morphemes = okt.morphs("정부 AI 정책관에 김철수를 추천합니다")
print(morphemes)
# ['정부', 'AI', '정책', '관', '에', '김철수', '를', '추천', '합니다']
```

### 💭 비유: 문장 = 레고 구조물
- **원시 문장**: 완성된 레고 성
- **토큰화**: 성을 개별 블록으로 분해
- **목적**: 각 블록(토큰)을 개별적으로 이해하기 위함

---

## 🚀 현대적 토큰화: Subword Tokenization

### "BPE: 최적의 조각 찾기"

### 🧮 Byte Pair Encoding (BPE) 알고리즘

```python
# BPE의 핵심 아이디어
vocabulary = {
    "안": 100, "녕": 150, "하": 200, "세": 50, "요": 80,
    "안녕": 300,  # 자주 등장하는 조합은 하나의 토큰으로
    "하세": 180,
    "세요": 120
}

# "안녕하세요" 토큰화
# 방법 1: ["안", "녕", "하", "세", "요"] (5개 토큰)
# 방법 2: ["안녕", "하세", "요"] (3개 토큰) ← 더 효율적!
```

### 🎯 장점들
- **효율성**: 자주 쓰이는 조합을 하나로 처리
- **일반화**: 새로운 단어도 기존 조각들로 표현 가능
- **언어 독립적**: 한국어, 영어, 중국어 모두 동일한 방식

### 📊 기하학적 직관
토큰화는 **고차원 문자열 공간**을 **저차원 토큰 공간**으로 매핑하는 과정
```
문자열 공간 (무한차원) → 토큰 공간 (유한차원)
"안녕하세요" → [토큰1, 토큰2, 토큰3]
```

---

## 🔬 TRAS에서의 토큰화 실습

### 💻 실제 코드 구현

```python
class TextProcessor:
    """TRAS의 텍스트 전처리기"""
    
    def __init__(self, ai_provider: str):
        self.ai_provider = ai_provider
        self.tokenizer = self._load_tokenizer()
    
    def preprocess_email(self, email_content: str) -> List[str]:
        """이메일 내용을 AI가 처리할 수 있도록 전처리"""
        # 1. 노이즈 제거 (HTML 태그, 서명 등)
        cleaned = self._remove_noise(email_content)
        
        # 2. 토큰화
        tokens = self.tokenizer.encode(cleaned)
        
        # 3. 특수 토큰 추가
        return self._add_special_tokens(tokens)
    
    def _remove_noise(self, text: str) -> str:
        """HTML 태그, 이메일 서명 등 제거"""
        # 정규식을 이용한 정제
        import re
        text = re.sub(r'<[^>]+>', '', text)  # HTML 태그 제거
        text = re.sub(r'--\s*\n.*', '', text, flags=re.DOTALL)  # 서명 제거
        return text.strip()
```

### 🎯 핵심 포인트
- **도메인 특화**: 이메일/소셜미디어에 특화된 전처리
- **노이즈 제거**: 분석에 불필요한 정보 제거
- **표준화**: AI 모델이 일관되게 처리할 수 있는 형태로 변환

---

## 📊 2단계: 임베딩 (Embedding)

### "단어를 좌표계에 배치하기"

### 🌌 단어 임베딩의 직관

```python
# 전통적 방법: One-Hot Encoding (비효율적)
vocab_size = 50000
king_onehot = [0] * vocab_size
king_onehot[1234] = 1  # 'king'은 1234번 위치에 1

# 현대적 방법: Dense Embedding (효율적)
king_embedding = [0.2, -0.1, 0.8, 0.3, -0.5, ...]  # 300차원 벡터
queen_embedding = [0.3, -0.2, 0.7, 0.4, -0.4, ...]
```

### 🎨 기하학적 비유: "의미의 지도"

벡터 공간을 **의미의 지도**로 생각해보세요:

```
   👑 king    queen 👸
      \       /
       \     /
        man-woman 축
       /     \
      /       \
   👨 boy    girl 👧
```

- **거리**: 의미적 유사성
- **방향**: 의미적 관계 (성별, 나이 등)
- **연산**: `king - man + woman ≈ queen`

---

## 🧮 임베딩의 수학적 원리

### ⚡ Word2Vec의 핵심 아이디어

```python
# Skip-gram 모델의 목적 함수 (기하학적 해석)
def word2vec_objective(center_word, context_words):
    """
    중심 단어로부터 주변 단어들을 예측
    
    기하학적 직관:
    - 자주 함께 등장하는 단어들은 벡터 공간에서 가까이 배치
    - 내적(dot product)이 클수록 높은 유사도
    """
    similarity_scores = []
    for context_word in context_words:
        # 내적 = 벡터 간 각도의 코사인 × 크기의 곱
        score = dot_product(center_word, context_word)
        similarity_scores.append(score)
    
    return softmax(similarity_scores)
```

### 📐 기하학적 직관: "벡터의 춤"

1. **내적 (Dot Product)**: 두 벡터가 같은 방향을 향할수록 큰 값
2. **코사인 유사도**: 벡터 간 각도로 유사성 측정
3. **클러스터링**: 유사한 의미의 단어들이 자연스럽게 모임

```
벡터 A · 벡터 B = |A| × |B| × cos(θ)
θ가 작을수록 (같은 방향) → 높은 유사도
```

---

## 🎭 문맥적 임베딩: "카멜레온 단어"

### 🦎 단어의 다의성 문제

```python
# 전통적 임베딩의 한계
bank_traditional = [0.1, 0.2, 0.3, ...]  # 항상 같은 벡터

# 문맥적 임베딩의 해답
sentences = [
    "강둑(bank)에서 낚시를 했다",      # 강가
    "은행(bank)에서 돈을 빌렸다"       # 금융기관
]

# BERT/GPT 같은 모델은 문맥에 따라 다른 임베딩 생성
bank_river = model.encode(sentences[0])["bank"]    # 강가 의미
bank_money = model.encode(sentences[1])["bank"]    # 금융 의미
```

### 🌈 비유: 카멜레온의 색깔 변화
- **전통적 임베딩**: 사진 속 고정된 카멜레온
- **문맥적 임베딩**: 환경에 따라 색깔을 바꾸는 살아있는 카멜레온
- **장점**: 같은 단어라도 상황에 맞는 의미 표현

---

## 💻 TRAS에서의 임베딩 활용

### 🛠️ 실제 구현

```python
class EmbeddingProcessor:
    """TRAS의 임베딩 처리기"""
    
    def __init__(self, model_name: str = "klue/bert-base"):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """문장 전체의 임베딩 벡터 생성"""
        # 토큰화
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True)
        
        # 모델 통과
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # [CLS] 토큰의 임베딩을 문장 표현으로 사용
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding.numpy()
    
    def similarity_score(self, text1: str, text2: str) -> float:
        """두 텍스트 간 유사도 계산"""
        emb1 = self.get_sentence_embedding(text1)
        emb2 = self.get_sentence_embedding(text2)
        
        # 코사인 유사도 계산
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(emb1, emb2)[0][0]
```

---

## 🔍 메타인지 체크포인트 #2

### 🤔 임베딩 이해도 점검

1. **개념 이해**
   - "임베딩이 왜 필요한지 한 문장으로 설명할 수 있나?"
   - "문맥적 임베딩과 정적 임베딩의 차이를 알고 있나?"

2. **수학적 직관**
   - "벡터 내적이 유사도와 어떤 관련이 있는지 기하학적으로 설명할 수 있나?"
   - "고차원 공간에서 의미적 관계가 어떻게 표현되는지 이해하고 있나?"

3. **실무 적용**
   - "TRAS에서 임베딩이 어떤 역할을 하는지 알고 있나?"
   - "임베딩 품질을 어떻게 평가할 수 있을까?"

---

## 👁️ 3단계: 어텐션 메커니즘 (Attention)

### "AI의 집중력 모델링하기"

### 🧠 인간의 주의집중과 AI의 어텐션

```python
# 인간이 문장을 읽는 방식
sentence = "정부 AI 정책관에 김철수를 강력히 추천합니다"

# 단어별 중요도 (인간의 직관)
human_attention = {
    "정부": 0.3,      # 어느 정도 중요
    "AI": 0.8,        # 매우 중요 (핵심 분야)
    "정책관에": 0.9,   # 가장 중요 (핵심 직책)
    "김철수를": 0.7,   # 중요 (추천 대상)
    "강력히": 0.6,     # 중요 (추천 강도)
    "추천합니다": 0.8   # 매우 중요 (행동)
}

# AI의 어텐션도 비슷한 가중치를 학습해야 함
```

### 🎯 어텐션의 핵심 아이디어

**"모든 정보를 똑같이 보지 말고, 중요한 것에 집중하자!"**

---

## 🧮 어텐션의 수학적 정의

### ⚡ 어텐션 함수의 구조

```python
def attention(query, key, value):
    """
    Query(Q): "무엇을 찾고 있는가?"
    Key(K): "각 위치에 무엇이 있는가?"
    Value(V): "실제 정보는 무엇인가?"
    """
    # 1. 유사도 계산 (Q와 K의 내적)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 2. 스케일링 (수치 안정성)
    scores = scores / math.sqrt(key.size(-1))
    
    # 3. 소프트맥스로 확률 분포 변환
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 4. 가중 평균으로 최종 출력
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### 📐 기하학적 해석: "조명과 그림자"

어텐션을 **조명 시스템**으로 생각해보세요:

```
🔦 Query: 손전등 (찾고자 하는 것)
🏠 Key: 방 안의 물건들 (각 위치의 정보)
💎 Value: 실제 보물들 (추출할 정보)

과정:
1. 손전등으로 방을 비춤 (Q·K)
2. 밝게 비춰진 곳일수록 높은 가중치
3. 가중치에 따라 보물들을 수집 (가중 평균)
```

---

## 🎭 Self-Attention: "문장 내 단어들의 대화"

### 🗣️ 단어들이 서로 대화한다면?

```python
# 예시 문장: "김철수는 AI 전문가로서 정책관에 적합하다"
words = ["김철수는", "AI", "전문가로서", "정책관에", "적합하다"]

# Self-attention이 발견하는 관계들
attention_map = {
    "김철수는": {
        "AI": 0.3,        # 김철수 ↔ AI (연관성 있음)
        "전문가로서": 0.8,  # 김철수 ↔ 전문가 (강한 연관)
        "적합하다": 0.6     # 김철수 ↔ 적합 (주어-서술어)
    },
    "정책관에": {
        "AI": 0.7,        # 정책관 ↔ AI (분야 연관)
        "전문가로서": 0.5,  # 정책관 ↔ 전문가 (자격 연관)
        "적합하다": 0.9     # 정책관 ↔ 적합 (직접 연관)
    }
}
```

### 💡 비유: 파티에서의 대화
- **각 단어**: 파티 참가자
- **어텐션 가중치**: 서로에 대한 관심도
- **Self-attention**: 모든 참가자가 동시에 대화하며 정보 교환
- **결과**: 각자가 다른 사람들로부터 얻은 정보로 자신을 업데이트

---

## 🌟 Multi-Head Attention: "여러 관점으로 보기"

### 👁️‍🗨️ 다양한 시각의 필요성

```python
class MultiHeadAttention:
    """여러 개의 어텐션 헤드로 다양한 관점 학습"""
    
    def __init__(self, d_model=512, num_heads=8):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 각 헤드별로 별도의 W_Q, W_K, W_V 학습
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # 1. 여러 헤드로 분할
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k)
        
        # 2. 각 헤드에서 독립적으로 어텐션 계산
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        
        # 3. 모든 헤드의 결과를 합침
        concat = attention_output.view(batch_size, -1, d_model)
        
        # 4. 최종 선형 변환
        return self.W_o(concat)
```

### 🎨 비유: 여러 명의 전문가 패널
- **헤드 1**: 문법 전문가 (주어-동사-목적어 관계 주목)
- **헤드 2**: 의미 전문가 (단어 간 의미적 연관성 주목)
- **헤드 3**: 감정 전문가 (긍정/부정 표현 주목)
- **최종 결과**: 모든 전문가 의견을 종합한 판단

---

## 🔬 TRAS에서의 어텐션 활용

### 💻 정부 직책 추출에서의 어텐션

```python
class GovernmentPositionExtractor:
    """어텐션을 활용한 정부 직책 추출기"""
    
    def __init__(self, model_name="klue/bert-base"):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 정부 직책 키워드들
        self.position_keywords = [
            "정책관", "과장", "국장", "차관", "장관", 
            "대통령", "총리", "비서관", "보좌관", "수석"
        ]
    
    def extract_with_attention(self, text: str):
        """어텐션 가중치를 활용한 직책 추출"""
        # 토큰화 및 모델 실행
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs, output_attentions=True)
        
        # 어텐션 가중치 분석
        attention_weights = outputs.attentions[-1]  # 마지막 레이어
        
        # 직책 키워드 주변의 어텐션 패턴 분석
        positions = []
        for keyword in self.position_keywords:
            if keyword in text:
                attention_score = self._analyze_keyword_attention(
                    keyword, text, attention_weights
                )
                if attention_score > 0.5:  # 임계값 이상
                    positions.append((keyword, attention_score))
        
        return sorted(positions, key=lambda x: x[1], reverse=True)
```

---

## 📊 어텐션 시각화: "AI의 사고 과정 엿보기"

### 🎨 어텐션 맵 해석

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(text, attention_weights):
    """어텐션 가중치를 히트맵으로 시각화"""
    tokens = text.split()
    
    # 어텐션 행렬 생성
    attention_matrix = attention_weights[0][0].detach().numpy()
    
    # 히트맵 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        annot=True,
        fmt='.2f'
    )
    plt.title("Self-Attention Visualization")
    plt.xlabel("Key (attending to)")
    plt.ylabel("Query (attending from)")
    plt.show()

# 사용 예시
text = "정부 AI 정책관에 김철수를 추천합니다"
attention_map = model.get_attention(text)
visualize_attention(text, attention_map)
```

### 🔍 해석 방법
- **밝은 색**: 높은 어텐션 가중치 (강한 연관성)
- **어두운 색**: 낮은 어텐션 가중치 (약한 연관성)
- **대각선**: 자기 자신에 대한 어텐션 (항상 높음)
- **패턴**: 언어적 구조와 의미적 관계 반영

---

## 🔍 메타인지 체크포인트 #3

### 🤔 어텐션 메커니즘 이해도 점검

1. **핵심 개념**
   - "어텐션의 Query, Key, Value가 무엇인지 설명할 수 있나?"
   - "Self-attention과 Cross-attention의 차이를 알고 있나?"

2. **수학적 이해**
   - "어텐션 가중치가 어떻게 계산되는지 단계별로 설명할 수 있나?"
   - "Softmax가 어텐션에서 왜 필요한지 알고 있나?"

3. **실무 적용**
   - "Multi-head attention이 왜 더 효과적인지 설명할 수 있나?"
   - "어텐션 시각화를 통해 무엇을 알 수 있는가?"

---

## 🛠️ 종합 실습: TRAS의 NLP 파이프라인

### 📋 전체 프로세스 구현

```python
class TRASNLPPipeline:
    """TRAS의 완전한 NLP 처리 파이프라인"""
    
    def __init__(self, model_name="klue/bert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.government_positions = self._load_government_positions()
    
    def analyze_text(self, text: str) -> dict:
        """텍스트 분석 전체 프로세스"""
        
        # 1단계: 토큰화
        tokens = self.tokenizer.tokenize(text)
        
        # 2단계: 임베딩 생성
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs, output_attentions=True)
        embeddings = outputs.last_hidden_state
        
        # 3단계: 어텐션 분석
        attention_weights = outputs.attentions[-1]
        
        # 4단계: 정보 추출
        results = {
            "tokens": tokens,
            "embeddings": embeddings.detach().numpy(),
            "attention_weights": attention_weights.detach().numpy(),
            "extracted_positions": self._extract_positions(text, attention_weights),
            "sentiment": self._analyze_sentiment(embeddings),
            "key_phrases": self._extract_key_phrases(text, attention_weights)
        }
        
        return results
    
    def _extract_positions(self, text, attention_weights):
        """어텐션 기반 정부 직책 추출"""
        # 어텐션 가중치 분석을 통한 중요 키워드 식별
        # 정부 직책 데이터베이스와 매칭
        pass
    
    def _analyze_sentiment(self, embeddings):
        """감정 분석 (추천의 강도 측정)"""
        # 임베딩 기반 감정 분류
        pass
    
    def _extract_key_phrases(self, text, attention_weights):
        """핵심 구문 추출"""
        # 높은 어텐션 가중치를 받는 구문 식별
        pass
```

---

## 🎯 성능 평가와 개선

### 📊 NLP 모델 평가 지표

```python
class NLPEvaluator:
    """NLP 모델 성능 평가기"""
    
    def evaluate_position_extraction(self, predictions, ground_truth):
        """정부 직책 추출 성능 평가"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='macro'
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    def evaluate_attention_quality(self, attention_weights, human_annotations):
        """어텐션 품질 평가 (인간 주석과 비교)"""
        # 인간이 중요하다고 표시한 단어들과 
        # AI가 높은 어텐션을 준 단어들 간의 일치도 측정
        
        correlation = self._calculate_attention_correlation(
            attention_weights, human_annotations
        )
        
        return correlation
```

### 🔧 모델 개선 전략

1. **데이터 증강**: 다양한 표현 방식의 추천 텍스트 수집
2. **도메인 적응**: 정부 용어에 특화된 사전 훈련
3. **앙상블**: 여러 모델의 결과를 조합
4. **인간 피드백**: 전문가 검토를 통한 지속적 개선

---

## 🌟 고급 주제: Transformer의 핵심

### 🏗️ Transformer 아키텍처 미리보기

```python
class TransformerBlock:
    """Transformer의 기본 구성 요소"""
    
    def __init__(self, d_model, num_heads):
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
    
    def forward(self, x):
        # 1. Multi-Head Self-Attention + Residual Connection
        attn_output = self.multi_head_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # 2. Feed-Forward + Residual Connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x
```

### 🎯 왜 Transformer가 혁명적인가?

1. **병렬 처리**: RNN과 달리 모든 위치를 동시에 처리
2. **장거리 의존성**: 어텐션으로 멀리 떨어진 단어 간 관계 포착
3. **확장성**: 더 많은 레이어와 파라미터로 성능 향상
4. **범용성**: 번역, 요약, 분류 등 다양한 작업에 적용

---

## 🎯 Section 2 요약

### ✅ 우리가 마스터한 것들

1. **🔤 토큰화**: 문장을 AI가 처리할 수 있는 조각으로 나누기
   - BPE 알고리즘의 효율성
   - 한국어 형태소 분석의 특수성

2. **📊 임베딩**: 단어를 의미있는 벡터로 변환
   - 정적 vs 문맥적 임베딩
   - 벡터 공간에서의 의미 표현

3. **👁️ 어텐션**: AI의 집중력 메커니즘
   - Self-attention의 강력함
   - Multi-head attention의 다양한 관점

4. **🔬 실습**: TRAS에서의 실제 적용
   - 정부 직책 추출
   - 어텐션 시각화

---

## 🔗 다음 섹션 예고

이제 우리는 **BERT의 세계**로 들어갑니다!

### 🧠 Section 3에서 다룰 내용
- Transformer → BERT로의 진화
- 사전훈련의 마법: MLM과 NSP
- 파인튜닝: 범용 모델을 전문가로 만들기
- TRAS에서 BERT 활용하기

### 💭 연결 고리
```
Section 2의 기초 → Section 3의 응용
토큰화 + 임베딩 + 어텐션 → BERT → 실제 문제 해결
```

---

## 💡 심화 학습 과제

### 🤓 도전 과제들

1. **어텐션 분석 프로젝트**
   - TRAS의 어텐션 패턴을 분석해보세요
   - 어떤 단어들이 서로 높은 어텐션을 갖는지 조사

2. **임베딩 시각화**
   - t-SNE나 PCA로 단어 임베딩을 2D로 시각화
   - 정부 관련 용어들의 클러스터 확인

3. **토큰화 비교 실험**
   - 다양한 토큰화 방법의 성능 비교
   - 한국어 특화 토크나이저 vs 다국어 토크나이저

### 📚 추천 읽을거리
- "The Illustrated Transformer" (Jay Alammar)
- "Word2Vec Tutorial" (TensorFlow)
- "Attention Is All You Need" (원논문)

---

## 🚀 BERT가 기다리고 있어요!

다음 60분은 **BERT 완전 정복** 시간입니다!

🧠 **Section 3: BERT 깊이 이해**에서 만나요! 