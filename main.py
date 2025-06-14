#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 소셜미디어 인재 추천 분석 시스템
Integrated Social Media Talent Recommendation Analysis System

이메일과 트위터 댓글을 AI로 분석하여 정부 인재 추천을 자동 분류하는 시스템

MIT License
Copyright (c) 2025 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sqlite3
import email
import re
import json
import requests
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import chardet
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """분석 설정 클래스"""
    ai_provider: str = "ollama"  # "ollama" 또는 "openai" 또는 "claude"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    claude_api_key: str = ""
    claude_model: str = "claude-3-haiku-20240307"
    max_tokens: int = 1000
    temperature: float = 0.1

@dataclass
class TwitterComment:
    """트위터 댓글 데이터 클래스"""
    comment_id: str
    username: str
    display_name: str
    content: str
    timestamp: str
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    parent_post_id: str = ""
    parent_post_content: str = ""

class AIProvider(ABC):
    """AI 제공자 추상 클래스"""
    
    @abstractmethod
    def analyze_content(self, content: str, subject: str = "", content_type: str = "email") -> Dict:
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        pass

class OllamaProvider(AIProvider):
    """Ollama AI 제공자"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.base_url = config.ollama_url
        self.model = config.ollama_model
    
    def test_connection(self) -> bool:
        """Ollama 연결 테스트"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama 연결 실패: {e}")
            return False
    
    def _make_request(self, prompt: str) -> str:
        """Ollama API 요청"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama API 오류: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Ollama 요청 실패: {e}")
            return ""
    
    def analyze_content(self, content: str, subject: str = "", content_type: str = "email") -> Dict:
        """컨텐츠 AI 분석 (이메일 또는 트위터 댓글)"""
        
        if content_type == "twitter":
            prompt = f"""
다음 트위터 댓글을 분석하여 JSON 형태로 결과를 제공해주세요:

댓글 내용: {content[:2000]}
원글 제목/내용: {subject}

분석 항목:
1. 추천여부: 이 댓글이 정부 인재 추천인지 판단 (true/false)
2. 정부직책: 언급된 정부 직책명들을 추출
3. 추천내용요약: 추천하는 인물이나 본인 소개 내용을 200자 이내로 요약
4. 핵심키워드: 학력, 경력, 전문분야 등 주요 키워드 5개 추출
5. 추천유형: "본인지원", "타인추천", "의견제시" 중 하나
6. 신뢰도: 분석 결과의 신뢰도 (1-10점)

응답 형식:
{{
  "is_recommendation": true/false,
  "government_positions": ["직책1", "직책2"],
  "summary": "추천/지원 내용 요약",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "recommendation_type": "본인지원/타인추천/의견제시",
  "confidence": 점수
}}

JSON만 응답하세요:
"""
        else:
            prompt = f"""
다음 이메일을 분석하여 JSON 형태로 결과를 제공해주세요:

제목: {subject}
내용: {content[:2000]}...

분석 항목:
1. 추천여부: 이 이메일이 정부 인재 추천 이메일인지 판단 (true/false)
2. 정부직책: 언급된 정부 직책명들을 추출
3. 자기소개요약: 발신자의 자기소개 내용을 200자 이내로 요약
4. 핵심키워드: 학력, 경력, 전문분야 등 주요 키워드 5개 추출
5. 추천유형: "본인지원", "타인추천", "의견제시" 중 하나
6. 신뢰도: 분석 결과의 신뢰도 (1-10점)

응답 형식:
{{
  "is_recommendation": true/false,
  "government_positions": ["직책1", "직책2"],
  "summary": "자기소개 요약",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "recommendation_type": "본인지원/타인추천/의견제시",
  "confidence": 점수
}}

JSON만 응답하세요:
"""
        
        response = self._make_request(prompt)
        
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            return self._extract_from_text(response, content_type)
    
    def _extract_from_text(self, text: str, content_type: str) -> Dict:
        """텍스트에서 정보 추출 (JSON 파싱 실패 시 대안)"""
        return {
            "is_recommendation": "추천" in text or "recommend" in text.lower() or "지원" in text,
            "government_positions": re.findall(r'[가-힣]+(?:장관|차관|국장|과장|원장)', text),
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "keywords": re.findall(r'[가-힣A-Za-z]+(?:대학|경력|전공|박사|석사)', text)[:5],
            "recommendation_type": "본인지원" if content_type == "twitter" else "타인추천",
            "confidence": 5
        }

class OpenAIProvider(AIProvider):
    """OpenAI GPT 제공자"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_key = config.openai_api_key
        self.model = config.openai_model
    
    def test_connection(self) -> bool:
        """OpenAI API 연결 테스트"""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"OpenAI 연결 실패: {e}")
            return False
    
    def analyze_content(self, content: str, subject: str = "", content_type: str = "email") -> Dict:
        """OpenAI를 이용한 컨텐츠 분석"""
        
        if content_type == "twitter":
            prompt = f"""
다음 트위터 댓글을 분석하여 JSON 형태로 결과를 제공해주세요:

댓글 내용: {content}
원글 내용: {subject}

분석 항목:
1. 추천여부: 이 댓글이 정부 인재 추천인지 판단
2. 정부직책: 언급된 정부 직책명들을 추출
3. 추천내용요약: 추천하는 인물이나 본인 소개 내용을 200자 이내로 요약
4. 핵심키워드: 학력, 경력, 전문분야 등 주요 키워드 5개 추출
5. 추천유형: "본인지원", "타인추천", "의견제시" 중 하나
6. 신뢰도: 분석 결과의 신뢰도 (1-10점)

JSON 형식으로만 응답:
{{
  "is_recommendation": true/false,
  "government_positions": ["직책1", "직책2"],
  "summary": "추천/지원 내용 요약",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "recommendation_type": "본인지원/타인추천/의견제시",
  "confidence": 점수
}}
"""
        else:
            prompt = f"""
다음 이메일을 분석하여 JSON 형태로 결과를 제공해주세요:

제목: {subject}
내용: {content}

분석 항목:
1. 추천여부: 이 이메일이 정부 인재 추천 이메일인지 판단
2. 정부직책: 언급된 정부 직책명들을 추출
3. 자기소개요약: 발신자의 자기소개 내용을 200자 이내로 요약
4. 핵심키워드: 학력, 경력, 전문분야 등 주요 키워드 5개 추출
5. 추천유형: "본인지원", "타인추천", "의견제시" 중 하나
6. 신뢰도: 분석 결과의 신뢰도 (1-10점)

JSON 형식으로만 응답:
{{
  "is_recommendation": true/false,
  "government_positions": ["직책1", "직책2"],
  "summary": "자기소개 요약",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "recommendation_type": "본인지원/타인추천/의견제시",
  "confidence": 점수
}}
"""
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return json.loads(content)
            else:
                logger.error(f"OpenAI API 오류: {response.status_code}")
                return self._default_analysis()
                
        except Exception as e:
            logger.error(f"OpenAI 분석 실패: {e}")
            return self._default_analysis()
    
    def _default_analysis(self) -> Dict:
        """기본 분석 결과"""
        return {
            "is_recommendation": False,
            "government_positions": [],
            "summary": "분석 실패",
            "keywords": [],
            "recommendation_type": "의견제시",
            "confidence": 1
        }

class ClaudeProvider(AIProvider):
    """Anthropic Claude 제공자"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.api_key = config.claude_api_key
        self.model = config.claude_model
    
    def test_connection(self) -> bool:
        """Claude API 연결 테스트"""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.model,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Hello"}]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Claude 연결 실패: {e}")
            return False
    
    def analyze_content(self, content: str, subject: str = "", content_type: str = "email") -> Dict:
        """Claude를 이용한 컨텐츠 분석"""
        
        if content_type == "twitter":
            prompt = f"""
다음 트위터 댓글을 분석하여 JSON 형태로 결과를 제공해주세요:

댓글 내용: {content}
원글 내용: {subject}

분석 항목:
1. 추천여부: 이 댓글이 정부 인재 추천인지 판단
2. 정부직책: 언급된 정부 직책명들을 추출
3. 추천내용요약: 추천하는 인물이나 본인 소개 내용을 200자 이내로 요약
4. 핵심키워드: 학력, 경력, 전문분야 등 주요 키워드 5개 추출
5. 추천유형: "본인지원", "타인추천", "의견제시" 중 하나
6. 신뢰도: 분석 결과의 신뢰도 (1-10점)

JSON 형식으로만 응답:
{{
  "is_recommendation": true/false,
  "government_positions": ["직책1", "직책2"],
  "summary": "추천/지원 내용 요약",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "recommendation_type": "본인지원/타인추천/의견제시",
  "confidence": 점수
}}
"""
        else:
            prompt = f"""
다음 이메일을 분석하여 JSON 형태로 결과를 제공해주세요:

제목: {subject}
내용: {content}

분석 항목:
1. 추천여부: 이 이메일이 정부 인재 추천 이메일인지 판단
2. 정부직책: 언급된 정부 직책명들을 추출
3. 자기소개요약: 발신자의 자기소개 내용을 200자 이내로 요약
4. 핵심키워드: 학력, 경력, 전문분야 등 주요 키워드 5개 추출
5. 추천유형: "본인지원", "타인추천", "의견제시" 중 하나
6. 신뢰도: 분석 결과의 신뢰도 (1-10점)

JSON 형식으로만 응답:
{{
  "is_recommendation": true/false,
  "government_positions": ["직책1", "직책2"],
  "summary": "자기소개 요약",
  "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"],
  "recommendation_type": "본인지원/타인추천/의견제시",
  "confidence": 점수
}}
"""
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": self.model,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["content"][0]["text"]
                return json.loads(content)
            else:
                logger.error(f"Claude API 오류: {response.status_code}")
                return self._default_analysis()
                
        except Exception as e:
            logger.error(f"Claude 분석 실패: {e}")
            return self._default_analysis()
    
    def _default_analysis(self) -> Dict:
        """기본 분석 결과"""
        return {
            "is_recommendation": False,
            "government_positions": [],
            "summary": "분석 실패",
            "keywords": [],
            "recommendation_type": "의견제시",
            "confidence": 1
        }

class IntegratedTalentAnalyzer:
    """통합 인재 추천 분석기 (이메일 + 트위터)"""
    
    def __init__(self, db_path: str = "integrated_talent_analysis.db", 
                 email_folder: str = "emails", twitter_folder: str = "twitter_data"):
        self.db_path = db_path
        self.email_folder = email_folder
        self.twitter_folder = twitter_folder
        self.config = self._load_config()
        self.ai_provider = self._initialize_ai_provider()
        self.init_database()
    
    def _load_config(self) -> AnalysisConfig:
        """설정 파일 로드 또는 생성"""
        config_file = "ai_config.json"
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    return AnalysisConfig(**config_data)
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}")
        
        config = AnalysisConfig()
        self._save_config(config)
        return config
    
    def _save_config(self, config: AnalysisConfig):
        """설정 파일 저장"""
        config_data = {
            "ai_provider": config.ai_provider,
            "ollama_url": config.ollama_url,
            "ollama_model": config.ollama_model,
            "openai_api_key": config.openai_api_key,
            "openai_model": config.openai_model,
            "claude_api_key": config.claude_api_key,
            "claude_model": config.claude_model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
        
        with open("ai_config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def _initialize_ai_provider(self) -> AIProvider:
        """AI 제공자 초기화"""
        if self.config.ai_provider == "ollama":
            return OllamaProvider(self.config)
        elif self.config.ai_provider == "openai":
            return OpenAIProvider(self.config)
        elif self.config.ai_provider == "claude":
            return ClaudeProvider(self.config)
        else:
            logger.error(f"지원하지 않는 AI 제공자: {self.config.ai_provider}")
            return OllamaProvider(self.config)
    
    def init_database(self):
        """통합 데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 통합 분석 결과 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_type TEXT NOT NULL,  -- 'email' or 'twitter'
                source_file TEXT,
                content_id TEXT,
                username TEXT,
                display_name TEXT,
                is_recommendation INTEGER NOT NULL,
                government_positions TEXT,
                ai_summary TEXT,
                received_date TEXT,
                sender_email TEXT,
                ai_keywords TEXT,
                recommendation_type TEXT,  -- '본인지원', '타인추천', '의견제시'
                confidence_score INTEGER,
                ai_provider TEXT,
                likes_count INTEGER DEFAULT 0,
                retweets_count INTEGER DEFAULT 0,
                replies_count INTEGER DEFAULT 0,
                parent_post_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("통합 분석 데이터베이스 초기화 완료")
    
    def load_twitter_data_from_csv(self, csv_file: str) -> List[TwitterComment]:
        """CSV 파일에서 트위터 데이터 로드"""
        twitter_comments = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    comment = TwitterComment(
                        comment_id=row.get('comment_id', ''),
                        username=row.get('username', ''),
                        display_name=row.get('display_name', ''),
                        content=row.get('content', ''),
                        timestamp=row.get('timestamp', ''),
                        likes=int(row.get('likes', 0)),
                        retweets=int(row.get('retweets', 0)),
                        replies=int(row.get('replies', 0)),
                        parent_post_id=row.get('parent_post_id', ''),
                        parent_post_content=row.get('parent_post_content', '')
                    )
                    twitter_comments.append(comment)
            
            logger.info(f"CSV에서 {len(twitter_comments)}개의 트위터 댓글을 로드했습니다.")
            
        except Exception as e:
            logger.error(f"CSV 로드 오류: {e}")
        
        return twitter_comments
    
    def load_twitter_data_from_json(self, json_file: str) -> List[TwitterComment]:
        """JSON 파일에서 트위터 데이터 로드"""
        twitter_comments = []
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSON 구조에 따라 파싱 (예시)
            if isinstance(data, list):
                for item in data:
                    comment = TwitterComment(
                        comment_id=item.get('id', ''),
                        username=item.get('user', {}).get('username', ''),
                        display_name=item.get('user', {}).get('name', ''),
                        content=item.get('text', ''),
                        timestamp=item.get('created_at', ''),
                        likes=item.get('public_metrics', {}).get('like_count', 0),
                        retweets=item.get('public_metrics', {}).get('retweet_count', 0),
                        replies=item.get('public_metrics', {}).get('reply_count', 0),
                        parent_post_content=item.get('referenced_tweets', [{}])[0].get('text', '') if item.get('referenced_tweets') else ''
                    )
                    twitter_comments.append(comment)
            
            logger.info(f"JSON에서 {len(twitter_comments)}개의 트위터 댓글을 로드했습니다.")
            
        except Exception as e:
            logger.error(f"JSON 로드 오류: {e}")
        
        return twitter_comments
    
    def analyze_twitter_comment(self, comment: TwitterComment) -> Optional[Dict]:
        """트위터 댓글 AI 분석"""
        logger.info(f"트위터 댓글 AI 분석 시작: @{comment.username}")
        
        # AI 분석 수행
        ai_result = self.ai_provider.analyze_content(
            content=comment.content,
            subject=comment.parent_post_content,
            content_type="twitter"
        )
        
        return {
            'content_type': 'twitter',
            'source_file': '',
            'content_id': comment.comment_id,
            'username': comment.username,
            'display_name': comment.display_name,
            'is_recommendation': 1 if ai_result.get('is_recommendation', False) else 0,
            'government_positions': ', '.join(ai_result.get('government_positions', [])),
            'ai_summary': ai_result.get('summary', ''),
            'received_date': comment.timestamp,
            'sender_email': '',
            'ai_keywords': ', '.join(ai_result.get('keywords', [])),
            'recommendation_type': ai_result.get('recommendation_type', '의견제시'),
            'confidence_score': ai_result.get('confidence', 0),
            'ai_provider': self.config.ai_provider,
            'likes_count': comment.likes,
            'retweets_count': comment.retweets,
            'replies_count': comment.replies,
            'parent_post_content': comment.parent_post_content
        }
    
    def detect_encoding(self, file_path: str) -> str:
        """파일 인코딩 감지"""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def parse_email_file(self, file_path: str) -> Optional[email.message.Message]:
        """이메일 파일 파싱"""
        try:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                return email.message_from_file(f)
        except Exception as e:
            logger.error(f"이메일 파싱 오류 ({file_path}): {e}")
            return None
    
    def extract_email_content(self, msg: email.message.Message) -> str:
        """이메일 본문 내용 추출"""
        content = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            content += payload.decode('utf-8', errors='ignore')
                        except:
                            content += str(payload)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    content = payload.decode('utf-8', errors='ignore')
                except:
                    content = str(payload)
        
        return content.strip()
    
    def parse_date(self, date_str: str) -> str:
        """날짜 문자열 파싱"""
        if not date_str:
            return ""
        
        try:
            parsed_date = email.utils.parsedate_to_datetime(date_str)
            return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return date_str
    
    def extract_sender_info(self, msg: email.message.Message) -> Tuple[str, str]:
        """발신자 정보 추출"""
        from_header = msg.get('From', '')
        
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        email_match = re.search(email_pattern, from_header)
        sender_email = email_match.group(1) if email_match else ''
        
        sender_name = from_header.replace(f'<{sender_email}>', '').strip()
        sender_name = sender_name.replace('"', '').strip()
        
        return sender_name, sender_email
    
    def analyze_email_with_ai(self, file_path: str) -> Optional[Dict]:
        """AI를 이용한 이메일 분석"""
        msg = self.parse_email_file(file_path)
        if not msg:
            return None
        
        # 기본 정보 추출
        subject = msg.get('Subject', '')
        content = self.extract_email_content(msg)
        received_date = self.parse_date(msg.get('Date', ''))
        sender_name, sender_email = self.extract_sender_info(msg)
        
        # AI 분석 수행
        logger.info(f"이메일 AI 분석 시작: {os.path.basename(file_path)}")
        ai_result = self.ai_provider.analyze_content(content, subject, "email")
        
        return {
            'content_type': 'email',
            'source_file': os.path.basename(file_path),
            'content_id': '',
            'username': sender_name,
            'display_name': sender_name,
            'is_recommendation': 1 if ai_result.get('is_recommendation', False) else 0,
            'government_positions': ', '.join(ai_result.get('government_positions', [])),
            'ai_summary': ai_result.get('summary', ''),
            'received_date': received_date,
            'sender_email': sender_email,
            'ai_keywords': ', '.join(ai_result.get('keywords', [])),
            'recommendation_type': ai_result.get('recommendation_type', '타인추천'),
            'confidence_score': ai_result.get('confidence', 0),
            'ai_provider': self.config.ai_provider,
            'likes_count': 0,
            'retweets_count': 0,
            'replies_count': 0,
            'parent_post_content': subject
        }
    
    def save_to_database(self, analysis_result: Dict):
        """분석 결과를 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO content_analysis 
            (content_type, source_file, content_id, username, display_name,
             is_recommendation, government_positions, ai_summary, received_date,
             sender_email, ai_keywords, recommendation_type, confidence_score,
             ai_provider, likes_count, retweets_count, replies_count, parent_post_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_result['content_type'],
            analysis_result['source_file'],
            analysis_result['content_id'],
            analysis_result['username'],
            analysis_result['display_name'],
            analysis_result['is_recommendation'],
            analysis_result['government_positions'],
            analysis_result['ai_summary'],
            analysis_result['received_date'],
            analysis_result['sender_email'],
            analysis_result['ai_keywords'],
            analysis_result['recommendation_type'],
            analysis_result['confidence_score'],
            analysis_result['ai_provider'],
            analysis_result['likes_count'],
            analysis_result['retweets_count'],
            analysis_result['replies_count'],
            analysis_result['parent_post_content']
        ))
        
        conn.commit()
        conn.close()
    
    def process_all_emails(self):
        """모든 이메일 AI 분석 처리"""
        if not os.path.exists(self.email_folder):
            logger.warning(f"이메일 폴더가 존재하지 않습니다: {self.email_folder}")
            return 0
        
        # AI 연결 테스트
        if not self.ai_provider.test_connection():
            logger.error(f"{self.config.ai_provider} 연결에 실패했습니다. 설정을 확인해주세요.")
            return 0
        
        email_files = []
        for ext in ['*.eml', '*.msg', '*.txt']:
            email_files.extend(Path(self.email_folder).glob(ext))
        
        logger.info(f"총 {len(email_files)}개의 이메일 파일을 AI로 분석합니다.")
        
        processed = 0
        for file_path in email_files:
            try:
                analysis_result = self.analyze_email_with_ai(str(file_path))
                
                if analysis_result:
                    self.save_to_database(analysis_result)
                    processed += 1
                    logger.info(f"이메일 완료: {file_path.name} (신뢰도: {analysis_result.get('confidence_score', 0)})")
                
            except Exception as e:
                logger.error(f"이메일 분석 오류 ({file_path.name}): {e}")
        
        logger.info(f"총 {processed}개의 이메일이 성공적으로 AI 분석되었습니다.")
        return processed
    
    def process_all_twitter_data(self):
        """모든 트위터 데이터 AI 분석 처리"""
        if not os.path.exists(self.twitter_folder):
            logger.warning(f"트위터 데이터 폴더가 존재하지 않습니다: {self.twitter_folder}")
            return 0
        
        # AI 연결 테스트
        if not self.ai_provider.test_connection():
            logger.error(f"{self.config.ai_provider} 연결에 실패했습니다. 설정을 확인해주세요.")
            return 0
        
        # CSV 및 JSON 파일 찾기
        data_files = []
        data_files.extend(Path(self.twitter_folder).glob('*.csv'))
        data_files.extend(Path(self.twitter_folder).glob('*.json'))
        
        if not data_files:
            logger.warning("분석할 트위터 데이터 파일이 없습니다.")
            return 0
        
        logger.info(f"총 {len(data_files)}개의 트위터 데이터 파일을 처리합니다.")
        logger.info(f"사용중인 AI: {self.config.ai_provider}")
        
        total_processed = 0
        
        for data_file in data_files:
            try:
                logger.info(f"트위터 데이터 로드 중: {data_file.name}")
                
                # 파일 형식에 따라 로드
                if data_file.suffix.lower() == '.csv':
                    twitter_comments = self.load_twitter_data_from_csv(str(data_file))
                elif data_file.suffix.lower() == '.json':
                    twitter_comments = self.load_twitter_data_from_json(str(data_file))
                else:
                    continue
                
                # 각 댓글 분석
                processed = 0
                for comment in twitter_comments:
                    try:
                        analysis_result = self.analyze_twitter_comment(comment)
                        
                        if analysis_result:
                            self.save_to_database(analysis_result)
                            processed += 1
                            total_processed += 1
                            
                            logger.info(f"트위터 완료: @{comment.username} (신뢰도: {analysis_result.get('confidence_score', 0)})")
                    
                    except Exception as e:
                        logger.error(f"트위터 댓글 분석 오류 (@{comment.username}): {e}")
                
                logger.info(f"{data_file.name}에서 {processed}개 댓글 분석 완료")
                
            except Exception as e:
                logger.error(f"파일 처리 오류 ({data_file.name}): {e}")
        
        logger.info(f"총 {total_processed}개의 트위터 댓글이 성공적으로 AI 분석되었습니다.")
        return total_processed
    
    def configure_ai_settings(self):
        """AI 설정 구성"""
        print("\n=== AI 설정 구성 ===")
        print("1. Ollama (로컬 AI)")
        print("2. OpenAI GPT")
        print("3. Claude")
        
        choice = input("AI 제공자 선택 (1-3): ").strip()
        
        if choice == '1':
            self.config.ai_provider = "ollama"
            self.config.ollama_url = input(f"Ollama URL [{self.config.ollama_url}]: ").strip() or self.config.ollama_url
            self.config.ollama_model = input(f"Ollama 모델 [{self.config.ollama_model}]: ").strip() or self.config.ollama_model
            
        elif choice == '2':
            self.config.ai_provider = "openai"
            api_key = input("OpenAI API 키: ").strip()
            if api_key:
                self.config.openai_api_key = api_key
            self.config.openai_model = input(f"OpenAI 모델 [{self.config.openai_model}]: ").strip() or self.config.openai_model
            
        elif choice == '3':
            self.config.ai_provider = "claude"
            api_key = input("Claude API 키: ").strip()
            if api_key:
                self.config.claude_api_key = api_key
            self.config.claude_model = input(f"Claude 모델 [{self.config.claude_model}]: ").strip() or self.config.claude_model
        
        # 공통 설정
        temp_input = input(f"Temperature (0.0-1.0) [{self.config.temperature}]: ").strip()
        if temp_input:
            try:
                self.config.temperature = float(temp_input)
            except ValueError:
                pass
        
        tokens_input = input(f"Max Tokens [{self.config.max_tokens}]: ").strip()
        if tokens_input:
            try:
                self.config.max_tokens = int(tokens_input)
            except ValueError:
                pass
        
        # 설정 저장
        self._save_config(self.config)
        
        # AI 제공자 재초기화
        self.ai_provider = self._initialize_ai_provider()
        
        # 연결 테스트
        if self.ai_provider.test_connection():
            print(f"✅ {self.config.ai_provider} 연결 성공!")
        else:
            print(f"❌ {self.config.ai_provider} 연결 실패!")
    
    def get_integrated_statistics(self) -> Dict:
        """통합 분석 통계 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 전체 통계
        cursor.execute("SELECT COUNT(*) FROM content_analysis")
        total_content = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM content_analysis WHERE content_type = 'email'")
        total_emails = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM content_analysis WHERE content_type = 'twitter'")
        total_twitter = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM content_analysis WHERE is_recommendation = 1")
        recommendation_content = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence_score) FROM content_analysis")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # 추천 유형별 통계
        cursor.execute("""
            SELECT recommendation_type, COUNT(*) 
            FROM content_analysis 
            WHERE is_recommendation = 1 
            GROUP BY recommendation_type 
            ORDER BY COUNT(*) DESC
        """)
        recommendation_type_stats = cursor.fetchall()
        
        # 정부 직책별 통계
        cursor.execute("""
            SELECT government_positions, COUNT(*) 
            FROM content_analysis 
            WHERE government_positions != '' 
            GROUP BY government_positions 
            ORDER BY COUNT(*) DESC
        """)
        position_stats = cursor.fetchall()
        
        # 플랫폼별 추천 통계
        cursor.execute("""
            SELECT content_type, 
                   COUNT(*) as total,
                   SUM(is_recommendation) as recommendations
            FROM content_analysis 
            GROUP BY content_type
        """)
        platform_stats = cursor.fetchall()
        
        # 트위터 인기도 통계 (추천된 댓글 중)
        cursor.execute("""
            SELECT AVG(likes_count), AVG(retweets_count), AVG(replies_count)
            FROM content_analysis 
            WHERE content_type = 'twitter' AND is_recommendation = 1
        """)
        twitter_engagement = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_content': total_content,
            'total_emails': total_emails,
            'total_twitter': total_twitter,
            'recommendation_content': recommendation_content,
            'non_recommendation_content': total_content - recommendation_content,
            'average_confidence': round(avg_confidence, 2),
            'recommendation_type_statistics': recommendation_type_stats,
            'position_statistics': position_stats,
            'platform_statistics': platform_stats,
            'twitter_engagement': {
                'avg_likes': round(twitter_engagement[0] or 0, 1),
                'avg_retweets': round(twitter_engagement[1] or 0, 1),
                'avg_replies': round(twitter_engagement[2] or 0, 1)
            }
        }
    
    def export_to_csv(self, output_file: str = "integrated_analysis_results.csv"):
        """통합 분석 결과를 CSV로 내보내기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content_type, source_file, content_id, username, display_name,
                   is_recommendation, government_positions, ai_summary, received_date,
                   sender_email, ai_keywords, recommendation_type, confidence_score,
                   ai_provider, likes_count, retweets_count, replies_count,
                   parent_post_content, created_at
            FROM content_analysis
            ORDER BY created_at DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        with open(output_file, 'w', encoding='utf-8-sig') as f:
            f.write("플랫폼,소스파일,컨텐츠ID,사용자명,표시명,추천여부,정부직책,AI요약,수신일자,이메일,AI키워드,추천유형,신뢰도,AI제공자,좋아요,리트윗,댓글수,원글내용,처리일시\n")
            
            for row in results:
                csv_row = []
                for field in row:
                    if field is None:
                        csv_row.append("")
                    else:
                        field_str = str(field).replace('"', '""')
                        if ',' in field_str or '\n' in field_str:
                            csv_row.append(f'"{field_str}"')
                        else:
                            csv_row.append(field_str)
                
                f.write(','.join(csv_row) + '\n')
        
        logger.info(f"통합 분석 결과가 {output_file}로 내보내졌습니다.")


def create_sample_twitter_data():
    """샘플 트위터 데이터 생성"""
    print("\n🐦 샘플 트위터 데이터를 생성합니다...")
    
    # twitter_data 폴더 생성
    os.makedirs("twitter_data", exist_ok=True)
    
    # 샘플 트위터 댓글 데이터
    sample_twitter_csv = """comment_id,username,display_name,content,timestamp,likes,retweets,replies,parent_post_id,parent_post_content
1001,kim_ai_expert,김AI전문가,"정부 AI 정책관 모집에 지원하고 싶습니다. 서울대 AI학과 박사과정이고 삼성에서 3년간 AI 연구했습니다. #정부채용 #AI정책",2025-06-13 10:30:00,15,3,2,post_001,"정부에서 AI 정책관을 모집합니다. 자격을 갖춘 분들의 많은 지원 바랍니다."
1002,lee_professor,이교수,"우리 연구실 박사과정 박영수님을 추천합니다. 환경공학 전공으로 기후변화 정책 연구에 탁월한 능력을 보여주고 있습니다.",2025-06-13 11:15:00,28,5,1,post_002,"환경부 기후변화 담당관 모집 공고"
1003,ordinary_citizen,일반시민,"정부에서 일하는 것도 좋겠네요. 그런데 자격 요건이 너무 까다로운 것 같아요.",2025-06-13 12:00:00,3,0,1,post_001,"정부에서 AI 정책관을 모집합니다. 자격을 갖춘 분들의 많은 지원 바랍니다."
1004,data_scientist,데이터과학자박사,"빅데이터 분석 경험 10년, 정부 자문위원 3년 경력 있습니다. 디지털정책관 지원 희망합니다.",2025-06-13 14:20:00,12,2,0,post_003,"디지털정부 혁신을 위한 정책관 모집"
1005,skeptical_user,회의적사용자,"또 정부 일자리 만들기네요. 실제로 성과가 있을까요?",2025-06-13 15:45:00,1,0,3,post_003,"디지털정부 혁신을 위한 정책관 모집"
"""
    
    # CSV 파일 생성
    csv_file = os.path.join("twitter_data", "sample_twitter_comments.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(sample_twitter_csv)
    
    # JSON 파일도 생성
    sample_twitter_json = [
        {
            "id": "1006",
            "user": {
                "username": "startup_ceo",
                "name": "스타트업대표"
            },
            "text": "정부 스타트업 정책관에 지원하겠습니다. 테크 스타트업 5개 창업 경험과 정부 R&D 과제 수행 경험 있습니다.",
            "created_at": "2025-06-13T16:30:00Z",
            "public_metrics": {
                "like_count": 20,
                "retweet_count": 4,
                "reply_count": 2
            },
            "referenced_tweets": [
                {
                    "text": "정부에서 스타트업 생태계 활성화를 위한 정책관을 모집합니다."
                }
            ]
        },
        {
            "id": "1007",
            "user": {
                "username": "policy_researcher",
                "name": "정책연구원"
            },
            "text": "국정원 출신 동료 추천드립니다. 보안 정책 전문가로 사이버보안 담당관에 적합합니다.",
            "created_at": "2025-06-13T17:15:00Z",
            "public_metrics": {
                "like_count": 8,
                "retweet_count": 1,
                "reply_count": 0
            },
            "referenced_tweets": [
                {
                    "text": "국가 사이버보안 강화를 위한 전문 담당관을 모집합니다."
                }
            ]
        }
    ]
    
    json_file = os.path.join("twitter_data", "sample_twitter_comments.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(sample_twitter_json, f, indent=2, ensure_ascii=False)
    
    print(f"✅ CSV 파일 생성: {csv_file}")
    print(f"✅ JSON 파일 생성: {json_file}")
    print(f"\n총 7개의 샘플 트위터 댓글이 생성되었습니다.")
    print("이제 '트위터 댓글 분석 실행' 메뉴를 사용하여 분석해보세요!")


def create_sample_emails():
    """샘플 이메일 파일 생성"""
    print("\n📧 샘플 이메일 파일을 생성합니다...")
    
    # emails 폴더 생성
    os.makedirs("emails", exist_ok=True)
    
    # 샘플 이메일 데이터
    sample_emails = [
        {
            "filename": "recommendation_001.eml",
            "content": """From: 김철수 <kim.cs@university.ac.kr>
To: talent@government.go.kr
Subject: 정부 AI 정책관 추천
Date: Mon, 13 Jun 2025 10:30:00 +0900

안녕하세요.

서울대학교 컴퓨터공학과 김철수 교수입니다.
정부의 AI 정책관 직책에 저희 연구실 박사과정 이영희 님을 추천하고자 합니다.

이영희 님은 다음과 같은 우수한 경력을 보유하고 있습니다:
- 서울대학교 컴퓨터공학과 박사과정 (AI 전공)
- 삼성전자 AI 연구소 인턴 3년 경력
- 정부 빅데이터 정책 자문위원 활동
- AI 윤리 관련 논문 10편 발표
- 한국정보과학회 우수논문상 수상

특히 AI 정책 수립과 기술적 이해를 동시에 갖춘 인재로서 정부 AI 정책관 업무에 매우 적합하다고 판단됩니다.

연락처: 010-1234-5678
이메일: lee.yh@snu.ac.kr

감사합니다.

김철수 교수
서울대학교 컴퓨터공학과""",
        },
        {
            "filename": "application_002.eml",
            "content": """From: 박민수 <park.ms@korea.ac.kr>
To: recruit@ministry.go.kr  
Subject: 환경부 기후변화 담당관 지원
Date: Tue, 14 Jun 2025 14:15:00 +0900

환경부 기후변화 담당관 모집에 지원합니다.

저는 고려대학교 환경공학과에서 박사학위를 취득하였으며, 다음과 같은 전문성을 보유하고 있습니다:

학력:
- 고려대학교 환경공학과 박사 (2023년)
- 연세대학교 환경공학과 석사 (2020년)

경력:
- 한국환경연구원 연구원 2년
- 기후변화센터 선임연구원 1년
- UN 기후변화협약 한국 대표단 참여

주요 성과:
- 기후변화 적응정책 연구 5편
- 탄소중립 시나리오 개발 참여
- 환경영향평가 전문가 자격

기후변화 대응 정책 수립에 기여하고 싶습니다.

박민수
연락처: 010-9876-5432""",
        }
    ]
    
    # 샘플 파일 생성
    for sample in sample_emails:
        file_path = os.path.join("emails", sample["filename"])
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample["content"])
        print(f"✅ 생성됨: {sample['filename']}")
    
    print(f"\n총 {len(sample_emails)}개의 샘플 이메일이 'emails' 폴더에 생성되었습니다.")


def install_ollama_guide():
    """Ollama 설치 가이드"""
    print("\n" + "="*60)
    print("Ollama 설치 가이드 (로컬 AI)")
    print("="*60)
    print("""
1. Ollama 다운로드 및 설치:
   - Windows/Mac: https://ollama.ai/download
   - Linux: curl -fsSL https://ollama.ai/install.sh | sh

2. 한국어 지원 모델 다운로드:
   ollama pull llama3.2:latest
   ollama pull qwen2:7b
   ollama pull solar:10.7b

3. 서비스 시작:
   ollama serve

4. 연결 확인:
   http://localhost:11434

권장 모델:
- llama3.2:latest (영어/한국어 균형)
- qwen2:7b (한국어 특화)  
- solar:10.7b (고성능, 더 큰 메모리 필요)

최소 시스템 요구사항:
- RAM: 8GB 이상
- GPU: 선택사항 (NVIDIA GPU 권장)
""")


def twitter_data_format_guide():
    """트위터 데이터 형식 가이드"""
    print("\n" + "="*70)
    print("🐦 트위터 데이터 형식 가이드")
    print("="*70)
    print("""
지원하는 파일 형식:
1. CSV 파일 (.csv)
2. JSON 파일 (.json)

=== CSV 파일 형식 ===
필수 컬럼:
- comment_id: 댓글 고유 ID
- username: 사용자명
- display_name: 표시명
- content: 댓글 내용
- timestamp: 작성 시간

선택 컬럼:
- likes: 좋아요 수
- retweets: 리트윗 수
- replies: 댓글 수
- parent_post_content: 원글 내용

CSV 예시:
comment_id,username,display_name,content,timestamp,likes,retweets,replies,parent_post_content
1001,user123,사용자명,"댓글 내용",2025-06-13 10:30:00,5,1,0,"원래 글 내용"

=== JSON 파일 형식 ===
Twitter API v2 형식 지원:
[
  {
    "id": "댓글ID",
    "user": {
      "username": "사용자명",
      "name": "표시명"
    },
    "text": "댓글 내용",
    "created_at": "2025-06-13T10:30:00Z",
    "public_metrics": {
      "like_count": 5,
      "retweet_count": 1,
      "reply_count": 0
    },
    "referenced_tweets": [
      {
        "text": "원래 글 내용"
      }
    ]
  }
]

=== 데이터 수집 방법 ===
1. Twitter API 사용
2. 타사 도구 (Tweepy, snscrape 등)
3. 수동 데이터 입력

파일은 'twitter_data' 폴더에 저장하세요.
""")


def requirements_guide():
    """필요 라이브러리 설치 가이드"""
    print("\n" + "="*60)
    print("필요 라이브러리 설치")
    print("="*60)
    print("""
다음 명령어로 필요한 라이브러리를 설치하세요:

pip install requests chardet

선택적 설치 (더 나은 성능):
pip install python-magic  # 파일 타입 감지 개선
pip install beautifulsoup4  # HTML 이메일 파싱
pip install pandas  # 데이터 분석
pip install tweepy  # Twitter API 접근

지원하는 파일 형식:
이메일: .eml, .msg, .txt
트위터: .csv, .json
""")


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🤖🐦 통합 소셜미디어 인재 추천 분석 시스템")
    print("Integrated Social Media Talent Recommendation Analysis System")
    print("이메일 + 트위터 댓글 AI 분석")
    print("=" * 80)
    
    # 통합 분석기 초기화
    analyzer = IntegratedTalentAnalyzer()
    
    while True:
        print(f"\n현재 AI 제공자: {analyzer.config.ai_provider.upper()}")
        print("\n📋 메뉴를 선택하세요:")
        print("=" * 50)
        print("🔍 분석 실행")
        print("1. 📧 이메일 분석 실행")
        print("2. 🐦 트위터 댓글 분석 실행")
        print("3. 🔄 통합 분석 실행 (이메일 + 트위터)")
        print("\n⚙️ 설정 및 관리")
        print("4. 🤖 AI 설정 구성")
        print("5. 📊 통합 통계 조회")
        print("6. 📄 결과 CSV 내보내기")
        print("\n🛠️ 도구 및 가이드")
        print("7. 🏠 Ollama 설치 가이드")
        print("8. 📦 라이브러리 설치 가이드")
        print("9. 🐦 트위터 데이터 형식 가이드")
        print("10. 🔌 AI 연결 테스트")
        print("\n📝 샘플 데이터")
        print("11. 📧 샘플 이메일 생성")
        print("12. 🐦 샘플 트위터 데이터 생성")
        print("13. 📊 전체 샘플 데이터 생성")
        print("\n14. ❌ 종료")
        print("=" * 50)
        
        choice = input("선택 (1-14): ").strip()
        
        if choice == '1':
            print(f"\n📧 {analyzer.config.ai_provider.upper()}를 이용한 이메일 분석을 시작합니다...")
            processed = analyzer.process_all_emails()
            if processed > 0:
                print(f"✅ {processed}개의 이메일이 성공적으로 분석되었습니다!")
            
        elif choice == '2':
            print(f"\n🐦 {analyzer.config.ai_provider.upper()}를 이용한 트위터 댓글 분석을 시작합니다...")
            processed = analyzer.process_all_twitter_data()
            if processed > 0:
                print(f"✅ {processed}개의 트위터 댓글이 성공적으로 분석되었습니다!")
            
        elif choice == '3':
            print(f"\n🔄 {analyzer.config.ai_provider.upper()}를 이용한 통합 분석을 시작합니다...")
            email_processed = analyzer.process_all_emails()
            twitter_processed = analyzer.process_all_twitter_data()
            total_processed = email_processed + twitter_processed
            if total_processed > 0:
                print(f"✅ 총 {total_processed}개 콘텐츠가 분석되었습니다! (이메일: {email_processed}, 트위터: {twitter_processed})")
            
        elif choice == '4':
            analyzer.configure_ai_settings()
            
        elif choice == '5':
            print("\n📊 통합 분석 통계:")
            stats = analyzer.get_integrated_statistics()
            print(f"- 전체 분석된 콘텐츠: {stats['total_content']}건")
            print(f"  └ 이메일: {stats['total_emails']}건")
            print(f"  └ 트위터: {stats['total_twitter']}건")
            print(f"- AI가 추천으로 분류: {stats['recommendation_content']}건")
            print(f"- AI가 일반으로 분류: {stats['non_recommendation_content']}건")
            print(f"- 평균 신뢰도: {stats['average_confidence']}/10")
            
            if stats['recommendation_type_statistics']:
                print("\n추천 유형별 통계:")
                for rec_type, count in stats['recommendation_type_statistics']:
                    print(f"  - {rec_type}: {count}건")
            
            if stats['platform_statistics']:
                print("\n플랫폼별 분석 통계:")
                for platform, total, recommendations in stats['platform_statistics']:
                    print(f"  - {platform.upper()}: {total}건 (추천: {recommendations}건)")
            
            if stats['twitter_engagement']['avg_likes'] > 0:
                engagement = stats['twitter_engagement']
                print(f"\n트위터 추천 댓글 평균 인기도:")
                print(f"  - 좋아요: {engagement['avg_likes']}개")
                print(f"  - 리트윗: {engagement['avg_retweets']}개")
                print(f"  - 댓글: {engagement['avg_replies']}개")
            
            if stats['position_statistics']:
                print("\n정부 직책별 통계 (상위 10개):")
                for position, count in stats['position_statistics'][:10]:
                    if position:
                        print(f"  - {position}: {count}건")
            
        elif choice == '6':
            output_file = input("CSV 파일명 (기본값: integrated_analysis_results.csv): ").strip()
            if not output_file:
                output_file = "integrated_analysis_results.csv"
            analyzer.export_to_csv(output_file)
            
        elif choice == '7':
            install_ollama_guide()
            
        elif choice == '8':
            requirements_guide()
            
        elif choice == '9':
            twitter_data_format_guide()
            
        elif choice == '10':
            print(f"\n🔌 {analyzer.config.ai_provider.upper()} 연결을 테스트합니다...")
            if analyzer.ai_provider.test_connection():
                print("✅ 연결 성공!")
            else:
                print("❌ 연결 실패! 설정을 확인해주세요.")
            
        elif choice == '11':
            create_sample_emails()
            
        elif choice == '12':
            create_sample_twitter_data()
            
        elif choice == '13':
            print("\n📊 전체 샘플 데이터를 생성합니다...")
            create_sample_emails()
            create_sample_twitter_data()
            print("✅ 모든 샘플 데이터가 생성되었습니다!")
            
        elif choice == '14':
            print("프로그램을 종료합니다. 감사합니다!")
            break
            
        else:
            print("❌ 잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()