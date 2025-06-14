#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정부 인재 추천 이메일 분석 및 데이터베이스화 시스템
Government Talent Recommendation Email Analysis System

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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import chardet
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmailAnalyzer:
    """이메일 분석 및 데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "government_talent.db", email_folder: str = "emails"):
        self.db_path = db_path
        self.email_folder = email_folder
        self.init_database()
        
        # 정부 직책명 키워드 (확장 가능)
        self.government_positions = [
            "장관", "차관", "실장", "국장", "과장", "담당관", "주무관",
            "청장", "원장", "소장", "팀장", "부장", "센터장", "본부장",
            "위원장", "위원", "이사장", "사장", "대표", "총장", "학장",
            "연구원", "교수", "박사", "전문위원", "고문", "자문위원",
            "정책관", "기획관", "심사관", "조사관", "분석관", "연구관"
        ]
        
        # 추천 관련 키워드
        self.recommendation_keywords = [
            "추천", "천거", "제안", "지원", "후보", "적합", "우수", "경험",
            "자격", "능력", "전문성", "expertise", "recommend", "candidate"
        ]
    
    def init_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                is_recommendation INTEGER NOT NULL,  -- 0: 아니오, 1: 예
                government_position TEXT,
                self_introduction_summary TEXT,
                received_date TEXT,
                sender_name TEXT,
                sender_email TEXT,
                keywords TEXT,  -- JSON 형태로 저장
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("데이터베이스 초기화 완료")
    
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
    
    def classify_recommendation(self, content: str, subject: str) -> bool:
        """추천 이메일 여부 분류"""
        text = (content + " " + subject).lower()
        
        recommendation_count = sum(1 for keyword in self.recommendation_keywords 
                                 if keyword in text)
        
        # 추천 키워드가 2개 이상 있으면 추천 이메일로 분류
        return recommendation_count >= 2
    
    def extract_government_position(self, content: str, subject: str) -> str:
        """정부 직책명 추출"""
        text = content + " " + subject
        found_positions = []
        
        for position in self.government_positions:
            if position in text:
                found_positions.append(position)
        
        return ", ".join(found_positions) if found_positions else ""
    
    def summarize_self_introduction(self, content: str) -> str:
        """자기소개 내용 요약"""
        # 간단한 요약: 첫 200자 + 주요 경력 키워드
        summary = content[:200] + "..." if len(content) > 200 else content
        
        # 경력 관련 키워드 추출
        career_keywords = ["경력", "경험", "근무", "재직", "졸업", "학위", "자격", "수상"]
        career_info = []
        
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line for keyword in career_keywords):
                career_info.append(line.strip())
        
        if career_info:
            summary += "\n\n[주요 경력]\n" + "\n".join(career_info[:3])
        
        return summary
    
    def extract_keywords(self, content: str) -> List[str]:
        """키워드 추출"""
        # 한국어 키워드 추출을 위한 간단한 방법
        keywords = []
        
        # 학력 관련
        education_pattern = r'(대학교|대학|석사|박사|학사|졸업|전공)'
        education_matches = re.findall(education_pattern, content)
        keywords.extend(education_matches)
        
        # 경력 관련
        career_pattern = r'(\d+년|경력|근무|재직|담당|업무|프로젝트)'
        career_matches = re.findall(career_pattern, content)
        keywords.extend(career_matches)
        
        # 기술/전문 분야
        tech_pattern = r'(IT|AI|빅데이터|정책|기획|관리|운영|개발|연구|분석)'
        tech_matches = re.findall(tech_pattern, content)
        keywords.extend(tech_matches)
        
        # 중복 제거 및 빈도 기반 정렬
        unique_keywords = list(set(keywords))
        return unique_keywords[:10]  # 상위 10개만 반환
    
    def parse_date(self, date_str: str) -> str:
        """날짜 문자열 파싱"""
        if not date_str:
            return ""
        
        try:
            # 이메일 날짜 형식 파싱
            parsed_date = email.utils.parsedate_to_datetime(date_str)
            return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return date_str
    
    def extract_sender_info(self, msg: email.message.Message) -> Tuple[str, str]:
        """발신자 정보 추출"""
        from_header = msg.get('From', '')
        
        # 이메일 주소 추출
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        email_match = re.search(email_pattern, from_header)
        sender_email = email_match.group(1) if email_match else ''
        
        # 발신자 이름 추출
        sender_name = from_header.replace(f'<{sender_email}>', '').strip()
        sender_name = sender_name.replace('"', '').strip()
        
        return sender_name, sender_email
    
    def analyze_email(self, file_path: str) -> Dict:
        """개별 이메일 분석"""
        msg = self.parse_email_file(file_path)
        if not msg:
            return None
        
        # 기본 정보 추출
        subject = msg.get('Subject', '')
        content = self.extract_email_content(msg)
        received_date = self.parse_date(msg.get('Date', ''))
        sender_name, sender_email = self.extract_sender_info(msg)
        
        # 분석 수행
        is_recommendation = self.classify_recommendation(content, subject)
        government_position = self.extract_government_position(content, subject)
        self_intro_summary = self.summarize_self_introduction(content)
        keywords = self.extract_keywords(content)
        
        return {
            'file_name': os.path.basename(file_path),
            'is_recommendation': 1 if is_recommendation else 0,
            'government_position': government_position,
            'self_introduction_summary': self_intro_summary,
            'received_date': received_date,
            'sender_name': sender_name,
            'sender_email': sender_email,
            'keywords': ', '.join(keywords)
        }
    
    def save_to_database(self, analysis_result: Dict):
        """분석 결과를 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO email_analysis 
            (file_name, is_recommendation, government_position, self_introduction_summary,
             received_date, sender_name, sender_email, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_result['file_name'],
            analysis_result['is_recommendation'],
            analysis_result['government_position'],
            analysis_result['self_introduction_summary'],
            analysis_result['received_date'],
            analysis_result['sender_name'],
            analysis_result['sender_email'],
            analysis_result['keywords']
        ))
        
        conn.commit()
        conn.close()
    
    def process_all_emails(self):
        """모든 이메일 처리"""
        if not os.path.exists(self.email_folder):
            logger.error(f"이메일 폴더가 존재하지 않습니다: {self.email_folder}")
            return
        
        email_files = []
        for ext in ['*.eml', '*.msg', '*.txt']:
            email_files.extend(Path(self.email_folder).glob(ext))
        
        logger.info(f"총 {len(email_files)}개의 이메일 파일을 처리합니다.")
        
        processed = 0
        for file_path in email_files:
            try:
                logger.info(f"처리 중: {file_path.name}")
                analysis_result = self.analyze_email(str(file_path))
                
                if analysis_result:
                    self.save_to_database(analysis_result)
                    processed += 1
                    logger.info(f"완료: {file_path.name}")
                
            except Exception as e:
                logger.error(f"처리 오류 ({file_path.name}): {e}")
        
        logger.info(f"총 {processed}개의 이메일이 성공적으로 처리되었습니다.")
    
    def get_statistics(self) -> Dict:
        """통계 정보 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 전체 이메일 수
        cursor.execute("SELECT COUNT(*) FROM email_analysis")
        total_emails = cursor.fetchone()[0]
        
        # 추천 이메일 수
        cursor.execute("SELECT COUNT(*) FROM email_analysis WHERE is_recommendation = 1")
        recommendation_emails = cursor.fetchone()[0]
        
        # 정부 직책별 통계
        cursor.execute("""
            SELECT government_position, COUNT(*) 
            FROM email_analysis 
            WHERE government_position != '' 
            GROUP BY government_position 
            ORDER BY COUNT(*) DESC
        """)
        position_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_emails': total_emails,
            'recommendation_emails': recommendation_emails,
            'non_recommendation_emails': total_emails - recommendation_emails,
            'position_statistics': position_stats
        }
    
    def export_to_csv(self, output_file: str = "email_analysis_results.csv"):
        """결과를 CSV로 내보내기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT file_name, is_recommendation, government_position, 
                   self_introduction_summary, received_date, sender_name, 
                   sender_email, keywords, created_at
            FROM email_analysis
            ORDER BY created_at DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        with open(output_file, 'w', encoding='utf-8-sig') as f:
            # CSV 헤더
            f.write("파일명,추천여부,정부직책,자기소개요약,수신일자,발신자명,발신자이메일,키워드,처리일시\n")
            
            for row in results:
                # CSV 형식으로 변환 (쉼표와 따옴표 처리)
                csv_row = []
                for field in row:
                    if field is None:
                        csv_row.append("")
                    else:
                        # 쉼표가 포함된 필드는 따옴표로 감싸기
                        field_str = str(field).replace('"', '""')
                        if ',' in field_str or '\n' in field_str:
                            csv_row.append(f'"{field_str}"')
                        else:
                            csv_row.append(field_str)
                
                f.write(','.join(csv_row) + '\n')
        
        logger.info(f"결과가 {output_file}로 내보내졌습니다.")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("정부 인재 추천 이메일 분석 시스템")
    print("Government Talent Recommendation Email Analysis System")
    print("=" * 60)
    
    # 이메일 분석기 초기화
    analyzer = EmailAnalyzer()
    
    while True:
        print("\n메뉴를 선택하세요:")
        print("1. 이메일 분석 실행")
        print("2. 통계 조회")
        print("3. CSV 내보내기")
        print("4. 종료")
        
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == '1':
            print("\n이메일 분석을 시작합니다...")
            analyzer.process_all_emails()
            
        elif choice == '2':
            print("\n통계 정보:")
            stats = analyzer.get_statistics()
            print(f"- 전체 이메일: {stats['total_emails']}건")
            print(f"- 추천 이메일: {stats['recommendation_emails']}건")
            print(f"- 일반 이메일: {stats['non_recommendation_emails']}건")
            
            if stats['position_statistics']:
                print("\n정부 직책별 통계:")
                for position, count in stats['position_statistics'][:10]:
                    print(f"  - {position}: {count}건")
            
        elif choice == '3':
            output_file = input("CSV 파일명 (기본값: email_analysis_results.csv): ").strip()
            if not output_file:
                output_file = "email_analysis_results.csv"
            analyzer.export_to_csv(output_file)
            
        elif choice == '4':
            print("프로그램을 종료합니다.")
            break
            
        else:
            print("잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()
