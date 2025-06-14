#!/usr/bin/env python3
"""
📝 지능적인 체인지로그 생성기
TRAS 프로젝트용 스마트 릴리스 노트 자동 생성 도구

기능:
- 커밋 메시지 의미 분석 및 분류
- 한국어 키워드 기반 카테고리 자동 분류
- 중요도별 정렬 및 그룹화
- 상세한 릴리스 노트 자동 생성

사용법:
    python scripts/changelog_generator.py v3.2.0 v3.3.0
    python scripts/changelog_generator.py v3.2.0            # 최신까지
    python scripts/changelog_generator.py --last            # 마지막 태그부터
"""

import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


class CommitAnalyzer:
    """커밋 메시지 분석기"""
    
    def __init__(self):
        # 한국어 키워드 기반 카테고리 분류
        self.categories = {
            '🚀 새로운 기능': [
                '새로운', '추가', '신규', '기능', '모듈', '시스템', '지원', 
                'BERT', '분석', 'AI', '통합', '구현', '개발', '생성'
            ],
            '🔧 개선 사항': [
                '개선', '향상', '최적화', '업그레이드', '강화', '성능', 
                '효율', '속도', '품질', '사용성', '편의'
            ],
            '🐛 버그 수정': [
                '수정', '버그', '오류', '문제', '해결', '픽스', 'fix', 
                '복구', '정정', '에러'
            ],
            '📚 문서 업데이트': [
                '문서', 'README', '가이드', '설명', '매뉴얼', '튜토리얼',
                '주석', '코멘트', '도움말', 'md'
            ],
            '⚙️ 설정 및 구성': [
                '설정', '구성', '환경', 'config', 'pyproject', 'toml',
                '의존성', '라이브러리', '패키지', '버전', '릴리스'
            ],
            '🎨 UI/UX 개선': [
                'UI', 'UX', '인터페이스', '디자인', '화면', '메뉴',
                '사용자', '경험', '편의성'
            ],
            '🧪 테스트': [
                '테스트', 'test', '검증', '확인', '점검', '시험'
            ],
            '🔒 보안': [
                '보안', 'security', '인증', '권한', '암호화', '취약점'
            ]
        }
        
        # 중요도 키워드
        self.importance_keywords = {
            'critical': ['긴급', '치명적', '중요', 'critical', '필수'],
            'major': ['주요', '대규모', 'major', '핵심', '메이저'],
            'minor': ['마이너', 'minor', '소규모', '작은'],
            'patch': ['패치', 'patch', '수정', '미세']
        }

    def classify_commit(self, message: str) -> Tuple[str, str, int]:
        """커밋 메시지를 분류하고 중요도를 평가"""
        message_lower = message.lower()
        
        # 카테고리 분류
        best_category = '🔄 기타 변경사항'
        max_score = 0
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > max_score:
                max_score = score
                best_category = category
        
        # 중요도 평가 (1-5)
        importance = 2  # 기본값
        
        for level, keywords in self.importance_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                if level == 'critical':
                    importance = 5
                elif level == 'major':
                    importance = 4
                elif level == 'minor':
                    importance = 2
                elif level == 'patch':
                    importance = 1
                break
        
        # 이모지나 특별한 패턴으로 중요도 조정
        if any(emoji in message for emoji in ['🚨', '⚠️', '❌']):
            importance = max(importance, 4)
        elif any(emoji in message for emoji in ['🚀', '✨', '🎉']):
            importance = max(importance, 3)
        
        return best_category, message, importance

    def extract_summary(self, message: str) -> str:
        """커밋 메시지에서 핵심 요약 추출"""
        # 이모지 및 태그 제거
        clean_message = re.sub(r'[🎉🚀✨🔧🐛📚⚙️🎨🧪🔒🔖📝🔄⚠️🚨❌✅]', '', message)
        clean_message = re.sub(r'^\w+:\s*', '', clean_message.strip())
        
        # 너무 긴 메시지는 첫 번째 문장만
        sentences = clean_message.split('.')
        if sentences and len(sentences[0]) > 10:
            return sentences[0].strip()
        
        return clean_message[:100] + ('...' if len(clean_message) > 100 else '')


class ChangelogGenerator:
    """체인지로그 생성기"""
    
    def __init__(self):
        self.analyzer = CommitAnalyzer()
    
    def get_commit_range(self, from_tag: str = None, to_tag: str = None) -> str:
        """커밋 범위 결정"""
        if not to_tag:
            to_tag = "HEAD"
        
        if not from_tag:
            # 이전 태그 자동 찾기
            try:
                result = subprocess.run([
                    "git", "describe", "--tags", "--abbrev=0", "HEAD^"
                ], capture_output=True, text=True, check=True)
                from_tag = result.stdout.strip()
            except subprocess.CalledProcessError:
                # 이전 태그가 없으면 처음부터
                return to_tag
        
        return f"{from_tag}..{to_tag}"
    
    def get_commits(self, commit_range: str) -> List[Dict]:
        """커밋 정보 가져오기"""
        try:
            # 커밋 로그 가져오기 (해시, 메시지, 날짜, 작성자)
            result = subprocess.run([
                "git", "log", commit_range,
                "--pretty=format:%H|%s|%ad|%an",
                "--date=short"
            ], capture_output=True, text=True, check=True)
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        hash_val, message, date, author = parts
                        
                        # 커밋 분류
                        category, clean_message, importance = self.analyzer.classify_commit(message)
                        summary = self.analyzer.extract_summary(clean_message)
                        
                        commits.append({
                            'hash': hash_val[:8],
                            'message': message,
                            'summary': summary,
                            'category': category,
                            'importance': importance,
                            'date': date,
                            'author': author
                        })
            
            # 중요도별 정렬
            commits.sort(key=lambda x: x['importance'], reverse=True)
            return commits
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 커밋 로그 가져오기 실패: {e}")
            return []
    
    def generate_changelog(self, commits: List[Dict], version: str = None) -> str:
        """체인지로그 생성"""
        if not commits:
            return "## 🔄 변경사항\n\n변경사항이 없습니다."
        
        # 카테고리별 그룹화
        categorized = {}
        for commit in commits:
            category = commit['category']
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(commit)
        
        # 체인지로그 생성
        changelog = []
        
        if version:
            changelog.append(f"# 🚀 릴리스 v{version}")
            changelog.append("")
        
        changelog.append("## 📋 주요 변경사항")
        changelog.append("")
        
        # 통계 정보
        total_commits = len(commits)
        high_importance = len([c for c in commits if c['importance'] >= 4])
        
        changelog.append(f"- 총 **{total_commits}개** 커밋")
        if high_importance:
            changelog.append(f"- 주요 변경사항 **{high_importance}개**")
        changelog.append("")
        
        # 카테고리별 변경사항
        for category, category_commits in categorized.items():
            if not category_commits:
                continue
                
            changelog.append(f"### {category}")
            changelog.append("")
            
            for commit in category_commits:
                importance_icon = "🔥" if commit['importance'] >= 4 else "⭐" if commit['importance'] >= 3 else ""
                changelog.append(f"- {importance_icon} {commit['summary']} ([`{commit['hash']}`](../../commit/{commit['hash']}))")
            
            changelog.append("")
        
        return "\n".join(changelog)
    
    def generate_release_notes(self, commits: List[Dict], version: str, previous_version: str = None) -> str:
        """완전한 릴리스 노트 생성"""
        changelog = self.generate_changelog(commits, version)
        
        # 릴리스 정보 헤더
        release_notes = [
            f"# 🚀 TRAS v{version} 릴리스",
            "",
            f"> **발표일**: {datetime.now().strftime('%Y년 %m월 %d일')}",
            "",
            "## ✨ 이번 릴리스의 하이라이트",
            ""
        ]
        
        # 주요 변경사항 하이라이트
        high_priority = [c for c in commits if c['importance'] >= 4]
        if high_priority:
            for commit in high_priority[:3]:  # 상위 3개만
                release_notes.append(f"- 🔥 **{commit['summary']}**")
            release_notes.append("")
        
        # 전체 체인지로그
        release_notes.append(changelog)
        
        # 설치 방법
        release_notes.extend([
            "",
            "## 📥 설치 및 업그레이드",
            "",
            "### UV 사용 (권장)",
            "```bash",
            "git clone https://github.com/Baryon-ai/TRAS.git",
            "cd TRAS",
            "uv sync --extra ai",
            "```",
            "",
            "### 기존 설치에서 업그레이드",
            "```bash",
            "git pull origin main",
            "uv sync --extra ai",
            "```",
            "",
            "## 🆘 문제 해결",
            "",
            "문제가 발생하면 다음을 확인하세요:",
            "- [이슈 트래커](https://github.com/Baryon-ai/TRAS/issues)",
            "- [디스커션](https://github.com/Baryon-ai/TRAS/discussions)", 
            "- 이메일: admin@barion.ai",
            "",
            "---",
            "",
            "**🎯 Made with ❤️ by BarionLabs**"
        ])
        
        return "\n".join(release_notes)


def main():
    generator = ChangelogGenerator()
    
    # 명령행 인수 처리
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print("📝 TRAS 지능적인 체인지로그 생성기")
        print()
        print("사용법:")
        print("  python scripts/changelog_generator.py v3.2.0 v3.3.0  # 특정 범위")
        print("  python scripts/changelog_generator.py v3.2.0         # v3.2.0부터 최신까지")
        print("  python scripts/changelog_generator.py --last         # 마지막 태그부터")
        print("  python scripts/changelog_generator.py --last --full  # 완전한 릴리스 노트")
        sys.exit(0)
    
    from_tag = None
    to_tag = None
    generate_full = "--full" in sys.argv
    
    if "--last" in sys.argv:
        # 마지막 태그부터 현재까지
        pass
    elif len(sys.argv) >= 3:
        from_tag = sys.argv[1]
        to_tag = sys.argv[2]
    elif len(sys.argv) >= 2:
        from_tag = sys.argv[1]
    
    # 커밋 범위 결정
    commit_range = generator.get_commit_range(from_tag, to_tag)
    print(f"📋 분석 범위: {commit_range}")
    
    # 커밋 분석
    commits = generator.get_commits(commit_range)
    if not commits:
        print("❌ 분석할 커밋이 없습니다.")
        sys.exit(1)
    
    print(f"✅ {len(commits)}개 커밋 분석 완료")
    
    # 버전 정보 추출
    version = to_tag.lstrip('v') if to_tag and to_tag != "HEAD" else None
    
    if generate_full and version:
        # 완전한 릴리스 노트 생성
        content = generator.generate_release_notes(commits, version, from_tag)
        
        # 파일로 저장
        output_file = f"RELEASE_NOTES_v{version}.md"
        Path(output_file).write_text(content, encoding='utf-8')
        print(f"📄 릴리스 노트 저장: {output_file}")
    else:
        # 간단한 체인지로그만 출력
        content = generator.generate_changelog(commits, version)
    
    print("\n" + "="*50)
    print(content)


if __name__ == "__main__":
    main() 