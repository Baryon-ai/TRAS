#!/usr/bin/env python3
"""
🚀 GitHub 릴리스 생성 스크립트
TRAS (Talent Recommendation Analysis System) GitHub 릴리스 자동화 도구

사용법:
    uv run python scripts/github_release.py            # 최신 태그로 릴리스 생성
    uv run python scripts/github_release.py v3.0.1     # 특정 태그로 릴리스 생성
    uv run python scripts/github_release.py --draft    # 드래프트 릴리스 생성

필요 조건:
    - GitHub CLI (gh) 설치 및 인증 필요
    - Git 태그가 이미 생성되어 있어야 함
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime


def check_github_cli():
    """GitHub CLI 설치 및 인증 확인"""
    try:
        # gh 명령어 존재 확인
        result = subprocess.run(
            ["gh", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"✅ GitHub CLI 감지: {result.stdout.split()[2]}")
        
        # 인증 확인
        auth_result = subprocess.run(
            ["gh", "auth", "status"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("✅ GitHub CLI 인증 확인됨")
        return True
        
    except FileNotFoundError:
        print("❌ GitHub CLI (gh)가 설치되지 않았습니다.")
        print("   설치 방법: https://cli.github.com/")
        return False
    except subprocess.CalledProcessError as e:
        print("❌ GitHub CLI 인증이 필요합니다.")
        print("   인증 방법: gh auth login")
        return False


def get_latest_tag():
    """가장 최근 Git 태그 가져오기"""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        tag = result.stdout.strip()
        print(f"🏷️  최신 태그: {tag}")
        return tag
    except subprocess.CalledProcessError:
        print("❌ Git 태그를 찾을 수 없습니다.")
        print("   먼저 'uv run python scripts/version_manager.py patch'로 태그를 생성하세요.")
        return None


def get_tag_info(tag):
    """태그 정보 가져오기"""
    try:
        # 태그 존재 확인
        subprocess.run(
            ["git", "rev-parse", tag], 
            capture_output=True, 
            check=True
        )
        
        # 태그 메시지 가져오기
        result = subprocess.run(
            ["git", "tag", "-l", "--format=%(contents)", tag], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        tag_message = result.stdout.strip()
        version = tag.lstrip('v')
        
        return {
            'tag': tag,
            'version': version,
            'message': tag_message or f"버전 {version} 릴리스"
        }
        
    except subprocess.CalledProcessError:
        print(f"❌ 태그 '{tag}'를 찾을 수 없습니다.")
        return None


def generate_smart_changelog(tag):
    """지능적인 체인지로그 생성"""
    try:
        # 새로운 체인지로그 생성기 사용
        changelog_script = Path(__file__).parent / "changelog_generator.py"
        
        # 이전 태그 찾기
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", f"{tag}^"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            prev_tag = result.stdout.strip()
            print(f"📝 스마트 체인지로그 생성: {prev_tag}..{tag}")
            
            # 체인지로그 생성기 실행
            changelog_result = subprocess.run([
                sys.executable, str(changelog_script), prev_tag, tag
            ], capture_output=True, text=True, check=True)
            
            # 출력에서 실제 체인지로그 부분만 추출 (=== 이후)
            output_lines = changelog_result.stdout.split('\n')
            changelog_start = False
            changelog_lines = []
            
            for line in output_lines:
                if '=' * 30 in line:
                    changelog_start = True
                    continue
                if changelog_start:
                    changelog_lines.append(line)
            
            if changelog_lines:
                return '\n'.join(changelog_lines).strip()
        
        # 폴백: 기본 체인지로그
        return generate_basic_changelog(tag)
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  스마트 체인지로그 생성 실패, 기본 방식 사용: {e}")
        return generate_basic_changelog(tag)


def generate_basic_changelog(tag):
    """기본 체인지로그 생성 (폴백용)"""
    try:
        # 이전 태그 찾기
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", f"{tag}^"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            prev_tag = result.stdout.strip()
            print(f"📝 기본 체인지로그 생성: {prev_tag}..{tag}")
            
            # 커밋 로그 가져오기
            log_result = subprocess.run(
                ["git", "log", f"{prev_tag}..{tag}", "--pretty=format:- %s (%h)"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            changelog = log_result.stdout.strip()
            if changelog:
                return f"## 🔄 변경사항\n\n{changelog}"
        
        # 이전 태그가 없으면 현재 태그부터 몇 개 커밋만
        log_result = subprocess.run(
            ["git", "log", "-10", "--pretty=format:- %s (%h)"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        return f"## 🔄 최근 변경사항\n\n{log_result.stdout.strip()}"
        
    except subprocess.CalledProcessError:
        return "## 🔄 변경사항\n\n릴리스 정보를 참고하세요."


def get_release_notes_template(version, changelog):
    """릴리스 노트 템플릿 생성"""
    return f"""# 🚀 TRAS v{version} 릴리스

> **Talent Recommendation Analysis System** - 정부 인재 추천 분석 시스템

## ✨ 주요 특징

- 🤖 **AI 기반 분석**: Ollama, OpenAI, Claude 지원
- 📧 **이메일 분석**: 정부 인재 추천 이메일 자동 분류
- 🐦 **소셜미디어 분석**: 트위터 댓글에서 인재 추천 발굴
- 🔄 **통합 플랫폼**: 멀티 플랫폼 데이터 통합 분석
- 📊 **스마트 분류**: 추천 유형, 정부 직책 자동 분류

{changelog}

## 📥 설치 방법

### UV를 사용한 빠른 설치 (권장)
```bash
git clone https://github.com/Baryon-ai/TRAS.git
cd TRAS
uv sync --extra ai
```

### 전통적인 설치
```bash
git clone https://github.com/Baryon-ai/TRAS.git
cd TRAS
pip install -r requirements.txt
```

## 🎬 실행

```bash
# UV 환경
uv run python main.py

# 일반 환경
python main.py
```

## 🔧 시스템 요구사항

- Python 3.8.1+
- RAM 4GB+ (Ollama 사용시 8GB+ 권장)
- 저장공간 2GB+

## 🆘 지원

- 📋 [이슈 리포트](https://github.com/Baryon-ai/TRAS/issues)
- 📖 [위키 문서](https://github.com/Baryon-ai/TRAS/wiki)
- 💬 [디스커션](https://github.com/Baryon-ai/TRAS/discussions)

---

**🎯 Made with ❤️ for Government Talent Management**
"""


def create_github_release(tag_info, is_draft=False, is_prerelease=False):
    """GitHub 릴리스 생성"""
    tag = tag_info['tag']
    version = tag_info['version']
    
    print(f"🚀 GitHub 릴리스 생성 중: {tag}")
    
    # 지능적인 체인지로그 생성
    changelog = generate_smart_changelog(tag)
    
    # 릴리스 노트 생성
    release_notes = get_release_notes_template(version, changelog)
    
    # 릴리스 생성 명령어 구성
    cmd = [
        "gh", "release", "create", tag,
        "--title", f"🚀 TRAS v{version}",
        "--notes", release_notes
    ]
    
    if is_draft:
        cmd.append("--draft")
    if is_prerelease:
        cmd.append("--prerelease")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ GitHub 릴리스가 생성되었습니다!")
        print(f"🔗 릴리스 URL: https://github.com/Baryon-ai/TRAS/releases/tag/{tag}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 릴리스 생성 실패: {e}")
        if e.stderr:
            print(f"   오류 상세: {e.stderr}")
        return False


def upload_assets(tag):
    """릴리스 에셋 업로드 (선택사항)"""
    assets_to_upload = []
    
    # pyproject.toml을 릴리스에 포함
    if Path("pyproject.toml").exists():
        assets_to_upload.append("pyproject.toml")
    
    # requirements.txt를 릴리스에 포함
    if Path("requirements.txt").exists():
        assets_to_upload.append("requirements.txt")
    
    # 스크린샷이 있으면 포함
    if Path("screenshot.png").exists():
        assets_to_upload.append("screenshot.png")
    
    for asset in assets_to_upload:
        try:
            subprocess.run([
                "gh", "release", "upload", tag, asset
            ], check=True)
            print(f"📎 에셋 업로드 완료: {asset}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  에셋 업로드 실패: {asset} - {e}")


def main():
    print("🚀 TRAS GitHub 릴리스 생성 도구")
    print()
    
    # GitHub CLI 확인
    if not check_github_cli():
        sys.exit(1)
    
    # 명령행 인수 처리
    is_draft = "--draft" in sys.argv
    is_prerelease = "--prerelease" in sys.argv
    
    # 태그 지정
    specified_tag = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            specified_tag = arg
            break
    
    # 태그 정보 가져오기
    if specified_tag:
        tag_info = get_tag_info(specified_tag)
    else:
        latest_tag = get_latest_tag()
        if not latest_tag:
            sys.exit(1)
        tag_info = get_tag_info(latest_tag)
    
    if not tag_info:
        print("❌ 유효한 태그를 찾을 수 없습니다.")
        sys.exit(1)
    
    print(f"📋 릴리스 정보:")
    print(f"   • 태그: {tag_info['tag']}")
    print(f"   • 버전: {tag_info['version']}")
    print(f"   • 메시지: {tag_info['message']}")
    
    if is_draft:
        print("   • 타입: 드래프트")
    if is_prerelease:
        print("   • 타입: 프리릴리스")
    
    print()
    
    # 확인
    confirm = input("GitHub 릴리스를 생성하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 취소되었습니다.")
        sys.exit(0)
    
    # 릴리스 생성
    if create_github_release(tag_info, is_draft, is_prerelease):
        # 에셋 업로드
        upload_confirm = input("추가 파일을 릴리스에 업로드하시겠습니까? (y/N): ")
        if upload_confirm.lower() == 'y':
            upload_assets(tag_info['tag'])
        
        print()
        print("🎉 GitHub 릴리스가 성공적으로 생성되었습니다!")
        print(f"🔗 https://github.com/Baryon-ai/TRAS/releases/tag/{tag_info['tag']}")


if __name__ == "__main__":
    main() 