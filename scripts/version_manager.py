#!/usr/bin/env python3
"""
🔢 버전 관리 스크립트
TRAS (Talent Recommendation Analysis System) 버전 관리 도구

사용법:
    uv run python scripts/version_manager.py patch    # 3.0.0 → 3.0.1
    uv run python scripts/version_manager.py minor    # 3.0.0 → 3.1.0
    uv run python scripts/version_manager.py major    # 3.0.0 → 4.0.0
    uv run python scripts/version_manager.py 3.2.1    # 직접 버전 지정
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from datetime import datetime


def get_current_version():
    """pyproject.toml에서 현재 버전 가져오기"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ pyproject.toml 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    content = pyproject_path.read_text(encoding='utf-8')
    version_match = re.search(r'version = "([^"]+)"', content)
    
    if not version_match:
        print("❌ pyproject.toml에서 버전을 찾을 수 없습니다.")
        sys.exit(1)
    
    return version_match.group(1)


def parse_version(version_str):
    """버전 문자열을 (major, minor, patch)로 파싱"""
    try:
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError("버전은 X.Y.Z 형식이어야 합니다.")
        
        return tuple(int(part) for part in parts)
    except ValueError as e:
        print(f"❌ 잘못된 버전 형식: {version_str} - {e}")
        sys.exit(1)


def increment_version(current_version, bump_type):
    """버전을 증가시키기"""
    major, minor, patch = parse_version(current_version)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        # 직접 버전 지정
        parse_version(bump_type)  # 유효성 검사
        return bump_type


def update_pyproject_version(new_version):
    """pyproject.toml의 [project] 섹션의 버전만 정확히 업데이트"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text(encoding='utf-8')
    
    # [project] 섹션의 version만 정확히 찾아서 교체
    # 다른 곳의 version은 절대 건드리지 않음
    pattern = r'(\[project\][\s\S]*?)version = "[^"]+"'
    
    def replace_project_version(match):
        return match.group(1) + f'version = "{new_version}"'
    
    new_content = re.sub(pattern, replace_project_version, content)
    
    # 안전성 검증: [project] 섹션의 version이 정확히 업데이트되었는지 확인
    project_match = re.search(r'\[project\][\s\S]*?version = "([^"]+)"', new_content)
    if not project_match or project_match.group(1) != new_version:
        print("❌ [project] 섹션의 버전 업데이트에 실패했습니다!")
        sys.exit(1)
    
    # 추가 안전성 검증: 다른 곳에 잘못된 버전이 들어가지 않았는지 확인
    forbidden_patterns = [
        (r'tras-version = "[0-9]+\.[0-9]+\.[0-9]+"', 'script entry에 버전 번호가 들어감'),
        (r'minversion = "[0-9]+\.[0-9]+\.[0-9]+"', 'pytest minversion에 프로젝트 버전이 들어감'),
        (r'python_version = "[0-9]+\.[0-9]+\.[0-9]+"', 'mypy python_version에 프로젝트 버전이 들어감'),
        (r'target-version = "[0-9]+\.[0-9]+\.[0-9]+"', 'ruff target-version에 프로젝트 버전이 들어감')
    ]
    
    for pattern, error_msg in forbidden_patterns:
        if re.search(pattern, new_content):
            print(f"❌ 안전성 검증 실패: {error_msg}")
            print("🔧 pyproject.toml을 수동으로 수정해야 합니다.")
            sys.exit(1)
    
    pyproject_path.write_text(new_content, encoding='utf-8')
    print(f"✅ pyproject.toml [project] 섹션의 버전을 {new_version}로 업데이트했습니다.")
    print("✅ 안전성 검증 완료: 다른 섹션은 변경되지 않았습니다.")


def update_readme_version(old_version, new_version):
    """README.md의 버전 정보 업데이트"""
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("⚠️  README.md를 찾을 수 없습니다. 건너뛰기.")
        return
    
    content = readme_path.read_text(encoding='utf-8')
    
    # 버전 번호 교체 (여러 패턴 지원)
    patterns = [
        (rf"v{re.escape(old_version)}", f"v{new_version}"),
        (rf"### 🚀 v{re.escape(old_version)}", f"### 🚀 v{new_version}"),
        (rf"version {re.escape(old_version)}", f"version {new_version}"),
    ]
    
    updated = False
    for old_pattern, new_pattern in patterns:
        if re.search(old_pattern, content):
            content = re.sub(old_pattern, new_pattern, content)
            updated = True
    
    if updated:
        readme_path.write_text(content, encoding='utf-8')
        print(f"✅ README.md의 버전 정보를 업데이트했습니다.")
    else:
        print("ℹ️  README.md에서 업데이트할 버전 정보를 찾지 못했습니다.")


def run_command(command, description):
    """명령어 실행"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(f"   {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 실패: {e}")
        if e.stderr:
            print(f"   오류: {e.stderr.strip()}")
        return False


def create_git_tag(version):
    """Git 태그 생성"""
    tag_name = f"v{version}"
    
    # 스테이징 및 커밋
    if not run_command("git add pyproject.toml README.md", "변경사항 스테이징"):
        return False
    
    commit_message = f"🔖 버전 {version} 릴리스 준비"
    if not run_command(f'git commit -m "{commit_message}"', "버전 업데이트 커밋"):
        # 이미 커밋된 경우 무시
        pass
    
    # 태그 생성
    tag_message = f"버전 {version} 릴리스"
    if not run_command(f'git tag -a {tag_name} -m "{tag_message}"', f"Git 태그 {tag_name} 생성"):
        return False
    
    print(f"✅ Git 태그 {tag_name}를 생성했습니다.")
    return True


def push_changes(version):
    """Git 푸시"""
    tag_name = f"v{version}"
    
    # 브랜치 푸시
    if not run_command("git push origin main", "메인 브랜치 푸시"):
        return False
    
    # 태그 푸시
    if not run_command(f"git push origin {tag_name}", f"태그 {tag_name} 푸시"):
        return False
    
    print(f"✅ 변경사항과 태그를 GitHub에 푸시했습니다.")
    return True


def main():
    if len(sys.argv) < 2:
        print("🔢 TRAS 버전 관리 도구")
        print()
        print("사용법:")
        print("  uv run python scripts/version_manager.py patch    # 패치 버전 증가")
        print("  uv run python scripts/version_manager.py minor    # 마이너 버전 증가")
        print("  uv run python scripts/version_manager.py major    # 메이저 버전 증가")
        print("  uv run python scripts/version_manager.py 3.2.1    # 직접 버전 지정")
        print()
        current_version = get_current_version()
        print(f"현재 버전: {current_version}")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    current_version = get_current_version()
    
    print(f"🔢 현재 버전: {current_version}")
    
    # 새 버전 계산
    try:
        new_version = increment_version(current_version, bump_type)
    except SystemExit:
        return
    
    print(f"🆕 새 버전: {new_version}")
    
    # 확인
    if bump_type in ["major", "minor", "patch"]:
        confirm = input(f"버전을 {current_version} → {new_version}로 업데이트하시겠습니까? (y/N): ")
        if confirm.lower() != 'y':
            print("❌ 취소되었습니다.")
            sys.exit(0)
    
    # 버전 업데이트
    update_pyproject_version(new_version)
    update_readme_version(current_version, new_version)
    
    # Git 태그 생성
    if create_git_tag(new_version):
        # 푸시할지 선택
        push_confirm = input("GitHub에 푸시하시겠습니까? (y/N): ")
        if push_confirm.lower() == 'y':
            push_changes(new_version)
            print()
            print("🎉 버전 업데이트가 완료되었습니다!")
            print(f"   • 새 버전: {new_version}")
            print(f"   • Git 태그: v{new_version}")
            print(f"   • GitHub: https://github.com/Baryon-ai/TRAS/releases/tag/v{new_version}")
        else:
            print("ℹ️  로컬에만 저장되었습니다. 나중에 'git push origin main && git push origin v{new_version}'로 푸시하세요.")


if __name__ == "__main__":
    main() 