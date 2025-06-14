#!/usr/bin/env python3
"""
🎯 TRAS 통합 릴리스 스크립트
버전 업데이트부터 GitHub 릴리스까지 한 번에 처리

사용법:
    uv run python scripts/release.py patch         # 패치 릴리스 (3.0.0 → 3.0.1)
    uv run python scripts/release.py minor         # 마이너 릴리스 (3.0.0 → 3.1.0)
    uv run python scripts/release.py major         # 메이저 릴리스 (3.0.0 → 4.0.0)
    uv run python scripts/release.py 3.2.1         # 직접 버전 지정
    uv run python scripts/release.py patch --draft # 드래프트 릴리스
"""

import sys
import subprocess
from pathlib import Path


def run_script(script_name, args):
    """스크립트 실행"""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args
    
    print(f"🔄 실행: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 스크립트 실행 실패: {script_name}")
        return False


def main():
    if len(sys.argv) < 2:
        print("🎯 TRAS 통합 릴리스 도구")
        print()
        print("사용법:")
        print("  uv run python scripts/release.py patch         # 패치 릴리스")
        print("  uv run python scripts/release.py minor         # 마이너 릴리스") 
        print("  uv run python scripts/release.py major         # 메이저 릴리스")
        print("  uv run python scripts/release.py 3.2.1         # 직접 버전 지정")
        print("  uv run python scripts/release.py patch --draft # 드래프트 릴리스")
        print()
        print("이 스크립트는 다음을 순차적으로 실행합니다:")
        print("  1. 버전 업데이트 (pyproject.toml, README.md)")
        print("  2. Git 태그 생성 및 푸시")
        print("  3. GitHub 릴리스 생성")
        sys.exit(1)
    
    # 명령행 인수 분석
    version_arg = sys.argv[1]
    additional_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    print("🎯 TRAS 통합 릴리스 프로세스 시작")
    print("=" * 50)
    
    # 1단계: 버전 관리
    print("\n📍 1단계: 버전 업데이트")
    version_args = [version_arg]
    if not run_script("version_manager.py", version_args):
        print("❌ 버전 업데이트 실패")
        sys.exit(1)
    
    # 2단계: GitHub 릴리스
    print("\n📍 2단계: GitHub 릴리스 생성")
    
    # 릴리스 스크립트에 전달할 인수 준비
    release_args = additional_args.copy()  # --draft, --prerelease 등
    
    if not run_script("github_release.py", release_args):
        print("❌ GitHub 릴리스 생성 실패")
        print("   • 수동으로 릴리스하려면: uv run python scripts/github_release.py")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 릴리스 프로세스가 완료되었습니다!")
    print()
    print("📋 완료된 작업:")
    print("  ✅ 버전 업데이트")
    print("  ✅ Git 태그 생성 및 푸시")
    print("  ✅ GitHub 릴리스 생성")
    print()
    print("🔗 릴리스 확인: https://github.com/Baryon-ai/TRAS/releases")


if __name__ == "__main__":
    main() 