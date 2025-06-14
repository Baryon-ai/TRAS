# 🛠️ TRAS 릴리스 관리 스크립트

이 디렉토리는 TRAS 프로젝트의 버전 관리와 GitHub 릴리스 자동화를 위한 스크립트들을 포함합니다.

## 📁 스크립트 구성

### 🎯 `release.py` - 통합 릴리스 도구 (권장)
원스톱 릴리스 프로세스를 제공합니다.

```bash
# 패치 릴리스 (3.0.0 → 3.0.1)
uv run tras-release patch
uv run python scripts/release.py patch

# 마이너 릴리스 (3.0.0 → 3.1.0)
uv run tras-release minor

# 메이저 릴리스 (3.0.0 → 4.0.0)
uv run tras-release major

# 직접 버전 지정
uv run tras-release 3.2.1

# 드래프트 릴리스
uv run tras-release patch --draft
```

### 🔢 `version_manager.py` - 버전 관리
프로젝트 버전을 업데이트하고 Git 태그를 생성합니다.

```bash
# UV 명령어
uv run tras-version patch

# 직접 실행
uv run python scripts/version_manager.py patch
uv run python scripts/version_manager.py minor
uv run python scripts/version_manager.py major
uv run python scripts/version_manager.py 3.2.1
```

**수행 작업:**
- `pyproject.toml`의 version 필드 업데이트
- `README.md`의 버전 정보 업데이트
- Git 태그 생성 (`v3.0.1` 형식)
- 변경사항 커밋 및 태그 푸시

### 🚀 `github_release.py` - GitHub 릴리스 생성
Git 태그를 기반으로 GitHub 릴리스를 생성합니다.

```bash
# UV 명령어  
uv run tras-release-github

# 직접 실행
uv run python scripts/github_release.py            # 최신 태그로 릴리스
uv run python scripts/github_release.py v3.0.1     # 특정 태그로 릴리스
uv run python scripts/github_release.py --draft    # 드래프트 릴리스
```

**수행 작업:**
- 지능적인 체인지로그 자동 생성
- 릴리스 노트 작성
- GitHub 릴리스 생성
- 프로젝트 파일 에셋 업로드

### 📝 `changelog_generator.py` - 지능적인 체인지로그 생성기 (신규)
커밋 메시지를 분석하여 의미 있는 체인지로그를 자동 생성합니다.

```bash
# 마지막 태그부터 현재까지
uv run python scripts/changelog_generator.py --last

# 특정 범위
uv run python scripts/changelog_generator.py v3.2.0 v3.3.0

# 완전한 릴리스 노트 생성
uv run python scripts/changelog_generator.py --last --full
```

**주요 기능:**
- 🧠 **커밋 메시지 의미 분석**: 한국어 키워드 기반 자동 분류
- 📊 **카테고리별 그룹화**: 신규 기능, 버그 수정, 문서, 설정 등
- ⭐ **중요도별 정렬**: 중요한 변경사항을 상위에 표시
- 🔗 **GitHub 링크 연결**: 각 커밋에 대한 직접 링크 제공
- 📈 **통계 정보**: 총 커밋 수, 주요 변경사항 수 요약

**분류 카테고리:**
- 🚀 새로운 기능 (BERT, AI, 시스템 추가 등)
- 🔧 개선 사항 (성능, 최적화, 강화 등)
- 🐛 버그 수정 (오류, 문제 해결 등)
- 📚 문서 업데이트 (README, 가이드 등)
- ⚙️ 설정 및 구성 (pyproject.toml, 환경 등)
- 🎨 UI/UX 개선 (사용자 경험 등)
- 🧪 테스트 (검증, 확인 등)
- 🔒 보안 (인증, 권한 등)

## ⚙️ 사전 요구사항

### 1. GitHub CLI 설치 및 인증
```bash
# macOS
brew install gh

# Windows
winget install --id GitHub.cli

# Linux
# https://github.com/cli/cli/blob/trunk/docs/install_linux.md 참고

# 인증 설정
gh auth login
```

### 2. Git 설정
```bash
# 기본 설정 확인
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 원격 저장소 확인
git remote -v
# origin  https://github.com/Baryon-ai/TRAS.git (fetch)
# origin  https://github.com/Baryon-ai/TRAS.git (push)
```

### 3. 권한 확인
- GitHub 저장소에 대한 **push 권한** 필요
- **releases 생성 권한** 필요 (일반적으로 push 권한과 함께 제공)

## 🔄 릴리스 프로세스

### 자동 프로세스 (`release.py` 사용) ⭐ 권장
1. **버전 결정**: patch/minor/major 또는 직접 지정
2. **버전 업데이트**: pyproject.toml, README.md 자동 수정
3. **Git 작업**: 변경사항 커밋, 태그 생성, 푸시
4. **지능적인 체인지로그**: 커밋 메시지 자동 분석 및 분류
5. **릴리스 생성**: GitHub 릴리스 자동 생성
6. **에셋 업로드**: 프로젝트 파일 첨부

### 수동 프로세스 (단계별)
```bash
# 1단계: 버전 업데이트
uv run tras-version patch

# 2단계: GitHub 릴리스 생성  
uv run tras-release-github
```

## 📋 생성되는 릴리스 내용

### 릴리스 노트 구성
- **프로젝트 소개** 및 주요 특징
- **지능적인 체인지로그** 🆕
  - 커밋 메시지 의미 분석 및 자동 분류
  - 카테고리별 그룹화 (기능, 버그, 문서, 설정 등)
  - 중요도별 정렬 및 시각적 표시
  - GitHub 커밋 링크 자동 연결
- **설치 가이드** (UV 및 pip 방법)
- **실행 방법**
- **시스템 요구사항**
- **지원 채널** 링크

### 첨부 파일
- `pyproject.toml` - 프로젝트 설정
- `requirements.txt` - pip 호환 의존성
- `screenshot.png` - 실행 화면 (있는 경우)

## 🐛 문제 해결

### GitHub CLI 인증 오류
```bash
# 현재 인증 상태 확인
gh auth status

# 재인증
gh auth login

# 토큰으로 인증 (필요시)
gh auth login --with-token < token.txt
```

### Git 권한 오류
```bash
# 원격 저장소 URL 확인
git remote get-url origin

# HTTPS에서 SSH로 변경 (권장)
git remote set-url origin git@github.com:Baryon-ai/TRAS.git

# 또는 GitHub CLI로 자동 설정
gh repo clone Baryon-ai/TRAS
```

### 태그 충돌
```bash
# 기존 태그 삭제 (로컬)
git tag -d v3.0.1

# 기존 태그 삭제 (원격)
git push --delete origin v3.0.1

# 릴리스도 삭제 (필요시)
gh release delete v3.0.1
```

## 🎉 성공 예시

릴리스가 성공하면 다음과 같은 결과를 얻습니다:

- **GitHub 릴리스**: https://github.com/Baryon-ai/TRAS/releases/tag/v3.0.1
- **자동 태그**: Git 히스토리에 `v3.0.1` 태그 생성  
- **버전 업데이트**: pyproject.toml과 README.md 자동 수정
- **릴리스 노트**: 체인지로그와 설치 가이드 포함
- **알림**: GitHub의 Watch/Star 사용자들에게 자동 알림

이제 `uv run tras-release patch` 한 번의 명령으로 완전한 릴리스를 만들 수 있습니다! 🚀 