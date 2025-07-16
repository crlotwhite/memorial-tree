# GitHub Packages 사용 가이드

Memorial Tree 패키지는 GitHub Packages를 통해 팀 내부에서 배포 및 사용할 수 있습니다. 이 문서는 GitHub Packages를 통해 패키지를 설치하고 사용하는 방법을 설명합니다.

## 패키지 설치하기

### 1. 인증 설정

GitHub Packages에서 패키지를 설치하려면 GitHub 인증이 필요합니다. 다음과 같이 설정할 수 있습니다:

#### 방법 1: 개인 액세스 토큰(PAT) 사용

1. GitHub에서 [개인 액세스 토큰](https://github.com/settings/tokens)을 생성합니다.
   - `read:packages` 권한이 필요합니다.

2. `~/.pypirc` 파일을 생성하거나 수정합니다:

```
[distutils]
index-servers =
    github

[github]
repository = https://github.com/memorialtree/memorial-tree/
username = YOUR_GITHUB_USERNAME
password = YOUR_PERSONAL_ACCESS_TOKEN
```

#### 방법 2: 환경 변수 사용

```bash
export GITHUB_USERNAME=YOUR_GITHUB_USERNAME
export GITHUB_TOKEN=YOUR_PERSONAL_ACCESS_TOKEN
```

### 2. 패키지 설치

#### pip 명령어로 설치

```bash
pip install --index-url https://github.com/memorialtree/memorial-tree/raw/main/dist/ memorial-tree
```

특정 버전을 설치하려면:

```bash
pip install --index-url https://github.com/memorialtree/memorial-tree/raw/main/dist/ memorial-tree==0.1.0
```

#### requirements.txt에 추가

```
memorial-tree @ https://github.com/memorialtree/memorial-tree/releases/download/v0.1.0/memorial_tree-0.1.0-py3-none-any.whl
```

## 개발자를 위한 정보

### 새 버전 배포하기

새 버전을 배포하는 방법은 두 가지가 있습니다:

#### 1. 태그 푸시하기

1. 새 태그를 생성합니다:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

2. GitHub Actions가 자동으로 패키지를 빌드하고 GitHub Packages에 배포합니다.

#### 2. 수동으로 워크플로우 실행하기

1. GitHub 저장소의 Actions 탭으로 이동합니다.
2. "Publish Package" 워크플로우를 선택합니다.
3. "Run workflow" 버튼을 클릭합니다.
4. 배포할 버전 번호를 입력합니다 (예: "0.1.1").
5. "Run workflow" 버튼을 클릭하여 배포를 시작합니다.

### 로컬에서 빌드 및 테스트

로컬에서 패키지를 빌드하고 테스트하려면:

```bash
# 개발 모드로 설치
pip install -e ".[dev]"

# 테스트 실행
pytest

# 패키지 빌드
python -m build
```

## 주의사항

- GitHub Packages는 GitHub 계정 인증이 필요합니다.
- 프라이빗 저장소의 경우, 패키지를 설치하려면 저장소에 대한 접근 권한이 필요합니다.
- PyPI와 달리, GitHub Packages는 팀 내부 또는 조직 내부에서 사용하기에 적합합니다.