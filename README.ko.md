# Memorial Tree

[![PyPI version](https://badge.fury.io/py/memorial-tree.svg)](https://badge.fury.io/py/memorial-tree)
[![Tests](https://github.com/crlotwhite/memorial-tree/actions/workflows/test.yml/badge.svg)](https://github.com/crlotwhite/memorial-tree/actions/workflows/test.yml)
[![Lint](https://github.com/crlotwhite/memorial-tree/actions/workflows/lint.yml/badge.svg)](https://github.com/crlotwhite/memorial-tree/actions/workflows/lint.yml)
[![Documentation Status](https://readthedocs.io/en/latest/?badge=latest)](https://memorial-tree.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Memorial Tree는 트리 데이터 구조를 사용하여 인간의 사고 과정과 의사 결정을 모델링하는 Python 패키지입니다. 이 패키지는 인간의 인지 과정에서 의식적인 선택과 무의식적인 영향(고스트 노드)을 모두 표현함으로써 계산 정신의학 연구를 위한 도구를 제공하는 것을 목표로 합니다.

## 기능

- 인간의 사고 과정과 의사 결정을 위한 트리 기반 모델링
- 의사 결정에 대한 무의식적 영향을 나타내는 "고스트 노드" 지원
- 기존 ML 워크플로우와의 통합을 위한 다중 백엔드 지원(NumPy, PyTorch, TensorFlow)
- 정신 건강 상태(ADHD, 우울증, 불안)에 대한 특수 모델
- 사고 패턴 및 의사 결정 경로 분석을 위한 시각화 도구

## 설치

### PyPI에서 설치 (곧 제공 예정)

```bash
# 기본 설치
pip install memorial-tree

# PyTorch 지원 포함
pip install memorial-tree[pytorch]

# TensorFlow 지원 포함
pip install memorial-tree[tensorflow]

# 개발용
pip install -e ".[dev,docs]"
```

### GitHub Packages에서 설치

GitHub Packages에서 직접 패키지를 설치할 수 있습니다:

```bash
# 인증 설정 (최초 1회 설정)
export GITHUB_USERNAME=your-github-username
export GITHUB_TOKEN=your-personal-access-token

# 최신 버전 설치
pip install --index-url https://github.com/crlotwhite/memorial-tree/raw/main/dist/ memorial-tree

# 또는 특정 버전 지정
pip install --index-url https://github.com/crlotwhite/memorial-tree/raw/main/dist/ memorial-tree==0.1.0
```

GitHub Packages 사용에 대한 자세한 지침은 [GitHub Packages 가이드](docs/github_packages.md)를 참조하세요.

## 기본 사용법

```python
from memorial_tree import MemorialTree

# 새로운 사고 트리 생성
tree = MemorialTree()

# 트리에 생각 추가
root_id = tree.add_thought(parent_id=None, content="산책을 갈까?")
yes_id = tree.add_thought(parent_id=root_id, content="네, 산책을 가겠습니다", weight=0.7)
no_id = tree.add_thought(parent_id=root_id, content="아니요, 집에 있겠습니다", weight=0.3)

# 고스트 노드 추가 (무의식적 영향)
tree.add_ghost_node(content="산책은 나를 불안하게 만든다", influence=0.4)

# 결정 내리기
decision = tree.make_choice(root_id)
print(f"결정: {decision.content}")

# 트리 시각화
tree.visualize()
```

## 고급 기능

### 정신 건강 모델

Memorial Tree는 다양한 정신 건강 상태에 대한 모델을 포함합니다:

```python
from memorial_tree.models import ADHDModel, DepressionModel, AnxietyModel

# ADHD 모델로 트리 생성
adhd_tree = MemorialTree(model=ADHDModel())

# 우울증 모델로 트리 생성
depression_tree = MemorialTree(model=DepressionModel())

# 불안 모델로 트리 생성
anxiety_tree = MemorialTree(model=AnxietyModel())
```

### 다중 백엔드

Memorial Tree는 여러 수치 계산 백엔드를 지원합니다:

```python
# NumPy 백엔드 사용 (기본값)
tree = MemorialTree(backend='numpy')

# PyTorch 백엔드 사용
tree = MemorialTree(backend='pytorch')

# TensorFlow 백엔드 사용
tree = MemorialTree(backend='tensorflow')
```

## 예제

더 자세한 예제는 [examples](examples/) 디렉토리를 확인하세요:

- [기본 사용법](examples/basic_usage.py)
- [고급 기능](examples/advanced_features_example.py)
- [ADHD 모델](examples/adhd_model_example.py)
- [우울증 모델](examples/depression_model_example.py)
- [불안 모델](examples/anxiety_model_example.py)
- [모델 비교](examples/model_comparison_example.py)
- [트리 시각화](examples/tree_visualization_example.py)
- [경로 분석](examples/path_analysis_example.py)

## 문서

전체 문서는 [memorialtree.readthedocs.io](https://memorialtree.readthedocs.io)에서 확인할 수 있습니다.

문서를 로컬에서 빌드하려면:

```bash
# 문서 의존성 설치
pip install -e ".[docs]"

# 문서 빌드
cd docs
make html
# docs/build/html/index.html에서 브라우저로 문서 보기
```

문서는 메인 브랜치에 변경 사항이 푸시되거나 새 릴리스가 생성될 때 자동으로 빌드되어 Read the Docs에 배포됩니다.

## 기여하기

기여를 환영합니다! 자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요. 질문이나 제안이 있으시면 이슈를 생성하거나 직접 연락해 주세요.

## 개발

### CI/CD 파이프라인

이 프로젝트는 지속적 통합 및 배포를 위해 GitHub Actions를 사용합니다:

- **테스트**: 모든 푸시 및 풀 리퀘스트에 대해 여러 Python 버전에서 자동으로 테스트 실행
- **린팅**: flake8, black 및 mypy를 사용하여 코드 품질 검사
- **게시**: 새 릴리스가 생성될 때 자동으로 PyPI에 새 릴리스 게시
- **문서**: 자동으로 문서를 빌드하고 Read the Docs에 배포

로컬에서 검사를 실행하려면:

```bash
# 테스트 실행
pytest --cov=memorial_tree tests/

# 코드 형식 검사
black --check src tests examples

# 린팅 실행
flake8 src tests examples

# 타입 검사 실행
mypy src tests examples
```

### 패키지 배포

패키지는 GitHub에서 새 릴리스가 생성될 때 자동으로 PyPI에 배포됩니다. 배포 프로세스는 다음을 포함합니다:

1. 패키지 빌드
2. TestPyPI에서 패키지 테스트
3. 공식 PyPI에 배포

제공된 스크립트를 사용하여 패키지를 수동으로 배포할 수도 있습니다:

```bash
# TestPyPI에만 배포
python scripts/deploy_to_pypi.py --test-only

# TestPyPI에 배포한 후 확인 후 PyPI에 배포
python scripts/deploy_to_pypi.py
```

PyPI 또는 TestPyPI에서 설치를 테스트하려면:

```bash
# PyPI에서 설치 테스트
python scripts/test_installation.py

# TestPyPI에서 설치 테스트
python scripts/test_installation.py --test-pypi
```

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 인용

연구에서 Memorial Tree를 사용하는 경우 다음과 같이 인용해 주세요:

```bibtex
@software{memorial_tree,
  author = {Noel Kim (crlotwhite)},
  title = {Memorial Tree: A Python Package for Modeling Human Thought Processes},
  year = {2025},
  url = {https://github.com/crlotwhite/memorial-tree},
  email = {crlotwhite@gmail.com}
}
```