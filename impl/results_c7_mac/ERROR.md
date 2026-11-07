# C7 Mac 실행 오류 — 1단계(환경) 실패

- 일시: 2026-11-07
- 단계: 1. 환경 (`python3 -m venv .venv-c1 && source .venv-c1/bin/activate && pip install -r impl/c1/requirements_c1.txt`)
- 기계: arm64 (Apple Silicon), CPU 18, 메모리 128 GiB
- 사용한 python3: /Users/sungsoo/.pyenv/shims/python3 → Python 3.15.2 (pyenv system)

## 오류 전문

```
ERROR: Could not find a version that satisfies the requirement torch==2.14.1 (from versions: 2.15.0)

[notice] A new release of pip is available: 26.2.2 -> 26.3
[notice] To update, run: pip install --upgrade pip
ERROR: No matching distribution found for torch==2.14.1
```

설치 확인 결과: `ModuleNotFoundError: No module named 'torch'` (requirements 전체가 설치되지 않음)

## 상태

- .venv-c1는 만들어졌지만 패키지는 설치되지 않음.
- 2단계 이후(잠금/단위 검사, 축소 점검, 주 실행)는 실행하지 않음.
- 코드, 설정, requirements 파일은 고치지 않음.

## 참고 (판단은 사용자 몫)

- 이 Python 3.15에서는 PyPI에 torch 2.15.0만 있음.
- 이 Mac에 있는 다른 Python: pyenv 3.12.14, 3.14.0, 3.15.0, /opt/homebrew/bin/python3.12
