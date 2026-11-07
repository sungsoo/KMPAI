# C1 Mac 실행 오류 — 1단계(환경 설치)

- 일시: 2026-11-07
- 장비: Apple M5 Max, 18코어, 128 GiB
- 실행 명령:
  `python3 -m venv .venv-c1 && source .venv-c1/bin/activate && pip install -r impl/c1/requirements_c1.txt`
- `python3` = /Users/sungsoo/.pyenv/shims/python3 → Python 3.15.2 (pyenv system)

## 오류 전문
```
ERROR: Could not find a version that satisfies the requirement torch==2.14.1 (from versions: 2.15.0)

[notice] A new release of pip is available: 26.2.2 -> 26.3
[notice] To update, run: pip install --upgrade pip
ERROR: No matching distribution found for torch==2.14.1
```

## 상태
- pip가 전체 설치를 중단해 .venv-c1에는 mujoco/torch/numpy/scipy 모두 설치되지 않음.
- 참고(확인만 함, 설치 안 함): Python 3.15용 torch 휠은 2.15.0만 있음.
  /opt/homebrew/bin/python3.12 기준으로는 2.14.1 휠이 있음. Linux 기준 실행은 Python 3.12.3 + torch 2.14.1.
- 2단계 이후는 실행하지 않음.
