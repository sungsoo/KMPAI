# manuscript-v1 한 번 실행 보고

- 실행 환경: {'platform': 'macOS-27.2-arm64-arm-64bit', 'machine': 'arm64', 'python': '3.12.16', 'numpy': '2.4.4', 'scipy': '1.17.1', 'torch': '2.14.1', 'mujoco': '4.0.0', 'matplotlib': '3.11.2'}
- 모드: 전체 재실행

| 단계 | 시간(초) | 결과 |
|---|---|---|
| preflight | 0.4 | {"platform": "macOS-27.2-arm64-arm-64bit", "machine": "arm64", "python": "3.12.16", "numpy": "2.4.4", "scipy": "1.17.1", "torch": "2.14.1", "mujoco": "4.0.0", " |
| locks | 0.1 | {"c1": "ok", "c7": "ok"} |
| tests | 12.6 | {"c1": "5 passed", "c7": "5 passed"} |
| study1 | 290.9 | "cases 408: ok 408, failed 0" |
| study1_analyze | 0.5 | "{\"cases\": 408, \"expected_cases\": 408} {\"planar_2link\": \"supported\", \"planar_3link\": \"supported\", \"franka_7dof\": \"not_supported\"} {\"planar_2lin |
| study2 | 320.6 | "cases 408: ok 408, failed 0" |
| study2_analyze | 0.6 | "{\"cases\": 408, \"expected_cases\": 408} N1 supported N2 {\"planar_2link\": \"supported\", \"planar_3link\": \"supported\", \"franka_7dof\": \"supported\"} N3 |
| compare | 0.2 | {"c1": {"cases_compared": 408, "missing_in_packaged": [], "max_rel_mse_diff": 0.0, "verdicts_identical": true, "claims_numbers_identical": false}, "c7": {"cases |
| assets | 19.0 | {"packaged": "ok", "gate_ms_line_packaged": ["\\newcommand{\\CxGateMs}{6}"], "gate_ms_line_rerun": ["\\newcommand{\\CxGateMs}{2}"], "numbers_c_identical_from_re |
| build | 3.5 | {"check_paper": true, "style": "style: 155 sentences, unjustified passive 0 (0.0% <= 10%? True), 22 warnings -> reports/style_report.md", "limitations": "limita |
| ict_cost | 2.1 | {"file": "ict_cost_darwin.json", "summary": ["planar_2link: gate 2.1 ms (p95 2.3), ratios 0.28 ms, apply 5.6 us, judge Y 162 kB, A 16 B", "planar_3link: gate 2. |

## 동봉 결과와의 대조
- c1: 비교 케이스 408, 결정 차이 0, 최대 MSE 상대 차이 0.00e+00, 판정 동일 True
- c7: 비교 케이스 408, 결정 차이 0, 최대 MSE 상대 차이 0.00e+00, 판정 동일 True
