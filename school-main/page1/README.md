# 신경망 실험실

뉴런 · 퍼셉트론 · 다층 신경망을 직접 탐험하는 교육용 인터랙티브 시뮬레이션입니다.

## 실행 방법

로컬 서버 없이도 열 수 있지만, ES 모듈 사용을 위해 간단한 서버를 권장합니다:

```bash
# Python
python -m http.server 8081

# Node.js (npx)
npx serve .
```

브라우저에서 `http://localhost:8081` 접속

## 구성

| 섹션 | 내용 |
|------|------|
| ① 생물학적 뉴런 | 시냅스 자극 전파, 역치 발화 시각화 |
| ② 퍼셉트론 | p(x,y)=ax+by+w, 결정 경계, 활성화 T/F |
| ③ AND·OR·XOR | 선형 분리 가능/불가능 비교 |
| ④ 다층 퍼셉트론 | OR+NAND+AND로 XOR 해결, 단층 vs 다층 비교 |

## 기술 스택

- HTML5 Canvas
- Vanilla JavaScript (ES Modules)
- 별도 빌드 도구 불필요
