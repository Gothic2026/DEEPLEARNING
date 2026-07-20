# 딥러닝 시각화 학습 웹앱

백엔드 없이 브라우저에서 동작하는 딥러닝 교육용 시각화 앱입니다.

## 실행 방법

```bash
npm install
npm run dev
```

브라우저에서 `http://localhost:5173` 접속

## 모델 재학습 (선택)

```bash
npm run train:cnn   # CNN 숫자 인식 가중치 생성
npm run train:nlp   # NLP 사과/오렌지 분류 가중치 생성
```

## 탭 구성

| 탭 | 내용 |
|---|---|
| 딥러닝 개요 | ML vs DL, CNN/음성/NLP 개념 소개 |
| 순전파·역전파 | XOR 4계층 MLP, 순전파/역전파 애니메이션, 결정경계 그래프 |
| CNN | 숫자 그리기 → 합성곱/풀링 시각화 → DNN 분류 |
| 음성인식 | Web Speech API + FFT/MFCC/음소 파이프라인 시각화 |
| 자연어처리 | 사과/오렌지 문장 분류 (외부 AI API 없음) |

## 기술 스택

- React + Vite
- 순수 JavaScript 신경망 구현 (MLP, CNN, NLP)
- Web Speech API, Web Audio API
