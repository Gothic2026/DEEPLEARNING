/**
 * 사과/오렌지 NLP 분류기 학습
 */
import { writeFileSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const VOCAB = [
  '<PAD>', '사과', '오렌지', 'apple', 'orange', 'red', 'green', '달콤', 'sweet', 'sour', '새콤',
  '과일', 'fruit', '이', 'that', 'the', 'a', '은', '는', '가', '를', '을', '좋아', 'like', 'love',
  '먹', 'eat', 'want', '원해', 'please', 'give', 'me', '저', '빨간', '주황', '노란',
  'yellow', 'round', '둥근', 'vitamin', '비타민', 'healthy', '건강', 'delicious', '맛있',
  'prefer', '선호', 'choose', '고르', 'select', 'pick', 'fresh', '신선', 'juicy', '과즙',
  'citrus', '시트러스', 'tree', '나무', 'color', '색', 'taste', '맛', 'flavor', '향',
];

const wordToIdx = Object.fromEntries(VOCAB.map((w, i) => [w, i]));

const appleSentences = [
  '사과가 맛있어', '빨간 사과를 주세요', 'apple please', 'I like apple', '달콤한 사과',
  'red apple', '사과를 먹고 싶어', 'give me an apple', '신선한 사과', 'apple is delicious',
  '저는 사과를 좋아해', 'the red fruit apple', '사과 과일', 'I prefer apple', '맛있는 사과',
  'apple is healthy', '빨간색 사과', 'I want apple', '사과 주세요', 'fresh apple',
  '사과가 달콤해', 'love apple', '건강한 사과', 'pick an apple', '사과를 원해',
];

const orangeSentences = [
  '오렌지가 맛있어', '주황 오렌지를 주세요', 'orange please', 'I like orange', '새콤한 오렌지',
  'orange fruit', '오렌지를 먹고 싶어', 'give me an orange', '신선한 오렌지', 'orange is delicious',
  '저는 오렌지를 좋아해', 'the citrus orange', '오렌지 과일', 'I prefer orange', '맛있는 오렌지',
  'orange is healthy', '주황색 오렌지', 'I want orange', '오렌지 주세요', 'fresh orange',
  '오렌지가 새콤해', 'love orange', 'vitamin orange', 'pick an orange', '오렌지를 원해',
  'citrus fruit orange', 'juicy orange', '시트러스 오렌지', 'round orange', '노란 오렌지',
];

function tokenize(text) {
  const lower = text.toLowerCase();
  const tokens = [];
  for (const word of VOCAB.slice(1)) {
    const w = word.toLowerCase();
    if (lower.includes(w) || lower.includes(word)) tokens.push(wordToIdx[word] ?? wordToIdx[w]);
  }
  return tokens.length ? tokens : [0];
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function train() {
  const rng = mulberry32(99);
  const vocabSize = VOCAB.length;
  const embedDim = 16;
  const hiddenSize = 32;
  const outSize = 2;

  const embeddings = Array.from({ length: vocabSize }, () =>
    Array.from({ length: embedDim }, () => (rng() - 0.5) * 0.3)
  );
  // 사과/오렌지 단어 임베딩 초기 bias
  for (let d = 0; d < embedDim; d++) {
    embeddings[wordToIdx['사과']][d] += 0.5;
    embeddings[wordToIdx['apple']][d] += 0.5;
    embeddings[wordToIdx['빨간']][d] += 0.3;
    embeddings[wordToIdx['red']][d] += 0.3;
    embeddings[wordToIdx['오렌지']][d] -= 0.5;
    embeddings[wordToIdx['orange']][d] -= 0.5;
    embeddings[wordToIdx['주황']][d] -= 0.3;
    embeddings[wordToIdx['citrus']][d] -= 0.3;
  }

  const fc1 = {
    weights: Array.from({ length: hiddenSize }, () => Array.from({ length: embedDim }, () => (rng() - 0.5) * 0.2)),
    biases: Array.from({ length: hiddenSize }, () => 0),
  };
  const fc2 = {
    weights: Array.from({ length: outSize }, () => Array.from({ length: hiddenSize }, () => (rng() - 0.5) * 0.2)),
    biases: [0.5, -0.5],
  };

  const dataset = [
    ...appleSentences.map(s => ({ text: s, label: [1, 0] })),
    ...orangeSentences.map(s => ({ text: s, label: [0, 1] })),
  ];

  const lr = 0.05;
  for (let e = 0; e < 500; e++) {
    let loss = 0;
    for (const { text, label } of dataset) {
      const tokens = tokenize(text);
      const embedded = Array(embedDim).fill(0);
      for (const t of tokens) for (let d = 0; d < embedDim; d++) embedded[d] += embeddings[t][d];
      for (let d = 0; d < embedDim; d++) embedded[d] /= tokens.length;

      const hidden = fc1.weights.map((row, i) => {
        const z = row.reduce((s, w, j) => s + w * embedded[j], 0) + fc1.biases[i];
        return Math.max(0, z);
      });
      const logits = fc2.weights.map((row, i) => row.reduce((s, w, j) => s + w * hidden[j], 0) + fc2.biases[i]);
      const probs = logits.map(l => 1 / (1 + Math.exp(-l)));
      const sum = probs[0] + probs[1];
      const norm = [probs[0] / sum, probs[1] / sum];

      loss += (norm[0] - label[0]) ** 2 + (norm[1] - label[1]) ** 2;

      const dNorm = [2 * (norm[0] - label[0]), 2 * (norm[1] - label[1])];
      const dLogits = [dNorm[0], dNorm[1]];

      const dHidden = Array(hiddenSize).fill(0);
      for (let i = 0; i < outSize; i++)
        for (let j = 0; j < hiddenSize; j++) {
          fc2.weights[i][j] -= lr * dLogits[i] * hidden[j];
          dHidden[j] += dLogits[i] * fc2.weights[i][j];
        }
      for (let i = 0; i < outSize; i++) fc2.biases[i] -= lr * dLogits[i];

      for (let j = 0; j < hiddenSize; j++) {
        const dh = hidden[j] > 0 ? dHidden[j] : 0;
        for (let k = 0; k < embedDim; k++) fc1.weights[j][k] -= lr * dh * embedded[k];
        fc1.biases[j] -= lr * dh;
        for (const t of tokens)
          for (let k = 0; k < embedDim; k++) embeddings[t][k] -= lr * dh * fc1.weights[j][k] / tokens.length;
      }
    }
    if ((e + 1) % 100 === 0) console.log(`Epoch ${e + 1}, loss: ${(loss / dataset.length).toFixed(4)}`);
  }

  let correct = 0;
  for (const { text, label } of dataset) {
    const tokens = tokenize(text);
    const embedded = Array(embedDim).fill(0);
    for (const t of tokens) for (let d = 0; d < embedDim; d++) embedded[d] += embeddings[t][d];
    for (let d = 0; d < embedDim; d++) embedded[d] /= tokens.length;
    const hidden = fc1.weights.map((row, i) => Math.max(0, row.reduce((s, w, j) => s + w * embedded[j], 0) + fc1.biases[i]));
    const logits = fc2.weights.map((row, i) => row.reduce((s, w, j) => s + w * hidden[j], 0) + fc2.biases[i]);
    const pred = logits[0] >= logits[1] ? 0 : 1;
    if ((pred === 0 && label[0] === 1) || (pred === 1 && label[1] === 1)) correct++;
  }
  console.log(`Training accuracy: ${(correct / dataset.length * 100).toFixed(1)}%`);

  return { embeddingDim: embedDim, embeddings, fc1, fc2 };
}

const weights = train();
const outPath = join(__dirname, '..', 'src', 'data', 'nlpWeights.json');
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(weights));
console.log('Saved to', outPath);
