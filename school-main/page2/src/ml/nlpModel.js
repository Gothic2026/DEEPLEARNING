/**
 * NLP - 사과/오렌지 분류용 임베딩 + MLP (외부 API 없음)
 */

const VOCAB = [
  '<PAD>', '사과', '오렌지', 'apple', 'orange', 'red', 'green', '달콤', 'sweet', 'sour', '새콤',
  '과일', 'fruit', '이', 'that', 'the', 'a', '은', '는', '가', '를', '을', '좋아', 'like', 'love',
  '먹', 'eat', 'want', '원해', '주세요', 'please', 'give', 'me', '저', '빨간', '주황', '노란',
  'yellow', 'round', '둥근', 'vitamin', '비타민', 'healthy', '건강', 'delicious', '맛있',
  'prefer', '선호', 'choose', '고르', 'select', 'pick', 'fresh', '신선', 'juicy', '과즙',
  'citrus', '시트러스', 'tree', '나무', 'color', '색', 'taste', '맛', 'flavor', '향',
];

const APPLE_IDX = 0;
const ORANGE_IDX = 1;

export class FruitClassifier {
  constructor(weights) {
    this.vocab = VOCAB;
    this.wordToIdx = Object.fromEntries(VOCAB.map((w, i) => [w, i]));
    this.embeddingDim = weights.embeddingDim;
    this.embeddings = weights.embeddings;
    this.fc1W = weights.fc1.weights;
    this.fc1B = weights.fc1.biases;
    this.fc2W = weights.fc2.weights;
    this.fc2B = weights.fc2.biases;
    this.lastHidden = null;
    this.lastLogits = null;
  }

  tokenize(text) {
    const lower = text.toLowerCase();
    const tokens = [];
    for (const word of VOCAB.slice(1)) {
      if (word === '<PAD>') continue;
      const w = word.toLowerCase();
      if (lower.includes(w) || lower.includes(word)) {
        tokens.push(this.wordToIdx[word] ?? this.wordToIdx[w]);
      }
    }
    if (tokens.length === 0) {
      for (const ch of lower.replace(/\s/g, '')) {
        if (ch === '사' || ch === '과') tokens.push(this.wordToIdx['사과']);
        if (ch === '오' || ch === '렌') tokens.push(this.wordToIdx['오렌지']);
        if (ch === 'a') tokens.push(this.wordToIdx['apple']);
        if (ch === 'o') tokens.push(this.wordToIdx['orange']);
      }
    }
    return tokens.length ? tokens : [0];
  }

  embed(tokens) {
    const vec = Array(this.embeddingDim).fill(0);
    for (const t of tokens) {
      for (let d = 0; d < this.embeddingDim; d++) vec[d] += this.embeddings[t][d];
    }
    const n = tokens.length || 1;
    return vec.map((v) => v / n);
  }

  relu(x) {
    return Math.max(0, x);
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
  }

  forward(text) {
    const tokens = this.tokenize(text);
    const embedded = this.embed(tokens);

    const hidden = this.fc1W.map((row, i) =>
      this.relu(row.reduce((s, w, j) => s + w * embedded[j], 0) + this.fc1B[i])
    );
    const logits = this.fc2W.map((row, i) =>
      row.reduce((s, w, j) => s + w * hidden[j], 0) + this.fc2B[i]
    );
    const probs = logits.map((l) => this.sigmoid(l));
    const sum = probs.reduce((a, b) => a + b, 0);
    const normalized = probs.map((p) => p / sum);

    this.lastHidden = hidden;
    this.lastLogits = logits;
    this.lastEmbedded = embedded;
    this.lastTokens = tokens;

    return {
      apple: normalized[APPLE_IDX],
      orange: normalized[ORANGE_IDX],
      prediction: normalized[APPLE_IDX] >= normalized[ORANGE_IDX] ? 'apple' : 'orange',
      tokens,
      hidden,
      logits,
    };
  }
}

export { VOCAB, APPLE_IDX, ORANGE_IDX };
