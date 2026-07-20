/**
 * 사전 학습된 가중치 로드 (JSON) — 페이지 진입 즉시 사용
 */
import cnnWeightsJson from './cnnWeights.json';
import nlpWeightsJson from './nlpWeights.json';

let cachedCNN = null;
let cachedNLP = null;

export function getCNNWeights() {
  if (cachedCNN) return cachedCNN;
  cachedCNN = cnnWeightsJson;
  return cachedCNN;
}

export function getNLPWeights() {
  if (cachedNLP) return cachedNLP;
  cachedNLP = nlpWeightsJson;
  return cachedNLP;
}
