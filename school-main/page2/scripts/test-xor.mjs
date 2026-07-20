import { MLP, XOR_DATASET } from '../src/ml/neuralNetwork.js';

const mlp = new MLP([2, 4, 4, 1], 42);
for (let e = 0; e < 15000; e++) {
  for (const d of XOR_DATASET) mlp.trainStep(d.input, d.output, 1.2);
  const err = XOR_DATASET.reduce((s, d) => s + Math.abs(mlp.predict(d.input) - d.output[0]), 0);
  if (err < 0.08) {
    console.log('Converged epoch', e + 1, 'err', err.toFixed(4));
    break;
  }
}
const acc = XOR_DATASET.filter((d) => Math.abs(mlp.predict(d.input) - d.output[0]) < 0.5).length;
console.log('Accuracy', acc + '/4');
XOR_DATASET.forEach((d) => console.log(d.input, '->', mlp.predict(d.input).toFixed(4), '(target', d.output[0] + ')'));
