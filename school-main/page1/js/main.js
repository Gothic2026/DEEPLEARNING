import { initBioNeuron } from './bio-neuron.js';
import { initPerceptron } from './perceptron.js';
import { initLogicGates } from './logic-gates.js';
import { initMLP } from './mlp.js';

const cleanups = [];

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();

  cleanups.push(initBioNeuron(document.getElementById('bioCanvas')));
  cleanups.push(initPerceptron(document.getElementById('perceptronCanvas')));
  cleanups.push(initLogicGates(document.getElementById('logicCanvas')));
  cleanups.push(initMLP(
    document.getElementById('mlpSingleCanvas'),
    document.getElementById('mlpMultiCanvas'),
    document.getElementById('mlpNetworkCanvas'),
  ));
});

function initNavigation() {
  const nav = document.getElementById('sectionNav');
  const sections = document.querySelectorAll('.lab-section');

  nav.addEventListener('click', (e) => {
    const btn = e.target.closest('.nav-btn');
    if (!btn) return;

    nav.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    sections.forEach(s => s.classList.remove('active'));
    document.getElementById(`section-${btn.dataset.section}`).classList.add('active');
  });
}

window.addEventListener('beforeunload', () => {
  cleanups.forEach(fn => fn && fn());
});
