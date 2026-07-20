import {
  drawGrid, drawDecisionLine, gateLabels, perceptron, toCanvas,
} from './utils.js';

const GATE_PRESETS = {
  AND: { a: 1, b: 1, bias: -1.5 },
  OR: { a: 1, b: 1, bias: -0.5 },
  XOR: { a: 1, b: 1, bias: -0.5 },
};

const GATE_DESC = {
  AND: '<p><strong>AND:</strong> 두 입력이 모두 1일 때만 출력 1 — <em>선형 분리 가능 ✓</em></p>',
  OR: '<p><strong>OR:</strong> 하나라도 1이면 출력 1 — <em>선형 분리 가능 ✓</em></p>',
  XOR: '<p><strong>XOR:</strong> 입력이 다를 때만 출력 1 — <em class="fail">단일 직선으로 분류 불가 ✗</em></p>',
};

const GATE_CAPTION = {
  AND: 'AND: p = x₁ + x₂ − 1.5 → 하나의 직선으로 0/1 분리',
  OR: 'OR: p = x₁ + x₂ − 0.5 → 하나의 직선으로 0/1 분리',
  XOR: 'XOR: 어떤 직선을 그어도 4점을 완벽히 분리할 수 없습니다 — 두 직선 또는 곡선 필요',
};

let currentGate = 'AND';
let a = 1, b = 1, bias = -1.5;
let rafId = null;

export function initLogicGates(canvas) {
  document.querySelectorAll('.gate-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.gate-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentGate = btn.dataset.gate;
      applyPreset(currentGate);
      updateTruthTable();
    });
  });

  document.getElementById('logicSlope').addEventListener('input', (e) => {
    a = parseFloat(e.target.value);
  });

  document.getElementById('logicOffset').addEventListener('input', (e) => {
    bias = parseFloat(e.target.value);
  });

  applyPreset('AND');
  updateTruthTable();

  function loop() {
    draw(canvas);
    rafId = requestAnimationFrame(loop);
  }
  loop();

  return () => { if (rafId) cancelAnimationFrame(rafId); };
}

function applyPreset(gate) {
  const p = GATE_PRESETS[gate];
  a = p.a;
  b = p.b;
  bias = p.bias;
  document.getElementById('logicSlope').value = a;
  document.getElementById('logicOffset').value = bias;
  document.getElementById('gateDescription').innerHTML = GATE_DESC[gate];
  document.getElementById('logicCaption').textContent = GATE_CAPTION[gate];
}

function updateTruthTable() {
  const points = gateLabels(currentGate);
  const tbody = document.querySelector('#truthTable tbody');
  tbody.innerHTML = points.map(p =>
    `<tr><td>${p.x}</td><td>${p.y}</td><td>${p.label}</td></tr>`
  ).join('');
}

function updateResult() {
  const points = gateLabels(currentGate);
  let correct = 0;
  points.forEach(p => {
    if (perceptron(a, b, bias, p.x, p.y) === p.label) correct++;
  });

  const el = document.getElementById('logicResult');
  if (correct === 4) {
    el.className = 'result-msg success';
    el.textContent = '✓ 단일 직선으로 완벽 분류 가능! (4/4)';
  } else {
    el.className = 'result-msg fail';
    el.textContent = currentGate === 'XOR'
      ? `✗ XOR는 단층 퍼셉트론으로 해결 불가 (${correct}/4 정확)`
      : `△ 직선 위치를 조절하세요 (${correct}/4 정확)`;
  }
}

function drawXorHint(ctx, w, h, padding) {
  const lines = [
    { a: 1, b: -1, bias: 0.5 },
    { a: -1, b: 1, bias: 0.5 },
  ];

  ctx.setLineDash([4, 6]);
  const colors = ['rgba(81, 207, 102, 0.6)', 'rgba(255, 212, 59, 0.6)'];
  lines.forEach((ln, idx) => {
    ctx.strokeStyle = colors[idx];
    ctx.lineWidth = 1.5;
    let started = false;
    for (let t = 0; t <= 1; t += 0.02) {
      const x = t;
      const y = (-ln.a * x - ln.bias) / ln.b;
      const { cx, cy } = toCanvas(x, y, w, h, padding);
      if (!started) { ctx.beginPath(); ctx.moveTo(cx, cy); started = true; }
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  });
  ctx.setLineDash([]);

  ctx.fillStyle = '#51cf66';
  ctx.font = '11px Malgun Gothic, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('필요한 경계 1', padding + 10, padding + 20);
  ctx.fillStyle = '#ffd43b';
  ctx.fillText('필요한 경계 2', padding + 10, padding + 38);
}

function draw(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const w = canvas.width;
  const h = canvas.height;
  const { padding } = drawGrid(ctx, w, h);

  const points = gateLabels(currentGate);

  drawDecisionLine(ctx, a, b, w, bias, w, h, padding, true);
  if (currentGate === 'XOR') drawXorHint(ctx, w, h, padding);

  // Show formula
  ctx.fillStyle = '#4dabf7';
  ctx.font = '12px Cambria Math, serif';
  ctx.textAlign = 'right';
  const biasStr = bias >= 0 ? `+ ${bias.toFixed(1)}` : `− ${Math.abs(bias).toFixed(1)}`;
  ctx.fillText(`p = ${a.toFixed(1)}x₁ + ${b.toFixed(1)}x₂ ${biasStr}`, w - padding, padding - 12);

  points.forEach(pt => {
    const pred = perceptron(a, b, bias, pt.x, pt.y);
    const correct = pred === pt.label;
    const { cx, cy } = toCanvas(pt.x, pt.y, w, h, padding);

    ctx.beginPath();
    ctx.arc(cx, cy, 14, 0, Math.PI * 2);
    ctx.fillStyle = pt.label === 1 ? '#69db7c' : '#ff8787';
    ctx.fill();
    ctx.strokeStyle = correct ? '#fff' : '#ffd43b';
    ctx.lineWidth = correct ? 1.5 : 3;
    ctx.stroke();

    ctx.fillStyle = '#0f1419';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(pt.label), cx, cy);

    if (!correct) {
      ctx.fillStyle = '#ffd43b';
      ctx.font = '9px sans-serif';
      ctx.fillText('✗', cx + 18, cy - 12);
    }
  });

  updateResult();
}
