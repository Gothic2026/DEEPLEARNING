import { drawGrid, toCanvas, step, gateLabels } from './utils.js';

let x1 = 0, x2 = 0;
let animPhase = 0;
let animating = false;
let rafId = null;

function orGate(a, b) { return step(a + b - 0.5); }
function nandGate(a, b) { return step(-a - b + 1.5); }
function andGate(a, b) { return step(a + b - 1.5); }
function xorGate(a, b) {
  const h1 = orGate(a, b);
  const h2 = nandGate(a, b);
  return andGate(h1, h2);
}

export function initMLP(singleCanvas, multiCanvas, networkCanvas) {
  const mlpX1 = document.getElementById('mlpX1');
  const mlpX2 = document.getElementById('mlpX2');
  const animateBtn = document.getElementById('mlpAnimateBtn');

  mlpX1.addEventListener('input', () => {
    x1 = parseInt(mlpX1.value, 10);
    document.getElementById('mlpX1Val').textContent = x1;
    highlightTruthRow();
  });

  mlpX2.addEventListener('input', () => {
    x2 = parseInt(mlpX2.value, 10);
    document.getElementById('mlpX2Val').textContent = x2;
    highlightTruthRow();
  });

  animateBtn.addEventListener('click', () => {
    animating = true;
    animPhase = 0;
  });

  buildTruthTable();
  highlightTruthRow();

  function loop() {
    if (animating) animPhase = Math.min(animPhase + 0.02, 1);
    if (animPhase >= 1) setTimeout(() => { animating = false; animPhase = 0; }, 500);

    drawSingleLayer(singleCanvas);
    drawMultiLayer(multiCanvas);
    drawNetwork(networkCanvas);
    rafId = requestAnimationFrame(loop);
  }
  loop();

  return () => { if (rafId) cancelAnimationFrame(rafId); };
}

function buildTruthTable() {
  const tbody = document.querySelector('#mlpTruthTable tbody');
  const rows = [[0,0],[0,1],[1,0],[1,1]];
  tbody.innerHTML = rows.map(([a, b]) => {
    const o = orGate(a, b);
    const n = nandGate(a, b);
    const x = andGate(o, n);
    return `<tr data-x1="${a}" data-x2="${b}"><td>${a}</td><td>${b}</td><td>${o}</td><td>${n}</td><td>${x}</td></tr>`;
  }).join('');
}

function highlightTruthRow() {
  document.querySelectorAll('#mlpTruthTable tbody tr').forEach(tr => {
    tr.classList.toggle('active-row',
      tr.dataset.x1 === String(x1) && tr.dataset.x2 === String(x2));
  });
}

function drawSingleLayer(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = '#121820';
  ctx.fillRect(0, 0, w, h);
  const { padding } = drawGrid(ctx, w, h, 40);

  // Any single line fails for XOR
  const a = 1, b = -1, bias = 0.5;
  drawLine(ctx, a, b, bias, w, h, padding, 'rgba(77, 171, 247, 0.8)', true);

  const points = gateLabels('XOR');
  points.forEach(pt => {
    const { cx, cy } = toCanvas(pt.x, pt.y, w, h, padding);
    ctx.beginPath();
    ctx.arc(cx, cy, 12, 0, Math.PI * 2);
    ctx.fillStyle = pt.label === 1 ? '#69db7c' : '#ff8787';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = '#0f1419';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(pt.label), cx, cy);
  });

  ctx.fillStyle = '#ff6b6b';
  ctx.font = '11px Malgun Gothic, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('단일 직선 — XOR 불가', w / 2, h - 12);
}

function drawMultiLayer(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = '#121820';
  ctx.fillRect(0, 0, w, h);
  const { padding } = drawGrid(ctx, w, h, 40);

  // Two boundaries forming XOR regions
  drawLine(ctx, 1, -1, 0.5, w, h, padding, 'rgba(81, 207, 102, 0.7)', false);
  drawLine(ctx, -1, 1, 0.5, w, h, padding, 'rgba(255, 212, 59, 0.7)', false);

  // Color XOR regions: label 1 at (0,1) and (1,0), label 0 at (0,0) and (1,1)
  const steps = 30;
  const plotW = w - padding * 2;
  const plotH = h - padding * 2;
  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < steps; j++) {
      const px = i / (steps - 1);
      const py = j / (steps - 1);
      const xor = px !== py ? 1 : 0;
      const { cx, cy } = toCanvas(px, py, w, h, padding);
      ctx.fillStyle = xor ? 'rgba(105, 219, 124, 0.15)' : 'rgba(255, 135, 135, 0.1)';
      ctx.fillRect(cx - plotW / steps / 2, cy - plotH / steps / 2, plotW / steps, plotH / steps);
    }
  }

  const points = gateLabels('XOR');
  points.forEach(pt => {
    const { cx, cy } = toCanvas(pt.x, pt.y, w, h, padding);
    ctx.beginPath();
    ctx.arc(cx, cy, 12, 0, Math.PI * 2);
    ctx.fillStyle = pt.label === 1 ? '#69db7c' : '#ff8787';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = '#0f1419';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(pt.label), cx, cy);
  });

  ctx.fillStyle = '#51cf66';
  ctx.font = '11px Malgun Gothic, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('다층 — XOR 해결 ✓', w / 2, h - 12);
}

function drawLine(ctx, a, b, bias, w, h, padding, color, dash) {
  if (dash) ctx.setLineDash([5, 4]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  let started = false;
  for (let t = 0; t <= 1; t += 0.02) {
    const x = t;
    const y = Math.abs(b) > 0.001 ? (-a * x - bias) / b : 0;
    const { cx, cy } = toCanvas(x, y, w, h, padding);
    if (!started) { ctx.beginPath(); ctx.moveTo(cx, cy); started = true; }
    else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawNetwork(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = '#121820';
  ctx.fillRect(0, 0, w, h);

  const nodes = {
    x1: { x: 80, y: 100, label: 'x₁', val: x1 },
    x2: { x: 80, y: 180, label: 'x₂', val: x2 },
    or: { x: 320, y: 90, label: 'OR', fn: () => orGate(x1, x2) },
    nand: { x: 320, y: 190, label: 'NAND', fn: () => nandGate(x1, x2) },
    and: { x: 560, y: 140, label: 'AND\n(XOR)', fn: () => xorGate(x1, x2) },
  };

  const edges = [
    ['x1', 'or'], ['x2', 'or'],
    ['x1', 'nand'], ['x2', 'nand'],
    ['or', 'and'], ['nand', 'and'],
  ];

  const phase = animPhase;
  edges.forEach(([from, to]) => {
    const a = nodes[from];
    const b = nodes[to];
    const lit = animating && phase > 0.2;
    ctx.strokeStyle = lit ? '#4dabf7' : '#2d3f56';
    ctx.lineWidth = lit ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.moveTo(a.x + 22, a.y);
    ctx.lineTo(b.x - 22, b.y);
    ctx.stroke();

    if (animating && phase > 0.1) {
      const t = Math.min((phase - 0.1) / 0.8, 1);
      const px = a.x + 22 + (b.x - 22 - a.x - 22) * t;
      const py = a.y + (b.y - a.y) * t;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffd43b';
      ctx.fill();
    }
  });

  Object.entries(nodes).forEach(([key, n]) => {
    const isInput = key.startsWith('x');
    const active = animating ? phase > (isInput ? 0 : 0.5) : true;
    let fill = '#243044';
    if (isInput && n.val === 1) fill = '#4dabf7';
    if (!isInput && active) {
      const v = n.fn();
      fill = v === 1 ? '#51cf66' : '#ff8787';
    }

    ctx.beginPath();
    ctx.arc(n.x, n.y, 22, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = active ? '#fff' : '#3d5270';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#e8edf4';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const lines = n.label.split('\n');
    lines.forEach((line, i) => {
      ctx.fillText(line, n.x, n.y + (i - (lines.length - 1) / 2) * 13);
    });

    if (isInput) {
      ctx.fillStyle = '#8b9cb3';
      ctx.font = '10px sans-serif';
      ctx.fillText(`= ${n.val}`, n.x, n.y + 32);
    } else if (!animating || phase > 0.5) {
      ctx.fillStyle = '#8b9cb3';
      ctx.font = '10px sans-serif';
      ctx.fillText(`→ ${n.fn()}`, n.x, n.y + 32);
    }
  });

  ctx.fillStyle = '#8b9cb3';
  ctx.font = '12px Malgun Gothic, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('입력층', 80, 30);
  ctx.fillText('은닉층', 320, 30);
  ctx.fillText('출력층', 560, 30);

  const out = xorGate(x1, x2);
  ctx.fillStyle = out === 1 ? '#51cf66' : '#ff8787';
  ctx.font = 'bold 14px sans-serif';
  ctx.fillText(`XOR(${x1}, ${x2}) = ${out}`, w / 2, h - 20);
}
