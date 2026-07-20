import { useState, useRef, useEffect, useCallback } from 'react';
import {
  DigitCNN, normalizeDrawing, renderFeatureMapsToCanvas,
  renderVectorBars, renderConvAnimation,
} from '../ml/cnn';
import { getCNNWeights } from '../data/initModels';
import { useFitCanvas, setupCanvas } from '../hooks/useFitCanvas';

const STEPS = [
  '① 입력', '② Conv1', '③ ReLU', '④ Pool', '⑤ Conv2',
  '⑥ ReLU', '⑦ Pool', '⑧ Flatten', '⑨ FC', '⑩ Softmax',
];

function mapSteps(steps) {
  const relus = steps.filter((s) => s.type === 'relu');
  return [
    steps.find((s) => s.type === 'input'),
    steps.find((s) => s.label?.includes('합성곱 1')),
    relus[0], steps.find((s) => s.label?.includes('풀링 1')),
    steps.find((s) => s.label?.includes('합성곱 2')),
    relus[1], steps.find((s) => s.label?.includes('풀링 2')),
    steps.find((s) => s.type === 'flatten'),
    steps.find((s) => s.type === 'dense'),
    steps.find((s) => s.type === 'output'),
  ].filter(Boolean);
}

function DNNLayerViz({ hidden, logits, probs }) {
  const canvasRef = useRef(null);
  const draw = useCallback(({ w: W, h: H, ctx }) => {
    ctx.clearRect(0, 0, W, H);
    const layers = [
      { label: 'Flat', size: 6, vals: hidden ? Array(6).fill(0.5) : [] },
      { label: 'FC1', size: 6, vals: hidden?.slice(0, 6) || [] },
      { label: 'Logit', size: 10, vals: logits || [] },
      { label: 'Prob', size: 10, vals: probs || [] },
    ];
    const lx = [24, W * 0.32, W * 0.58, W - 24];
    layers.forEach((layer, li) => {
      ctx.fillStyle = '#8b949e';
      ctx.font = '8px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(layer.label, lx[li], 10);
      for (let ni = 0; ni < layer.size; ni++) {
        const y = 18 + (ni / Math.max(layer.size - 1, 1)) * (H - 28);
        const v = layer.vals[ni] ?? 0;
        const norm = li === 3 ? v : Math.min(1, Math.abs(v) / (Math.max(...layer.vals.map(Math.abs), 0.01) || 1));
        ctx.beginPath();
        ctx.arc(lx[li], y, 10, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(88,166,255,${0.2 + norm * 0.8})`;
        ctx.fill();
        ctx.fillStyle = '#aaa';
        ctx.font = 'bold 10px monospace';
        ctx.fillText(li === 3 ? `${(v * 100).toFixed(0)}` : v.toFixed(1), lx[li], y + 2);
      }
    });
  }, [hidden, logits, probs]);
  useFitCanvas(canvasRef, draw, [hidden, logits, probs]);
  return (
    <div className="canvas-wrap canvas-wrap-sm">
      <canvas ref={canvasRef} />
    </div>
  );
}

export default function CNNTab() {
  const drawCanvasRef = useRef(null);
  const featureCanvasRef = useRef(null);
  const probCanvasRef = useRef(null);
  const featureWrapRef = useRef(null);
  const [model] = useState(() => new DigitCNN(getCNNWeights()));
  const [isDrawing, setIsDrawing] = useState(false);
  const [stepIndex, setStepIndex] = useState(-1);
  const [probs, setProbs] = useState(Array(10).fill(0.1));
  const [topDigit, setTopDigit] = useState('-');
  const [topConf, setTopConf] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [hidden, setHidden] = useState(null);
  const [logits, setLogits] = useState(null);
  const [allSteps, setAllSteps] = useState([]);
  const [inputGrid, setInputGrid] = useState(null);

  const DRAW_SIZE = 140;

  useEffect(() => {
    const canvas = drawCanvasRef.current;
    if (!canvas) return;
    canvas.width = DRAW_SIZE;
    canvas.height = DRAW_SIZE;
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
  }, []);

  const getPos = (e) => {
    const canvas = drawCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (cx - rect.left) * (DRAW_SIZE / rect.width),
      y: (cy - rect.top) * (DRAW_SIZE / rect.height),
    };
  };

  const draw = (x, y) => {
    const ctx = drawCanvasRef.current.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
  };

  const getImageGrid = () => {
    const off = document.createElement('canvas');
    off.width = 28;
    off.height = 28;
    const octx = off.getContext('2d');
    octx.drawImage(drawCanvasRef.current, 0, 0, 28, 28);
    const imgData = octx.getImageData(0, 0, 28, 28);
    const raw = Array.from({ length: 28 }, (_, y) =>
      Array.from({ length: 28 }, (_, x) => imgData.data[(y * 28 + x) * 4] / 255)
    );
    return normalizeDrawing(raw);
  };

  const resizeFeatureCanvas = () => {
    const fc = featureCanvasRef.current;
    if (fc) setupCanvas(fc);
  };

  const showStep = async (step, idx, grid, weights) => {
    setStepIndex(idx);
    resizeFeatureCanvas();
    const fc = featureCanvasRef.current;
    if (!fc) return;

    if (step.type === 'input') {
      const dims = setupCanvas(fc);
      if (!dims) return;
      const img = dims.ctx.createImageData(28, 28);
      for (let y = 0; y < 28; y++)
        for (let x = 0; x < 28; x++) {
          const v = Math.floor(step.data[y][x] * 255);
          const i = (y * 28 + x) * 4;
          img.data[i] = v; img.data[i + 1] = v; img.data[i + 2] = v; img.data[i + 3] = 255;
        }
      const off = document.createElement('canvas');
      off.width = 28; off.height = 28;
      off.getContext('2d').putImageData(img, 0, 0);
      dims.ctx.drawImage(off, dims.w * 0.1, dims.h * 0.05, dims.w * 0.8, dims.h * 0.9);
    } else if (step.label?.includes('합성곱 1')) {
      for (let s = 0; s < 6; s++) {
        resizeFeatureCanvas();
        renderConvAnimation(grid, weights.conv1[0].kernel, fc, s);
        await new Promise((r) => setTimeout(r, 80));
      }
      renderFeatureMapsToCanvas(step.data, fc, 8, 0);
    } else if (['conv', 'relu', 'pool'].includes(step.type) || step.label?.includes('합성곱 2')) {
      renderFeatureMapsToCanvas(step.data, fc);
    } else if (step.type === 'flatten' || step.type === 'dense') {
      const dims = setupCanvas(fc);
      if (!dims) return;
      dims.ctx.fillStyle = '#0a0a12';
      dims.ctx.fillRect(0, 0, dims.w, dims.h);
      const data = step.data;
      const n = Math.min(data.length, step.type === 'flatten' ? 48 : 32);
      const barW = (dims.w - 16) / n;
      const maxV = Math.max(...data.slice(0, n).map(Math.abs), 0.01);
      data.slice(0, n).forEach((v, i) => {
        const barH = (Math.abs(v) / maxV) * (dims.h - 20);
        dims.ctx.fillStyle = '#58a6ff';
        dims.ctx.fillRect(8 + i * barW, dims.h - barH - 8, barW - 1, barH);
      });
    } else if (step.type === 'output') {
      resizeFeatureCanvas();
      renderVectorBars(step.data, fc, ['0','1','2','3','4','5','6','7','8','9']);
    }
    await new Promise((r) => setTimeout(r, 700));
  };

  const applyResult = (probabilities, h, l) => {
    setHidden(h);
    setLogits(l);
    setProbs(probabilities);
    const top = probabilities.indexOf(Math.max(...probabilities));
    setTopDigit(String(top));
    setTopConf(probabilities[top]);
    if (probCanvasRef.current) {
      setupCanvas(probCanvasRef.current);
      renderVectorBars(probabilities, probCanvasRef.current, ['0','1','2','3','4','5','6','7','8','9']);
    }
  };

  const runInference = useCallback(async () => {
    setIsRunning(true);
    const grid = getImageGrid();
    setInputGrid(grid);
    const weights = getCNNWeights();
    const { steps, probabilities, hidden: h, logits: l } = model.forward(grid, true);
    const mapped = mapSteps(steps);
    setAllSteps(mapped);
    for (let i = 0; i < mapped.length; i++) await showStep(mapped[i], i, grid, weights);
    applyResult(probabilities, h, l);
    setIsRunning(false);
  }, [model]);

  const runStep = async (idx) => {
    const grid = inputGrid || getImageGrid();
    const weights = getCNNWeights();
    if (!allSteps.length) {
      const { steps, probabilities, hidden: h, logits: l } = model.forward(grid, true);
      const mapped = mapSteps(steps);
      setAllSteps(mapped);
      setInputGrid(grid);
      applyResult(probabilities, h, l);
      await showStep(mapped[idx], idx, grid, weights);
    } else {
      await showStep(allSteps[idx], idx, grid, weights);
    }
  };

  const handleClear = () => {
    const canvas = drawCanvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, DRAW_SIZE, DRAW_SIZE);
    setStepIndex(-1);
    setTopDigit('-');
    setTopConf(0);
    setAllSteps([]);
    setInputGrid(null);
    setProbs(Array(10).fill(0.1));
    setHidden(null);
    setLogits(null);
  };

  const probDraw = useCallback(({ w, h, ctx }) => {
    ctx.fillStyle = '#0a0a12';
    ctx.fillRect(0, 0, w, h);
    renderVectorBars(probs, probCanvasRef.current, ['0','1','2','3','4','5','6','7','8','9']);
  }, [probs]);
  useFitCanvas(probCanvasRef, probDraw, [probs]);

  return (
    <div className="tab-panel">
      <div className="panel-row">
        <div className="panel-col" style={{ flex: '0 0 170px' }}>
          <div className="panel-box" style={{ height: '100%' }}>
            <div className="panel-title">숫자 그리기</div>
            <div className="canvas-wrap canvas-wrap-draw">
              <canvas ref={drawCanvasRef} style={{ cursor: 'crosshair', border: '1px solid #30363d', borderRadius: 3 }}
                onMouseDown={(e) => { setIsDrawing(true); draw(...Object.values(getPos(e))); }}
                onMouseMove={(e) => { if (isDrawing) draw(...Object.values(getPos(e))); }}
                onMouseUp={() => setIsDrawing(false)} onMouseLeave={() => setIsDrawing(false)}
                onTouchStart={(e) => { e.preventDefault(); setIsDrawing(true); draw(...Object.values(getPos(e))); }}
                onTouchMove={(e) => { e.preventDefault(); if (isDrawing) draw(...Object.values(getPos(e))); }}
                onTouchEnd={() => setIsDrawing(false)} />
            </div>
            <div className="btn-row">
              <button className="btn" onClick={handleClear}>지우기</button>
              <button className="btn primary" onClick={runInference} disabled={isRunning}>
                {isRunning ? '분석 중' : '분석'}
              </button>
            </div>
          </div>
        </div>
        <div className="panel-box" style={{ flex: 1, position: 'relative' }}>
          <div className="panel-title">CNN · {stepIndex >= 0 ? STEPS[stepIndex] : '합성곱 / 풀링'}</div>
          <div className="canvas-wrap canvas-wrap-md" ref={featureWrapRef}>
            <canvas ref={featureCanvasRef} />
          </div>
          <div className="prediction-badge">
            <div className="metric metric-sm">{topDigit}</div>
            <div className="metric-sub">{topConf > 0 ? `${(topConf * 100).toFixed(0)}%` : '예측'}</div>
          </div>
        </div>
      </div>

      <div className="step-row">
        {STEPS.map((s, i) => (
          <button key={i} className={`btn ${stepIndex === i ? 'primary' : ''}`}
            onClick={() => runStep(i)} disabled={isRunning}>{s}</button>
        ))}
      </div>

      <div className="panel-row" style={{ flex: '0 0 auto', minHeight: 70 }}>
        <div className="panel-box" style={{ flex: 1 }}>
          <div className="panel-title">DNN 층별 입·출력</div>
          <DNNLayerViz hidden={hidden} logits={logits} probs={probs} />
        </div>
        <div className="panel-box" style={{ flex: 1 }}>
          <div className="panel-title">Softmax (0~9)</div>
          <div className="canvas-wrap canvas-wrap-sm">
            <canvas ref={probCanvasRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
