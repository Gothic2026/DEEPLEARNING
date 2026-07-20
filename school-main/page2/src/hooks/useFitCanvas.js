import { useEffect, useCallback } from 'react';

/** 캔버스 내부 해상도를 부모 컨테이너 크기에 맞춤 */
export function setupCanvas(canvas) {
  const parent = canvas.parentElement;
  if (!parent) return null;
  const w = parent.clientWidth;
  const h = parent.clientHeight;
  if (w <= 0 || h <= 0) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, ctx, dpr };
}

export function useFitCanvas(canvasRef, draw, deps = []) {
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dims = setupCanvas(canvas);
    if (dims) draw(dims);
  }, [canvasRef, draw, ...deps]);

  useEffect(() => {
    redraw();
    const canvas = canvasRef.current;
    if (!canvas?.parentElement) return undefined;
    const ro = new ResizeObserver(redraw);
    ro.observe(canvas.parentElement);
    return () => ro.disconnect();
  }, [redraw]);

  return redraw;
}
