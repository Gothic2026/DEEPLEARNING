/* [RNN 모듈] 순환신경망 아코디언 언롤링 및 기억 소실 시각화 */
const rnnCanvas = document.getElementById('rnnCanvas');
const rnnCtx = rnnCanvas ? rnnCanvas.getContext('2d') : null;

let rnnIsUnfolded = false;
let unfoldProgress = 0;

function initRnn() {
    if (!rnnCtx) return;
    rnnIsUnfolded = false;
    unfoldProgress = 0;
}

function toggleRnnUnroll() {
    rnnIsUnfolded = !rnnIsUnfolded;
}

function animateRnn() {
    if (!rnnCtx || currentTab !== 'rnn') return;
    rnnCtx.clearRect(0, 0, rnnCanvas.width, rnnCanvas.height);

    if (rnnIsUnfolded && unfoldProgress < 1) unfoldProgress += 0.03;
    if (!rnnIsUnfolded && unfoldProgress > 0) unfoldProgress -= 0.03;

    const cy = 200; 

    rnnCtx.fillStyle = '#ffffff';
    rnnCtx.font = 'bold 13px Segoe UI';
    rnnCtx.fillText("시간 분해에 의한 동적 아코디언 펼침 시뮬레이터 (RNN Unrolling)", 30, 30);

    if (unfoldProgress < 0.01) {
        const cx = 350;
        
        rnnCtx.beginPath();
        rnnCtx.arc(cx, cy - 20, 25, 0.2 * Math.PI, 1.8 * Math.PI);
        rnnCtx.strokeStyle = '#ec4899'; 
        rnnCtx.lineWidth = 3;
        rnnCtx.stroke();
        
        rnnCtx.beginPath();
        rnnCtx.moveTo(cx + 17, cy - 10);
        rnnCtx.lineTo(cx + 25, cy - 3);
        rnnCtx.lineTo(cx + 31, cy - 12);
        rnnCtx.fillStyle = '#ec4899';
        rnnCtx.fill();

        drawArrow(cx, cy + 70, cx, cy + 20, 'rgba(148, 163, 184, 0.6)');
        drawArrow(cx, cy - 20, cx, cy - 70, 'rgba(148, 163, 184, 0.6)');

        drawRnnNode(cx, cy + 80, "입력 x", "#3b82f6");
        drawRnnNode(cx, cy, "순환층 h", "#ec4899");
        drawRnnNode(cx, cy - 80, "출력 y", "#ef4444");
    } else {
        const steps = 4;
        const startX = 350; 
        
        const getX = (index) => {
            const targetX = 120 + index * 150; 
            return lerp(startX, targetX, unfoldProgress);
        };

        for (let i = 0; i < steps - 1; i++) {
            const x1 = getX(i);
            const x2 = getX(i+1);
            
            const gradient = rnnCtx.createLinearGradient(x1, cy, x2, cy);
            const op1 = Math.max(0.05, 1.0 - i * 0.3); 
            const op2 = Math.max(0.05, 1.0 - (i+1) * 0.3); 

            gradient.addColorStop(0, `rgba(236, 72, 153, ${op1})`); 
            gradient.addColorStop(1, `rgba(236, 72, 153, ${op2})`);
            
            rnnCtx.beginPath();
            rnnCtx.moveTo(x1, cy);
            rnnCtx.lineTo(x2, cy);
            rnnCtx.strokeStyle = gradient;
            rnnCtx.lineWidth = 4;
            rnnCtx.stroke();

            drawArrowHead(x2 - 20, cy, 'right', `rgba(236, 72, 153, ${op2})`);
        }

        for (let i = 0; i < steps; i++) {
            const cx = getX(i);
            drawArrow(cx, cy + 70, cx, cy + 20, 'rgba(148, 163, 184, 0.4)');
            drawArrow(cx, cy - 20, cx, cy - 70, 'rgba(148, 163, 184, 0.4)');

            const nodeColorAlpha = Math.max(0.1, 1.0 - i * 0.3); 
            const nodeGradient = rnnCtx.createRadialGradient(cx, cy, 5, cx, cy, 20);
            nodeGradient.addColorStop(0, `rgba(236, 72, 153, ${nodeColorAlpha})`); 
            nodeGradient.addColorStop(1, '#14b8a6'); 

            drawRnnNode(cx, cy + 80, `x(t-${3-i})`, "#3b82f6");
            drawRnnNode(cx, cy, `h(t-${3-i})`, nodeGradient, true);
            drawRnnNode(cx, cy - 80, `y(t-${3-i})`, "#ef4444");

            if (i === 3 && unfoldProgress > 0.9) {
                rnnCtx.fillStyle = '#fda4af';
                rnnCtx.font = '11px Segoe UI';
                rnnCtx.fillText("이전 정보 유실 상태 (장기 의존성 한계)", cx - 100, cy + 35);
            }
        }
    }
}

function drawRnnNode(x, y, text, colorStyle, isGrad = false) {
    rnnCtx.beginPath();
    rnnCtx.arc(x, y, 18, 0, Math.PI * 2);
    rnnCtx.fillStyle = colorStyle;
    rnnCtx.fill();
    rnnCtx.strokeStyle = '#ffffff';
    rnnCtx.lineWidth = 1.5;
    rnnCtx.stroke();

    rnnCtx.fillStyle = '#ffffff';
    rnnCtx.font = '10px Segoe UI';
    rnnCtx.textAlign = 'center';
    rnnCtx.textBaseline = 'middle';
    rnnCtx.fillText(text, x, y);
}

function drawArrow(x1, y1, x2, y2, color) {
    rnnCtx.beginPath();
    rnnCtx.moveTo(x1, y1);
    rnnCtx.lineTo(x2, y2);
    rnnCtx.strokeStyle = color;
    rnnCtx.lineWidth = 2;
    rnnCtx.stroke();

    const dir = (y2 < y1) ? 'up' : 'down';
    drawArrowHead(x2, y2 + (dir === 'up' ? 5 : -5), dir, color);
}

function drawArrowHead(x, y, direction, color) {
    rnnCtx.beginPath();
    rnnCtx.fillStyle = color;
    if (direction === 'up') {
        rnnCtx.moveTo(x - 5, y); rnnCtx.lineTo(x + 5, y); rnnCtx.lineTo(x, y - 8);
    } else if (direction === 'down') {
        rnnCtx.moveTo(x - 5, y); rnnCtx.lineTo(x + 5, y); rnnCtx.lineTo(x, y + 8);
    } else if (direction === 'right') {
        rnnCtx.moveTo(x, y - 5); rnnCtx.lineTo(x, y + 5); rnnCtx.lineTo(x + 8, y);
    }
    rnnCtx.fill();
}