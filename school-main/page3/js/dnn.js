/* [DNN 모듈] 심층신경망 특징 추출 및 3D 매니폴드 분리 */
const dnnCanvas = document.getElementById('dnnCanvas');
const dnnCtx = dnnCanvas ? dnnCanvas.getContext('2d') : null;

let dnnPoints = [];
let dnnIsClassified = false;
let dnnTransition = 0;
let dnnRotationAngle = 0;

function initDnn() {
    if (!dnnCtx) return;
    dnnPoints = [];
    dnnIsClassified = false;
    dnnTransition = 0;
    dnnRotationAngle = 0;

    for (let i = 0; i < 120; i++) {
        const theta = (i / 120) * Math.PI * 4;
        const r = (i / 120) * 80 + 10;
        const x3d = r * Math.cos(theta);
        const z3d = r * Math.sin(theta);
        const y3d = (Math.random() - 0.5) * 50;
        
        const targetX = -70 + (Math.random() - 0.5) * 40;
        const targetY = (Math.random() - 0.5) * 100;
        const targetZ = (Math.random() - 0.5) * 40;

        dnnPoints.push({ x3d, y3d, z3d, targetX, targetY, targetZ, color: '#f43f5e', class: 'A' });
    }

    for (let i = 0; i < 120; i++) {
        const theta = (i / 120) * Math.PI * 4 + Math.PI;
        const r = (i / 120) * 80 + 35;
        const x3d = r * Math.cos(theta);
        const z3d = r * Math.sin(theta);
        const y3d = (Math.random() - 0.5) * 50;

        const targetX = 70 + (Math.random() - 0.5) * 40;
        const targetY = (Math.random() - 0.5) * 100;
        const targetZ = (Math.random() - 0.5) * 40;

        dnnPoints.push({ x3d, y3d, z3d, targetX, targetY, targetZ, color: '#38bdf8', class: 'B' });
    }
}

function toggleDnnClassification() {
    dnnIsClassified = !dnnIsClassified;
}

function animateDnn() {
    if (!dnnCtx || currentTab !== 'dnn') return;
    dnnCtx.clearRect(0, 0, dnnCanvas.width, dnnCanvas.height);

    // 왼쪽 특징 추출 그리드뷰
    dnnCtx.fillStyle = '#ffffff';
    dnnCtx.font = 'bold 13px Segoe UI';
    dnnCtx.fillText("레이어별 피처 추출 지도 (Feature Hierarchy)", 30, 30);

    const layers = [
        { title: "L1: 경계면 검출", draw: drawL1 },
        { title: "L2: 모형 조합", draw: drawL2 },
        { title: "L3: 구체적 객체(얼굴)", draw: drawL3 }
    ];

    for (let i = 0; i < 3; i++) {
        const bx = 30;
        const by = 60 + i * 110;
        const bw = 240;
        const bh = 80;

        dnnCtx.fillStyle = '#1e293b';
        dnnCtx.fillRect(bx, by, bw, bh);
        dnnCtx.strokeStyle = '#334155';
        dnnCtx.strokeRect(bx, by, bw, bh);

        dnnCtx.fillStyle = '#94a3b8';
        dnnCtx.font = '11px Segoe UI';
        dnnCtx.fillText(layers[i].title, bx + 10, by + 20);

        layers[i].draw(bx + 110, by + 15, bh - 30);
    }

    function drawL1(x, y, size) {
        dnnCtx.strokeStyle = '#64748b';
        dnnCtx.lineWidth = 2;
        for (let j = 0; j < 4; j++) {
            const ox = x + j * 30;
            dnnCtx.strokeRect(ox, y, size, size);
            dnnCtx.beginPath();
            if (j === 0) { dnnCtx.moveTo(ox + size/2, y); dnnCtx.lineTo(ox + size/2, y+size); }
            else if (j === 1) { dnnCtx.moveTo(ox, y + size/2); dnnCtx.lineTo(ox+size, y+size/2); }
            else { dnnCtx.moveTo(ox, y); dnnCtx.lineTo(ox+size, y+size); }
            dnnCtx.stroke();
        }
    }

    function drawL2(x, y, size) {
        dnnCtx.strokeStyle = '#10b981';
        dnnCtx.lineWidth = 2;
        for (let j = 0; j < 4; j++) {
            const ox = x + j * 30;
            dnnCtx.strokeRect(ox, y, size, size);
            dnnCtx.beginPath();
            if (j === 0) { dnnCtx.arc(ox + size/2, y + size/2, size/3, 0, Math.PI*2); }
            else if (j === 1) { dnnCtx.moveTo(ox + size/2, y+5); dnnCtx.lineTo(ox+5, y+size-5); dnnCtx.lineTo(ox+size-5, y+size-5); dnnCtx.closePath(); }
            else if (j === 2) { dnnCtx.arc(ox + 5, y + 5, size/2, 0, Math.PI/2); }
            else { dnnCtx.moveTo(ox+size/2, y); dnnCtx.lineTo(ox+size/2, y+size); dnnCtx.moveTo(ox, y+size/2); dnnCtx.lineTo(ox+size, y+size/2); }
            dnnCtx.stroke();
        }
    }

    function drawL3(x, y, size) {
        dnnCtx.strokeStyle = '#38bdf8';
        dnnCtx.lineWidth = 1.5;
        for (let j = 0; j < 4; j++) {
            const ox = x + j * 30;
            dnnCtx.strokeRect(ox, y, size, size);
            dnnCtx.beginPath();
            if (j === 0) { dnnCtx.arc(ox + size/3, y + size/2, 4, 0, Math.PI*2); dnnCtx.arc(ox + (size/3)*2, y + size/2, 4, 0, Math.PI*2); }
            else if (j === 1) { dnnCtx.moveTo(ox+size/2, y+10); dnnCtx.lineTo(ox+size/2, y+size-15); dnnCtx.lineTo(ox+size/2+8, y+size-15); }
            else if (j === 2) { dnnCtx.arc(ox+size/2, y+10, size/2, 0.2*Math.PI, 0.8*Math.PI); }
            else { dnnCtx.arc(ox + size/2, y + size/2, size/2.5, 0, Math.PI*2); dnnCtx.moveTo(ox+size/3, y+size/2); dnnCtx.lineTo(ox+size/3+3, y+size/2); dnnCtx.moveTo(ox+(size/3)*2, y+size/2); dnnCtx.lineTo(ox+(size/3)*2-3, y+size/2); }
            dnnCtx.stroke();
        }
    }

    // 오른쪽 3D 만능 공간
    const centerX = 480;
    const centerY = 200;
    dnnCtx.fillStyle = '#ffffff';
    dnnCtx.font = 'bold 13px Segoe UI';
    dnnCtx.fillText("3D 잠재 공간(Latent Space)의 차원 매니폴드 풀림", 320, 30);

    if (dnnIsClassified && dnnTransition < 1) dnnTransition += 0.02;
    if (!dnnIsClassified && dnnTransition > 0) dnnTransition -= 0.02;

    dnnRotationAngle += 0.01;

    dnnCtx.strokeStyle = 'rgba(71, 85, 105, 0.2)';
    dnnCtx.lineWidth = 1;
    for (let g = -5; g <= 5; g++) {
        const start3D = { x: -100, y: 50, z: g * 20 };
        const end3D = { x: 100, y: 50, z: g * 20 };
        const pStart = project3D(start3D.x, start3D.y, start3D.z, dnnRotationAngle);
        const pEnd = project3D(end3D.x, end3D.y, end3D.z, dnnRotationAngle);
        dnnCtx.beginPath();
        dnnCtx.moveTo(centerX + pStart.x, centerY + pStart.y);
        dnnCtx.lineTo(centerX + pEnd.x, centerY + pEnd.y);
        dnnCtx.stroke();
    }

    if (dnnTransition > 0.1) {
        dnnCtx.fillStyle = `rgba(16, 185, 129, ${dnnTransition * 0.15})`;
        dnnCtx.strokeStyle = `rgba(16, 185, 129, ${dnnTransition * 0.5})`;
        dnnCtx.lineWidth = 1.5;
        
        const wallH = 80;
        const corner1 = project3D(0, -wallH, -60, dnnRotationAngle);
        const corner2 = project3D(0, wallH, -60, dnnRotationAngle);
        const corner3 = project3D(0, wallH, 60, dnnRotationAngle);
        const corner4 = project3D(0, -wallH, 60, dnnRotationAngle);
        
        dnnCtx.beginPath();
        dnnCtx.moveTo(centerX + corner1.x, centerY + corner1.y);
        dnnCtx.lineTo(centerX + corner2.x, centerY + corner2.y);
        dnnCtx.lineTo(centerX + corner3.x, centerY + corner3.y);
        dnnCtx.lineTo(centerX + corner4.x, centerY + corner4.y);
        dnnCtx.closePath();
        dnnCtx.fill();
        dnnCtx.stroke();
    }

    function project3D(x, y, z, angle) {
        const rotX = x * Math.cos(angle) - z * Math.sin(angle);
        const rotZ = x * Math.sin(angle) + z * Math.cos(angle);
        const distance = 250;
        const fov = 200;
        const scale = fov / (distance + rotZ);
        return { x: rotX * scale * 1.5, y: y * scale * 1.5, depth: rotZ };
    }

    const renderList = dnnPoints.map(p => {
        const currentX = lerp(p.x3d, p.targetX, dnnTransition);
        const currentY = lerp(p.y3d, p.targetY, dnnTransition);
        const currentZ = lerp(p.z3d, p.targetZ, dnnTransition);
        const proj = project3D(currentX, currentY, currentZ, dnnRotationAngle);
        return { proj, color: p.color };
    });
    
    renderList.sort((a, b) => b.proj.depth - a.proj.depth);

    renderList.forEach(p => {
        dnnCtx.beginPath();
        dnnCtx.arc(centerX + p.proj.x, centerY + p.proj.y, Math.max(1, 4 * (180 / (200 + p.proj.depth))), 0, Math.PI * 2);
        dnnCtx.fillStyle = p.color;
        dnnCtx.fill();
    });
}