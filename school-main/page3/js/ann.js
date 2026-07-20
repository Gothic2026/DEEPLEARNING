/* [ANN 모듈] 인공신경망 순방향 및 역전파 시각화 */
const annCanvas = document.getElementById('annCanvas');
const annCtx = annCanvas ? annCanvas.getContext('2d') : null;

let annNodes = [];
let annConnections = [];
let annParticles = [];
let backpropParticles = [];
let annState = 'idle'; 
let pulseProgress = [0, 0, 0]; 

function initAnn() {
    if (!annCtx) return;
    annNodes = [];
    annConnections = [];
    annParticles = [];
    backpropParticles = [];
    annState = 'idle';
    pulseProgress = [0, 0, 0];
    const bpBtn = document.getElementById('backprop-btn');
    if (bpBtn) bpBtn.style.display = 'none';

    const layout = [3, 4, 2];
    const colors = ['#38bdf8', '#10b981', '#f43f5e'];
    
    for (let l = 0; l < layout.length; l++) {
        const numNodes = layout[l];
        const x = 120 + l * 230;
        const nodeArr = [];
        for (let i = 0; i < numNodes; i++) {
            const y = 200 - ((numNodes - 1) * 45) + (i * 90);
            nodeArr.push({ x, y, size: 16, originalSize: 16, color: colors[l], pulse: 0 });
        }
        annNodes.push(nodeArr);
    }

    for (let l = 0; l < annNodes.length - 1; l++) {
        const currentLayer = annNodes[l];
        const nextLayer = annNodes[l+1];
        for (let i = 0; i < currentLayer.length; i++) {
            for (let j = 0; j < nextLayer.length; j++) {
                annConnections.push({
                    from: currentLayer[i],
                    to: nextLayer[j],
                    width: 1.5,
                    color: 'rgba(71, 85, 105, 0.4)'
                });
            }
        }
    }
}

function startAnnForward() {
    if (annState !== 'idle') return;
    annState = 'forward';
    annParticles = [];
    
    annConnections.slice(0, 12).forEach(conn => {
        annParticles.push({ conn, progress: 0, speed: 0.02, active: true });
    });
    pulseProgress[0] = 1; 
}

function startAnnBackward() {
    if (annState !== 'forward_done') return;
    annState = 'backward';
    backpropParticles = [];

    annConnections.slice(12).forEach(conn => {
        backpropParticles.push({ conn, progress: 1, speed: -0.02, active: true });
    });
    pulseProgress[2] = 1; 
    const bpBtn = document.getElementById('backprop-btn');
    if (bpBtn) bpBtn.style.display = 'none';
}

function animateAnn() {
    if (!annCtx || currentTab !== 'ann') return;
    annCtx.clearRect(0, 0, annCanvas.width, annCanvas.height);

    annConnections.forEach(conn => {
        annCtx.beginPath();
        annCtx.moveTo(conn.from.x, conn.from.y);
        annCtx.lineTo(conn.to.x, conn.to.y);
        annCtx.lineWidth = conn.width;
        annCtx.strokeStyle = conn.color;
        annCtx.stroke();
    });

    for (let l = 0; l < annNodes.length; l++) {
        if (pulseProgress[l] > 0) pulseProgress[l] -= 0.05;
        annNodes[l].forEach(node => {
            const scale = 1 + Math.max(0, pulseProgress[l]) * 0.4;
            
            annCtx.beginPath();
            annCtx.arc(node.x, node.y, node.size * scale * 1.5, 0, Math.PI * 2);
            annCtx.fillStyle = node.color + '1a';
            annCtx.fill();

            annCtx.beginPath();
            annCtx.arc(node.x, node.y, node.size * scale, 0, Math.PI * 2);
            annCtx.fillStyle = node.color;
            annCtx.fill();
            annCtx.strokeStyle = '#ffffff';
            annCtx.lineWidth = 2;
            annCtx.stroke();
        });
    }

    if (annState === 'forward') {
        let allReachedHidden = true;
        annParticles.forEach(p => {
            if (p.active) {
                p.progress += p.speed;
                if (p.progress >= 1) {
                    p.active = false;
                } else {
                    allReachedHidden = false;
                    const x = lerp(p.conn.from.x, p.conn.to.x, p.progress);
                    const y = lerp(p.conn.from.y, p.conn.to.y, p.progress);
                    annCtx.beginPath();
                    annCtx.arc(x, y, 5, 0, Math.PI * 2);
                    annCtx.fillStyle = '#f59e0b';
                    annCtx.shadowBlur = 10;
                    annCtx.shadowColor = '#f59e0b';
                    annCtx.fill();
                    annCtx.shadowBlur = 0;
                }
            }
        });

        if (allReachedHidden && annParticles.length > 0 && annParticles[0].conn.from.x === annNodes[0][0].x) {
            pulseProgress[1] = 1;
            annParticles = [];
            annConnections.slice(12).forEach(conn => {
                annParticles.push({ conn, progress: 0, speed: 0.02, active: true });
            });
        } else if (allReachedHidden && annParticles.length > 0 && annParticles[0].conn.from.x === annNodes[1][0].x) {
            pulseProgress[2] = 1;
            annParticles = [];
            annState = 'forward_done';
            const bpBtn = document.getElementById('backprop-btn');
            if (bpBtn) bpBtn.style.display = 'block';
        }
    }

    if (annState === 'backward') {
        let allReachedHidden = true;
        backpropParticles.forEach(p => {
            if (p.active) {
                p.progress += p.speed;
                p.conn.width = lerp(4, 1.5, p.progress);
                p.conn.color = `rgba(239, 68, 68, ${lerp(0.8, 0.4, p.progress)})`;
                
                if (p.progress <= 0) {
                    p.active = false;
                } else {
                    allReachedHidden = false;
                    const x = lerp(p.conn.from.x, p.conn.to.x, p.progress);
                    const y = lerp(p.conn.from.y, p.conn.to.y, p.progress);
                    annCtx.beginPath();
                    annCtx.arc(x, y, 6, 0, Math.PI * 2);
                    annCtx.fillStyle = '#ef4444';
                    annCtx.shadowBlur = 12;
                    annCtx.shadowColor = '#ef4444';
                    annCtx.fill();
                    annCtx.shadowBlur = 0;
                }
            }
        });

        if (allReachedHidden && backpropParticles.length > 0 && backpropParticles[0].conn.to.x === annNodes[2][0].x) {
            pulseProgress[1] = 1;
            backpropParticles = [];
            annConnections.slice(0, 12).forEach(conn => {
                backpropParticles.push({ conn, progress: 1, speed: -0.02, active: true });
            });
        } else if (allReachedHidden && backpropParticles.length > 0 && backpropParticles[0].conn.to.x === annNodes[1][0].x) {
            pulseProgress[0] = 1;
            backpropParticles = [];
            annState = 'idle';

            annConnections.forEach((conn, index) => {
                conn.width = (index % 3 === 0) ? 3.5 : 1.0;
                conn.color = (index % 3 === 0) ? 'rgba(16, 185, 129, 0.8)' : 'rgba(56, 189, 248, 0.3)';
            });
        }
    }
}