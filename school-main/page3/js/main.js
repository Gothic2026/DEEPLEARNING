/* [Main Controller] 탭 관리 및 공통 함수 */
let currentTab = 'ann';
let animationFrameId;

// 공통 보간(Interpolation) 수학 함수
const lerp = (start, end, amt) => (1 - amt) * start + amt * end;

function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.content-section').forEach(sec => sec.classList.remove('active'));
    
    // 버튼 및 섹션 활성화
    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(btn => btn.getAttribute('onclick').includes(tabName));
    if (activeBtn) activeBtn.classList.add('active');
    
    const targetSection = document.getElementById(`${tabName}-section`);
    if (targetSection) targetSection.classList.add('active');
    
    initTabAnimations();
}

function initTabAnimations() {
    if (currentTab === 'ann') initAnn();
    else if (currentTab === 'dnn') initDnn();
    else if (currentTab === 'rnn') initRnn();
}

// 전체 통합 애니메이션 렌더링 루프
function renderLoop() {
    if (currentTab === 'ann') animateAnn();
    else if (currentTab === 'dnn') animateDnn();
    else if (currentTab === 'rnn') animateRnn();

    animationFrameId = requestAnimationFrame(renderLoop);
}

// 브라우저가 모두 로드되면 시뮬레이터 자동 실행
window.addEventListener('DOMContentLoaded', () => {
    initAnn();
    initDnn();
    initRnn();
    renderLoop();
});