"use strict";

const API = window.location.origin;

const MSGS = ['Initializing neural network…', 'Loading ONNX model…', 'Calibrating…', 'Ready.'];
let pct = 0;
let mi = 0;
const lf = document.getElementById('lf');
const lm = document.getElementById('lm');

function tickLoader() {
    pct += Math.random() * 20 + 8;
    if (pct > 100) pct = 100;
    
    lf.style.width = pct + '%';
    lm.textContent = MSGS[Math.min(mi++, MSGS.length - 1)];
    
    if (pct < 100) {
        setTimeout(tickLoader, 180);
    } else {
        setTimeout(() => {
            const el = document.getElementById('ldr');
            el.style.transition = 'opacity .35s';
            el.style.opacity = '0';
            setTimeout(() => {
                el.style.display = 'none';
                initHero();
            }, 360);
        }, 220);
    }
}

window.addEventListener('DOMContentLoaded', tickLoader);

function initHero() {
    gsap.set(['#hb', '#h1 span', '#hp', '#hbtns', '.hero-right', '#sc'], { opacity: 0, y: 16 });
    gsap.timeline({ defaults: { ease: 'power2.out' } })
        .to('#hb', { opacity: 1, y: 0, duration: .5, delay: .05 })
        .to('#h1 span', { opacity: 1, y: 0, duration: .7, stagger: .12 }, '-=.25')
        .to('#hp', { opacity: 1, y: 0, duration: .5 }, '-=.25')
        .to('#hbtns', { opacity: 1, y: 0, duration: .45 }, '-=.2')
        .to('.hero-right', { opacity: 1, y: 0, duration: .6 }, '-=.35')
        .to('#sc', { opacity: 1, duration: .4 }, '-=.15');
        
    initObs();
    loadCharts();
    
    window.addEventListener('scroll', () => {
        document.getElementById('nav').classList.toggle('stuck', scrollY > 50)
    }, { passive: true });
    
    fetch(`${API}/health`)
        .then(r => r.json())
        .then(d => {
            if (!d.model_loaded) showToast('Model offline', 'err');
        })
        .catch(() => {});
}

function initObs() {
    const ro = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.classList.add('in');
                ro.unobserve(e.target);
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.reveal').forEach(el => ro.observe(el));
    
    const co = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (!e.isIntersecting) return;
            const el = e.target;
            const target = +el.dataset.count;
            const div = +(el.dataset.div || 1);
            const dur = 1400;
            const s = performance.now();
            const ease = t => t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
            
            function step(now) {
                const p = Math.min((now - s) / dur, 1);
                const v = ease(p) * target / div;
                el.textContent = div > 1 ? v.toFixed(2) : Math.round(v).toLocaleString();
                if (p < 1) requestAnimationFrame(step);
            }
            step(s);
            co.unobserve(el);
        });
    }, { threshold: .5 });
    
    document.querySelectorAll('[data-count]').forEach(el => co.observe(el));
}

function toggleMenu() {
    document.getElementById('mob-nav').classList.toggle('open');
}

function closeMob() {
    document.getElementById('mob-nav').classList.remove('open');
}

function switchTab(m) {
    const panels = {
        draw: document.getElementById('draw-panel'),
        upload: document.getElementById('upload-panel'),
        camera: document.getElementById('camera-panel')
    };
    const tabs = {
        draw: document.getElementById('td'),
        upload: document.getElementById('tu'),
        camera: document.getElementById('tc')
    };
    
    for (const k in panels) {
        if (panels[k]) panels[k].style.display = k === m ? 'block' : 'none';
    }
    for (const k in tabs) {
        if (tabs[k]) {
            if (k === m) tabs[k].classList.add('active');
            else tabs[k].classList.remove('active');
        }
    }
    
    if (m === 'camera') startCamera();
    else stopCamera();
    
    clearResult();
}

(function() {
    const c = document.getElementById('draw-canvas');
    const ctx = c.getContext('2d');
    const bEl = document.getElementById('brush');
    const bVal = document.getElementById('bval');
    const hint = document.getElementById('cv-hint');
    
    let drawing = false, lx = 0, ly = 0, drawn = false;
    
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    function pos(e) {
        const r = c.getBoundingClientRect();
        const sx = c.width / r.width;
        const sy = c.height / r.height;
        const s = e.touches ? e.touches[0] : e;
        return { x: (s.clientX - r.left) * sx, y: (s.clientY - r.top) * sy };
    }
    
    function start(e) {
        drawing = true;
        const p = pos(e);
        lx = p.x;
        ly = p.y;
        e.preventDefault();
    }
    
    function draw(e) {
        if (!drawing) return;
        if (!drawn) {
            drawn = true;
            hint.classList.add('gone');
        }
        const p = pos(e);
        const sz = +bEl.value;
        ctx.beginPath();
        ctx.moveTo(lx, ly);
        ctx.lineTo(p.x, p.y);
        ctx.strokeStyle = '#111827';
        ctx.lineWidth = sz;
        ctx.stroke();
        lx = p.x;
        ly = p.y;
        e.preventDefault();
    }
    
    function stop() {
        drawing = false;
    }
    
    c.addEventListener('mousedown', start);
    c.addEventListener('mousemove', draw);
    c.addEventListener('mouseup', stop);
    c.addEventListener('mouseleave', stop);
    c.addEventListener('touchstart', start, { passive: false });
    c.addEventListener('touchmove', draw, { passive: false });
    c.addEventListener('touchend', stop);
    bEl.addEventListener('input', () => bVal.textContent = bEl.value);
    
    window._cv = c;
    window._cx = ctx;
    window._hd = () => drawn;
    window._rd = () => { drawn = false; };
})();

function clearCanvas() {
    window._cx.fillStyle = '#ffffff';
    window._cx.fillRect(0, 0, window._cv.width, window._cv.height);
    document.getElementById('cv-hint').classList.remove('gone');
    window._rd();
    clearResult();
}

let uFile = null;

function fsv(e) {
    if (e.target.files[0]) setFile(e.target.files[0]);
}

function dov(e) {
    e.preventDefault();
    document.getElementById('dropz').classList.add('drag');
}

function dlv() {
    document.getElementById('dropz').classList.remove('drag');
}

function ddp(e) {
    e.preventDefault();
    dlv();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) setFile(f);
}

function setFile(f) {
    uFile = f;
    const r = new FileReader();
    r.onload = e => {
        const img = document.getElementById('pimg');
        img.src = e.target.result;
        img.style.display = 'block';
        document.getElementById('ubtns').style.display = 'block';
        document.getElementById('dropz').style.display = 'none';
    };
    r.readAsDataURL(f);
}

function showScanning() {
    document.getElementById('res-empty').style.display = 'none';
    document.getElementById('scanning').style.display = 'flex';
    document.getElementById('res-body').style.display = 'none';
}

function showResult(data) {
    document.getElementById('scanning').style.display = 'none';
    const body = document.getElementById('res-body');
    body.style.display = 'flex';
    
    gsap.fromTo('#rdig', { scale: 1.5, opacity: 0 }, { scale: 1, opacity: 1, duration: .4, ease: 'back.out(1.4)' });
    document.getElementById('rdig').textContent = data.digit;
    document.getElementById('rconf').textContent = data.confidence.toFixed(1) + '%';
    
    const wrap = document.getElementById('probs');
    wrap.innerHTML = '';
    data.probabilities.forEach((p, i) => {
        const top = i === data.digit;
        const row = document.createElement('div');
        row.className = 'pb-row';
        row.innerHTML = `<span class="pb-d">${i}</span><div class="pb-tr"><div class="pb-fill${top ? ' top' : ''}" data-w="${p}"></div></div><span class="pb-p">${p.toFixed(1)}%</span>`;
        wrap.appendChild(row);
    });
    
    requestAnimationFrame(() => document.querySelectorAll('.pb-fill').forEach(b => b.style.width = b.dataset.w + '%'));
}

function clearResult() {
    document.getElementById('res-empty').style.display = 'flex';
    document.getElementById('scanning').style.display = 'none';
    document.getElementById('res-body').style.display = 'none';
}

async function predictCanvas() {
    if (!window._hd()) {
        showToast('Draw a digit first', 'err');
        return;
    }
    
    const btn = document.getElementById('pred-btn');
    const sp = document.getElementById('ps');
    const lbl = document.getElementById('pl');
    
    btn.disabled = true;
    sp.classList.add('on');
    lbl.textContent = 'Analyzing…';
    showScanning();
    
    window._cv.toBlob(async blob => {
        try {
            const fd = new FormData();
            fd.append('file', blob, 'digit.png');
            const res = await fetch(`${API}/predict`, { method: 'POST', body: fd });
            if (!res.ok) throw new Error((await res.json()).detail || 'Error');
            const data = await res.json();
            showResult(data);
            showToast(`Predicted: ${data.digit} · ${data.confidence.toFixed(1)}%`, 'ok');
        } catch (e) {
            clearResult();
            showToast(e.message, 'err');
        } finally {
            btn.disabled = false;
            sp.classList.remove('on');
            lbl.textContent = 'Predict Digit';
        }
    }, 'image/png');
}

async function predictUpload() {
    if (!uFile) {
        showToast('Upload an image first', 'err');
        return;
    }
    
    const btn = document.getElementById('upbtn');
    const sp = document.getElementById('us');
    const lbl = document.getElementById('ul');
    
    btn.disabled = true;
    sp.classList.add('on');
    lbl.textContent = 'Analyzing…';
    showScanning();
    
    try {
        const fd = new FormData();
        fd.append('file', uFile);
        const res = await fetch(`${API}/predict`, { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail || 'Error');
        const data = await res.json();
        showResult(data);
        showToast(`Predicted: ${data.digit} · ${data.confidence.toFixed(1)}%`, 'ok');
    } catch (e) {
        clearResult();
        showToast(e.message, 'err');
    } finally {
        btn.disabled = false;
        sp.classList.remove('on');
        lbl.textContent = 'Predict Digit';
    }
}

async function loadCharts() {
    try {
        const data = await fetch(`${API}/metrics`).then(r => r.json());
        if (!data.accuracy?.length) return;
        
        const labels = data.accuracy.map((_, i) => `E${i + 1}`);
        const base = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#6b7280', font: { family: 'JetBrains Mono', size: 10 } } } },
            scales: {
                x: { ticks: { color: '#9ca3af', font: { size: 10 } }, grid: { color: '#f4f4f5' } },
                y: { ticks: { color: '#9ca3af', font: { size: 10 } }, grid: { color: '#f4f4f5' } }
            }
        };
        
        new Chart(document.getElementById('acc-ch'), {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'train', data: data.accuracy, borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,.06)', fill: true, tension: .4, pointRadius: 3, pointBackgroundColor: '#2563eb' },
                    { label: 'val', data: data.val_accuracy, borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,.06)', fill: true, tension: .4, pointRadius: 3, pointBackgroundColor: '#7c3aed' }
                ]
            },
            options: base
        });
        
        new Chart(document.getElementById('loss-ch'), {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'train', data: data.loss, borderColor: '#0891b2', backgroundColor: 'rgba(8,145,178,.06)', fill: true, tension: .4, pointRadius: 3, pointBackgroundColor: '#0891b2' },
                    { label: 'val', data: data.val_loss, borderColor: '#d97706', backgroundColor: 'rgba(217,119,6,.06)', fill: true, tension: .4, pointRadius: 3, pointBackgroundColor: '#d97706' }
                ]
            },
            options: base
        });
    } catch (e) {
        console.warn('Charts:', e);
    }
}

let _tt = null;
function showToast(msg, type = 'ok') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = `toast ${type} show`;
    clearTimeout(_tt);
    _tt = setTimeout(() => t.className = 'toast', 3500);
}

let camStream = null;

async function startCamera() {
    const video = document.getElementById('cam-video');
    const hint = document.getElementById('cam-hint');
    const btn = document.getElementById('cam-btn');
    
    if (camStream) return;
    
    video.style.display = 'none';
    hint.textContent = 'Requesting camera...';
    hint.style.display = 'flex';
    btn.disabled = true;
    
    try {
        camStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: { ideal: 'environment' } } 
        });
        video.srcObject = camStream;
        await video.play();
        video.style.display = 'block';
        hint.style.display = 'none';
        btn.disabled = false;
    } catch (err) {
        hint.textContent = 'Camera access denied or unavailable.';
        showToast('Could not access camera', 'err');
        console.error(err);
    }
}

function stopCamera() {
    if (camStream) {
        camStream.getTracks().forEach(t => t.stop());
        camStream = null;
    }
}

async function predictCamera() {
    const video = document.getElementById('cam-video');
    if (!camStream || !video.videoWidth) {
        showToast('Camera not ready', 'err');
        return;
    }
    
    const btn = document.getElementById('cam-btn');
    const sp = document.getElementById('cs');
    const lbl = document.getElementById('cl');
    
    btn.disabled = true;
    sp.classList.add('on');
    lbl.textContent = 'Analyzing…';
    showScanning();
    
    try {
        const size = Math.min(video.videoWidth, video.videoHeight);
        const cvs = document.createElement('canvas');
        cvs.width = size;
        cvs.height = size;
        const ctx = cvs.getContext('2d');
        
        const sx = (video.videoWidth - size) / 2;
        const sy = (video.videoHeight - size) / 2;
        
        // Match CSS horizontal mirror
        ctx.translate(size, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, sx, sy, size, size, 0, 0, size, size);
        
        const blob = await new Promise(r => cvs.toBlob(r, 'image/png'));
        
        const fd = new FormData();
        fd.append('file', blob, 'camera.png');
        const res = await fetch(`${API}/predict`, { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail || 'Error');
        const data = await res.json();
        showResult(data);
        showToast(`Predicted: ${data.digit} · ${data.confidence.toFixed(1)}%`, 'ok');
    } catch (e) {
        clearResult();
        showToast(e.message, 'err');
    } finally {
        btn.disabled = false;
        sp.classList.remove('on');
        lbl.textContent = 'Capture & Predict';
    }
}
