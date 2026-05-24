// 맨 위 첫 줄을 아래와 같이 교체하거나 파일 전체를 유지해주세요.
let botRegistry = {};
try {
    botRegistry = ServerBotRegistry;
} catch(e) {
    console.error("데이터 로드 실패, 기본값으로 대체합니다.", e);
}

let selectedBotId = null;
let logInterval = null;

// 초기 로딩 엔진
function initDashboard() {
    renderBots();
    loadTop10("KRW");
    setInterval(() => {
        const currentExchange = document.getElementById("form-exchange").value;
        const unit = currentExchange === 'Upbit' ? 'KRW' : 'USD';
        loadTop10(unit);
    }, 5000);
}

function renderBots() {
    const idleList = document.getElementById("idle-list");
    const activeList = document.getElementById("active-list");
    
    if(!idleList || !activeList) return;
    
    idleList.innerHTML = ""; activeList.innerHTML = "";
    let idleCount = 0, activeCount = 0;

    for (let key in botRegistry) {
        const bot = botRegistry[key];
        const card = document.createElement("div");
        card.className = "bot-item";
        card.draggable = true;
        card.id = bot.id;
        card.setAttribute("ondragstart", "drag(event)");
        card.setAttribute("onclick", `selectBot('${bot.id}')`);
        
        if(bot.status === "running") card.classList.add("running-glow");

        card.innerHTML = `
            <div>
                <div class="bot-name">${bot.name} ${bot.status === 'running' ? '🟢' : ''}</div>
                <div class="bot-strategy">ID: ${bot.id} · Python Bot</div>
            </div>
            <span style="font-size:12px; font-weight:bold; color:var(--green);">${bot.status.toUpperCase()}</span>
        `;

        if (bot.status === "running") { activeList.appendChild(card); activeCount++; }
        else { idleList.appendChild(card); idleCount++; }
    }

    document.getElementById("idle-count").innerText = idleCount;
    document.getElementById("active-count").innerText = activeCount;
    document.getElementById("active-count-txt").innerText = `${activeCount} / ${Object.keys(botRegistry).length}`;
    updatePositionTable();
}

function allowDrop(ev) { ev.preventDefault(); }
function drag(ev) { ev.dataTransfer.setData("text", ev.target.id); }

function drop(ev, zone) {
    ev.preventDefault();
    const id = ev.dataTransfer.getData("text");
    const bot = botRegistry[id];
    if(!bot) return;
    
    if (zone === 'active' && bot.status === 'idle') {
        bot.status = 'ready';
        selectBot(id);
    } else if (zone === 'idle' && bot.status === 'running') {
        selectBot(id);
        submitToggle();
    } else if (zone === 'idle') {
        bot.status = 'idle';
    }
    renderBots();
}

function selectBot(id) {
    selectedBotId = id;
    const bot = botRegistry[id];
    if(!bot) return;
    
    document.getElementById("panel-placeholder").style.display = "none";
    document.getElementById("panel-form").style.display = "flex";
    
    document.getElementById("form-bot-title").innerText = bot.name;
    document.getElementById("form-seed").value = bot.seed_money;
    document.getElementById("form-coin").value = bot.target_coin;
    document.getElementById("form-leverage").value = bot.leverage || 1;
    document.getElementById("form-exchange").value = bot.currency === 'KRW' ? 'Upbit' : 'Binance';
    changeExchangeUnit(document.getElementById("form-exchange").value);
    updateLevLabel(bot.leverage || 1);

    document.getElementById("form-bot-status").innerText = bot.status.toUpperCase();
    
    if (bot.status === "running") {
        document.getElementById("main-submit-btn").innerText = "🛑 가동 중지 (STOP)";
        document.getElementById("main-submit-btn").classList.add("stop-mode");
    } else {
        document.getElementById("main-submit-btn").innerText = "🚀 설정값으로 가동 시작 (START)";
        document.getElementById("main-submit-btn").classList.remove("stop-mode");
    }
    
    setMode(bot.mode || 'Mock');
    setMarket(bot.market_type || 'Spot');
    startLogPolling(id);
}

function changeExchangeUnit(val) {
    const unit = val === 'Upbit' ? 'KRW' : 'USD';
    document.getElementById("currency-unit-txt").innerText = unit;
    document.getElementById("top10-unit-txt").innerText = unit;
    loadTop10(unit);
}

function updateLevLabel(val) { document.getElementById("lev-label").innerText = `Leverage 슬라이더: ${val}배`; }

function setMode(mode) {
    if(!selectedBotId) return;
    botRegistry[selectedBotId].mode = mode;
    ['Backtest', 'Mock', 'Live'].forEach(m => {
        const el = document.getElementById(`m-${m}`);
        if(el) el.classList.remove('active');
    });
    const targetEl = document.getElementById(`m-${mode}`);
    if(targetEl) targetEl.classList.add('active');
}

function setMarket(type) {
    if(!selectedBotId) return;
    botRegistry[selectedBotId].market_type = type;
    ['Spot', 'Futures'].forEach(t => {
        const el = document.getElementById(`t-${t}`);
        if(el) el.classList.remove('active');
    });
    const targetEl = document.getElementById(`t-${type}`);
    if(targetEl) targetEl.classList.add('active');
    
    const levArea = document.getElementById("leverage-area");
    if(levArea) levArea.style.display = (type === 'Futures') ? 'flex' : 'none';
}

function submitToggle() {
    if(!selectedBotId) return;
    const bot = botRegistry[selectedBotId];
    const seed = document.getElementById("form-seed").value;
    const coin = document.getElementById("form-coin").value;
    const lev = document.getElementById("form-leverage").value;
    const exchange = document.getElementById("form-exchange").value;
    const unit = exchange === 'Upbit' ? 'KRW' : 'USD';

    fetch(`/action/toggle?bot_id=${bot.id}&seed=${seed}&coin=${coin}&unit=${unit}&lev=${lev}&market=${bot.market_type}&mode=${bot.mode}`)
    .then(res => res.json())
    .then(data => {
        if(data.status !== "error") {
            bot.status = data.status;
            botRegistry[selectedBotId] = data.bot;
            renderBots();
            selectBot(selectedBotId);
        }
    });
}

function loadTop10(unit) {
    fetch(`/api/top10?currency=${unit}`)
    .then(res => res.json())
    .then(data => {
        const container = document.getElementById("top10-list");
        if(!container) return;
        if(!data || data.length === 0) { container.innerHTML = "데이터 로드 실패"; return; }
        
        let html = "";
        data.forEach(c => {
            let color = c.change > 0 ? "var(--green)" : c.change < 0 ? "var(--red)" : "var(--text-muted)";
            html += `
                <div style="display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid var(--border-color); font-size:14px;">
                    <span style="font-weight:bold;">${c.rank}. ${c.symbol}</span>
                    <span style="color:${color}; font-weight:bold;">${c.price} (${c.change > 0 ? '+' : ''}${c.change.toFixed(2)}%)</span>
                    <span style="font-size:12px; color:var(--text-muted);">${c.volume}</span>
                </div>
            `;
        });
        container.innerHTML = html;
    });
}

function startLogPolling(id) {
    if(logInterval) clearInterval(logInterval);
    const fetchLogs = () => {
        fetch(`/api/logs?bot_id=${id}`)
        .then(res => res.json())
        .then(data => {
            const box = document.getElementById("live-bot-logs");
            if(!box) return;
            if(!data.logs || data.logs.length === 0) { box.innerHTML = "<li>대기 중...</li>"; return; }
            box.innerHTML = data.logs.map(log => `<li>${log}</li>`).join("");
        });
    };
    fetchLogs();
    logInterval = setInterval(fetchLogs, 3000);
}

function updatePositionTable() {
    const rows = document.getElementById("position-rows");
    if(!rows) return;
    let html = "";
    let activeFutures = false;
    
    for(let key in botRegistry) {
        const bot = botRegistry[key];
        if(bot.status === 'running' && bot.market_type === 'Futures') {
            activeFutures = true;
            html += `
                <tr>
                    <td><strong>${bot.name}</strong></td>
                    <td>${bot.target_coin}/USDT</td>
                    <td><span class="badge-pos long">LONG</span></td>
                    <td style="color:var(--blue); font-weight:bold;">${bot.leverage}x</td>
                    <td>$64,250.0</td>
                    <td style="color:var(--green); font-weight:bold;">+$182.40 (+5.6%)</td>
                </tr>
            `;
        }
    }
    if(activeFutures) rows.innerHTML = html;
}

window.onload = initDashboard;
