const botRegistry = ServerBotRegistry;
let selectedBotId = null;

function renderBots() {
    const idleList = document.getElementById("idle-list");
    const activeList = document.getElementById("active-list");
    
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
                <div class="bot-strategy">ID: ${bot.id} · 알고리즘</div>
            </div>
            <span style="font-size:12px; font-weight:bold; color:var(--green);">+15.2%</span>
        `;

        if (bot.status === "running") { activeList.appendChild(card); activeCount++; }
        else { idleList.appendChild(card); idleCount++; }
    }

    document.getElementById("idle-count").innerText = idleCount;
    document.getElementById("active-count").innerText = activeCount;
    document.getElementById("active-count-txt").innerText = `${activeCount} / 5`;
    updatePositionTable();
}

function allowDrop(ev) { ev.preventDefault(); }
function drag(ev) { ev.dataTransfer.setData("text", ev.target.id); }

function drop(ev, zone) {
    ev.preventDefault();
    const id = ev.dataTransfer.getData("text");
    const bot = botRegistry[id];
    
    if (zone === 'active' && bot.status === 'idle') {
        bot.status = 'running_ready';
        selectBot(id);
    } else if (zone === 'idle' && bot.status === 'running') {
        selectBot(id); submitToggle();
    }
    renderBots();
}

function selectBot(id) {
    selectedBotId = id;
    const bot = botRegistry[id];
    
    document.getElementById("panel-placeholder").style.display = "none";
    document.getElementById("panel-form").style.display = "flex";
    document.getElementById("form-bot-title").innerText = bot.name;
    document.getElementById("form-seed").value = bot.seed_money;
    document.getElementById("form-coin").value = bot.target_coin;
    document.getElementById("form-leverage").value = bot.leverage;
    updateLevLabel(bot.leverage);

    document.getElementById("form-bot-status").innerText = bot.status.toUpperCase();
    
    if (bot.status === "running") {
        document.getElementById("main-submit-btn").innerText = "🛑 시스템 가동 중지 (STOP)";
        document.getElementById("main-submit-btn").classList.add("stop-mode");
    } else {
        document.getElementById("main-submit-btn").innerText = bot.mode === 'Live' ? "🚀 실전 매매 가동" : "▷ 모의 거래 시작";
        document.getElementById("main-submit-btn").classList.remove("stop-mode");
    }
    setMode(bot.mode); setMarket(bot.market_type);
}

function updateLevLabel(val) { document.getElementById("lev-label").innerText = `Leverage 슬라이더: ${val}배`; }

function setMode(mode) {
    if(!selectedBotId) return;
    botRegistry[selectedBotId].mode = mode;
    ['Backtest', 'Mock', 'Live'].forEach(m => document.getElementById(`m-${m}`).classList.remove('active'));
    document.getElementById(`m-${mode}`).classList.add('active');
}

function setMarket(type) {
    if(!selectedBotId) return;
    botRegistry[selectedBotId].market_type = type;
    ['Spot', 'Futures'].forEach(t => document.getElementById(`t-${t}`).classList.remove('active'));
    document.getElementById(`t-${type}`).classList.add('active');
    document.getElementById("leverage-area").style.display = (type === 'Futures') ? 'flex' : 'none';
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
        bot.status = data.status;
        renderBots(); selectBot(selectedBotId);
    });
}

function updatePositionTable() {
    const rows = document.getElementById("position-rows"); rows.innerHTML = "";
    for(let key in botRegistry) {
        const bot = botRegistry[key];
        if(bot.status === 'running' && bot.market_type === 'Futures') {
            rows.innerHTML += `<tr><td><strong>${bot.name}</strong></td><td>${bot.target_coin}/USDT</td><td><span class="badge-pos long">LONG</span></td><td style="color:var(--blue); font-weight:bold;">${bot.leverage}x</td><td>$64,250.0</td><td>$65,120.5</td><td style="color:var(--green); font-weight:bold;">+$182.40</td></tr>`;
        }
    }
}

window.onload = renderBots;
