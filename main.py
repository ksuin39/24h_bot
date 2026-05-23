import os
import asyncio
import importlib
import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI()
bot_registry = {}

@app.on_event("startup")
def scan_bots_folder():
    global bot_registry
    BOTS_DIR = "bots"
    if not os.path.exists(BOTS_DIR):
        os.makedirs(BOTS_DIR)
        
    files = os.listdir(BOTS_DIR)
    for file in files:
        if file.endswith(".py"):
            bot_id = file.replace(".py", "")
            bot_registry[bot_id] = {
                "name": f"{bot_id} 봇",
                "status": "idle",
                "seed_money": 1000000,
                "target_coin": "BTC",
                "currency": "KRW",
                "leverage": 1, # 💡 레버리지 기본값 1배 세팅
                "logs": [f"📁 bots/ 폴더에서 {file} 탐지 완료."]
            }

async def run_bot_loop(bot_id: str):
    bot = bot_registry[bot_id]
    bot["logs"].append(f"🚀 {bot['name']} 가동! ({bot['target_coin']}/{bot['currency']} | 레버리지: {bot['leverage']}배)")
    
    while bot["status"] == "running":
        try:
            module = importlib.import_module(f"bots.{bot_id}")
            importlib.reload(module)
            
            if hasattr(module, "trade_logic"):
                # 💡 레버리지 배율(leverage)도 봇에게 정밀 배달합니다.
                await module.trade_logic(
                    seed_money=bot["seed_money"],
                    target_coin=bot["target_coin"],
                    currency=bot["currency"],
                    leverage=bot["leverage"]
                )
                bot["logs"].append(f"✅ [{bot_id}] {bot['target_coin']} 포지션 추격 연산 중 (레버리지 {bot['leverage']}x)")
            else:
                bot["logs"].append(f"❌ 에러: trade_logic 함수 없음")
                bot["status"] = "idle"
                break
        except Exception as e:
            bot["logs"].append(f"❌ 실행 에러: {str(e)}")
            bot["status"] = "idle"
            break
            
        await asyncio.sleep(10)

def get_crypto_top10(currency="KRW"):
    try:
        if currency == "KRW":
            markets = requests.get("https://api.upbit.com/v1/market/all").json()
            krw_markets = [m['market'] for m in markets if m['market'].startswith("KRW-")]
            tickers = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(krw_markets[:100])}").json()
            sorted_tickers = sorted(tickers, key=lambda x: x['acc_trade_price_24h'], reverse=True)
            top10 = []
            for i, t in enumerate(sorted_tickers[:10]):
                symbol = t['market'].replace("KRW-", "")
                change_rate = t['change_rate'] * 100 if t['change'] != "FALL" else -t['change_rate'] * 100
                top10.append({"rank": i+1, "symbol": symbol, "price": f"{t['trade_price']:,}원", "volume": f"{int(t['acc_trade_price_24h']/100000000):,}억", "change": change_rate})
            return top10
        else:
            res = requests.get("https://api.binance.com/api/v3/ticker/24hr").json()
            usdt_markets = [r for r in res if r['symbol'].endswith("USDT")]
            sorted_binance = sorted(usdt_markets, key=lambda x: float(x['quoteVolume']), reverse=True)
            top10 = []
            for i, t in enumerate(sorted_binance[:10]):
                symbol = t['symbol'].replace("USDT", "")
                price = float(t['lastPrice'])
                price_str = f"${price:,.2f}" if price >= 1 else f"${price:.6f}"
                top10.append({"rank": i+1, "symbol": symbol, "price": price_str, "volume": f"{int(float(t['quoteVolume'])/1000000):,}M", "change": float(t['priceChangePercent'])})
            return top10
    except: return []

@app.get("/action/toggle")
def toggle_bot(background_tasks: BackgroundTasks, seed: int = 1000000, coin: str = "BTC", unit: str = "KRW", lev: int = 1):
    bot_id = "martingale_bot"
    if bot_id not in bot_registry: return {"status": "error"}
    
    bot = bot_registry[bot_id]
    if bot["status"] == "running":
        bot["status"] = "idle"
        bot["logs"].append("🛑 대시보드 제어판에서 유저가 원격 중지 시켰습니다.")
        return {"status": "idle"}
    else:
        bot["seed_money"] = seed
        bot["target_coin"] = coin.upper()
        bot["currency"] = unit.upper()
        bot["leverage"] = lev # 💡 입력받은 레버리지 배율 저장
        bot["status"] = "running"
        
        background_tasks.add_task(run_bot_loop, bot_id)
        bot["logs"].append(f"🟩 설정 반영 완료 -> 자금: {seed:,}{unit} | 대상: {coin} | 레버리지: {lev}배")
        return {"status": "running"}

@app.get("/", response_class=HTMLResponse)
def dashboard_page():
    bot_id = "martingale_bot"
    bot_data = bot_registry.get(bot_id, {"status": "idle", "seed_money": 1000000, "target_coin": "BTC", "currency": "KRW", "leverage": 1, "logs": []})
    status = bot_data.get("status", "idle")
    
    is_run = (status == "running")
    status_text = "🟢 시스템 가동 중" if is_run else "🟡 시스템 대기 중"
    btn_text = "🛑 가동 중지 (STOP)" if is_run else "🚀 설정값으로 가동 시작 (START)"
    btn_class = "btn-stop" if is_run else "btn-start"
    disabled_attr = "disabled" if is_run else ""
    
    logs_html = "".join([f"<li>{log}</li>" for log in bot_data.get("logs", [])[-5:][::-1]])
    
    top10_data = get_crypto_top10(bot_data["currency"])
    top10_html = ""
    for c in top10_data:
        color = "#e66767" if c['change'] > 0 else "#3867d6" if c['change'] < 0 else "#2f3640"
        top10_html += f"""
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px dashed #dcdde1; font-size:14px;">
            <span style="font-weight:bold; color:#718093;">{c['rank']}. {c['symbol']}</span>
            <span style="color:{color}; font-weight:bold;">{c['price']} ({c['change']:+.2f}%)</span>
            <span style="font-size:12px; color:#7f8c8d;">{c['volume']}</span>
        </div>
        """

    # 레버리지 셀렉트 박스 옵션 동적 생성 구문
    lev_options = "".join([f'<option value="{x}" {"selected" if bot_data["leverage"]==x else ""}>{x}배</option>' for x in [1, 2, 3, 5, 10, 20, 50]])

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>24h 하이엔드 제어 콘솔</title>
        <style>
            body {{ font-family: 'Malgun Gothic', sans-serif; background-color: #f1f2f6; margin: 0; padding: 30px; }}
            .layout {{ display: flex; max-width: 1100px; margin: 0 auto; gap: 25px; }}
            .panel {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            .left-panel {{ flex: 1.2; }}
            .right-panel {{ flex: 0.8; height: fit-content; }}
            h1 {{ margin: 0 0 15px 0; color: #1e2f97; font-size: 22px; border-bottom: 2px solid #f1f2f6; padding-bottom: 10px; }}
            .status {{ font-size: 16px; font-weight: bold; background: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 15px; border: 1px solid #dcdde1; }}
            .form-group {{ margin-bottom: 12px; }}
            .form-group label {{ display: block; font-size: 14px; font-weight: bold; margin-bottom: 5px; color: #57606f; }}
            .form-group input, .form-group select {{ width: 95%; padding: 10px; border: 1px solid #00000020; border-radius: 6px; font-size: 14px; }}
            .btn {{ color: white; border: none; padding: 14px; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%; font-size: 16px; margin-top: 10px; transition: 0.2s; }}
            .btn-start {{ background: #20bf6b; }} .btn-start:hover {{ background: #26de81; }}
            .btn-stop {{ background: #eb3b5a; }} .btn-stop:hover {{ background: #ff5252; }}
            .log-box {{ background: #2f3640; color: #f5f6fa; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 12px; height: 120px; overflow-y: auto; margin-top: 15px; }}
            .log-box ul {{ list-style: none; padding: 0; margin: 0; }}
            .log-box li {{ margin-bottom: 5px; border-bottom: 1px solid #4b5563; padding-bottom: 3px; }}
        </style>
        <script>
            function toggleBot() {{
                const seed = document.getElementById('seed').value;
                const coin = document.getElementById('coin').value;
                const unit = document.getElementById('unit').value;
                const lev = document.getElementById('lev').value;
                
                fetch(`/action/toggle?seed=${{seed}}&coin=${{coin}}&unit=${{unit}}&lev=${{lev}}`)
                .then(() => {{ location.reload(); }});
            }}
        </script>
    </head>
    <body>
        <div class="layout">
            <div class="panel left-panel">
                <h1>⚙️ 24h 매매 봇 무한 커스텀</h1>
                <div class="status">{status_text}</div>
                
                <div class="form-group">
                    <label>💰 가동 시드머니 금액</label>
                    <input type="number" id="seed" value="{bot_data['seed_money']}" {disabled_attr}>
                </div>
                <div class="form-group">
                    <label>⚡ 숏/롱 선물 레버리지 배율 설정</label>
                    <select id="lev" {disabled_attr}>
                        {lev_options}
                    </select>
                </div>
                <div class="form-group">
                    <label>💱 거래 기준 화폐 단위</label>
                    <select id="unit" {disabled_attr}>
                        <option value="KRW" {"selected" if bot_data['currency']=='KRW' else ""}>원화 (KRW 마켓)</option>
                        <option value="USD" {"selected" if bot_data['currency']=='USD' else ""}>달러 (글로벌 USDT 마켓)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>🪙 대상 타겟 코인 심볼</label>
                    <input type="text" id="coin" value="{bot_data['target_coin']}" {disabled_attr}>
                </div>
                
                <button class="btn {btn_class}" onclick="toggleBot()">{btn_text}</button>
                
                <h3>📋 현재 세션 실시간 로그</h3>
                <div class="log-box"><ul>{logs_html}</ul></div>
            </div>
            
            <div class="panel right-panel">
                <h1>🔥 실시간 거래대금 TOP 10 ({bot_data['currency']})</h1>
                <div>{top10_html}</div>
            </div>
        </div>
    </body>
    </html>
    """
