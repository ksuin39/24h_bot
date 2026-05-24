import os
import asyncio
import importlib
import json
import requests
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# 폴더 자동 생성 안전장치
if not os.path.exists("static"): os.makedirs("static")
if not os.path.exists("templates"): os.makedirs("templates")
if not os.path.exists("bots"): os.makedirs("bots")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

bot_registry = {}

@app.on_event("startup")
def scan_bots_folder():
    global bot_registry
    BOTS_DIR = "bots"
    files = os.listdir(BOTS_DIR)
    
    # 봇이 하나도 없을 때를 대비해 기본 더미 봇 하나 강제 등록 (에러 방지)
    bot_registry["martingale_bot"] = {
        "id": "martingale_bot",
        "name": "Martingale Bot",
        "status": "idle",
        "seed_money": 1000000,
        "target_coin": "BTC",
        "currency": "KRW",
        "leverage": 1,
        "market_type": "Spot",
        "mode": "Mock",
        "logs": ["📁 시스템 초기화 완료."]
    }
    
    for file in files:
        if file.endswith(".py"):
            bot_id = file.replace(".py", "")
            if bot_id not in bot_registry:
                bot_registry[bot_id] = {
                    "id": bot_id,
                    "name": f"{bot_id.replace('_', ' ').title()}",
                    "status": "idle",
                    "seed_money": 1000000,
                    "target_coin": "BTC",
                    "currency": "KRW",
                    "leverage": 1,
                    "market_type": "Spot",
                    "mode": "Mock",
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
                await module.trade_logic(
                    seed_money=bot["seed_money"],
                    target_coin=bot["target_coin"],
                    currency=bot["currency"],
                    leverage=bot["leverage"]
                )
                bot["logs"].append(f"✅ [{bot_id}] 연산 완료 (레버리지 {bot['leverage']}x)")
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
            tickers = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(krw_markets[:20])}").json()
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
    except: 
        return []

@app.get("/api/top10")
def api_top10(currency: str = "KRW"):
    return get_crypto_top10(currency)

@app.get("/action/toggle")
def toggle_bot(background_tasks: BackgroundTasks, bot_id: str, seed: int, coin: str, unit: str, lev: int, market: str, mode: str):
    if bot_id not in bot_registry: return {"status": "error"}
    
    bot = bot_registry[bot_id]
    if bot["status"] == "running":
        bot["status"] = "idle"
        bot["logs"].append("🛑 대시보드 제어판에서 유저가 원격 중지 시켰습니다.")
        return {"status": "idle", "bot": bot}
    else:
        bot["seed_money"] = seed
        bot["target_coin"] = coin.upper()
        bot["currency"] = unit.upper()
        bot["leverage"] = lev
        bot["market_type"] = market
        bot["mode"] = mode
        bot["status"] = "running"
        
        background_tasks.add_task(run_bot_loop, bot_id)
        bot["logs"].append(f"🟩 설정 반영 완료 -> 자금: {seed:,}{unit} | 대상: {coin} | 레버리지: {lev}배")
        return {"status": "running", "bot": bot}

@app.get("/api/logs")
def get_bot_logs(bot_id: str):
    if bot_id in bot_registry:
        return {"logs": bot_registry[bot_id].get("logs", [])[-5:][::-1]}
    return {"logs": []}

@app.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    # 💡 Jinja2가 따옴표 이스케이프 오류를 내지 않도록 미리 가공해서 안전하게 주입
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "bots_json": json.dumps(bot_registry).replace("'", "\\'")
    })
