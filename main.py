import os
import asyncio
import importlib
import json
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# 💡 정적 파일(CSS, JS)과 HTML 템플릿 폴더 경로 지정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

bot_registry = {}

@app.on_event("startup")
def scan_bots_folder():
    global bot_registry
    BOTS_DIR = "bots"
    if not os.path.exists(BOTS_DIR): os.makedirs(BOTS_DIR)
        
    files = os.listdir(BOTS_DIR)
    bot_names = {
        "martingale_bot": "DCA Master",
        "rsi_bot": "RSI Master",
        "macd_bot": "MACD Hunter"
    }
    
    for file in files:
        if file.endswith(".py"):
            bot_id = file.replace(".py", "")
            display_name = bot_names.get(bot_id, f"{bot_id} Bot")
            bot_registry[bot_id] = {
                "id": bot_id,
                "name": display_name,
                "status": "idle",
                "seed_money": 1000,
                "target_coin": "BTC",
                "currency": "USD",
                "leverage": 3,
                "market_type": "Futures",
                "mode": "Mock"
            }

@app.get("/action/toggle")
def toggle_bot(background_tasks: BackgroundTasks, bot_id: str, seed: int, coin: str, unit: str, lev: int, market: str, mode: str):
    if bot_id not in bot_registry: return {"status": "error"}
    bot = bot_registry[bot_id]
    
    if bot["status"] == "running":
        bot["status"] = "idle"
        return {"status": "idle"}
    else:
        bot["seed_money"] = seed
        bot["target_coin"] = coin.upper()
        bot["currency"] = unit.upper()
        bot["leverage"] = lev
        bot["market_type"] = market
        bot["mode"] = mode
        bot["status"] = "running"
        return {"status": "running"}

# 💡 HTML 파일을 읽어서 장부 데이터(JSON)와 함께 프론트엔드로 조립 배달
@app.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "bots_json": json.dumps(bot_registry)
    })
