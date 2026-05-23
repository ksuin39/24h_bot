import os
import asyncio
import importlib
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse  # 💡 화면 출력을 위해 반드시 필요한 라이브러리
from pydantic import BaseModel

app = FastAPI()
bot_registry = {}

# [1단계] 서버가 켜질 때 폴더 자동 스캔
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
                "seed_money": 0,
                "logs": [f"📁 bots/ 폴더에서 {file} 탐지 완료."]
            }

class BotStartRequest(BaseModel):
    seed_money: int

# [2단계] 봇 파일 동적 로드 및 무한 루프 가동
async def run_bot_loop(bot_id: str):
    bot = bot_registry[bot_id]
    bot["logs"].append(f"🚀 {bot['name']}의 가상 루프 가동 시작!")
    
    while bot["status"] == "running":
        try:
            module = importlib.import_module(f"bots.{bot_id}")
            importlib.reload(module)
            
            if hasattr(module, "trade_logic"):
                await module.trade_logic(bot["seed_money"])
                # 봇 파일 내부에서 프린트한 것과 별개로 대시보드 출력용 로그도 한 줄 남겨줍니다.
                bot["logs"].append(f"✅ [{bot_id}] 실시간 시세 및 RSI 파동 계산 완료")
            else:
                bot["logs"].append(f"❌ 에러: trade_logic 함수를 찾을 수 없습니다.")
                bot["status"] = "idle"
                break
        except Exception as e:
            bot["logs"].append(f"❌ 실행 중 치명적 에러: {str(e)}")
            bot["status"] = "idle"
            break
            
        await asyncio.sleep(10)

# [3단계] API 제어 창구
@app.get("/api/bots")
def get_all_bots():
    return bot_registry

@app.post("/api/bots/{bot_id}/start")
def start_bot(bot_id: str, data: BotStartRequest, background_tasks: BackgroundTasks):
    if bot_id not in bot_registry or bot_registry[bot_id]["status"] == "running":
        return {"success": False, "message": "존재하지 않거나 이미 가동 중입니다."}
    
    bot_registry[bot_id]["seed_money"] = data.seed_money
    bot_registry[bot_id]["status"] = "running"
    background_tasks.add_task(run_bot_loop, bot_id)
    return {"success": True, "message": f"{bot_id} 가동 시작"}

@app.post("/api/bots/{bot_id}/stop")
def stop_bot(bot_id: str):
    if bot_id in bot_registry:
        bot_registry[bot_id]["status"] = "idle"
        bot_registry[bot_id]["logs"].append("🛑 유저 요청으로 가동 중지됨.")
        return {"success": True, "message": f"{bot_id} 중지 완료"}
    return {"success": False, "message": "봇 없음"}

# -------------------------------------------------------------
# 🎨 [최종 추가] 메인 접속 시 나타나는 실시간 대시보드 화면
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard_page():
    bot_id = "martingale_bot"
    bot_data = bot_registry.get(bot_id, {"status": "offline", "seed_money": 0, "logs": ["봇이 아직 로드되지 않았습니다."]})
    status = bot_data.get("status", "offline")
    
    status_badge = "🟢 가동 중 (RUNNING)" if status == "running" else "🟡 대기 중 (IDLE)"
    if status == "offline": status_badge = "🔴 오프라인 (OFFLINE)"

    # 최근 로그 10개 추출하여 이쁘게 줄바꿈 처리
    logs_html = "".join([f"<li>{log}</li>" for log in bot_data.get("logs", [])[-10:][::-1]])

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>사지방 24h 비트코인 봇</title>
        <style>
            body {{ font-family: 'Malgun Gothic', sans-serif; background-color: #f5f6fa; margin: 0; padding: 40px; color: #2f3640; }}
            .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ margin-top: 0; color: #1e2f97; border-bottom: 2px solid #f1f2f6; padding-bottom: 15px; font-size: 24px; }}
            .status {{ font-size: 18px; font-weight: bold; margin-bottom: 20px; }}
            .card {{ background: #f8f9fa; border-left: 5px solid #1e2f97; padding: 15px; border-radius: 4px; margin-bottom: 20px; }}
            .card p {{ margin: 5px 0; font-size: 16px; }}
            .log-box {{ background: #2f3640; color: #f5f6fa; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 13px; height: 220px; overflow-y: auto; }}
            .log-box ul {{ list-style: none; padding: 0; margin: 0; }}
            .log-box li {{ margin-bottom: 8px; border-bottom: 1px solid #487eb0; padding-bottom: 4px; }}
            .btn-refresh {{ background: #1e2f97; color: white; border: none; padding: 12px 15px; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; margin-bottom: 15px; font-size: 15px; }}
            .btn-refresh:hover {{ background: #273c75; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📡 24h 자동매매 시스템 대시보드</h1>
            <div class="status">서버 상태: {status_badge}</div>
            
            <button class="btn-refresh" onclick="location.reload()">🔄 실시간 데이터 새로고침</button>
            
            <div class="card">
                <p><strong>🤖 대상 전략 파일:</strong> {bot_id}.py</p>
                <p><strong>💰 가동 시드머니:</strong> {bot_data.get("seed_money", 0):,} 원</p>
            </div>

            <h3>📋 최근 실시간 시스템 로그 (최신순)</h3>
            <div class="log-box">
                <ul>
                    {logs_html}
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content
