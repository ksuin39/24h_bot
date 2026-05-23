import os
import asyncio
import importlib  # 💡 파일 이름으로 코드를 동적으로 불러오는 마법의 도구
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import HTMLResponse

app = FastAPI()
bot_registry = {}

# [1단계에서 만든 스캐너] 서버 켜질 때 폴더 읽기
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

# -------------------------------------------------------------
# 💡 [2단계 핵심] 봇 파일을 동적으로 읽어서 무한 루프를 돌리는 함수
# -------------------------------------------------------------
async def run_bot_loop(bot_id: str):
    bot = bot_registry[bot_id]
    bot["logs"].append(f"🚀 {bot['name']}의 가상 루프 가동 시작!")
    
    while bot["status"] == "running":
        try:
            # 1. bots.sample_strategy 형식으로 파일 동적 임포트(소환)
            module = importlib.import_module(f"bots.{bot_id}")
            # 2. 강제로 새로고침(코드 변경 대비)
            importlib.reload(module)
            
            # 3. 파일 안에 약속된 이름인 'trade_logic' 함수가 있는지 확인하고 실행!
            if hasattr(module, "trade_logic"):
                # 봇 파일 내부의 함수를 호출하면서 시드머니를 전달합니다.
                await module.trade_logic(bot["seed_money"])
                bot["logs"].append(f"✅ [{bot_id}] 내부 로직 실행 성공!")
            else:
                bot["logs"].append(f"❌ 에러: {bot_id}.py 내부에 trade_logic 함수가 없습니다.")
                bot["status"] = "idle"
                break
                
        except Exception as e:
            bot["logs"].append(f"❌ 실행 중 치명적 에러 발생: {str(e)}")
            bot["status"] = "idle"
            break
            
        # 10초마다 반복 실행
        await asyncio.sleep(10)

# -------------------------------------------------------------
# [3단계] API 창구 연동
# -------------------------------------------------------------
@app.get("/api/bots")
def get_all_bots():
    return bot_registry

@app.post("/api/bots/{bot_id}/start")
def start_bot(bot_id: str, data: BotStartRequest, background_tasks: BackgroundTasks):
    if bot_id not in bot_registry or bot_registry[bot_id]["status"] == "running":
        return {"success": False, "message": "존재하지 않거나 이미 가동 중입니다."}
    
    # 장부 세팅
    bot_registry[bot_id]["seed_money"] = data.seed_money
    bot_registry[bot_id]["status"] = "running"
    
    # 백그라운드에서 무한 루프 작동시키기
    background_tasks.add_task(run_bot_loop, bot_id)
    return {"success": True, "message": f"{bot_id} 가동 시작"}

@app.post("/api/bots/{bot_id}/stop")
def stop_bot(bot_id: str):
    if bot_id in bot_registry:
        bot_registry[bot_id]["status"] = "idle"
        bot_registry[bot_id]["logs"].append("🛑 유저 요청으로 가동 중지됨.")
        return {"success": True, "message": f"{bot_id} 중지 완료"}
    return {"success": False, "message": "봇 없음"}

from fastapi.responses import HTMLResponse  # 💡 맨 위에 이 import가 없다면 같이 추가해주세요!

# -------------------------------------------------------------
# 🎨 [프론트엔드] 실시간 모니터링 심플 대시보드 화면
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard_page():
    bot_id = "martingale_bot"
    
    # 봇 상태 데이터 가져오기 (없으면 기본값)
    bot_data = bot_registry.get(bot_id, {"status": "offline", "logs": ["봇이 아직 로드되지 않았습니다."]})
    status = bot_data.get("status", "offline")
    
    # 상태에 따른 이쁜 색상 태그 분기
    status_badge = "🟢 가동 중 (RUNNING)" if status == "running" else "🟡 대기 중 (IDLE)"
    if status == "offline": status_badge = "🔴 오프라인 (OFFLINE)"

    # 최근 로그 5개만 역순으로 긁어와서 HTML 태그로 조립
    logs_html = "".join([f"<li>{log}</li>" for log in bot_data.get("logs", [])[-5:][::-1]])

    # 화면 디자인 (HTML / CSS / JavaScript)
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
            .log-box {{ background: #2f3640; color: #f5f6fa; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 13px; height: 180px; overflow-y: auto; }}
            .log-box ul {{ list-style: none; padding: 0; margin: 0; }}
            .log-box li {{ margin-bottom: 8px; border-bottom: 1px solid #487eb0; padding-bottom: 4px; }}
            .btn-refresh {{ background: #1e2f97; color: white; border: none; padding: 10px 15px; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; margin-bottom: 15px; }}
            .btn-refresh:hover {{ background: #273c75; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📡 24h 자동매매 시스템 대시보드</h1>
            <div class="status">서버 상태: {status_badge}</div>
            
            <button class="btn-refresh" onclick="location.reload()">🔄 실시간 데이터 새로고침</button>
            
            <div class="card">
                <p><strong>🤖 대상 파일:</strong> {bot_id}.py</p>
                <p><strong>💰 설정 시드머니:</strong> {bot_data.get("seed_money", 0):,} 원</p>
            </div>

            <h3>📋 최근 시스템 실시간 로그 (최신순)</h3>
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
