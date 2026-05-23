import os
import asyncio
import importlib  # 💡 파일 이름으로 코드를 동적으로 불러오는 마법의 도구
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

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
