import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 1. 처음엔 비어있는 장부 (서버가 켜지면서 자동으로 채울 예정)
bot_registry = {}

# 2. 서버가 켜질 때 자동으로 실행되는 '스캐너' 함수
@app.on_event("startup")
def scan_bots_folder():
    global bot_registry
    BOTS_DIR = "bots"
    
    # 만약 bots 폴더가 없으면 에러 방지를 위해 자동으로 만듬
    if not os.path.exists(BOTS_DIR):
        os.makedirs(BOTS_DIR)
        
    # bots/ 폴더 안의 모든 파일 목록을 긁어옴
    files = os.listdir(BOTS_DIR)
    
    for file in files:
        # 확장자가 .py로 끝나는 진짜 파이썬 파일만 골라내기
        if file.endswith(".py"):
            bot_id = file.replace(".py", "") # 'sample_strategy.py' -> 'sample_strategy' (아이디로 사용)
            
            # 장부에 봇 자동 등록!
            bot_registry[bot_id] = {
                "name": f"{bot_id} (자동 탐지됨)",
                "status": "idle",
                "seed_money": 0,
                "current_balance": 0,
                "holding_amount": 0,
                "logs": [f"📁 bots/ 폴더에서 {file} 파일이 성공적으로 탐지되었습니다."]
            }
            print(f"✅ 봇 등록 완료: {bot_id}")

# 3. 데이터 검증용 모델
class BotStartRequest(BaseModel):
    seed_money: int

# 4. 탐지된 봇 명단을 브라우저에 뿌려주는 창구
@app.get("/api/bots")
def get_all_bots():
    return bot_registry
