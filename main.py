import asyncio
import requests
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI()

# 봇들의 상태와 가상 시드머니를 저장하는 임시 메모리
bot_registry = {
    "bot_1": {
        "name": "비트코인 봇",
        "market": "KRW-BTC",
        "status": "idle",
        "seed_money": 0,
        "current_balance": 0,
        "holding_amount": 0,
        "logs": []
    },
    "bot_2": {
        "name": "이더리움 봇",
        "market": "KRW-ETH",
        "status": "idle",
        "seed_money": 0,
        "current_balance": 0,
        "holding_amount": 0,
        "logs": []
    }
}

class BotStartRequest(BaseModel):
    seed_money: int

# 백그라운드에서 10초마다 돌아갈 매매 루프
async def trading_loop(bot_id: str):
    bot = bot_registry[bot_id]
    bot["logs"].append(f"🤖 {bot['name']} 모의투자 가동 시작!")
    
    while bot["status"] == "running":
        try:
            # 업비트에서 로그인 없이 실시간 현재가 가져오기
            url = f"https://api.upbit.com/v1/ticker?markets={bot['market']}"
            response = requests.get(url).json()
            current_price = float(response[0]["trade_price"])
            
            bot["logs"].append(f"⏰ 현재 {bot['name']} 시세: {current_price:,.0f}원")
            
            # 가상 매수 로직 (돈이 있으면 일단 다 산다!)
            if bot["current_balance"] > 0 and bot["holding_amount"] == 0:
                buy_amount = bot["current_balance"] / current_price
                bot["holding_amount"] = buy_amount
                bot["current_balance"] = 0
                bot["logs"].append(f"🛒 [가상 매수] {current_price:,.0f}원에 매수 완료! 보유량: {buy_amount:.6f}개")
                
        except Exception as e:
            bot["logs"].append(f"❌ 에러 발생: {str(e)}")
            
        await asyncio.sleep(10) # 10초 대기
        
    bot["logs"].append(f"🛑 {bot['name']} 가동 중지됨.")

# 대시보드용 API 창구들
@app.get("/api/bots")
def get_all_bots():
    return bot_registry

@app.get("/api/bots/{bot_id}/logs")
def get_bot_logs(bot_id: str):
    if bot_id in bot_registry:
        return {"logs": bot_registry[bot_id]["logs"]}
    return {"logs": ["봇이 없습니다."]}

@app.post("/api/bots/{bot_id}/start")
def start_bot(bot_id: str, data: BotStartRequest, background_tasks: BackgroundTasks):
    if bot_id not in bot_registry or bot_registry[bot_id]["status"] == "running":
        return {"success": False, "message": "시작할 수 없습니다."}
    
    bot_registry[bot_id]["seed_money"] = data.seed_money
    bot_registry[bot_id]["current_balance"] = data.seed_money
    bot_registry[bot_id]["holding_amount"] = 0
    bot_registry[bot_id]["status"] = "running"
    bot_registry[bot_id]["logs"] = []
    
    background_tasks.add_task(trading_loop, bot_id)
    return {"success": True, "message": f"{bot_registry[bot_id]['name']} 작동 시작"}

@app.post("/api/bots/{bot_id}/stop")
def stop_bot(bot_id: str):
    if bot_id in bot_registry:
        bot_registry[bot_id]["status"] = "idle"
        return {"success": True, "message": "중지 완료"}
    return {"success": False, "message": "봇 없음"}
