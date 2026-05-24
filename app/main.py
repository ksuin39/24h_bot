from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# 프로젝트 최상위 폴더에 있는 index.html 파일을 읽어옵니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    # 최상위 폴더에 바로 위치한 index.html을 찾습니다.
    html_path = os.path.join(BASE_DIR, "index.html")
    
    # 만약 index.html이 다른 폴더(예: templates 등)에 있다면 경로를 수정해야 합니다.
    if not os.path.exists(html_path):
        return "<h1>index.html 파일을 찾을 수 없습니다. 경로를 확인해주세요!</h1>"
        
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# 프론트엔드 대시보드와 통신할 가짜 데이터 API 예시
@app.get("/api/stats")
async def get_stats():
    return {
        "visitors": "1,234",
        "sales": "4,560,000",
        "pending": "7"
    }
