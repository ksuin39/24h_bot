from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# HTML 파일이 저장된 경로 지정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    # templates 폴더 안의 index.html 파일을 읽어서 화면에 띄웁니다.
    html_path = os.path.join(TEMPLATE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# 추후 실제 대시보드 데이터를 전달할 API 공간 (백엔드 기능)
@format.get("/api/stats")
async def get_stats():
    return {
        "visitors": 1234,
        "sales": 4560000,
        "pending_inquiries": 7
    }
