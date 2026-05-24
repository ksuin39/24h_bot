import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# 현재 main.py가 있는 위치를 기준으로 templates 폴더 경로를 정확하게 지정합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # templates 폴더 안의 index.html 파일을 찾아서 화면에 그려줍니다.
    return templates.TemplateResponse("index.html", {"request": request})
