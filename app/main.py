from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    # 현재 이 파일(main.py)이 있는 위치를 기준으로 절대 경로를 계산합니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))      # app 폴더 위치
    parent_dir = os.path.dirname(current_dir)                     # 최상위 프로젝트 폴더 위치
    
    # 1. 혹시 app 폴더 안에 html이 있는지 확인
    path_inside_app = os.path.join(current_dir, "index.html")
    # 2. 혹시 최상위 폴더에 html이 있는지 확인
    path_outside_app = os.path.join(parent_dir, "index.html")
    # 3. 혹시 app/templates 폴더 안에 있는지 확인
    path_templates = os.path.join(current_dir, "templates", "index.html")
    
    # 순서대로 찾아서 먼저 발견되는 파일을 읽어옵니다.
    target_path = None
    if os.path.exists(path_outside_app):
        target_path = path_outside_app
    elif os.path.exists(path_inside_app):
        target_path = path_inside_app
    elif os.path.exists(path_templates):
        target_path = path_templates

    # 만약 세 군데 모두 파일이 없다면 에러 메시지와 함께 서버 컴퓨터의 실제 폴더 구조를 화면에 띄웁니다.
    if target_path is None:
        error_html = f"""
        <html>
            <body style="font-family: sans-serif; padding: 50px; line-height: 1.6;">
                <h1 style="color: #dc2626;">❌ index.html 파일을 찾을 수 없습니다!</h1>
                <p>백엔드 서버는 정상 작동 중이나, HTML 파일의 위치가 맞지 않습니다.</p>
                <h3>서버 내부 탐색 경로 리스트:</h3>
                <ul>
                    <li>최상위 경로 (없음): <code>{path_outside_app}</code></li>
                    <li>app 폴더 내부 (없음): <code>{path_inside_app}</code></li>
                    <li>templates 폴더 내부 (없음): <code>{path_templates}</code></li>
                </ul>
                <p><b>해결 방법:</b> 깃허브 저장소에 <code>index.html</code>이 어떤 폴더 안에 들어있는지 확인해 주세요.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=404)
        
    # 파일을 찾았다면 정상적으로 읽어서 대시보드 화면을 띄웁니다.
    with open(target_path, "r", encoding="utf-8") as f:
        return f.read()

# 프론트엔드가 호출할 샘플 API
@app.get("/api/stats")
async def get_stats():
    return {
        "visitors": "1,234",
        "sales": "4,560,000",
        "pending": "7"
    }
