from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

# 1. 화면에 띄울 대시보드 HTML을 파이썬 코드 안에 글로 직접 박아 넣습니다. (경로 에러 해결)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>나의 대시보드 - 메인</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-gray-100 flex h-screen overflow-hidden">

    <aside class="w-64 bg-slate-800 text-white flex flex-col justify-between p-5">
        <div>
            <h1 class="text-2xl font-bold mb-10">Dashboard</h1>
            <nav class="space-y-4">
                <a href="#" class="block py-2.5 px-4 rounded bg-slate-700 text-white font-medium">🏠 홈 (Overview)</a>
                <a href="#" class="block py-2.5 px-4 rounded text-slate-300 hover:bg-slate-700 hover:text-white">📊 통계 분석</a>
                <a href="#" class="block py-2.5 px-4 rounded text-slate-300 hover:bg-slate-700 hover:text-white">⚙️ 설정</a>
            </nav>
        </div>
        <div class="text-sm text-slate-400">v1.0.0 (FastAPI Backend)</div>
    </aside>

    <div class="flex-1 flex flex-col">
        <header class="bg-white shadow-xs px-6 py-4 flex justify-between items-center">
            <h2 class="text-xl font-semibold text-gray-800">대시보드 홈 (백엔드 연동 완료)</h2>
            <div class="text-sm text-gray-600">접속 중</div>
        </header>

        <main class="p-6 space-y-6 overflow-y-auto flex-1">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="bg-white p-6 rounded-xl shadow-xs border border-gray-200">
                    <p class="text-sm font-medium text-gray-500">오늘 방문자 수</p>
                    <p class="text-3xl font-bold text-gray-900 mt-2">1,234 명</p>
                </div>
                <div class="bg-white p-6 rounded-xl shadow-xs border border-gray-200">
                    <p class="text-sm font-medium text-gray-500">당월 매출</p>
                    <p class="text-3xl font-bold text-gray-900 mt-2">₩4,560,000</p>
                </div>
                <div class="bg-white p-6 rounded-xl shadow-xs border border-gray-200">
                    <p class="text-sm font-medium text-gray-500">처리 대기 문의</p>
                    <p class="text-3xl font-bold text-gray-900 mt-2">7 건</p>
                </div>
            </div>
            <div class="bg-white p-6 rounded-xl shadow-xs border border-gray-200 h-64 flex items-center justify-center text-gray-400">
                백엔드 웹 서비스 서버가 정상적으로 화면을 전송하고 있습니다.
            </div>
        </main>
    </div>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    # 주소창에 접속하면 위의 HTML 글자를 그대로 화면에 뿌려줍니다.
    return DASHBOARD_HTML

# 임시 API 데이터 (프론트엔드가 데이터를 요구할 때 줄 방)
@app.get("/api/stats")
async def get_stats():
    return {
        "visitors": "1,234",
        "sales": "4,560,000",
        "pending": "7"
    }
