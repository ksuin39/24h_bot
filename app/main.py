from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_root():
    # 요청하신 항목들을 포함한 테일윈드 CSS 기반의 다크모드 트레이딩 대시보드 HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crypto Bot Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-gray-100 font-sans antialiased">

        <nav class="bg-gray-800 border-b border-gray-700 p-4">
            <div class="container mx-auto flex justify-between items-center">
                <h1 class="text-xl font-bold text-green-400">⚡ 24H Trading Bot System</h1>
                <div class="flex items-center space-x-4">
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-900 text-green-300">
                        ● API 연결 완료 (정상)
                    </span>
                </div>
            </div>
        </nav>

        <div class="container mx-auto p-6 space-y-6">
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <p class="text-gray-400 text-sm font-medium">연결 계좌 총잔고</p>
                    <p class="text-2xl font-bold mt-1 text-white">$14,520.80</p>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <p class="text-gray-400 text-sm font-medium">가동 중인 봇 수</p>
                    <p class="text-2xl font-bold mt-1 text-green-400">4 <span class="text-sm text-gray-400">/ 5개</span></p>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <p class="text-gray-400 text-sm font-medium">오늘 PnL</p>
                    <p class="text-2xl font-bold mt-1 text-green-400">+$120.50</p>
                    <span class="text-xs text-green-400 font-medium">+1.24%</span>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <p class="text-gray-400 text-sm font-medium">7일 누적 PnL</p>
                    <p class="text-2xl font-bold mt-1 text-green-400">+$840.00</p>
                    <span class="text-xs text-green-400 font-medium">+6.15%</span>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <p class="text-gray-400 text-sm font-medium">30일 누적 PnL</p>
                    <p class="text-2xl font-bold mt-1 text-red-400">-$310.20</p>
                    <span class="text-xs text-red-400 font-medium">-2.10%</span>
                </div>
            </div>

            <div class="bg-gray-800 rounded-lg border border-gray-700 p-6">
                <h2 class="text-lg font-bold mb-4 text-gray-200">📊 현재 진입 포지션 현황</h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-left border-collapse">
                        <thead>
                            <tr class="border-b border-gray-700 text-gray-400 text-sm">
                                <th class="pb-3 font-medium">봇 이름</th>
                                <th class="pb-3 font-medium">투자 자산</th>
                                <th class="pb-3 font-medium">포지션</th>
                                <th class="pb-3 font-medium">진입가</th>
                                <th class="pb-3 font-medium">현재가</th>
                                <th class="pb-3 font-medium">실시간 수익률</th>
                            </tr>
                        </thead>
                        <tbody class="text-sm divide-y divide-gray-700">
                            <tr>
                                <td class="py-3 font-medium text-white">Grid_Bot_V1</td>
                                <td class="py-3 text-gray-300">BTC/USDT</td>
                                <td class="py-3"><span class="px-2 py-0.5 rounded text-xs font-bold bg-green-900 text-green-300">LONG</span></td>
                                <td class="py-3 text-gray-300">$64,200</td>
                                <td class="py-3 text-gray-300">$64,850</td>
                                <td class="py-3 text-green-400 font-semibold">+1.01%</td>
                            </tr>
                            <tr>
                                <td class="py-3 font-medium text-white">Trend_Follow_Eth</td>
                                <td class="py-3 text-gray-300">ETH/USDT</td>
                                <td class="py-3"><span class="px-2 py-0.5 rounded text-xs font-bold bg-red-900 text-red-300">SHORT</span></td>
                                <td class="py-3 text-gray-300">$3,450</td>
                                <td class="py-3 text-gray-300">$3,410</td>
                                <td class="py-3 text-green-400 font-semibold">+1.15%</td>
                            </tr>
                            <tr>
                                <td class="py-3 font-medium text-white">RSI_Scalper</td>
                                <td class="py-3 text-gray-300">SOL/USDT</td>
                                <td class="py-3"><span class="px-2 py-0.5 rounded text-xs font-bold bg-green-900 text-green-300">LONG</span></td>
                                <td class="py-3 text-gray-300">$145.2</td>
                                <td class="py-3 text-gray-300">$142.1</td>
                                <td class="py-3 text-red-400 font-semibold">-2.13%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

        </div>
    </body>
    </html>
    """
    return html_content
