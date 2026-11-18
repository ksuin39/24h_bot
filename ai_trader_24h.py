"""
ai_trader_24h.py
----------------
지표/규칙 없이 '가격 변화'만 보고 스스로 배우는 강화학습(Q-learning) 모의투자 봇.
- 데이터: 거래소 공개 웹소켓 (API 키 불필요, 실제 주문 없음 = 돈 안 씀)
- 10초마다 1스텝: 최근 가격 변화 4개를 보고 [관망 / 롱 / 숏] 중 하나 선택
- 보상: 포지션 수익 - 수수료(0.05%)
- 이어 달리기 지원: 중간 상태를 저장했다가 다음 실행에서 이어서 학습

실행 예:
    pip install websockets
    python ai_trader_24h.py --hours 4.8 --source coinbase   # 4.8시간 실행 (이전 상태 있으면 이어서)
    python ai_trader_24h.py --sim                           # 오프라인 테스트 (가짜 가격)
옵션:
    --hours N      이번 실행 시간 (기본 24)
    --source X     okx 또는 coinbase (기본 okx)
"""
import asyncio, json, random, time, csv, sys, os

# ===== 설정 =====
STEP_SEC    = 10                # 몇 초마다 판단할지
FEE         = 0.0005            # 포지션 바꿀 때 수수료 (0.05%)
LOOKBACK    = 4                 # 최근 몇 개의 가격 변화를 볼지
FLAT_TH     = 0.0002            # 이보다 작은 변화는 '보합' (0.02%)
ALPHA       = 0.1               # 학습률
GAMMA       = 0.95              # 미래 보상 할인
EPS_START, EPS_END = 1.0, 0.05  # 탐험 비율 (처음 100% 랜덤 → 점점 감소)
TOTAL_HOURS = 24                # 탐험률을 줄여나갈 기준 전체 시간
OUT_DIR     = "ai_trader_output"
STATE_FILE  = os.path.join(OUT_DIR, "state.json")

SOURCES = {
    "okx": {
        "url": "wss://ws.okx.com:8443/ws/v5/public",
        "sub": {"op": "subscribe", "args": [{"channel": "tickers", "instId": "BTC-USDT-SWAP"}]},
    },
    "coinbase": {
        "url": "wss://ws-feed.exchange.coinbase.com",
        "sub": {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]},
    },
}

ACTIONS = {0: 0, 1: 1, 2: -1}   # 행동 -> 목표 포지션 (0 관망, 1 롱, -1 숏)
ACT_NAME = {0: "FLAT", 1: "LONG", 2: "SHORT"}


def arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def parse_price(source, d):
    if source == "okx" and "data" in d:
        return float(d["data"][0]["last"])
    if source == "coinbase" and d.get("type") == "ticker":
        return float(d["price"])
    return None


def bucket(ret):
    if ret > FLAT_TH:  return "U"
    if ret < -FLAT_TH: return "D"
    return "F"


class Trader:
    def __init__(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        self.total_steps = int(TOTAL_HOURS * 3600 / STEP_SEC)
        # 기본값
        self.q, self.step, self.pos, self.equity = {}, 0, 0, 1.0
        self.first_price, self.trades, self.elapsed_before = None, 0, 0.0
        self.prices = []
        # 이전 실행 상태가 있으면 이어서
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, encoding="utf-8") as f:
                st = json.load(f)
            self.q, self.step, self.pos = st["q"], st["step"], st["pos"]
            self.equity, self.first_price = st["equity"], st["first_price"]
            self.trades, self.elapsed_before = st["trades"], st["elapsed"]
            self.prices = st.get("prices", [])
            print(f"이전 상태 불러옴: {self.elapsed_before/3600:.2f}시간 학습된 상태에서 이어서 시작", flush=True)
        self.prev = None
        self.pending_fee = 0.0
        self.start = time.time()
        new = not os.path.exists(os.path.join(OUT_DIR, "log.csv"))
        self.f = open(os.path.join(OUT_DIR, "log.csv"), "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "price", "action", "position", "reward",
                             "ai_equity", "buyhold_equity", "epsilon"])

    # ---- 학습 로직 ----
    def eps(self):
        frac = min(1.0, self.step / max(1, self.total_steps * 0.7))
        return EPS_START + (EPS_END - EPS_START) * frac

    def qv(self, s):
        return self.q.setdefault(s, [0.0, 0.0, 0.0])

    def choose(self, s):
        if random.random() < self.eps():
            return random.randrange(3)
        v = self.qv(s)
        return max(range(3), key=lambda a: v[a])

    def state(self):
        rets = [self.prices[i] / self.prices[i - 1] - 1 for i in range(-LOOKBACK, 0)]
        return "".join(bucket(r) for r in rets) + f"|{self.pos}"

    def on_price(self, price):
        if self.first_price is None:
            self.first_price = price
        self.prices.append(price)
        self.prices = self.prices[-50:]
        if len(self.prices) < LOOKBACK + 2:
            return

        # 1) 지난 행동 결과(보상)
        ret = self.prices[-1] / self.prices[-2] - 1
        reward = self.pos * ret
        self.equity *= (1 + reward)

        s2 = self.state()
        if self.prev:
            s, a = self.prev
            v = self.qv(s)
            r = (reward - self.pending_fee) * 100      # 수수료 = 벌점
            v[a] += ALPHA * (r + GAMMA * max(self.qv(s2)) - v[a])
            self.step += 1

        # 2) 새 행동
        a2 = self.choose(s2)
        new_pos = ACTIONS[a2]
        self.pending_fee = 0.0
        if new_pos != self.pos:
            self.pending_fee = FEE * abs(new_pos - self.pos)
            self.equity *= (1 - self.pending_fee)
            self.trades += 1
            self.pos = new_pos
        self.prev = (s2, a2)

        bh = price / self.first_price
        self.w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), price, ACT_NAME[a2], self.pos,
                         round(reward, 6), round(self.equity, 6), round(bh, 6), round(self.eps(), 3)])
        if self.step % 60 == 0:
            self.f.flush()
            self.save()
            self.report()

    # ---- 저장/출력 ----
    def elapsed(self):
        return self.elapsed_before + (time.time() - self.start)

    def save(self):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"q": self.q, "step": self.step, "pos": self.pos, "equity": self.equity,
                       "first_price": self.first_price, "trades": self.trades,
                       "elapsed": self.elapsed(), "prices": self.prices}, f)

    def summary_lines(self):
        bh = self.prices[-1] / self.first_price if self.prices else 1.0
        best = {"FLAT": 0, "LONG": 0, "SHORT": 0}
        for v in self.q.values():
            best[ACT_NAME[max(range(3), key=lambda a: v[a])]] += 1
        return [
            f"누적 학습 시간: {self.elapsed()/3600:.2f}시간 (판단 {self.step}회)",
            f"AI 수익률: {(self.equity-1)*100:+.3f}%",
            f"그냥 들고 있었을 때: {(bh-1)*100:+.3f}%",
            f"거래 횟수: {self.trades}회",
            f"현재 탐험률(랜덤 행동 비율): {self.eps():.2f}",
            f"상황별로 선호하게 된 행동: 관망 {best['FLAT']} / 롱 {best['LONG']} / 숏 {best['SHORT']}",
        ]

    def report(self):
        print(" | ".join(self.summary_lines()), flush=True)

    def finish(self):
        self.f.close()
        self.save()
        lines = self.summary_lines()
        print("\n===== 이번 실행 종료 =====")
        for l in lines:
            print(l)
        # GitHub Actions 화면에 요약 표시
        summ = os.environ.get("GITHUB_STEP_SUMMARY")
        if summ:
            with open(summ, "a", encoding="utf-8") as f:
                f.write("## AI 트레이더 결과\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")


async def run_live(hours, source):
    import websockets
    cfg = SOURCES[source]
    t = Trader()
    end = time.time() + hours * 3600
    last = {"p": None}
    try:
        while time.time() < end:
            try:
                async with websockets.connect(cfg["url"], ping_interval=20) as ws:
                    await ws.send(json.dumps(cfg["sub"]))
                    print(f"{source} 연결됨, 학습 진행 중…", flush=True)

                    async def reader():
                        async for msg in ws:
                            p = parse_price(source, json.loads(msg))
                            if p:
                                last["p"] = p

                    task = asyncio.create_task(reader())
                    while time.time() < end and not task.done():
                        await asyncio.sleep(STEP_SEC)
                        if last["p"]:
                            t.on_price(last["p"])
                    task.cancel()
            except Exception as e:
                print("연결 끊김, 5초 후 재연결:", e, flush=True)
                await asyncio.sleep(5)
    finally:
        t.finish()


def run_sim():
    t = Trader()
    p = 60000.0
    for _ in range(8640 // 2):          # 반나절 분량씩 (두 번 실행하면 이어 달리기 테스트)
        p *= 1 + random.gauss(0, 0.0008)
        t.on_price(p)
    t.finish()


if __name__ == "__main__":
    if "--sim" in sys.argv:
        run_sim()
    else:
        hours = float(arg("--hours", "24"))
        source = arg("--source", "okx")
        try:
            asyncio.run(run_live(hours, source))
        except KeyboardInterrupt:
            pass
