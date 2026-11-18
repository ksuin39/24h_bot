"""
replay_train.py
---------------
과거 데이터를 빠르게 재생하면서 AI를 학습시키는 '압축 학습' 버전.
(ai_trader_24h.py 와 같은 AI: 지표 없이 가격 변화만 보고 관망/롱/숏 선택)

1) Coinbase 공개 API에서 BTC-USD 1분봉을 DAYS일치 받음 (API 키 불필요)
2) 앞 70%로 학습 (EPOCHS번 반복 재생)
3) 뒤 30%는 처음 보는 데이터로 시험 (학습 끔, 랜덤 행동 없음) ← 진짜 실력
4) 결과를 GitHub Actions Summary에 표 + 그래프로 출력

실행:
    python replay_train.py --days 10 --epochs 5
    python replay_train.py --sim          # 네트워크 없이 가짜 데이터로 테스트
"""
import json, os, random, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

FEE, LOOKBACK, FLAT_TH = 0.0005, 4, 0.0002
ALPHA, GAMMA = 0.1, 0.95
EPS_START, EPS_END = 1.0, 0.05
ACTIONS = {0: 0, 1: 1, 2: -1}
NAME = {0: "관망", 1: "롱", 2: "숏"}


def arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


# ---------- 데이터 ----------
def fetch_coinbase(days):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    rows, cur = {}, start
    while cur < end:
        nxt = min(cur + timedelta(minutes=300), end)
        url = ("https://api.exchange.coinbase.com/products/BTC-USD/candles"
               f"?granularity=60&start={cur.isoformat()}&end={nxt.isoformat()}")
        req = urllib.request.Request(url, headers={"User-Agent": "ai-trader-replay"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    for t, lo, hi, op, cl, vol in json.load(r):
                        rows[t] = cl
                break
            except Exception as e:
                print("재시도:", e, flush=True)
                time.sleep(2 + attempt * 2)
        cur = nxt
        time.sleep(0.35)  # 요청 제한 배려
    prices = [rows[t] for t in sorted(rows)]
    print(f"1분봉 {len(prices)}개 받음 ({days}일치)", flush=True)
    return prices


def fake_prices(n):
    p, out = 60000.0, []
    for _ in range(n):
        p *= 1 + random.gauss(0, 0.0008)
        out.append(p)
    return out


# ---------- AI ----------
def bucket(r):
    return "U" if r > FLAT_TH else "D" if r < -FLAT_TH else "F"


def run(prices, q, learn, eps_fn):
    """prices 를 처음부터 끝까지 재생. learn=False 면 시험 모드."""
    pos, equity, trades, prev, fee_pend = 0, 1.0, 0, None, 0.0
    curve = []
    for i in range(LOOKBACK + 1, len(prices)):
        ret = prices[i] / prices[i - 1] - 1
        reward = pos * ret
        equity *= 1 + reward
        s2 = "".join(bucket(prices[k] / prices[k - 1] - 1) for k in range(i - LOOKBACK + 1, i + 1)) + f"|{pos}"
        v2 = q.setdefault(s2, [0.0, 0.0, 0.0])
        if learn and prev:
            s, a = prev
            v = q[s]
            v[a] += ALPHA * ((reward - fee_pend) * 100 + GAMMA * max(v2) - v[a])
        eps = eps_fn() if learn else 0.0
        a2 = random.randrange(3) if random.random() < eps else max(range(3), key=lambda a: v2[a])
        new_pos = ACTIONS[a2]
        fee_pend = 0.0
        if new_pos != pos:
            fee_pend = FEE * abs(new_pos - pos)
            equity *= 1 - fee_pend
            trades += 1
            pos = new_pos
        prev = (s2, a2)
        curve.append(equity)
    return equity, trades, curve


def sample(curve, n=40):
    if len(curve) <= n:
        return curve
    step = len(curve) / n
    return [curve[int(i * step)] for i in range(n)] + [curve[-1]]


def main():
    days = float(arg("--days", "10"))
    epochs = int(arg("--epochs", "5"))
    prices = fake_prices(int(days * 1440)) if "--sim" in sys.argv else fetch_coinbase(days)

    split = int(len(prices) * 0.7)
    train, test = prices[:split], prices[split - LOOKBACK - 1:]
    q = {}

    total = epochs * len(train)
    counter = {"n": 0}

    def eps_fn():
        counter["n"] += 1
        frac = min(1.0, counter["n"] / (total * 0.7))
        return EPS_START + (EPS_END - EPS_START) * frac

    t0 = time.time()
    lines = []
    for ep in range(1, epochs + 1):
        eq, tr, _ = run(train, q, True, eps_fn)
        msg = f"학습 {ep}회차: 수익률 {(eq-1)*100:+.2f}%, 거래 {tr}회"
        print(msg, flush=True)
        lines.append(msg)

    # 학습 데이터 성적 (랜덤 없이) / 처음 보는 데이터 성적
    tr_eq, tr_tr, _ = run(train, q, False, None)
    te_eq, te_tr, curve = run(test, q, False, None)
    bh_train = train[-1] / train[0]
    bh_test = test[-1] / test[LOOKBACK + 1]
    best = {"관망": 0, "롱": 0, "숏": 0}
    for v in q.values():
        best[NAME[max(range(3), key=lambda a: v[a])]] += 1
    secs = time.time() - t0

    train_days, test_days = days * 0.7, days * 0.3
    md = [
        "## 압축 학습 결과",
        "",
        f"- 데이터: BTC 1분봉 {len(prices)}개 ({days:g}일치) → 앞 {train_days:.1f}일 학습, 뒤 {test_days:.1f}일 시험",
        f"- 학습 반복: {epochs}회 (실제 시간 약 {days*epochs:g}일분을 {secs:.1f}초 만에 재생)",
        "",
        "| 구간 | AI 수익률 | 그냥 보유 | 거래 횟수 |",
        "|---|---|---|---|",
        f"| 학습한 구간 (본 적 있음) | {(tr_eq-1)*100:+.2f}% | {(bh_train-1)*100:+.2f}% | {tr_tr} |",
        f"| **시험 구간 (처음 봄)** | **{(te_eq-1)*100:+.2f}%** | {(bh_test-1)*100:+.2f}% | {te_tr} |",
        "",
        f"- 상황별로 선호하게 된 행동: 관망 {best['관망']} / 롱 {best['롱']} / 숏 {best['숏']}",
        "",
        "<details><summary>회차별 학습 기록</summary>",
        "",
        *[f"- {l}" for l in lines],
        "",
        "</details>",
        "",
        "### 시험 구간 AI 자산 변화 (시작 = 100)",
        "```mermaid",
        "xychart-beta",
        '    x-axis "시간 흐름" 0 --> 100',
        '    y-axis "자산"',
        "    line [" + ", ".join(f"{v*100:.2f}" for v in sample(curve)) + "]",
        "```",
    ]
    out = "\n".join(md)
    print(out)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
