"""
evolve.py — 매일 스스로 진화하는 트레이딩 AI (모의투자, 실제 주문 없음)
=====================================================================
사람이 정한 매매 규칙 없음. AI 200마리가 각자
  - 어떤 정보를 볼지 (13가지 중 선택 = 유전자)
  - 그 정보를 어떻게 조합해 판단할지 (작은 신경망 가중치 = 유전자)
  - 얼마나 확신할 때 진입할지 (문턱값 = 유전자)
를 갖고 태어나서, 과거 데이터 시합 → 상위만 생존 → 교배/돌연변이 를 반복한다.

매일 실행될 때마다:
  1) 최근 60일 BTC 5분봉 수집 (Coinbase 공개 API)
  2) 어제까지 살아남은 개체군을 이어받아 N세대 더 진화 (학습 구간)
  3) 상위 10마리를 '검증 구간'에서 시험 → 오늘의 도전자 선발
  4) 도전자 vs 기존 챔피언을 '시험 구간(가장 최근, 학습에 안 쓴 데이터)'에서 대결
  5) 이기면 챔피언 교체 → 기록/그래프를 evolution/ 폴더에 저장

실행:  python evolve.py            (실제 데이터)
       python evolve.py --sim      (가짜 데이터로 테스트)
"""
import json, os, sys, time, random, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# ---------------- 설정 ----------------
DAYS        = 60      # 몇 일치 데이터를 쓸지
POP         = 200     # 개체 수
ELITE       = 20      # 매 세대 그대로 살아남는 수
GENERATIONS = 300     # 하루 실행당 진화 세대 수
IMMIGRANTS  = 0.15    # 매일 새로 섞어 넣는 랜덤 개체 비율 (다양성 유지)
HIDDEN      = 4       # 신경망 은닉 뉴런 수
FEE         = 0.0005  # 포지션 변경 수수료 0.05%
PENALTY_W   = 1.0     # 손실 보유 감점 세기: '손실 깊이 x 손실로 버틴 일수' 만큼 점수 차감
OUT         = "evolution"
KST         = timezone(timedelta(hours=9))

FEATURES = ["5분 변화", "15분 변화", "1시간 변화", "4시간 변화", "1일 변화",
            "1시간 평균 대비 위치", "4시간 평균 대비 위치", "1일 평균 대비 위치",
            "거래량 (1시간 대비)", "거래량 (1일 대비)", "변동성 수준",
            "시간대 (sin)", "시간대 (cos)"]
F = len(FEATURES)
GENOME_LEN = F + HIDDEN * F + HIDDEN + HIDDEN + 1 + 1   # mask, W1, b1, w2, b2, th


# ---------------- 데이터 ----------------
def fetch(days):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    cur = end - timedelta(days=days)
    rows = {}
    while cur < end:
        nxt = min(cur + timedelta(minutes=5 * 300), end)
        url = ("https://api.exchange.coinbase.com/products/BTC-USD/candles"
               f"?granularity=300&start={cur.isoformat()}&end={nxt.isoformat()}")
        req = urllib.request.Request(url, headers={"User-Agent": "evolve-ai"})
        for att in range(5):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    for t, lo, hi, op, cl, vol in json.load(r):
                        rows[t] = (cl, vol)
                break
            except Exception as e:
                print("재시도:", e, flush=True)
                time.sleep(2 + 2 * att)
        cur = nxt
        time.sleep(0.35)
    ts = sorted(rows)
    df = pd.DataFrame({"close": [rows[t][0] for t in ts], "volume": [rows[t][1] for t in ts]},
                      index=pd.to_datetime(ts, unit="s", utc=True))
    print(f"5분봉 {len(df)}개 수집", flush=True)
    return df


def fake(days):
    n = int(days * 288)
    p = 60000 * np.exp(np.cumsum(np.random.normal(0, 0.0015, n)))
    v = np.abs(np.random.normal(10, 3, n))
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="5min")
    return pd.DataFrame({"close": p, "volume": v}, index=idx)


def make_features(df):
    c, v = df["close"], df["volume"]
    lr = np.log(c).diff()
    vol = lr.rolling(288).std()                     # 1일 변동성
    X = pd.DataFrame(index=df.index)
    for i, k in enumerate([1, 3, 12, 48, 288]):
        X[i] = np.log(c / c.shift(k)) / (vol * np.sqrt(k))
    for i, w in zip([5, 6, 7], [12, 48, 288]):
        X[i] = np.log(c / c.rolling(w).mean()) / (vol * np.sqrt(w))
    X[8] = np.log((v + 1e-9) / (v.rolling(12).mean() + 1e-9))
    X[9] = np.log((v + 1e-9) / (v.rolling(288).mean() + 1e-9))
    X[10] = np.log(vol / lr.rolling(2016).std())    # 1일 vs 1주 변동성
    hour = df.index.hour + df.index.minute / 60
    X[11] = np.sin(2 * np.pi * hour / 24)
    X[12] = np.cos(2 * np.pi * hour / 24)
    X = X.clip(-5, 5)
    ret = c.shift(-1) / c - 1                       # 다음 5분 수익률
    ok = X.notna().all(axis=1) & ret.notna()
    return X[ok].to_numpy(np.float32), ret[ok].to_numpy(np.float64), c[ok].to_numpy()


# ---------------- AI (유전자 → 행동) ----------------
def random_genome():
    g = np.random.normal(0, 1, GENOME_LEN)
    g[-1] = np.random.uniform(0.1, 0.9)             # 문턱값
    return g


def decode(g):
    i = 0
    mask = (g[i:i + F] > 0).astype(np.float32); i += F
    W1 = g[i:i + HIDDEN * F].reshape(HIDDEN, F); i += HIDDEN * F
    b1 = g[i:i + HIDDEN]; i += HIDDEN
    w2 = g[i:i + HIDDEN]; i += HIDDEN
    b2 = g[i]; th = float(np.clip(g[i + 1], 0.0, 0.98))
    return mask, W1, b1, w2, b2, th


def positions(g, X):
    mask, W1, b1, w2, b2, th = decode(g)
    h = np.tanh((X * mask) @ W1.T + b1)
    s = np.tanh(h @ w2 + b2)
    return np.where(s > th, 1.0, np.where(s < -th, -1.0, 0.0))


def evaluate(g, X, ret, curve=False):
    pos = positions(g, X)
    change = np.abs(np.diff(np.concatenate([[0.0], pos])))
    pnl = pos * ret - FEE * change
    logp = np.log1p(pnl)

    # --- 손실 보유 감점 ---
    # 포지션을 잡은 뒤 '진입 시점 대비 손익'이 마이너스인 동안, 매 5분마다
    # (손실 깊이 / 하루 봉 수) 만큼 감점 → 2% 손실로 하루 버티면 2점(%p) 감점
    c = np.cumsum(logp)
    c_prev = np.concatenate([[0.0], c[:-1]])
    start = np.maximum.accumulate(np.where(change > 0, np.arange(len(pos)), 0))
    unreal = c - c_prev[start]                       # 진입 후 누적 손익
    underwater = np.where(pos != 0, np.maximum(0.0, -unreal), 0.0)
    pen = PENALTY_W * underwater / 288

    out = {"ret": float(np.expm1(logp.sum())), "trades": int((change > 0).sum()),
           "logp": logp, "pen": pen,
           "penalty": float(pen.sum()),
           "score": float(logp.sum() - pen.sum()),   # 최종 점수 = 수익(로그) - 감점
           "max_underwater": float(underwater.max()) if len(underwater) else 0.0}
    if curve:
        out["curve"] = np.exp(c)
    return out


def fitness(g, X, ret):
    """학습 점수: (수익 - 손실보유 감점)을 4구간으로 나눠 '꾸준히' 잘해야 높음"""
    e = evaluate(g, X, ret)
    net = e["logp"] - e["pen"]
    chunks = np.array([c.sum() for c in np.array_split(net, 4)])
    return float(chunks.mean() - 0.5 * chunks.std())


# ---------------- 진화 ----------------
def crossover(a, b):
    m = np.random.rand(GENOME_LEN) < 0.5
    return np.where(m, a, b)


def mutate(g, rate=0.1, sigma=0.3):
    g = g.copy()
    m = np.random.rand(GENOME_LEN) < rate
    g[m] += np.random.normal(0, sigma, m.sum())
    return g


def evolve(pop, X, ret, gens):
    for gen in range(gens):
        scores = np.array([fitness(g, X, ret) for g in pop])
        order = np.argsort(-scores)
        pop = [pop[i] for i in order]
        scores = scores[order]
        if gen % 30 == 0 or gen == gens - 1:
            print(f"  세대 {gen+1:3d}/{gens}: 최고 점수 {scores[0]:+.4f}, 평균 {scores.mean():+.4f}", flush=True)
        nxt = pop[:ELITE]
        while len(nxt) < POP:
            i, j = random.sample(range(POP // 2), 2)          # 상위 절반에서 부모 선택
            nxt.append(mutate(crossover(pop[i], pop[j])))
        pop = nxt
    return pop


# ---------------- 저장/불러오기 ----------------
def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def describe(g):
    mask, W1, *_ = decode(g)
    imp = np.abs(W1).sum(axis=0) * mask
    used = [(FEATURES[i], imp[i]) for i in np.argsort(-imp) if mask[i] > 0]
    return [n for n, _ in used[:5]], int(mask.sum())


# ---------------- 메인 ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(int(time.time()) % 2**31)
    df = fake(DAYS) if "--sim" in sys.argv else fetch(DAYS)
    X, ret, close = make_features(df)
    n = len(X)
    a, b = int(n * 0.70), int(n * 0.85)
    tr, va, te = slice(0, a), slice(a, b), slice(b, n)
    days_of = lambda s: (s.stop - s.start) / 288
    print(f"학습 {days_of(tr):.1f}일 / 검증 {days_of(va):.1f}일 / 시험 {days_of(te):.1f}일", flush=True)

    state = load_json(f"{OUT}/state.json", {"day": 0, "total_gens": 0, "champion": None,
                                             "champion_id": None, "champion_born": None})
    pop_saved = load_json(f"{OUT}/population.json", [])
    pop = [np.array(g) for g in pop_saved if len(g) == GENOME_LEN]
    n_imm = int(POP * IMMIGRANTS) if pop else POP
    pop = pop[:POP - n_imm] + [random_genome() for _ in range(n_imm)]
    state["day"] += 1
    day = state["day"]
    print(f"\n=== {day}일차 진화 시작 (이어받은 개체 {POP - n_imm}, 새 개체 {n_imm}) ===", flush=True)

    t0 = time.time()
    pop = evolve(pop, X[tr], ret[tr], GENERATIONS)
    state["total_gens"] += GENERATIONS

    # 도전자 선발: 학습 상위 10마리 중 검증 구간 최고
    top = pop[:10]
    val_scores = [evaluate(g, X[va], ret[va])["score"] for g in top]
    ch = top[int(np.argmax(val_scores))]

    # 시험 구간 대결
    ch_te = evaluate(ch, X[te], ret[te], curve=True)
    champ = np.array(state["champion"]) if state["champion"] else None
    cp_te = evaluate(champ, X[te], ret[te], curve=True) if champ is not None else None
    bh = float(close[te][-1] / close[te][0] - 1)

    replaced = cp_te is None or ch_te["score"] > cp_te["score"]
    if replaced:
        state["champion"] = ch.tolist()
        state["champion_id"] = f"G{state['total_gens']}-D{day}"
        state["champion_born"] = datetime.now(KST).strftime("%Y-%m-%d")
        cp_te = ch_te
    champ = np.array(state["champion"])
    feats, nfeat = describe(champ)
    secs = time.time() - t0

    # 기록
    hist = load_json(f"{OUT}/history.json", [])
    hist.append({"day": day, "date": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                 "total_gens": state["total_gens"], "champion_id": state["champion_id"],
                 "champion_test": cp_te["ret"], "champion_trades": cp_te["trades"],
                 "champion_score": cp_te["score"], "champion_penalty": cp_te["penalty"],
                 "challenger_test": ch_te["ret"], "challenger_val": max(val_scores),
                 "replaced": replaced, "buyhold_test": bh, "features": feats})
    save_json(f"{OUT}/history.json", hist)
    save_json(f"{OUT}/population.json", [np.round(g, 4).tolist() for g in pop])
    save_json(f"{OUT}/state.json", state)

    # 리포트
    def pct(x): return f"{x*100:+.2f}%"
    curve = cp_te["curve"]
    pts = [curve[int(i * len(curve) / 40)] * 100 for i in range(40)] + [curve[-1] * 100]
    hist_rows = "\n".join(
        f"| {h['day']} | {h['date'][:10]} | {h['total_gens']} | "
        f"{(format(h['champion_score']*100, '+.2f') if 'champion_score' in h else '-')} | {pct(h['champion_test'])} | "
        f"{pct(h['buyhold_test'])} | {h['champion_trades']} | {'교체' if h['replaced'] else '유지'} |"
        for h in hist[-30:][::-1])
    trend = ", ".join(f"{h['champion_test']*100:.2f}" for h in hist[-30:])
    md = f"""# 🧬 진화형 트레이딩 AI 리포트

_마지막 실행: {datetime.now(KST):%Y-%m-%d %H:%M} KST · {day}일차 · 누적 {state['total_gens']}세대 · 계산 {secs:.0f}초_

> 모의투자 실험입니다. 실제 주문은 하지 않습니다.
> 점수는 모두 **AI가 학습에 쓰지 않은 가장 최근 {days_of(te):.1f}일(시험 구간)** 기준입니다.

## 오늘의 결과

| | **점수** | 수익률 | 손실 보유 감점 | 최대 손실 상태 | 거래 |
|---|---|---|---|---|---|
| 👑 챔피언 `{state['champion_id']}` | **{cp_te['score']*100:+.2f}** | {pct(cp_te['ret'])} | -{cp_te['penalty']*100:.2f} | -{cp_te['max_underwater']*100:.2f}% | {cp_te['trades']} |
| 🥊 오늘의 도전자 | {ch_te['score']*100:+.2f} | {pct(ch_te['ret'])} | -{ch_te['penalty']*100:.2f} | -{ch_te['max_underwater']*100:.2f}% | {ch_te['trades']} |
| 그냥 보유 | - | {pct(bh)} | - | - | 1 |
| 아무것도 안 함 | +0.00 | +0.00% | 0 | 0% | 0 |

> **점수 = 수익률(로그) − 손실 보유 감점.** 포지션을 잡고 손실 상태로 버틴 시간만큼 깎입니다
> (예: 2% 손실 상태로 하루 버티면 −2점). 챔피언 교체도 이 점수로 결정합니다.

- 결과: **{'도전자가 이겨서 챔피언 교체!' if replaced and day > 1 else '첫 챔피언 등장' if day == 1 else '챔피언 방어 성공'}**
- 챔피언이 보는 정보 {nfeat}개 중 가장 중요하게 쓰는 것: {', '.join(feats) if feats else '없음 (아무 정보도 안 봄)'}

## 챔피언 시험 구간 자산 변화 (시작 = 100)
```mermaid
xychart-beta
    x-axis "시간 흐름" 0 --> 100
    y-axis "자산"
    line [{', '.join(f'{p:.2f}' for p in pts)}]
```

## 날짜별 챔피언 성적 (최근 30일, %)
```mermaid
xychart-beta
    x-axis "일차" 1 --> {max(2, len(hist[-30:]))}
    y-axis "시험 구간 수익률 %"
    line [{trend}]
```

| 일차 | 날짜 | 누적 세대 | 챔피언 점수 | 챔피언 수익률 | 그냥 보유 | 거래 | 결과 |
|---|---|---|---|---|---|---|---|
{hist_rows}
"""
    with open(f"{OUT}/README.md", "w", encoding="utf-8") as f:
        f.write(md)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    main()
