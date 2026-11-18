"""
evolve.py — 매일 스스로 진화하는 '데이트레이딩' AI (모의투자, 실제 주문 없음)
=========================================================================
목표: 하루에 계좌의 1% 수익을 꾸준히. 레버리지 최대 20배.

AI 200마리가 각자 유전자로
  - 어떤 정보를 볼지 (14가지 중 선택)
  - 정보를 어떻게 조합해 판단할지 (작은 신경망)
  - 얼마나 확신할 때 진입할지 (문턱값)
  - 레버리지 몇 배로 할지 (1~20배)
  - 계좌의 몇 %를 투입할지 (10~100%)
를 갖고 태어나서, 과거 데이터 시합 → 상위만 생존 → 교배/돌연변이를 반복한다.

점수 규칙 (Sin_jin 설계)
  포지션 진입                         +0.1  (x 거래 크기)
  보유 중 수익 상태                    +0.01 x 수익률(계좌 대비 %, 레버리지 반영) x 보유 일수
  보유 중 손실 상태                    -0.02 x 손실률(계좌 대비 %, 레버리지 반영) x 보유 일수
  하루를 넘겨 보유 (날 넘길 때마다)    -0.5    (하루 경계 = 오전 9시 KST, OKX 일봉)
  하루 안에 포지션 종료               +0.5  (x 거래 크기)
  익절 종료 (수수료 빼고 플러스)       +0.5  (x 거래 크기)
  손절 종료                          -0.8  (x 거래 크기)
  청산 당함                          -10    (투입한 증거금 전부 잃음)
  수수료                             -10 x 낸 수수료(계좌 대비 %)
  * 거래 크기 = min(1, 투입비율 x 레버리지): 계좌 10%만 1배로 걸면 0.1배만 인정
    (작게 걸고 수십 번 거래해 고정 보너스만 챙기는 꼼수 방지)
  * 보유 중 점수는 5분마다 1/288씩 = 하루 보유 시 위 값 그대로
    (5분마다 통째로 주면 20배로 추세를 며칠 들고 있는 게 압도적 이득이 되는 꼼수 방지)
→ 기간 전체 점수를 '하루 평균 점수'로 환산해서 비교

실행:  python evolve.py            (실제 데이터)
       python evolve.py --sim      (가짜 데이터로 테스트)
"""
import json, os, sys, time, random, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from numba import njit

# ---------------- 설정 ----------------
DAYS         = 60      # 몇 일치 데이터를 쓸지
POP          = 200     # 개체 수
ELITE        = 20      # 매 세대 그대로 살아남는 수
GENERATIONS  = 300     # 하루 실행당 진화 세대 수
IMMIGRANTS   = 0.15    # 매일 새로 섞어 넣는 랜덤 개체 비율
HIDDEN       = 4       # 신경망 은닉 뉴런 수
FEE          = 0.0005  # 거래 수수료 (포지션 금액 기준 0.05%, 진입/청산 각각)
MAINT_MARGIN = 0.005   # 유지 증거금률 (청산 가격 계산용)
MAX_LEV      = 20
DAILY_TARGET = 0.01    # 하루 목표 수익 1% (대시보드 표시용)

# 점수표
P_ENTRY      = 0.1
P_HOLD_WIN   = 0.01    # x 수익률% x 보유 일수 (5분마다 1/288씩)
P_HOLD_LOSS  = 0.02    # x 손실률% x 보유 일수
P_OVERNIGHT  = -0.5
P_SAMEDAY    = 0.5
P_TAKEPROFIT = 0.5
P_STOPLOSS   = -0.8
P_LIQUIDATED = -10.0
P_FEE_W      = 10.0

OUT          = "evolution"
KST          = timezone(timedelta(hours=9))
BARS_PER_DAY = 288

FEATURES = ["5분 변화", "15분 변화", "1시간 변화", "4시간 변화", "1일 변화",
            "1시간 평균 대비 위치", "4시간 평균 대비 위치", "1일 평균 대비 위치",
            "거래량 (1시간 대비)", "거래량 (1일 대비)", "변동성 수준",
            "시간대 (sin)", "시간대 (cos)", "오늘 시작가 대비 위치"]
F = len(FEATURES)
GENOME_LEN = F + HIDDEN * F + HIDDEN + HIDDEN + 1 + 1 + 2   # mask, W1, b1, w2, b2, th, lev, frac

# 점수 항목 (시뮬레이터 내부 순서)
CATS = ["진입", "보유 중 수익", "보유 중 손실", "하루 넘김", "당일 종료", "익절", "손절", "청산", "수수료"]


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
                        rows[t] = (cl, hi, lo, vol)
                break
            except Exception as e:
                print("재시도:", e, flush=True)
                time.sleep(2 + 2 * att)
        cur = nxt
        time.sleep(0.35)
    ts = sorted(rows)
    df = pd.DataFrame({k: [rows[t][i] for t in ts] for i, k in enumerate(["close", "high", "low", "volume"])},
                      index=pd.to_datetime(ts, unit="s", utc=True))
    print(f"5분봉 {len(df)}개 수집", flush=True)
    return df


def fake(days):
    n = int(days * BARS_PER_DAY)
    p = 60000 * np.exp(np.cumsum(np.random.normal(0, 0.0015, n)))
    wig = np.abs(np.random.normal(0, 0.001, n))
    idx = pd.date_range(end=datetime.now(timezone.utc).replace(second=0, microsecond=0),
                        periods=n, freq="5min")
    return pd.DataFrame({"close": p, "high": p * (1 + wig), "low": p * (1 - wig),
                         "volume": np.abs(np.random.normal(10, 3, n))}, index=idx)


def make_features(df):
    c, v = df["close"], df["volume"]
    lr = np.log(c).diff()
    vol = lr.rolling(BARS_PER_DAY).std()
    X = pd.DataFrame(index=df.index)
    for i, k in enumerate([1, 3, 12, 48, 288]):
        X[i] = np.log(c / c.shift(k)) / (vol * np.sqrt(k))
    for i, w in zip([5, 6, 7], [12, 48, 288]):
        X[i] = np.log(c / c.rolling(w).mean()) / (vol * np.sqrt(w))
    X[8] = np.log((v + 1e-9) / (v.rolling(12).mean() + 1e-9))
    X[9] = np.log((v + 1e-9) / (v.rolling(288).mean() + 1e-9))
    X[10] = np.log(vol / lr.rolling(2016).std())
    hour = df.index.hour + df.index.minute / 60
    X[11] = np.sin(2 * np.pi * hour / 24)
    X[12] = np.cos(2 * np.pi * hour / 24)
    day_open = c.groupby(df.index.floor("D")).transform("first")
    X[13] = np.log(c / day_open) / vol
    X = X.clip(-5, 5)
    ok = X.notna().all(axis=1).to_numpy()
    day = np.asarray((df.index.normalize() - pd.Timestamp("1970-01-01", tz="UTC")).days, dtype=np.int64)
    return {"X": X[ok].to_numpy(np.float32), "close": df["close"].to_numpy()[ok],
            "high": df["high"].to_numpy()[ok], "low": df["low"].to_numpy()[ok],
            "day": day[ok], "ts": df.index[ok]}


def subset(D, m):
    return {k: v[m] for k, v in D.items()}


# ---------------- AI 유전자 ----------------
TH_IDX = F + HIDDEN * F + 2 * HIDDEN + 1


def random_genome():
    g = np.random.normal(0, 1, GENOME_LEN)
    g[TH_IDX] = np.random.uniform(0.1, 0.9)   # 문턱값
    g[-2] = np.random.normal(-1.5, 1.0)       # 레버리지: 처음엔 대체로 낮게 시작
    return g


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode(g):
    i = 0
    mask = (g[i:i + F] > 0).astype(np.float32); i += F
    W1 = g[i:i + HIDDEN * F].reshape(HIDDEN, F); i += HIDDEN * F
    b1 = g[i:i + HIDDEN]; i += HIDDEN
    w2 = g[i:i + HIDDEN]; i += HIDDEN
    b2 = g[i]; th = float(np.clip(g[i + 1], 0.0, 0.98)); i += 2
    lev = float(np.clip(np.round(1 + (MAX_LEV - 1) * sigmoid(g[i])), 1, MAX_LEV))
    frac = float(0.1 + 0.9 * sigmoid(g[i + 1]))
    return mask, W1, b1, w2, b2, th, lev, frac


def signals(g, X):
    mask, W1, b1, w2, b2, th, lev, frac = decode(g)
    h = np.tanh((X * mask) @ W1.T + b1)
    s = np.tanh(h @ w2 + b2)
    return np.where(s > th, 1, np.where(s < -th, -1, 0)).astype(np.int8), lev, frac


# ---------------- 시뮬레이터 (점수 규칙 적용) ----------------
@njit(cache=True)
def simulate(sig, close, high, low, day, lev, frac, fee, mm,
             p_entry, p_hw, p_hl, p_on, p_sd, p_tp, p_sl, p_liq, p_fee):
    n = len(sig)
    pts = np.zeros(n)              # 봉별 점수
    cat = np.zeros(9)              # 항목별 점수 합계
    eq_arr = np.zeros(n)           # 봉별 계좌 평가금액 (시작 1.0)
    stats = np.zeros(8)            # 진입, 익절, 손절, 청산, 날넘김, 당일종료, 총수수료, 파산여부
    eq = 1.0
    size = min(1.0, frac * lev)    # 거래 크기 비율: 진입/종료/익절/손절 점수에 곱함
    d = 0                          # 현재 포지션 방향
    entry = 0.0; margin = 0.0; notional = 0.0; entry_day = 0; entry_fee = 0.0; liq = 0.0
    for t in range(n):
        if eq <= 0.01:             # 파산
            stats[7] = 1.0
            eq_arr[t] = eq
            continue
        # 1) 보유 중이면 이번 5분 동안 청산 가격 터치 확인
        if d != 0:
            hit = (d == 1 and low[t] <= liq) or (d == -1 and high[t] >= liq)
            if hit:
                eq -= margin
                pts[t] += p_liq; cat[7] += p_liq; stats[3] += 1
                d = 0
            else:
                # 2) 날 넘김 감점
                if t > 0 and day[t] != day[t - 1]:
                    pts[t] += p_on; cat[3] += p_on; stats[4] += 1
                # 3) 보유 중 수익/손실 점수 (계좌 대비 %)
                upnl = d * (close[t] / entry - 1.0) * notional
                acct_pct = upnl / eq * 100.0
                if acct_pct > 0:
                    v = p_hw * acct_pct / 288.0; pts[t] += v; cat[1] += v
                else:
                    v = -p_hl * (-acct_pct) / 288.0; pts[t] += v; cat[2] += v
        # 4) 판단
        want = sig[t]
        if want != d:
            if d != 0:             # 기존 포지션 종료
                pnl = d * (close[t] / entry - 1.0) * notional
                xfee = notional * (close[t] / entry) * fee
                before = eq
                eq += pnl - xfee
                stats[6] += xfee
                fp = -p_fee * xfee / before * 100.0
                pts[t] += fp; cat[8] += fp
                if day[t] == entry_day:
                    v = p_sd * size; pts[t] += v; cat[4] += v; stats[5] += 1
                if pnl - xfee - entry_fee > 0:
                    v = p_tp * size; pts[t] += v; cat[5] += v; stats[1] += 1
                else:
                    v = p_sl * size; pts[t] += v; cat[6] += v; stats[2] += 1
                d = 0
            if want != 0 and eq > 0.01:   # 새 포지션 진입
                d = want
                entry = close[t]
                margin = eq * frac
                notional = margin * lev
                entry_fee = notional * fee
                fp = -p_fee * entry_fee / eq * 100.0
                eq -= entry_fee
                stats[6] += entry_fee
                entry_day = day[t]
                if d == 1:
                    liq = entry * (1.0 - 1.0 / lev + mm)
                else:
                    liq = entry * (1.0 + 1.0 / lev - mm)
                pts[t] += p_entry * size + fp; cat[0] += p_entry * size; cat[8] += fp; stats[0] += 1
        # 5) 평가금액 기록
        if d != 0:
            eq_arr[t] = eq + d * (close[t] / entry - 1.0) * notional
        else:
            eq_arr[t] = eq
    return pts, cat, eq_arr, stats


def run(g, D):
    sig, lev, frac = signals(g, D["X"])
    return simulate(sig, D["close"], D["high"], D["low"], D["day"], lev, frac, FEE, MAINT_MARGIN,
                    P_ENTRY, P_HOLD_WIN, P_HOLD_LOSS, P_OVERNIGHT, P_SAMEDAY,
                    P_TAKEPROFIT, P_STOPLOSS, P_LIQUIDATED, P_FEE_W)


def evaluate(g, D):
    pts, cat, eq_arr, stats = run(g, D)
    _, d_idx = np.unique(D["day"], return_inverse=True)
    nd = int(d_idx.max()) + 1
    d_pts = np.bincount(d_idx, weights=pts, minlength=nd)
    last = np.zeros(nd, dtype=int)
    last[d_idx] = np.arange(len(d_idx))          # 각 날의 마지막 봉
    d_eq = eq_arr[last]
    prev = np.concatenate([[1.0], d_eq[:-1]])
    d_ret = d_eq / np.maximum(prev, 1e-9) - 1
    entries = int(stats[0])
    _, lev, frac = signals(g, D["X"][:1])
    return {
        "score": float(d_pts.mean()), "d_pts": d_pts, "daily": d_ret,
        "cats": {CATS[i]: float(cat[i] / nd) for i in range(9)},          # 항목별 하루 평균 점수
        "ret": float(eq_arr[-1] - 1), "avg_daily": float(np.expm1(np.log(max(eq_arr[-1], 1e-9)) / nd)),
        "days": nd, "hit_days": int((d_ret >= DAILY_TARGET).sum()), "loss_days": int((d_ret < 0).sum()),
        "trades": entries, "trades_per_day": entries / nd,
        "wins": int(stats[1]), "losses": int(stats[2]), "liquidations": int(stats[3]),
        "overnights": int(stats[4]), "sameday": int(stats[5]),
        "fee_pct": float(stats[6] * 100), "bankrupt": bool(stats[7]),
        "lev": lev, "frac": frac, "curve": eq_arr,
    }


def fitness(g, D):
    """학습 점수: 기간을 4등분해 각 구간 '하루 평균 점수'가 꾸준히 높아야 함"""
    pts, cat, eq_arr, stats = run(g, D)
    _, d_idx = np.unique(D["day"], return_inverse=True)
    d_pts = np.bincount(d_idx, weights=pts)
    parts = np.array([c.mean() for c in np.array_split(d_pts, 4)])
    return float(parts.mean() - 0.5 * parts.std())


# ---------------- 진화 ----------------
def crossover(a, b):
    m = np.random.rand(GENOME_LEN) < 0.5
    return np.where(m, a, b)


def mutate(g, rate=0.1, sigma=0.3):
    g = g.copy()
    m = np.random.rand(GENOME_LEN) < rate
    g[m] += np.random.normal(0, sigma, m.sum())
    return g


def evolve(pop, D, gens, log):
    for gen in range(gens):
        scores = np.array([fitness(g, D) for g in pop])
        order = np.argsort(-scores)
        pop = [pop[i] for i in order]
        scores = scores[order]
        log.append({"gen": gen + 1, "best": float(scores[0]), "mean": float(np.mean(np.maximum(scores, -50))),
                    "top20": float(scores[:ELITE].mean())})
        if gen % 30 == 0 or gen == gens - 1:
            _, lev, frac = signals(pop[0], D["X"][:1])
            print(f"  세대 {gen+1:3d}/{gens}: 최고 {scores[0]:+.3f}점/일 (레버리지 {lev:.0f}배, 투입 {frac*100:.0f}%), "
                  f"상위20 평균 {scores[:ELITE].mean():+.3f}", flush=True)
        nxt = pop[:ELITE]
        while len(nxt) < POP:
            i, j = random.sample(range(POP // 2), 2)
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


# ---------------- 메인 ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(int(time.time()) % 2**31)
    df = fake(DAYS) if "--sim" in sys.argv else fetch(DAYS)
    D = make_features(df)
    day = D["day"]
    udays = np.unique(day)
    nd = len(udays)
    d_va, d_te = udays[int(nd * 0.70)], udays[int(nd * 0.85)]
    TR, VA, TE = subset(D, day < d_va), subset(D, (day >= d_va) & (day < d_te)), subset(D, day >= d_te)
    n_days = lambda S: len(np.unique(S["day"]))
    print(f"학습 {n_days(TR)}일 / 검증 {n_days(VA)}일 / 시험 {n_days(TE)}일", flush=True)

    state = load_json(f"{OUT}/state.json", {"day": 0, "total_gens": 0, "champion": None,
                                             "champion_id": None, "champion_born": None})
    pop = [np.array(g) for g in load_json(f"{OUT}/population.json", []) if len(g) == GENOME_LEN]
    if state["champion"] is not None and len(state["champion"]) != GENOME_LEN:
        state["champion"] = None
    n_imm = int(POP * IMMIGRANTS) if pop else POP
    pop = pop[:POP - n_imm] + [random_genome() for _ in range(n_imm)]
    state["day"] += 1
    dnum = state["day"]
    print(f"\n=== {dnum}일차 진화 시작 (이어받은 개체 {POP - n_imm}, 새 개체 {n_imm}) ===", flush=True)

    t0 = time.time()
    gen_log = []
    pop = evolve(pop, TR, GENERATIONS, gen_log)
    state["total_gens"] += GENERATIONS

    top = pop[:10]
    val_scores = [evaluate(g, VA)["score"] for g in top]
    ch = top[int(np.argmax(val_scores))]
    ch_te = evaluate(ch, TE)
    champ = np.array(state["champion"]) if state["champion"] else None
    cp_te = evaluate(champ, TE) if champ is not None else None
    bh = float(TE["close"][-1] / TE["close"][0] - 1)
    replaced = cp_te is None or ch_te["score"] > cp_te["score"]
    if replaced:
        state["champion"] = ch.tolist()
        state["champion_id"] = f"G{state['total_gens']}-D{dnum}"
        state["champion_born"] = datetime.now(KST).strftime("%Y-%m-%d")
        cp_te = ch_te
    champ = np.array(state["champion"])
    secs = time.time() - t0

    mask, W1, b1, w2, b2, th, lev, frac = decode(champ)
    imp = np.abs(W1).sum(axis=0) * mask
    imp = imp / imp.max() if imp.max() > 0 else imp
    feats = [FEATURES[i] for i in np.argsort(-imp) if mask[i] > 0][:5]

    # ---- 기록 ----
    hist = load_json(f"{OUT}/history.json", [])
    hist.append({"day": dnum, "date": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                 "total_gens": state["total_gens"], "champion_id": state["champion_id"],
                 "champion_score": cp_te["score"], "champion_avg_daily": cp_te["avg_daily"],
                 "champion_test": cp_te["ret"], "champion_trades": cp_te["trades"],
                 "champion_hit_days": cp_te["hit_days"], "champion_loss_days": cp_te["loss_days"],
                 "champion_liquidations": cp_te["liquidations"], "test_days": cp_te["days"],
                 "lev": lev, "frac": frac, "challenger_score": ch_te["score"],
                 "replaced": replaced, "buyhold_test": bh, "features": feats})
    save_json(f"{OUT}/history.json", hist)
    save_json(f"{OUT}/population.json", [np.round(g, 4).tolist() for g in pop])
    save_json(f"{OUT}/state.json", state)

    # ---- 대시보드 데이터 ----
    sig_now, _, _ = signals(champ, D["X"][-1:])
    curve = cp_te["curve"]
    idx = np.linspace(0, len(curve) - 1, min(240, len(curve))).astype(int)
    bh_curve = TE["close"] / TE["close"][0]
    labels = [datetime.fromtimestamp(int(x) * 86400 + 9 * 3600, KST).strftime("%m-%d") for x in np.unique(TE["day"])]
    final_scores = sorted([fitness(g, TR) for g in pop], reverse=True)
    keys = ["score", "ret", "avg_daily", "days", "hit_days", "loss_days", "trades", "trades_per_day", "wins",
            "losses", "liquidations", "overnights", "sameday", "fee_pct", "bankrupt", "lev", "frac", "cats"]
    latest = {
        "mode": "daytrading-v2", "updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "day": dnum,
        "total_gens": state["total_gens"], "generations_today": GENERATIONS, "pop": POP,
        "seconds": round(secs, 1),
        "split_days": {"train": n_days(TR), "val": n_days(VA), "test": n_days(TE)},
        "settings": {"fee": FEE, "daily_target": DAILY_TARGET, "max_lev": MAX_LEV,
                     "points": {"진입": f"+{P_ENTRY} × 크기", "보유 중 수익": f"+{P_HOLD_WIN} × 수익률% × 보유일수",
                                "보유 중 손실": f"−{P_HOLD_LOSS} × 손실률% × 보유일수", "하루 넘김": f"{P_OVERNIGHT}",
                                "당일 종료": f"+{P_SAMEDAY} × 크기", "익절": f"+{P_TAKEPROFIT} × 크기", "손절": f"{P_STOPLOSS} × 크기",
                                "청산": f"{P_LIQUIDATED}", "수수료": f"−{P_FEE_W:g} × 수수료%"}},
        "replaced": bool(replaced),
        "champion": {"id": state["champion_id"], "born": state["champion_born"],
                     **{k: cp_te[k] for k in keys}, "threshold": th, "now_position": float(sig_now[0]),
                     "features": [{"name": FEATURES[i], "used": bool(mask[i] > 0), "weight": float(imp[i])}
                                  for i in range(F)]},
        "challenger": {k: ch_te[k] for k in keys},
        "buyhold": bh, "last_price": float(D["close"][-1]),
        "test_daily": {"labels": labels, "ai": [float(x) for x in cp_te["daily"]],
                       "pts": [float(x) for x in cp_te["d_pts"]]},
        "test_curve": {"t": [TE["ts"][i].tz_convert(KST).strftime("%m-%d %H:%M") for i in idx],
                       "ai": [float(curve[i]) for i in idx], "buyhold": [float(bh_curve[i]) for i in idx]},
        "gen_log": gen_log,
        "population_scores": [round(x, 4) for x in final_scores],
    }
    save_json(f"{OUT}/latest.json", latest)

    # ---- README ----
    def pct(x, d=2): return f"{x*100:+.{d}f}%"
    c = cp_te
    cat_rows = "\n".join(f"| {k} | {v:+.3f} |" for k, v in c["cats"].items())
    hist_rows = "\n".join(
        f"| {h['day']} | {h['date'][:10]} | {h['total_gens']} | {h['champion_score']:+.3f} | "
        f"{pct(h['champion_avg_daily'], 3)} | {h['champion_hit_days']}/{h['test_days']} | "
        f"{h.get('lev', 1):.0f}배 / {h.get('frac', 1)*100:.0f}% | {h.get('champion_liquidations', 0)} | {'교체' if h['replaced'] else '유지'} |"
        for h in hist[-30:][::-1])
    md = f"""# 🧬 진화형 데이트레이딩 AI 리포트

_마지막 실행: {datetime.now(KST):%Y-%m-%d %H:%M} KST · {dnum}일차 · 누적 {state['total_gens']}세대 · 계산 {secs:.0f}초_

> 모의투자 실험입니다. 실제 주문은 하지 않습니다. **목표: 하루 +1% · 레버리지 최대 {MAX_LEV}배**
> 모든 성적은 **AI가 학습에 쓰지 않은 가장 최근 {c['days']}일(시험 구간)** 기준입니다.
> 📊 실시간 대시보드: https://ksuin39.github.io/24h_bot/dashboard/

## 오늘의 챔피언 `{state['champion_id']}` ({'첫 등장' if dnum == 1 else '새로 등극' if replaced else '방어 성공'})

- **하루 평균 점수: {c['score']:+.3f}점**
- 하루 평균 수익 {pct(c['avg_daily'], 3)} · 시험 {c['days']}일 합계 {pct(c['ret'])} (그냥 보유 {pct(bh)})
- 목표(+1%) 달성 {c['hit_days']}/{c['days']}일 · 손실일 {c['loss_days']}일
- 레버리지 **{lev:.0f}배**, 계좌의 **{frac*100:.0f}%** 투입 · 하루 거래 {c['trades_per_day']:.1f}회
- 익절 {c['wins']} · 손절 {c['losses']} · 청산 {c['liquidations']} · 날 넘김 {c['overnights']} · 낸 수수료 계좌의 {c['fee_pct']:.2f}%
- 주로 보는 정보: {', '.join(feats) if feats else '없음'}

### 점수 항목별 (하루 평균)
| 항목 | 점수 |
|---|---|
{cat_rows}

## 일차별 기록
| 일차 | 날짜 | 누적 세대 | 하루 점수 | 하루 평균 수익 | 목표 달성일 | 레버리지/투입 | 청산 | 결과 |
|---|---|---|---|---|---|---|---|---|
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
