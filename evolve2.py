"""
evolve2.py — 진화형 트레이딩 AI v2 (모의투자, 실제 주문 없음)
================================================================
목표: 처음 보는 기간에서 하루 +1% (5개 코인 평균)

v1 에서 배운 것 → v2 에서 바꾼 것
  1. 5분봉은 잡음·수수료 부담이 큼           → 1시간봉으로 판단
  2. 학습 구간을 외우기만 함                 → '처음 보는 달' 성적으로만 챔피언/앙상블 선발
  3. 1등 하나가 롱↔숏으로 치우침             → 상위 10마리 투표(앙상블)로 매매
  4. 수수료를 시장가로 과대 계산               → 지정가 0.02% + 미끄러짐 0.01% = 0.03%
  5. 진화가 느림                            → GitHub 서버 16대에서 '섬' 16개 동시 진화, 라운드마다 우수 개체 교환

데이터 구간 (코인마다)
  학습  : 처음 ~ (검증 시작 전)          → 세대마다 '코인-달' 10개를 랜덤으로 뽑아 시합 (외우기 방지)
  검증  : 마지막 30일 앞의 150일 (약 5달) → 앙상블/챔피언 선발 (학습에 안 씀)
  시험  : 마지막 30일                    → 최종 성적 보고용 (학습·선발 모두 안 씀)

점수 (초반 방식): 수익률 + 수익 보너스(플러스면 2배) − 손실 보유 감점(손실 깊이 x 버틴 일수)
생존: 하루 평균 +0.1% 이상 & 거래한 날 70% 이상 · 포지션 최대 24시간 · BTC 레버리지 1~5배(AI 선택)

실행:
  python evolve2.py island --id 3 --minutes 50   # 섬 하나 진화 (병렬 작업 하나)
  python evolve2.py merge                        # 섬 결과 합치기, 앙상블 선발, 리포트
  python evolve2.py selftest                     # 가짜 데이터로 전체 흐름 시험
"""
import glob, json, os, sys, time, random
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from numba import njit

# ---------------- 설정 ----------------
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
COIN_NAMES = {"BTC": "비트코인", "ETH": "이더리움", "SOL": "솔라나", "XRP": "리플", "DOGE": "도지코인"}
COIN_MAX_LEV = {"BTC": 5}
DATA_DIR = "data"
OUT = "v2"
ISLAND_OUT = "v2_out"
KST = timezone(timedelta(hours=9))

BPD = 24                 # 하루 1시간봉 수
TEST_DAYS = 30
VAL_DAYS = 150
POP = 120                # 섬 하나의 개체 수
ELITE = 12
HIDDEN = 6
FEE = 0.0003             # 지정가 0.02% + 미끄러짐 0.01% (포지션 금액 기준, 진입/청산 각각)
MAINT_MARGIN = 0.005
MAX_HOLD = 24            # 최대 보유 24시간
PROFIT_BONUS = 1.0
PENALTY_W = 1.0
SURVIVE_DAILY = 0.001
SURVIVE_TRADE_DAYS = 0.7
MAX_SURVIVORS = 60
BATCH = 10               # 세대마다 뽑는 학습 '코인-달' 수
MIGRANTS = 10            # 라운드마다 다른 섬에서 받아오는 개체 수
ENSEMBLE = 10
GOAL_DAILY = 0.01

MARKET_FEATURES = ["1시간 변화", "4시간 변화", "12시간 변화", "1일 변화", "3일 변화", "1주 변화",
                   "20시간 평균 대비", "50시간 평균 대비", "SMA200 대비 (1시간봉)", "SMA200 대비 (4시간봉)",
                   "SMA200 대비 (일봉)", "RSI14 (1시간봉)", "RSI14 (4시간봉)", "RSI14 (일봉)",
                   "매수세 (12시간)", "매수세 (3일)", "하루 고저 범위 내 위치", "1주 고저 범위 내 위치",
                   "변동성 수준", "거래량 수준", "시간대 (sin)", "시간대 (cos)", "오늘 시작가 대비"]
STATE_FEATURES = ["오늘 내 계좌 수익률", "지금 포지션 손익"]
FEATURES = MARKET_FEATURES + STATE_FEATURES
FM, F = len(MARKET_FEATURES), len(MARKET_FEATURES) + len(STATE_FEATURES)
GENOME_LEN = F + HIDDEN * F + HIDDEN + HIDDEN + 1 + 1 + 1     # mask, W1, b1, w2, b2, th, lev


# ---------------- 데이터 ----------------
def read_coin_5m(coin):
    files = sorted(glob.glob(f"{DATA_DIR}/{coin.lower()}_5m_*.csv.gz"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in files])
    df.index = pd.to_datetime(df.index, utc=True)
    return df[~df.index.duplicated(keep="last")].sort_index().astype(float)


def fake_5m(days=730, seed=None):
    rng = np.random.default_rng(seed)
    n = days * 288
    regime = np.repeat(rng.normal(0, 0.0002, n // 5760 + 1), 5760)[:n]
    p = 60000 * np.exp(np.cumsum(rng.normal(0, 0.0015, n) + regime))
    wig = np.abs(rng.normal(0, 0.001, n))
    idx = pd.date_range(end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0), periods=n, freq="5min")
    return pd.DataFrame({"close": p, "high": p * (1 + wig), "low": p * (1 - wig),
                         "volume": np.abs(rng.normal(10, 3, n))}, index=idx)


def to_hourly(df5):
    h = df5.resample("1h").agg({"close": "last", "high": "max", "low": "min", "volume": "sum"})
    return h.dropna()


def make_features(df):
    c, v = df["close"], df["volume"]
    lr = np.log(c).diff()
    vol = lr.rolling(BPD * 7, min_periods=BPD * 2).std()
    X = pd.DataFrame(index=df.index)
    for i, k in enumerate([1, 4, 12, 24, 72, 168]):
        X[i] = np.log(c / c.shift(k)) / (vol * np.sqrt(k))
    for i, (w, mp) in zip([6, 7, 8, 9, 10], [(20, 20), (50, 50), (200, 200), (800, 400), (4800, 2400)]):
        X[i] = np.log(c / c.rolling(w, min_periods=mp).mean()) / (vol * np.sqrt(min(w, 168)))
    for i, bars in zip([11, 12, 13], [1, 4, 24]):
        ch = c.diff(bars)
        up = ch.clip(lower=0).rolling(14 * bars).mean()
        dn = (-ch).clip(lower=0).rolling(14 * bars).mean()
        X[i] = (up / (up + dn + 1e-12) - 0.5) * 4
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((2 * c - df["high"] - df["low"]) / rng).fillna(0)
    for i, w in zip([14, 15], [12, 72]):
        X[i] = (clv * v).rolling(w).sum() / (v.rolling(w).sum() + 1e-9) * 3
    for i, w in zip([16, 17], [24, 168]):
        hi, lo = df["high"].rolling(w).max(), df["low"].rolling(w).min()
        X[i] = ((c - lo) / (hi - lo + 1e-9) - 0.5) * 4
    X[18] = np.log(lr.rolling(BPD).std() / (vol + 1e-12))
    X[19] = np.log((v.rolling(BPD).mean() + 1e-9) / (v.rolling(BPD * 7).mean() + 1e-9))
    hour = df.index.hour
    X[20] = np.sin(2 * np.pi * hour / 24)
    X[21] = np.cos(2 * np.pi * hour / 24)
    X[22] = np.log(c / c.groupby(df.index.floor("D")).transform("first")) / (vol * np.sqrt(12))
    X = X.replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    ok = X.notna().all(axis=1).to_numpy()
    day = np.asarray((df.index.normalize() - pd.Timestamp("1970-01-01", tz="UTC")).days, dtype=np.int64)
    return {"X": np.ascontiguousarray(X[ok].to_numpy(np.float64)), "close": df["close"].to_numpy()[ok],
            "high": df["high"].to_numpy()[ok], "low": df["low"].to_numpy()[ok],
            "day": day[ok], "ts": df.index[ok]}


def seg(D, a, b, extra=None):
    S = {k: (v[a:b] if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in D.items()}
    if extra:
        S.update(extra)
    return S


def build_sets(sim=False):
    """코인별 데이터 → 학습 '코인-달' 목록, 검증 '코인-달' 목록, 시험(마지막 30일) 구간"""
    train, val, test, coins = [], [], {}, {}
    for i, coin in enumerate(COINS):
        df5 = fake_5m(seed=i + (int(time.time()) if sim == "random" else 0)) if sim else read_coin_5m(coin)
        if df5 is None or len(df5) < 288 * 200:
            print(f"[{coin}] 데이터 없음/부족 → 제외", flush=True)
            continue
        D = make_features(to_hourly(df5))
        D["max_lev"] = COIN_MAX_LEV.get(coin, 1)
        D["coin"] = coin
        coins[coin] = D
        end = D["ts"][-1]
        t_test = end - pd.Timedelta(days=TEST_DAYS)
        t_val = t_test - pd.Timedelta(days=VAL_DAYS)
        ts = D["ts"]
        a_val, a_test = int(np.searchsorted(ts, t_val)), int(np.searchsorted(ts, t_test))
        test[coin] = seg(D, a_test, len(ts))
        mk = pd.Index(ts).strftime("%Y-%m")
        for lo, hi, bucket in [(0, a_val, train), (a_val, a_test, val)]:
            months = mk[lo:hi]
            for m in pd.unique(months):
                ix = np.nonzero(months == m)[0] + lo
                if len(ix) >= BPD * 10:
                    bucket.append(seg(D, ix[0], ix[-1] + 1, {"month": m}))
    return coins, train, val, test


# ---------------- AI ----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def random_genome():
    g = np.random.normal(0, 1, GENOME_LEN)
    g[GENOME_LEN - 2] = np.random.uniform(0.1, 0.9)    # 문턱
    g[-1] = np.random.normal(-1.5, 1.0)                # 레버리지 성향 (낮게 시작)
    return g


def decode(g):
    i = 0
    mask = (g[i:i + F] > 0).astype(np.float64); i += F
    W1 = g[i:i + HIDDEN * F].reshape(HIDDEN, F); i += HIDDEN * F
    b1 = g[i:i + HIDDEN]; i += HIDDEN
    w2 = g[i:i + HIDDEN]; i += HIDDEN
    b2 = float(g[i]); th = float(np.clip(g[i + 1], 0.0, 0.98))
    return mask, W1, b1, w2, b2, th


def coin_lev(g, max_lev):
    return float(np.round(1 + (max_lev - 1) * sigmoid(g[-1]))) if max_lev > 1 else 1.0


@njit(cache=True)
def simulate(base, ws, w2, b2, th, q, close, high, low, day, fee, pen_w, lev, mm, max_hold, bpd):
    """K마리가 투표(K=1이면 혼자)해서 롱/숏/관망. 반환: logp, pen, eq, st"""
    K, n, H = base.shape
    logp = np.zeros(n); pen = np.zeros(n); eq_arr = np.ones(n)
    # st: 0 진입, 1 수익거래, 2 손실거래, 3 거래한 날, 4 청산, 5 수수료합
    st = np.zeros(6)
    eq = 1.0; d = 0; cum = 0.0; day_start = 1.0; entry = 0.0; entry_t = 0; last_td = -1
    dead = False
    for t in range(n):
        if dead:
            eq_arr[t] = eq
            continue
        if t > 0:
            if day[t] != day[t - 1]:
                day_start = eq
            if d != 0 and lev > 1.0 and ((d == 1 and low[t] <= entry * (1.0 - 1.0 / lev + mm)) or
                                         (d == -1 and high[t] >= entry * (1.0 + 1.0 / lev - mm))):
                logp[t] += np.log(0.01); eq *= 0.01; pen[t] += pen_w * 5.0
                st[4] += 1; st[2] += 1; dead = True; eq_arr[t] = eq
                continue
            if d != 0:
                r = max(d * lev * (close[t] / close[t - 1] - 1.0), -0.99)
                lp = np.log1p(r)
                logp[t] += lp; eq *= 1.0 + r; cum += lp
                if cum < 0:
                    pen[t] += pen_w * (-cum) / bpd
        tp = min(5.0, max(-5.0, (eq / day_start - 1.0) * 100.0))
        pp = min(5.0, max(-5.0, (np.exp(cum) - 1.0) * 100.0)) if d != 0 else 0.0
        votes = 0
        for k in range(K):
            s = b2[k]
            for j in range(H):
                s += np.tanh(base[k, t, j] + ws[k, j, 0] * tp + ws[k, j, 1] * pp) * w2[k, j]
            s = np.tanh(s)
            if s > th[k]:
                votes += 1
            elif s < -th[k]:
                votes -= 1
        want = 1 if votes >= q else (-1 if votes <= -q else 0)
        if d != 0 and t - entry_t >= max_hold:
            want = 0
        if want != d:
            if day[t] != last_td:
                st[3] += 1; last_td = day[t]
            if d != 0:
                if cum + np.log1p(-fee * lev) > 0:
                    st[1] += 1
                else:
                    st[2] += 1
            f = min(0.5, fee * lev * abs(want - d))
            lp = np.log1p(-f)
            logp[t] += lp; eq *= 1.0 - f; st[5] += f
            if want != 0:
                st[0] += 1
                cum = lp if d == 0 else np.log1p(-fee * lev)
                entry = close[t]; entry_t = t
            d = want
        eq_arr[t] = eq
    return logp, pen, eq_arr, st


def pack(genomes, S):
    K = len(genomes)
    base = np.empty((K, len(S["X"]), HIDDEN)); ws = np.empty((K, HIDDEN, 2))
    w2 = np.empty((K, HIDDEN)); b2 = np.empty(K); th = np.empty(K)
    levs = []
    for k, g in enumerate(genomes):
        mask, W1, b1, w2k, b2k, thk = decode(g)
        base[k] = (S["X"] * mask[:FM]) @ W1[:, :FM].T + b1
        ws[k] = W1[:, FM:] * mask[FM:]
        w2[k] = w2k; b2[k] = b2k; th[k] = thk
        levs.append(coin_lev(g, S.get("max_lev", 1)))
    return base, ws, w2, b2, th, float(np.median(levs))


def run(genomes, S):
    base, ws, w2, b2, th, lev = pack(genomes, S)
    K = len(genomes)
    q = 1 if K == 1 else max(1, int(round(K * 0.3)))
    return simulate(base, ws, w2, b2, th, q, S["close"], S["high"], S["low"], S["day"],
                    FEE, PENALTY_W, lev, MAINT_MARGIN, MAX_HOLD, BPD)


def ndays(S):
    return len(np.unique(S["day"]))


def evaluate(genomes, S):
    logp, pen, eq, st = run(genomes, S)
    nd = ndays(S)
    tot = logp.sum()
    score = (tot + PROFIT_BONUS * max(0.0, tot) - pen.sum()) * 100
    _, di = np.unique(S["day"], return_inverse=True)
    last = np.zeros(nd, dtype=int); last[di] = np.arange(len(di))
    d_eq = eq[last]
    d_ret = d_eq / np.concatenate([[1.0], d_eq[:-1]]) - 1
    return {"score": float(score), "ret": float(eq[-1] - 1), "avg_daily": float(np.expm1(tot / nd)),
            "days": nd, "trade_days": int(st[3]), "trades": int(st[0]), "trades_per_day": float(st[0] / nd),
            "wins": int(st[1]), "losses": int(st[2]), "liq": int(st[4]), "hit_days": int((d_ret >= 0.01).sum()),
            "daily": d_ret, "curve": eq}


def fitness(g, batch):
    """(학습 점수, 하루 평균 수익, 거래한 날 비율) — 뽑힌 '코인-달' 10개에서"""
    sc, lsum, days, tdays = [], 0.0, 0, 0
    for S in batch:
        logp, pen, eq, st = run([g], S)
        tot = logp.sum()
        nd = ndays(S)
        sc.append((tot + PROFIT_BONUS * max(0.0, tot) - pen.sum()) * 100 / nd * 30)   # 30일 기준으로 맞춤
        lsum += tot; days += nd; tdays += st[3]
    sc = np.array(sc)
    return float(sc.mean() - 0.5 * sc.std()), float(np.expm1(lsum / days)), float(tdays / days)


def crossover(a, b):
    return np.where(np.random.rand(GENOME_LEN) < 0.5, a, b)


def mutate(g, rate=0.1, sigma=0.3):
    g = g.copy()
    m = np.random.rand(GENOME_LEN) < rate
    g[m] += np.random.normal(0, sigma, m.sum())
    return g


# ---------------- 저장 ----------------
def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


# ---------------- 섬 하나 진화 ----------------
def island(iid, minutes, sim=False):
    t0 = time.time()
    np.random.seed((int(time.time()) + iid * 7919) % 2**31); random.seed(time.time() + iid)
    coins, train, val, test = build_sets(sim)
    print(f"[섬 {iid}] 학습 코인-달 {len(train)}개 · 검증 {len(val)}개 · 준비 {time.time()-t0:.0f}초", flush=True)
    pop = [np.array(g) for g in load_json(f"{OUT}/islands/{iid}.json", []) if len(g) == GENOME_LEN]
    migr = [np.array(g) for g in load_json(f"{OUT}/migrants.json", []) if len(g) == GENOME_LEN]
    random.shuffle(migr)
    pop = pop[:POP - MIGRANTS] + migr[:MIGRANTS]
    if len(pop) < POP:
        pop += [random_genome() for _ in range(POP - len(pop))]
    log, gen = [], 0
    budget = minutes * 60 - (time.time() - t0)
    t1 = time.time()
    while time.time() - t1 < budget:
        batch = random.sample(train, min(BATCH, len(train)))
        res = [fitness(g, batch) for g in pop]
        sc = np.array([r[0] for r in res]); dl = np.array([r[1] for r in res]); td = np.array([r[2] for r in res])
        o = np.argsort(-sc); pop = [pop[i] for i in o]; sc, dl, td = sc[o], dl[o], td[o]
        alive = [i for i in range(len(pop)) if dl[i] >= SURVIVE_DAILY and td[i] >= SURVIVE_TRADE_DAYS][:MAX_SURVIVORS]
        gen += 1
        if gen % 25 == 1:
            log.append({"gen": gen, "best": float(sc[0]), "top": float(sc[:ELITE].mean()),
                        "best_daily": float(dl[0]), "alive": len(alive)})
        if gen % 200 == 1:
            print(f"  [섬 {iid}] {gen}세대: 1등 점수 {sc[0]:+.2f}, 하루 {dl[0]*100:+.3f}%, 생존 {len(alive)}/{POP}", flush=True)
        if len(alive) >= 2:
            parents = [pop[i] for i in alive]
            nxt = list(parents)
        else:
            parents = pop[:5]
            nxt = [pop[i] for i in alive] + [random_genome() for _ in range(POP // 2)]
        while len(nxt) < POP:
            a, b = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
            nxt.append(mutate(crossover(a, b)))
        pop = nxt
    # 마무리: 넓은 학습 샘플로 최종 순위
    big = random.sample(train, min(BATCH * 3, len(train)))
    sc = np.array([fitness(g, big)[0] for g in pop])
    o = np.argsort(-sc); pop = [pop[i] for i in o]
    save_json(f"{ISLAND_OUT}/island_{iid}.json",
              {"id": iid, "gens": gen, "pop": [np.round(g, 4).tolist() for g in pop],
               "best_train": float(sc[o][0]), "log": log, "minutes": (time.time() - t0) / 60})
    print(f"[섬 {iid}] 완료: {gen}세대, 학습 1등 점수 {sc[o][0]:+.2f}", flush=True)


# ---------------- 합치기 + 선발 + 리포트 ----------------
def merge(sim=False):
    t0 = time.time()
    if not sim:
        try:
            import evolve as v1                      # 새 시세 받아서 data/ 갱신 (v1 함수 재사용)
            for coin in COINS:
                v1.load_data(coin)
        except Exception as e:
            print("데이터 갱신 실패(이전 데이터로 진행):", e, flush=True)
    coins, train, val, test = build_sets(sim)
    state = load_json(f"{OUT}/state.json", {"round": 0, "total_gens": 0, "started": datetime.now(KST).strftime("%Y-%m-%d %H:%M")})
    state["round"] += 1
    files = sorted(glob.glob(f"{ISLAND_OUT}/island_*.json"))
    islands, cands, round_gens = [], [], 0
    for f in files:
        r = load_json(f, {})
        pop = [np.array(g) for g in r.get("pop", []) if len(g) == GENOME_LEN]
        if not pop:
            continue
        save_json(f"{OUT}/islands/{r['id']}.json", [g.tolist() for g in pop])
        islands.append({"id": r["id"], "gens": r["gens"], "best_train": r["best_train"], "log": r["log"][-40:]})
        round_gens += r["gens"]
        cands += pop[:6]
    state["total_gens"] += round_gens
    print(f"라운드 {state['round']}: 섬 {len(islands)}개, 이번 라운드 {round_gens}세대", flush=True)
    if not cands:
        cands = [np.array(g) for g in load_json(f"{OUT}/ensemble.json", [])]
    if not cands:
        print("후보 없음 → 종료"); return

    # 다른 섬으로 보낼 이주민
    save_json(f"{OUT}/migrants.json", [g.tolist() for g in cands])

    # 검증(처음 보는 약 5달)에서 각 후보 평가
    def val_stats(gs):
        es = [evaluate(gs, S) for S in val]
        rets = np.array([e["ret"] for e in es])
        days = sum(e["days"] for e in es)
        return {"score": float(np.mean([e["score"] for e in es]) - 0.5 * np.std([e["score"] for e in es])),
                "pos_months": float((rets > 0).mean()), "mean_month_ret": float(rets.mean()),
                "avg_daily": float(np.expm1(np.sum([np.log1p(max(r, -0.99)) for r in rets]) / max(days, 1))),
                "trade_day_ratio": float(sum(e["trade_days"] for e in es) / max(days, 1))}
    prev = [np.array(g) for g in load_json(f"{OUT}/ensemble.json", []) if len(g) == GENOME_LEN]
    pool = cands + prev
    scored = []
    for g in pool:
        vs = val_stats([g])
        if vs["trade_day_ratio"] >= 0.5:
            scored.append((vs["score"], g, vs))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        scored = [(val_stats([g])["score"], g, val_stats([g])) for g in pool[:ENSEMBLE]]
    ens = [g for _, g, _ in scored[:ENSEMBLE]]
    best = scored[0][1]
    save_json(f"{OUT}/ensemble.json", [g.tolist() for g in ens])

    # 성적표: 앙상블 / 1등 단독, 검증·시험
    ens_val = val_stats(ens); best_val = scored[0][2]
    val_months = []
    for S in val:
        e = evaluate(ens, S)
        val_months.append({"coin": S["coin"], "month": S["month"], "ret": e["ret"], "avg_daily": e["avg_daily"],
                           "bh": float(S["close"][-1] / S["close"][0] - 1), "trades_per_day": e["trades_per_day"]})
    test_rows, curves = {}, {}
    for coin, S in test.items():
        e = evaluate(ens, S); b = evaluate([best], S)
        test_rows[coin] = {"ens_avg_daily": e["avg_daily"], "ens_ret": e["ret"], "ens_hit": e["hit_days"],
                           "ens_tpd": e["trades_per_day"], "ens_trade_days": e["trade_days"], "days": e["days"],
                           "best_avg_daily": b["avg_daily"], "best_ret": b["ret"], "liq": e["liq"],
                           "bh": float(S["close"][-1] / S["close"][0] - 1)}
        ix = np.linspace(0, len(e["curve"]) - 1, min(200, len(e["curve"]))).astype(int)
        curves[coin] = {"t": [S["ts"][i].tz_convert(KST).strftime("%m-%d %H시") for i in ix],
                        "ai": [float(e["curve"][i]) for i in ix],
                        "bh": [float(S["close"][i] / S["close"][0]) for i in ix]}
    t_avg = float(np.mean([r["ens_avg_daily"] for r in test_rows.values()]))
    t_best = float(np.mean([r["best_avg_daily"] for r in test_rows.values()]))
    goal = {"test_avg_daily": t_avg, "val_pos_months": ens_val["pos_months"],
            "met": bool(t_avg >= GOAL_DAILY and ens_val["pos_months"] >= 0.7)}

    # 앙상블이 보는 정보 (10마리 평균 중요도)
    imp = np.zeros(F)
    for g in ens:
        mask, W1, *_ = decode(g)
        x = np.abs(W1).sum(axis=0) * mask
        imp += x / (x.max() + 1e-9)
    imp /= len(ens)
    feats = [{"name": FEATURES[i], "weight": float(imp[i]), "state": bool(i >= FM)} for i in np.argsort(-imp)]
    lev_btc = float(np.median([coin_lev(g, 5) for g in ens]))
    # 지금 판단 (비트코인 마지막 시점, 내 상태 0 가정)
    now_pos = None
    if "BTC" in coins:
        Dn = seg(coins["BTC"], len(coins["BTC"]["X"]) - 1, len(coins["BTC"]["X"]))
        base, ws, w2, b2, th, _ = pack(ens, Dn)
        v = 0
        for k in range(len(ens)):
            s = np.tanh(np.tanh(base[k, 0]) @ w2[k] + b2[k])
            v += 1 if s > th[k] else (-1 if s < -th[k] else 0)
        q = max(1, int(round(len(ens) * 0.3)))
        now_pos = 1 if v >= q else (-1 if v <= -q else 0)

    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    hist = load_json(f"{OUT}/history.json", [])
    hist.append({"round": state["round"], "time": now, "total_gens": state["total_gens"], "round_gens": round_gens,
                 "islands": len(islands), "test_avg_daily": t_avg, "test_best_avg_daily": t_best,
                 "val_pos_months": ens_val["pos_months"], "val_avg_daily": ens_val["avg_daily"],
                 "val_mean_month_ret": ens_val["mean_month_ret"], "goal_met": goal["met"], "lev_btc": lev_btc})
    save_json(f"{OUT}/history.json", hist[-500:])
    save_json(f"{OUT}/state.json", state)
    latest = {"updated": now, "round": state["round"], "started": state["started"], "total_gens": state["total_gens"],
              "round_gens": round_gens, "islands": islands, "settings": {
                  "fee": FEE, "timeframe": "1시간봉", "pop_per_island": POP, "ensemble": ENSEMBLE,
                  "test_days": TEST_DAYS, "val_days": VAL_DAYS, "goal_daily": GOAL_DAILY,
                  "train_sets": len(train), "val_sets": len(val)},
              "goal": goal, "ensemble_val": ens_val, "best_val": best_val, "val_months": val_months,
              "test": test_rows, "test_curves": curves, "features": feats, "lev_btc": lev_btc,
              "now_position": now_pos, "coin_names": {k: COIN_NAMES[k] for k in test}}
    save_json(f"{OUT}/latest.json", latest)

    def pct(x, d=2): return f"{x*100:+.{d}f}%"
    rows = "\n".join(f"| {COIN_NAMES[k]} | **{pct(r['ens_avg_daily'], 3)}** | {pct(r['ens_ret'])} | {pct(r['bh'])} | "
                     f"{r['ens_hit']}/{r['days']} | {r['ens_tpd']:.1f} | {pct(r['best_avg_daily'], 3)} |" for k, r in test_rows.items())
    vrows = "\n".join(f"| {m['coin']} | {m['month']} | {pct(m['ret'])} | {pct(m['bh'])} |" for m in val_months)
    hrows = "\n".join(f"| {h['round']} | {h['time']} | {h['total_gens']:,} | {pct(h['test_avg_daily'], 3)} | "
                      f"{h['val_pos_months']*100:.0f}% | {'✅' if h['goal_met'] else '-'} |" for h in hist[-30:][::-1])
    md = f"""# 🧬 진화형 트레이딩 AI v2 리포트

_마지막: {now} KST · 라운드 {state['round']} · 누적 {state['total_gens']:,}세대 (이번 라운드 {round_gens:,}세대, 섬 {len(islands)}개 병렬)_

> 모의투자 실험 (실제 주문 없음) · **목표: 처음 보는 기간에서 하루 +1% (5코인 평균)**
> 1시간봉 · 섬 16개 병렬 진화 · 상위 {ENSEMBLE}마리 투표(앙상블) · 수수료 {FEE*100:.2f}% · BTC 레버리지 1~5배
> 학습: 오래된 기간의 '코인-달' 중 세대마다 랜덤 10개 · 선발: 처음 보는 약 5달(검증) · 최종 시험: 마지막 30일
> 📊 대시보드: https://ksuin39.github.io/24h_bot/dashboard2/

## 목표 달성 여부: {'✅ 달성' if goal['met'] else '❌ 아직'}
- 시험(마지막 30일) 5코인 평균 하루 **{pct(t_avg, 3)}** (목표 +1%) · 1등 단독 {pct(t_best, 3)}
- 검증 약 5달: 플러스인 코인-달 **{ens_val['pos_months']*100:.0f}%** (목표 70%) · 하루 평균 {pct(ens_val['avg_daily'], 3)}
- 비트코인 레버리지(앙상블 중앙값) {lev_btc:.0f}배 · 지금 판단(BTC): {'롱' if now_pos == 1 else '숏' if now_pos == -1 else '관망'}

## 시험 (마지막 30일, 학습·선발에 안 씀)
| 코인 | 앙상블 하루 평균 | 30일 합계 | 그냥 보유 | +1% 달성일 | 하루 거래 | 1등 단독 하루 |
|---|---|---|---|---|---|---|
{rows}

## 검증 달별 (앙상블)
| 코인 | 달 | AI | 그냥 보유 |
|---|---|---|---|
{vrows}

## 라운드 기록
| 라운드 | 시각 | 누적 세대 | 시험 하루 평균 | 검증 플러스 달 | 목표 |
|---|---|---|---|---|---|
{hrows}
"""
    with open(f"{OUT}/README.md", "w", encoding="utf-8") as f:
        f.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)
    print(md[:2500])
    print(f"합치기 {time.time()-t0:.0f}초", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    if mode == "island":
        island(int(arg("--id", "0")), float(arg("--minutes", "50")), sim="--sim" in sys.argv)
    elif mode == "merge":
        merge(sim="--sim" in sys.argv)
    else:   # selftest: 가짜 데이터로 섬 2개 + 합치기
        for i in range(2):
            island(i, float(arg("--minutes", "2")), sim=True)
        merge(sim=True)
