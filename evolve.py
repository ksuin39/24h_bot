"""
evolve.py — 매일 스스로 진화하는 '데이트레이딩' AI (모의투자, 실제 주문 없음)
=========================================================================
목표: 수익은 크게, 손실 상태로 오래 버티지 않기. (점수 = 초반 방식)

[학습 방식]
  - 비트코인 5분봉 2년치를 data/ 에 저장 (매일 새 데이터만 추가)
  - 세대마다 2년 중 '랜덤한 7일 구간 6개'를 뽑아 200마리가 같은 구간에서 시합
    → 매 세대 시험 범위가 바뀌어 외우기 불가, 엘리트도 매번 새 구간에서 재평가
  - 학습 점수 = 6개 구간 하루평균 점수의 평균 - 0.5 x 들쭉날쭉함 (고르게 잘해야 높음)
  - 가장 최근 17일은 학습에 안 씀: 앞 8일 = 검증(도전자 선발), 뒤 9일 = 시험(챔피언 대결)

[AI 유전자]
  - 볼 정보 선택 (시장 정보 14 + 내 상태 2 = 16가지)
      내 상태: 오늘 내 계좌 수익률, 지금 포지션 손익
  - 판단 신경망 가중치, 진입 문턱   (레버리지 없음 = 1배, 계좌 100%)

[점수] (초반 방식으로 복원)
  점수 = 수익률(로그, %) + 수익 보너스 - 손실 보유 감점
    수익 보너스     : 그 구간 순수익이 플러스면 같은 만큼 한 번 더 (수익 2배 인정)
    손실 보유 감점  : 포지션을 잡고 '진입 대비 손실 상태'로 버티는 동안
                     손실 깊이 x 버틴 시간(일) 만큼 감점 (2% 손실로 하루 버티면 -2점)
  - 수수료 0.05% (포지션 바꿀 때) 는 수익률에 그대로 반영
  - 날 넘김 허용, 강제 청산 없음
  - 구간 점수 = 7일 구간 전체 점수, 학습 점수 = 6개 구간 평균 - 0.5 x 들쭉날쭉함

실행:  python evolve.py          (실제 데이터)
       python evolve.py --sim    (가짜 데이터로 테스트)
"""
import glob, json, os, sys, time, random, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from numba import njit

# ---------------- 설정 ----------------
HISTORY_DAYS = 730     # 저장할 데이터 기간 (2년)
WIN_DAYS     = 7       # 랜덤 구간 길이 (일)
N_WINDOWS    = 6       # 세대마다 뽑는 구간 수
VAL_DAYS     = 8
TEST_DAYS    = 9
POP          = 200
ELITE        = 20
GENERATIONS  = 300
IMMIGRANTS   = 0.15
HIDDEN       = 4
FEE          = 0.0005  # 포지션 금액 기준 0.05% (진입/청산 각각)
MAINT_MARGIN = 0.005
MAX_LEV      = 20
DATA_DIR     = "data"
OUT          = "evolution"
KST          = timezone(timedelta(hours=9))
BPD          = 288     # 하루 5분봉 수

# 점수 (초반 방식)
PROFIT_BONUS = 1.0     # 순수익이 플러스면 그 수익을 (1+PROFIT_BONUS)배로 인정
PENALTY_W    = 1.0     # 손실 보유 감점: 손실 깊이 x 버틴 일수
CATS = ["수익", "수익 보너스", "손실 보유 감점"]
RULES = {"수익": "구간 수익률 (로그, %, 수수료 반영)",
         "수익 보너스": "순수익이 플러스면 같은 만큼 한 번 더 (2배)",
         "손실 보유 감점": "손실 깊이 × 버틴 일수 (2% 손실로 하루 = -2점)"}

MARKET_FEATURES = ["5분 변화", "15분 변화", "1시간 변화", "4시간 변화", "1일 변화",
                   "1시간 평균 대비 위치", "4시간 평균 대비 위치", "1일 평균 대비 위치",
                   "거래량 (1시간 대비)", "거래량 (1일 대비)", "변동성 수준",
                   "시간대 (sin)", "시간대 (cos)", "오늘 시작가 대비 위치"]
STATE_FEATURES = ["오늘 내 계좌 수익률", "지금 포지션 손익"]
FEATURES = MARKET_FEATURES + STATE_FEATURES
FM, FS = len(MARKET_FEATURES), len(STATE_FEATURES)
F = FM + FS
GENOME_LEN = F + HIDDEN * F + HIDDEN + HIDDEN + 1 + 1 + 2   # mask, W1, b1, w2, b2, th, lev, frac
TH_IDX = F + HIDDEN * F + 2 * HIDDEN + 1


# ---------------- 데이터 (월별 파일로 저장, 매일 새 것만 추가) ----------------
def fetch_range(start, end):
    rows = {}
    cur = start
    n = 0
    while cur < end:
        nxt = min(cur + timedelta(minutes=5 * 300), end)
        url = ("https://api.exchange.coinbase.com/products/BTC-USD/candles"
               f"?granularity=300&start={cur.isoformat()}&end={nxt.isoformat()}")
        req = urllib.request.Request(url, headers={"User-Agent": "evolve-ai"})
        for att in range(6):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    for t, lo, hi, op, cl, vol in json.load(r):
                        rows[t] = (cl, hi, lo, vol)
                break
            except Exception as e:
                print("재시도:", e, flush=True)
                time.sleep(2 + 2 * att)
        cur = nxt
        n += 1
        if n % 100 == 0:
            print(f"  데이터 받는 중... {cur:%Y-%m-%d}", flush=True)
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame(columns=["close", "high", "low", "volume"])
    ts = sorted(rows)
    return pd.DataFrame({k: [rows[t][i] for t in ts] for i, k in enumerate(["close", "high", "low", "volume"])},
                        index=pd.to_datetime(ts, unit="s", utc=True))


def load_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{DATA_DIR}/btc_5m_*.csv.gz"))
    old = (pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in files]) if files
           else pd.DataFrame(columns=["close", "high", "low", "volume"]))
    if len(old):
        old.index = pd.to_datetime(old.index, utc=True)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = (old.index.max().to_pydatetime() + timedelta(minutes=5)) if len(old) else now - timedelta(days=HISTORY_DAYS)
    print(f"새 데이터 받기: {start:%Y-%m-%d %H:%M} ~ 지금", flush=True)
    new = fetch_range(start, now)
    df = pd.concat([old, new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index >= pd.Timestamp(now - timedelta(days=HISTORY_DAYS + 31))]
    # 새 데이터가 들어간 달의 파일만 다시 저장
    touched = set(new.index.strftime("%Y-%m")) if len(new) else set()
    for m in touched:
        part = df[df.index.strftime("%Y-%m") == m]
        part.to_csv(f"{DATA_DIR}/btc_5m_{m}.csv.gz", compression="gzip", float_format="%.6g")
    keep = set(df.index.strftime("%Y-%m"))
    for f in files:
        if os.path.basename(f)[7:14] not in keep:
            os.remove(f)
    df = df[df.index >= pd.Timestamp(now - timedelta(days=HISTORY_DAYS))]
    print(f"전체 데이터: 5분봉 {len(df):,}개 ({df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d})", flush=True)
    return df.astype(float)


def fake(days):
    n = int(days * BPD)
    regime = np.repeat(np.random.normal(0, 0.0002, n // (BPD * 20) + 1), BPD * 20)[:n]
    p = 60000 * np.exp(np.cumsum(np.random.normal(0, 0.0015, n) + regime))
    wig = np.abs(np.random.normal(0, 0.001, n))
    idx = pd.date_range(end=datetime.now(timezone.utc).replace(second=0, microsecond=0), periods=n, freq="5min")
    return pd.DataFrame({"close": p, "high": p * (1 + wig), "low": p * (1 - wig),
                         "volume": np.abs(np.random.normal(10, 3, n))}, index=idx)


def make_features(df):
    c, v = df["close"], df["volume"]
    lr = np.log(c).diff()
    vol = lr.rolling(BPD).std()
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
    X[13] = np.log(c / c.groupby(df.index.floor("D")).transform("first")) / vol
    X = X.replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    ok = X.notna().all(axis=1).to_numpy()
    day = np.asarray((df.index.normalize() - pd.Timestamp("1970-01-01", tz="UTC")).days, dtype=np.int64)
    return {"X": np.ascontiguousarray(X[ok].to_numpy(np.float64)), "close": df["close"].to_numpy()[ok],
            "high": df["high"].to_numpy()[ok], "low": df["low"].to_numpy()[ok],
            "day": day[ok], "ts": df.index[ok]}


def seg(D, a, b):
    return {k: v[a:b] for k, v in D.items()}


# ---------------- AI 유전자 ----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def random_genome():
    g = np.random.normal(0, 1, GENOME_LEN)
    g[TH_IDX] = np.random.uniform(0.1, 0.9)
    g[-2] = np.random.normal(-1.5, 1.0)   # 레버리지는 낮게 시작
    return g


def decode(g):
    i = 0
    mask = (g[i:i + F] > 0).astype(np.float64); i += F
    W1 = g[i:i + HIDDEN * F].reshape(HIDDEN, F); i += HIDDEN * F
    b1 = g[i:i + HIDDEN]; i += HIDDEN
    w2 = g[i:i + HIDDEN]; i += HIDDEN
    b2 = float(g[i]); th = float(np.clip(g[i + 1], 0.0, 0.98)); i += 2
    lev = float(np.clip(np.round(1 + (MAX_LEV - 1) * sigmoid(g[i])), 1, MAX_LEV))
    frac = float(0.1 + 0.9 * sigmoid(g[i + 1]))
    return mask, W1, b1, w2, b2, th, lev, frac


# ---------------- 시뮬레이터 (초반 방식) ----------------
@njit(cache=True)
def simulate(base, ws, w2, b2, th, close, day, fee, pen_w):
    """레버리지 1배, 계좌 100%. 5분마다 롱/숏/관망.
    반환: logp(봉별 로그수익), pen(봉별 손실보유 감점), eq(봉별 계좌), st(통계)"""
    n = close.shape[0]
    H = base.shape[1]
    logp = np.zeros(n)
    pen = np.zeros(n)
    eq_arr = np.ones(n)
    # st: 0 진입, 1 수익 거래, 2 손실 거래, 3 날 넘김, 4 최대 손실 상태, 5 수수료합
    st = np.zeros(6)
    eq = 1.0
    d = 0
    cum = 0.0            # 진입 후 누적 로그손익
    day_start = 1.0
    h = np.zeros(H)
    for t in range(n):
        if t > 0:
            if day[t] != day[t - 1]:
                day_start = eq
                if d != 0:
                    st[3] += 1
            # 지난 5분 포지션 결과
            if d != 0:
                r = d * (close[t] / close[t - 1] - 1.0)
                lp = np.log1p(r)
                logp[t] += lp
                eq *= 1.0 + r
                cum += lp
                if cum < 0:
                    pen[t] += pen_w * (-cum) / 288.0
                    if -cum > st[4]:
                        st[4] = -cum
        # 내 상태
        today_pct = (eq / day_start - 1.0) * 100.0
        pos_pct = (np.exp(cum) - 1.0) * 100.0 if d != 0 else 0.0
        tp = min(5.0, max(-5.0, today_pct))
        pp = min(5.0, max(-5.0, pos_pct))
        s = b2
        for k in range(H):
            h[k] = np.tanh(base[t, k] + ws[k, 0] * tp + ws[k, 1] * pp)
            s += h[k] * w2[k]
        s = np.tanh(s)
        want = 1 if s > th else (-1 if s < -th else 0)
        if want != d:
            if d != 0:                      # 종료
                if cum + np.log1p(-fee) > 0:     # 청산 수수료까지 빼고 플러스면 수익 거래
                    st[1] += 1
                else:
                    st[2] += 1
            f = fee * abs(want - d)
            lp = np.log1p(-f)
            logp[t] += lp
            eq *= 1.0 - f
            st[5] += f
            if want != 0:
                st[0] += 1
                cum = lp if d == 0 else np.log1p(-fee)   # 새 포지션의 진입 수수료부터 시작
            d = want
        eq_arr[t] = eq
    return logp, pen, eq_arr, st


def run(g, S):
    mask, W1, b1, w2, b2, th, lev, frac = decode(g)
    base = (S["X"] * mask[:FM]) @ W1[:, :FM].T + b1
    ws = np.ascontiguousarray(W1[:, FM:] * mask[FM:])
    return simulate(np.ascontiguousarray(base), ws, np.ascontiguousarray(w2), b2, th,
                    S["close"], S["day"], FEE, PENALTY_W)


def n_days(S):
    return len(np.unique(S["day"]))


def score_of(logp_sum, pen_sum):
    """점수(%p) = 수익 + 수익 보너스 - 손실 보유 감점"""
    bonus = PROFIT_BONUS * max(0.0, logp_sum)
    return (logp_sum + bonus - pen_sum) * 100.0, bonus * 100.0


def window_score(g, S):
    logp, pen, *_ = run(g, S)
    return score_of(logp.sum(), pen.sum())[0]


def evaluate(g, S):
    logp, pen, eq, st = run(g, S)
    score, bonus = score_of(logp.sum(), pen.sum())
    _, d_idx = np.unique(S["day"], return_inverse=True)
    nd = int(d_idx.max()) + 1
    last = np.zeros(nd, dtype=int); last[d_idx] = np.arange(len(d_idx))
    d_eq = eq[last]
    d_ret = d_eq / np.concatenate([[1.0], d_eq[:-1]]) - 1
    d_pts = (np.bincount(d_idx, weights=logp, minlength=nd) - np.bincount(d_idx, weights=pen, minlength=nd)) * 100
    th = decode(g)[5]
    streak = best = 0
    for r in d_ret:
        streak = streak + 1 if r >= 0.01 else 0
        best = max(best, streak)
    return {
        "score": float(score), "d_pts": d_pts, "daily": d_ret,
        "cats": {"수익": float(logp.sum() * 100), "수익 보너스": float(bonus), "손실 보유 감점": float(-pen.sum() * 100)},
        "ret": float(eq[-1] - 1), "avg_daily": float(np.expm1(np.log(max(eq[-1], 1e-9)) / nd)),
        "days": nd, "hit_days": int((d_ret >= 0.01).sum()), "loss_days": int((d_ret < 0).sum()), "best_streak": int(best),
        "trades": int(st[0]), "trades_per_day": float(st[0] / nd), "wins": int(st[1]),
        "small_losses": int(st[2]), "big_losses": 0, "liquidations": 0,
        "overnights": int(st[3]), "dd_days": 0, "fee_pct": float(st[5] * 100), "max_underwater": float(-np.expm1(-st[4]) * 100),
        "bankrupt": False, "lev": 1.0, "frac": 1.0, "th": th, "curve": eq,
    }


# ---------------- 진화 (세대마다 랜덤 구간) ----------------
def crossover(a, b):
    return np.where(np.random.rand(GENOME_LEN) < 0.5, a, b)


def mutate(g, rate=0.1, sigma=0.3):
    g = g.copy()
    m = np.random.rand(GENOME_LEN) < rate
    g[m] += np.random.normal(0, sigma, m.sum())
    return g


def fitness_on(g, windows):
    s = np.array([window_score(g, W) for W in windows])
    return float(s.mean() - 0.5 * s.std())


def evolve(pop, D, day_starts, n_train_days, gens, log):
    for gen in range(gens):
        picks = np.random.randint(0, n_train_days - WIN_DAYS, N_WINDOWS)
        windows = [seg(D, day_starts[p], day_starts[p + WIN_DAYS]) for p in picks]
        scores = np.array([fitness_on(g, windows) for g in pop])
        order = np.argsort(-scores)
        pop = [pop[i] for i in order]
        scores = scores[order]
        lbl = [datetime.fromtimestamp(int(D["day"][day_starts[p]]) * 86400, KST).strftime("%y-%m-%d") for p in sorted(picks)]
        log.append({"gen": gen + 1, "best": float(scores[0]), "top20": float(scores[:ELITE].mean()),
                    "mean": float(np.mean(np.maximum(scores, -30))), "windows": lbl})
        if gen % 30 == 0 or gen == gens - 1:
            *_, lev, frac = decode(pop[0])
            print(f"  세대 {gen+1:3d}/{gens}: 1등 {scores[0]:+.2f}점, "
                  f"상위20 {scores[:ELITE].mean():+.2f} | 이번 구간 {', '.join(lbl)}", flush=True)
        nxt = pop[:ELITE]
        while len(nxt) < POP:
            i, j = random.sample(range(POP // 2), 2)
            nxt.append(mutate(crossover(pop[i], pop[j])))
        pop = nxt
    # 마지막으로 넓은 구간에서 한 번 더 순위 매기기
    picks = np.random.randint(0, n_train_days - WIN_DAYS, N_WINDOWS * 2)
    windows = [seg(D, day_starts[p], day_starts[p + WIN_DAYS]) for p in picks]
    scores = np.array([fitness_on(g, windows) for g in pop])
    order = np.argsort(-scores)
    return [pop[i] for i in order], scores[order]


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
    np.random.seed(int(time.time()) % 2**31); random.seed(int(time.time()))
    df = fake(HISTORY_DAYS) if "--sim" in sys.argv else load_data()
    D = make_features(df)
    day = D["day"]
    starts = np.concatenate([[0], np.nonzero(np.diff(day))[0] + 1, [len(day)]])   # 날마다 시작 위치
    nd = len(starts) - 1
    tr_days = nd - VAL_DAYS - TEST_DAYS
    VA = seg(D, starts[tr_days], starts[tr_days + VAL_DAYS])
    TE = seg(D, starts[tr_days + VAL_DAYS], starts[nd])
    print(f"학습용 {tr_days}일 (랜덤 {WIN_DAYS}일 x {N_WINDOWS}구간씩) / 검증 {VAL_DAYS}일 / 시험 {TEST_DAYS}일", flush=True)

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
    pop, final_scores = evolve(pop, D, starts, tr_days, GENERATIONS, gen_log)
    state["total_gens"] += GENERATIONS

    val_scores = [evaluate(g, VA)["score"] for g in pop[:10]]
    ch = pop[int(np.argmax(val_scores))]
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

    # ---- 월별 성적표 (2년 전체, 달마다 새 계좌로) ----
    months = []
    mk = pd.Index(D["ts"]).strftime("%Y-%m")
    for m in sorted(set(mk)):
        idx = np.nonzero(mk == m)[0]
        if len(idx) < BPD * 5:
            continue
        S = seg(D, idx[0], idx[-1] + 1)
        e = evaluate(champ, S)
        months.append({"month": m, "score": e["score"], "ret": e["ret"], "hit_days": e["hit_days"],
                       "days": e["days"], "liq": e["liquidations"], "bankrupt": e["bankrupt"],
                       "btc": float(S["close"][-1] / S["close"][0] - 1)})
    secs = time.time() - t0

    mask, W1, b1, w2, b2, th, lev, frac = decode(champ)
    lev, frac = 1.0, 1.0     # 초반 방식: 레버리지 없음, 계좌 100%
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
                 "months_positive": sum(1 for m in months if m["score"] > 0), "months_total": len(months),
                 "replaced": replaced, "buyhold_test": bh, "features": feats})
    save_json(f"{OUT}/history.json", hist)
    save_json(f"{OUT}/population.json", [np.round(g, 4).tolist() for g in pop])
    save_json(f"{OUT}/state.json", state)

    # ---- 대시보드 데이터 ----
    st_now = {"today": 0.0, "pos": 0.0}
    base_now = (D["X"][-1:] * mask[:FM]) @ W1[:, :FM].T + b1
    s_now = np.tanh(np.tanh(base_now[0]) @ w2 + b2)
    now_pos = 1.0 if s_now > th else (-1.0 if s_now < -th else 0.0)
    curve = cp_te["curve"]
    idx = np.linspace(0, len(curve) - 1, min(240, len(curve))).astype(int)
    bh_curve = TE["close"] / TE["close"][0]
    labels = [datetime.fromtimestamp(int(x) * 86400 + 9 * 3600, KST).strftime("%m-%d") for x in np.unique(TE["day"])]
    keys = ["score", "ret", "avg_daily", "days", "hit_days", "loss_days", "best_streak", "trades", "trades_per_day",
            "wins", "small_losses", "big_losses", "liquidations", "overnights", "dd_days", "fee_pct", "bankrupt",
            "lev", "frac", "cats"]
    latest = {
        "mode": "daytrading-v3", "updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "day": dnum,
        "total_gens": state["total_gens"], "generations_today": GENERATIONS, "pop": POP, "seconds": round(secs, 1),
        "data": {"from": str(D["ts"][0].date()), "to": str(D["ts"][-1].date()), "train_days": tr_days,
                 "win_days": WIN_DAYS, "n_windows": N_WINDOWS, "val_days": VAL_DAYS, "test_days": TEST_DAYS},
        "settings": {"fee": FEE, "daily_target": 0.01, "max_lev": 1, "rules": RULES, "mode": "classic",
                     "profit_bonus": PROFIT_BONUS, "penalty_w": PENALTY_W},
        "replaced": bool(replaced),
        "champion": {"id": state["champion_id"], "born": state["champion_born"],
                     **{k: cp_te[k] for k in keys}, "threshold": th, "now_position": now_pos,
                     "features": [{"name": FEATURES[i], "used": bool(mask[i] > 0), "weight": float(imp[i]),
                                   "state": i >= FM} for i in range(F)]},
        "challenger": {k: ch_te[k] for k in keys},
        "buyhold": bh, "last_price": float(D["close"][-1]),
        "test_daily": {"labels": labels, "ai": [float(x) for x in cp_te["daily"]],
                       "pts": [float(x) for x in cp_te["d_pts"]]},
        "test_curve": {"t": [TE["ts"][i].tz_convert(KST).strftime("%m-%d %H:%M") for i in idx],
                       "ai": [float(curve[i]) for i in idx], "buyhold": [float(bh_curve[i]) for i in idx]},
        "months": months,
        "gen_log": [{k: v for k, v in x.items() if k != "windows"} for x in gen_log],
        "last_windows": gen_log[-1]["windows"] if gen_log else [],
        "population_scores": [round(float(x), 3) for x in final_scores],
    }
    save_json(f"{OUT}/latest.json", latest)

    # ---- README ----
    def pct(x, d=2): return f"{x*100:+.{d}f}%"
    c = cp_te
    cat_rows = "\n".join(f"| {k} | {RULES[k]} | {v:+.3f} |" for k, v in c["cats"].items())
    month_rows = "\n".join(f"| {m['month']} | {m['score']:+.2f} | {pct(m['ret'])} | {pct(m['btc'])} | "
                           f"{m['hit_days']}/{m['days']} | {m['liq']} |" for m in months[::-1])
    hist_rows = "\n".join(
        f"| {h['day']} | {h['date'][:10]} | {h['total_gens']} | {h['champion_score']:+.2f} | "
        f"{pct(h['champion_avg_daily'], 3)} | {h['champion_hit_days']}/{h['test_days']} | "
        f"{h.get('months_positive', '-')}/{h.get('months_total', '-')} | {h.get('lev', 1):.0f}배/{h.get('frac', 1)*100:.0f}% | "
        f"{'교체' if h['replaced'] else '유지'} |" for h in hist[-30:][::-1])
    md = f"""# 🧬 진화형 데이트레이딩 AI 리포트

_마지막 실행: {datetime.now(KST):%Y-%m-%d %H:%M} KST · {dnum}일차 · 누적 {state['total_gens']}세대 · 계산 {secs:.0f}초_

> 모의투자 실험입니다. 실제 주문은 하지 않습니다. **점수 = 수익 + 수익 보너스(플러스면 2배) − 손실 보유 감점 · 레버리지 없음**
> 학습: {latest['data']['from']} ~ 데이터 {tr_days}일 중 세대마다 랜덤 {WIN_DAYS}일 x {N_WINDOWS}구간 · 시험: 가장 최근 {c['days']}일 (학습에 안 씀)
> 📊 실시간 대시보드: https://ksuin39.github.io/24h_bot/dashboard/

## 오늘의 챔피언 `{state['champion_id']}` ({'첫 등장' if dnum == 1 else '새로 등극' if replaced else '방어 성공'})

- **시험 구간 점수: {c['score']:+.2f}점** · 하루 평균 수익 {pct(c['avg_daily'], 3)} · 시험 {c['days']}일 합계 {pct(c['ret'])} (그냥 보유 {pct(bh)})
- 목표(+1%) 달성 {c['hit_days']}/{c['days']}일 · 최장 연속 {c['best_streak']}일 · 손실일 {c['loss_days']}일
- 하루 거래 {c['trades_per_day']:.1f}회 · 수익 거래 {c['wins']} · 손실 거래 {c['small_losses']} · 날 넘김 {c['overnights']} · 최대 손실 상태 -{c['max_underwater']:.2f}%
- 2년 월별 성적: 플러스인 달 **{sum(1 for m in months if m['score'] > 0)}/{len(months)}개월**
- 주로 보는 정보: {', '.join(feats) if feats else '없음'}

### 점수 구성 (시험 구간)
| 항목 | 규칙 | 점수 |
|---|---|---|
{cat_rows}

### 월별 성적표 (2년, 달마다 새 계좌로 시작)
| 월 | 점수 | AI 수익 | 비트코인 | +1% 달성일 | - |
|---|---|---|---|---|---|
{month_rows}

## 일차별 기록
| 일차 | 날짜 | 누적 세대 | 시험 점수 | 하루 평균 수익 | +1% 달성일 | 플러스 달 | 레버리지/투입 | 결과 |
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
