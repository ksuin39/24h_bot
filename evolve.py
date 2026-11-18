"""
evolve.py — 매일 스스로 진화하는 '데이트레이딩' AI (모의투자, 실제 주문 없음)
=========================================================================
목표: 하루에 계좌의 +1% 를 꾸준히. 레버리지 최대 20배.

[학습 방식]
  - 비트코인 5분봉 2년치를 data/ 에 저장 (매일 새 데이터만 추가)
  - 세대마다 2년 중 '랜덤한 7일 구간 6개'를 뽑아 200마리가 같은 구간에서 시합
    → 매 세대 시험 범위가 바뀌어 외우기 불가, 엘리트도 매번 새 구간에서 재평가
  - 학습 점수 = 6개 구간 하루평균 점수의 평균 - 0.5 x 들쭉날쭉함 (고르게 잘해야 높음)
  - 가장 최근 17일은 학습에 안 씀: 앞 8일 = 검증(도전자 선발), 뒤 9일 = 시험(챔피언 대결)

[AI 유전자]
  - 볼 정보 선택 (시장 정보 14 + 내 상태 2 = 16가지)
      내 상태: 오늘 내 계좌 수익률, 지금 포지션 손익
  - 판단 신경망 가중치, 진입 문턱, 레버리지(1~20배), 계좌 투입 비율(10~100%)

[점수표] (Sin_jin 설계) — 모든 % 는 계좌 대비
  1. 하루 성적표 (매일 오전 9시 KST 정산)
     +1% 이상(목표 달성)  +3점, 1% 넘는 부분 1%당 +0.5 (최대 +2 추가)
     0 ~ +1%             수익% x 2
     손실                 손실% x 3   (예: -0.5% → -1.5)
     연속 목표 달성        2일 연속 +0.5, 3일 이상 연속 매일 +1
  2. 거래 하나하나
     진입                 +0.1 (그 거래가 수익으로 끝났을 때만)
     익절                 +0.5 x min(1, 수익% / 0.5%)
     손절 (손실 -1% 이내)  -0.3
     손절 (손실 -1% 초과)  -0.8 x 손실% / 1%
     당일 종료            +0.2 (하루 3번까지)
     보유 중 손실 상태     -0.1 x 손실% x 버틴 시간(시간)
  3. 위험 관리
     청산                 -10
     장중 계좌 -3% 아래    -2 (하루 1번)
     계좌 50% 아래 (파산)  -20, 그 구간 시합 종료
     포지션 들고 날 넘김    -0.5
     하루 10번 넘는 거래    11번째부터 1번당 -0.2
     수수료               -5 x 낸 수수료%

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

# 점수표 (시뮬레이터에 배열로 전달)
SC = dict(
    day_target_pts=3.0, day_extra_per_pct=0.5, day_extra_cap=2.0,
    day_gain_mult=2.0, day_loss_mult=3.0, streak2=0.5, streak3=1.0,
    entry_win=0.1, tp=0.5, tp_full_pct=0.5, sl_small=-0.3, sl_big=-0.8, sl_big_pct=1.0,
    sameday=0.2, sameday_cap=3.0, hold_loss=0.1,
    liq=-10.0, dd=-2.0, dd_pct=3.0, bankrupt=-20.0, bankrupt_level=0.5,
    overnight=-0.5, overtrade_n=10.0, overtrade=-0.2, fee_w=5.0,
)
SC_KEYS = list(SC.keys())
SC_ARR = np.array([SC[k] for k in SC_KEYS])
I = {k: i for i, k in enumerate(SC_KEYS)}

CATS = ["하루 성적", "연속 달성", "진입", "익절", "손절", "당일 종료", "손실 보유",
        "청산", "장중 -3%", "파산", "날 넘김", "과잉 거래", "수수료"]
RULES = {"하루 성적": "+1%↑ 3점(+초과 1%당 0.5) / 수익%×2 / 손실%×3",
         "연속 달성": "2일 +0.5, 3일↑ 매일 +1", "진입": "+0.1 (수익 거래만)",
         "익절": "+0.5 × min(1, 수익%/0.5%)", "손절": "-0.3 (−1% 이내) / -0.8×손실%",
         "당일 종료": "+0.2 (하루 3번)", "손실 보유": "-0.1 × 손실% × 시간",
         "청산": "-10", "장중 -3%": "-2", "파산": "-20 (계좌 50%↓)", "날 넘김": "-0.5",
         "과잉 거래": "11번째부터 -0.2", "수수료": "-5 × 수수료%"}

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


# ---------------- 시뮬레이터 ----------------
@njit(cache=True)
def day_points(r, sc, streak):
    """하루 계좌 수익률 r(%) → (하루 성적 점수, 연속 보너스, 새 streak)"""
    if r >= 1.0:
        p = sc[0] + min(sc[2], sc[1] * (r - 1.0))
        streak += 1
        b = sc[6] if streak >= 3 else (sc[5] if streak == 2 else 0.0)
        return p, b, streak
    if r > 0:
        return sc[3] * r, 0.0, 0
    return sc[4] * r, 0.0, 0


@njit(cache=True)
def simulate(base, ws, w2, b2, th, close, high, low, day, lev, frac, fee, mm, sc):
    n = close.shape[0]
    H = base.shape[1]
    pts = np.zeros(n)
    cat = np.zeros(13)
    mtm_arr = np.zeros(n)
    # stats: 0 진입, 1 익절, 2 작은손절, 3 큰손절, 4 청산, 5 날넘김, 6 당일종료, 7 수수료합, 8 파산, 9 목표달성일, 10 최대연속, 11 장중-3%
    st = np.zeros(12)
    eq = 1.0
    d = 0
    entry = 0.0; notional = 0.0; margin = 0.0; entry_eq = 1.0; entry_fee = 0.0; liq = 0.0; entry_day = 0
    day_start = 1.0; dd_hit = False; streak = 0; trades_today = 0; sameday_today = 0
    dead = False
    last = 1.0
    h = np.zeros(H)
    for t in range(n):
        if dead:
            mtm_arr[t] = last
            continue
        # ---- 하루 정산 (날이 바뀌면 어제 성적 매기기) ----
        if t > 0 and day[t] != day[t - 1]:
            r = (mtm_arr[t - 1] / day_start - 1.0) * 100.0
            p, b, streak = day_points(r, sc, streak)
            pts[t] += p + b; cat[0] += p; cat[1] += b
            if r >= 1.0:
                st[9] += 1
            if streak > st[10]:
                st[10] = streak
            day_start = mtm_arr[t - 1]
            dd_hit = False; trades_today = 0; sameday_today = 0
            if d != 0:
                v = sc[21]; pts[t] += v; cat[10] += v; st[5] += 1
        # ---- 청산 확인 ----
        if d != 0:
            if (d == 1 and low[t] <= liq) or (d == -1 and high[t] >= liq):
                eq -= margin
                v = sc[16]; pts[t] += v; cat[7] += v; st[4] += 1
                d = 0
        # ---- 내 상태 ----
        upnl = d * (close[t] / entry - 1.0) * notional if d != 0 else 0.0
        mtm = eq + upnl
        pos_pct = upnl / entry_eq * 100.0 if d != 0 else 0.0
        today_pct = (mtm / day_start - 1.0) * 100.0
        if d != 0 and pos_pct < 0:
            v = -sc[15] * (-pos_pct) * (5.0 / 60.0); pts[t] += v; cat[6] += v
        # ---- 파산 ----
        if mtm < sc[20]:
            v = sc[19]; pts[t] += v; cat[9] += v; st[8] = 1
            dead = True; last = mtm; mtm_arr[t] = mtm
            continue
        # ---- 장중 -3% ----
        if (not dd_hit) and mtm < day_start * (1.0 - sc[18] / 100.0):
            v = sc[17]; pts[t] += v; cat[8] += v; dd_hit = True; st[11] += 1
        # ---- 판단 (신경망) ----
        tp = min(5.0, max(-5.0, today_pct))
        pp = min(5.0, max(-5.0, pos_pct))
        s = b2
        for k in range(H):
            h[k] = np.tanh(base[t, k] + ws[k, 0] * tp + ws[k, 1] * pp)
            s += h[k] * w2[k]
        s = np.tanh(s)
        want = 1 if s > th else (-1 if s < -th else 0)
        if want != d:
            if d != 0:   # 종료
                pnl = d * (close[t] / entry - 1.0) * notional
                xfee = notional * (close[t] / entry) * fee
                before = eq
                eq += pnl - xfee
                st[7] += xfee
                v = -sc[24] * xfee / before * 100.0; pts[t] += v; cat[12] += v
                net = (pnl - xfee - entry_fee) / entry_eq * 100.0
                if net > 0:
                    v = sc[7]; pts[t] += v; cat[2] += v
                    v = sc[8] * min(1.0, net / sc[9]); pts[t] += v; cat[3] += v; st[1] += 1
                elif net >= -sc[12]:
                    v = sc[10]; pts[t] += v; cat[4] += v; st[2] += 1
                else:
                    v = sc[11] * (-net / sc[12]); pts[t] += v; cat[4] += v; st[3] += 1
                if day[t] == entry_day and sameday_today < sc[14]:
                    v = sc[13]; pts[t] += v; cat[5] += v; sameday_today += 1; st[6] += 1
                d = 0
            if want != 0:   # 진입
                d = want
                entry = close[t]; entry_eq = eq
                margin = eq * frac; notional = margin * lev
                entry_fee = notional * fee
                v = -sc[24] * entry_fee / eq * 100.0; pts[t] += v; cat[12] += v
                eq -= entry_fee; st[7] += entry_fee
                entry_day = day[t]
                liq = entry * (1.0 - 1.0 / lev + mm) if d == 1 else entry * (1.0 + 1.0 / lev - mm)
                trades_today += 1; st[0] += 1
                if trades_today > sc[22]:
                    v = sc[23]; pts[t] += v; cat[11] += v
        mtm_arr[t] = eq + (d * (close[t] / entry - 1.0) * notional if d != 0 else 0.0)
    # 마지막 날 정산
    if not dead and n > 0:
        r = (mtm_arr[n - 1] / day_start - 1.0) * 100.0
        p, b, streak = day_points(r, sc, streak)
        pts[n - 1] += p + b; cat[0] += p; cat[1] += b
        if r >= 1.0:
            st[9] += 1
        if streak > st[10]:
            st[10] = streak
    return pts, cat, mtm_arr, st


def run(g, S):
    mask, W1, b1, w2, b2, th, lev, frac = decode(g)
    base = (S["X"] * mask[:FM]) @ W1[:, :FM].T + b1
    ws = np.ascontiguousarray(W1[:, FM:] * mask[FM:])
    return simulate(np.ascontiguousarray(base), ws, np.ascontiguousarray(w2), b2, th,
                    S["close"], S["high"], S["low"], S["day"], lev, frac, FEE, MAINT_MARGIN, SC_ARR)


def n_days(S):
    return len(np.unique(S["day"]))


def window_score(g, S):
    pts, *_ = run(g, S)
    return pts.sum() / n_days(S)


def evaluate(g, S):
    pts, cat, mtm, st = run(g, S)
    _, d_idx = np.unique(S["day"], return_inverse=True)
    nd = int(d_idx.max()) + 1
    last = np.zeros(nd, dtype=int); last[d_idx] = np.arange(len(d_idx))
    d_eq = mtm[last]
    d_ret = d_eq / np.concatenate([[1.0], d_eq[:-1]]) - 1
    _, _, _, _, _, th, lev, frac = decode(g)
    return {
        "score": float(pts.sum() / nd), "d_pts": np.bincount(d_idx, weights=pts, minlength=nd), "daily": d_ret,
        "cats": {CATS[i]: float(cat[i] / nd) for i in range(13)},
        "ret": float(mtm[-1] - 1), "avg_daily": float(np.expm1(np.log(max(mtm[-1], 1e-9)) / nd)),
        "days": nd, "hit_days": int(st[9]), "loss_days": int((d_ret < 0).sum()), "best_streak": int(st[10]),
        "trades": int(st[0]), "trades_per_day": float(st[0] / nd), "wins": int(st[1]),
        "small_losses": int(st[2]), "big_losses": int(st[3]), "liquidations": int(st[4]),
        "overnights": int(st[5]), "dd_days": int(st[11]), "fee_pct": float(st[7] * 100),
        "bankrupt": bool(st[8]), "lev": lev, "frac": frac, "th": th, "curve": mtm,
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
            print(f"  세대 {gen+1:3d}/{gens}: 1등 {scores[0]:+.2f}점/일 (레버리지 {lev:.0f}배, 투입 {frac*100:.0f}%), "
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
        "settings": {"fee": FEE, "daily_target": 0.01, "max_lev": MAX_LEV, "rules": RULES},
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

> 모의투자 실험입니다. 실제 주문은 하지 않습니다. **목표: 하루 +1% · 레버리지 최대 {MAX_LEV}배**
> 학습: {latest['data']['from']} ~ 데이터 {tr_days}일 중 세대마다 랜덤 {WIN_DAYS}일 x {N_WINDOWS}구간 · 시험: 가장 최근 {c['days']}일 (학습에 안 씀)
> 📊 실시간 대시보드: https://ksuin39.github.io/24h_bot/dashboard/

## 오늘의 챔피언 `{state['champion_id']}` ({'첫 등장' if dnum == 1 else '새로 등극' if replaced else '방어 성공'})

- **하루 평균 점수: {c['score']:+.2f}점** · 하루 평균 수익 {pct(c['avg_daily'], 3)} · 시험 {c['days']}일 합계 {pct(c['ret'])} (그냥 보유 {pct(bh)})
- 목표(+1%) 달성 {c['hit_days']}/{c['days']}일 · 최장 연속 {c['best_streak']}일 · 손실일 {c['loss_days']}일
- 레버리지 **{lev:.0f}배**, 계좌 **{frac*100:.0f}%** 투입 · 하루 거래 {c['trades_per_day']:.1f}회
- 익절 {c['wins']} · 작은 손절 {c['small_losses']} · 큰 손절 {c['big_losses']} · 청산 {c['liquidations']} · 날 넘김 {c['overnights']}
- 2년 월별 성적: 플러스인 달 **{sum(1 for m in months if m['score'] > 0)}/{len(months)}개월**
- 주로 보는 정보: {', '.join(feats) if feats else '없음'}

### 점수 항목별 (시험 구간 하루 평균)
| 항목 | 규칙 | 점수 |
|---|---|---|
{cat_rows}

### 월별 성적표 (2년, 달마다 새 계좌로 시작)
| 월 | 하루 점수 | AI 수익 | 비트코인 | 목표 달성일 | 청산 |
|---|---|---|---|---|---|
{month_rows}

## 일차별 기록
| 일차 | 날짜 | 누적 세대 | 하루 점수 | 하루 평균 수익 | 목표 달성일 | 플러스 달 | 레버리지/투입 | 결과 |
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
