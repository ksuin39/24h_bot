"""
evolve.py — 단계별로 스스로 진화하는 트레이딩 AI (모의투자, 실제 주문 없음)
=========================================================================
최종 목표: 하루 +1% 수익을 내는 AI. (점수 = 초반 방식)

[학습 방식] — 단계별 진화 (여러 코인 x 랜덤 30일)
  - 비트코인·이더리움·솔라나·리플·도지코인 5분봉 2년치를 data/ 에 저장 (실행마다 새 데이터만 추가)
  - 각 단계 = 랜덤 코인 x 랜덤 30일. 그 30일을 반복 학습
  - 졸업 기준: 1등 AI가 그 30일에서 하루 평균 +1% 이상 → 다음 랜덤 코인·30일로 (개체군은 이어감)
      → STAGE_MAX_GENS 세대가 지나도 못 졸업하면 '미달'로 기록하고 다음으로
  - 30일을 4토막으로 나눠 '꾸준히' 잘해야 점수가 높음
  - 생존 규칙: 매 세대 하루 평균 +0.1% 이상인 개체만 생존 (미만은 탈락, 빈자리는 생존자의 자식)
  - 각 코인의 가장 최근 17일은 학습에 안 씀: 앞 8일 = 검증, 뒤 9일 = 시험 (5개 코인 평균으로 챔피언 결정)
  - 매시간(연속) 자동 실행, 한 번에 약 RUN_MINUTES 분 진화하고 저장

[AI 유전자]
  - 볼 정보 선택 (시장 정보 25 + 내 상태 2 = 27가지)
      시장: 가격 변화(5분~1일), 평균 대비 위치, 거래량, 변동성, 시간대, 오늘 시작가 대비,
            SMA200 대비(5분/1시간/4시간/일봉), RSI14(5분/1시간/4시간), 매수세(1시간/4시간),
            하루/1주 고저 범위 내 위치
      내 상태: 오늘 내 계좌 수익률, 지금 포지션 손익
  - 판단 신경망 가중치, 진입 문턱   (레버리지 없음 = 1배, 계좌 100%)

[점수] (초반 방식으로 복원)
  점수 = 수익률(로그, %) + 수익 보너스 - 손실 보유 감점
    수익 보너스     : 그 구간 순수익이 플러스면 같은 만큼 한 번 더 (수익 2배 인정)
    손실 보유 감점  : 포지션을 잡고 '진입 대비 손실 상태'로 버티는 동안
                     손실 깊이 x 버틴 시간(일) 만큼 감점 (2% 손실로 하루 버티면 -2점)
  - 수수료 0.05% (포지션 바꿀 때) 는 수익률에 그대로 반영
  - 날 넘김 허용, 강제 청산 없음
  - 학습 점수 = 60일을 4토막 낸 각 토막 점수의 평균 - 0.5 x 들쭉날쭉함

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
COINS        = ["BTC", "ETH", "SOL", "XRP", "DOGE"]   # 학습에 쓰는 코인 (Coinbase 현물 USD)
COIN_NAMES   = {"BTC": "비트코인", "ETH": "이더리움", "SOL": "솔라나", "XRP": "리플", "DOGE": "도지코인"}
BLOCK_DAYS   = 30      # 한 단계에서 학습할 기간 (일) — 랜덤 코인 x 랜덤 30일
CHUNKS       = 4       # 꾸준함 평가용 토막 수
GOAL_DAILY   = 0.01    # 졸업 기준: 하루 평균 +1%
STAGE_MAX_GENS = 6000  # 한 단계 최대 세대 (넘으면 미달로 다음 단계)
RUN_MINUTES  = float(os.environ.get("RUN_MINUTES", "45"))   # 한 번 실행에 진화할 시간(분)
CHECK_EVERY  = 25      # 몇 세대마다 졸업 여부 확인
VAL_DAYS     = 8
TEST_DAYS    = 9
POP          = 200
ELITE        = 20
IMMIGRANTS   = 0.15
SURVIVE_DAILY = 0.001   # 생존 기준: 학습 60일에서 하루 평균 +0.1% 이상이면 생존, 미만이면 탈락
MAX_SURVIVORS = 100     # 생존자가 너무 많으면 점수 높은 100마리까지만 (자식 자리 확보)
HIDDEN       = 8       # 신경망 은닉 뉴런 (목표가 높아 4 → 8로 키움)
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
                   "시간대 (sin)", "시간대 (cos)", "오늘 시작가 대비 위치",
                   "SMA200 대비 (5분봉)", "SMA200 대비 (1시간봉)", "SMA200 대비 (4시간봉)", "SMA200 대비 (일봉)",
                   "RSI14 (5분봉)", "RSI14 (1시간봉)", "RSI14 (4시간봉)",
                   "매수세 (1시간)", "매수세 (4시간)",
                   "하루 고저 범위 내 위치", "1주 고저 범위 내 위치"]
STATE_FEATURES = ["오늘 내 계좌 수익률", "지금 포지션 손익"]
FEATURES = MARKET_FEATURES + STATE_FEATURES
FM, FS = len(MARKET_FEATURES), len(STATE_FEATURES)
F = FM + FS
GENOME_LEN = F + HIDDEN * F + HIDDEN + HIDDEN + 1 + 1 + 2   # mask, W1, b1, w2, b2, th, lev, frac
TH_IDX = F + HIDDEN * F + 2 * HIDDEN + 1


# ---------------- 데이터 (월별 파일로 저장, 매일 새 것만 추가) ----------------
def fetch_range(start, end, product="BTC-USD"):
    rows = {}
    cur = start
    n = 0
    while cur < end:
        nxt = min(cur + timedelta(minutes=5 * 300), end)
        url = (f"https://api.exchange.coinbase.com/products/{product}/candles"
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
            print(f"  {product} 데이터 받는 중... {cur:%Y-%m-%d}", flush=True)
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame(columns=["close", "high", "low", "volume"])
    ts = sorted(rows)
    return pd.DataFrame({k: [rows[t][i] for t in ts] for i, k in enumerate(["close", "high", "low", "volume"])},
                        index=pd.to_datetime(ts, unit="s", utc=True))


def load_data(coin="BTC"):
    os.makedirs(DATA_DIR, exist_ok=True)
    pre = f"{coin.lower()}_5m_"
    files = sorted(glob.glob(f"{DATA_DIR}/{pre}*.csv.gz"))
    old = (pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in files]) if files
           else pd.DataFrame(columns=["close", "high", "low", "volume"]))
    if len(old):
        old.index = pd.to_datetime(old.index, utc=True)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = (old.index.max().to_pydatetime() + timedelta(minutes=5)) if len(old) else now - timedelta(days=HISTORY_DAYS)
    print(f"[{coin}] 새 데이터 받기: {start:%Y-%m-%d %H:%M} ~ 지금", flush=True)
    new = fetch_range(start, now, f"{coin}-USD")
    df = pd.concat([old, new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index >= pd.Timestamp(now - timedelta(days=HISTORY_DAYS + 31))]
    # 새 데이터가 들어간 달의 파일만 다시 저장
    touched = set(new.index.strftime("%Y-%m")) if len(new) else set()
    for m in touched:
        part = df[df.index.strftime("%Y-%m") == m]
        part.to_csv(f"{DATA_DIR}/{pre}{m}.csv.gz", compression="gzip", float_format="%.7g")
    keep = set(df.index.strftime("%Y-%m"))
    for f in files:
        if os.path.basename(f)[len(pre):len(pre) + 7] not in keep:
            os.remove(f)
    df = df[df.index >= pd.Timestamp(now - timedelta(days=HISTORY_DAYS))]
    print(f"[{coin}] 전체 데이터: 5분봉 {len(df):,}개 ({df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d})", flush=True)
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
    # SMA200 대비 위치 (5분봉 / 1시간봉 / 4시간봉 / 일봉 기준 200개 평균) — 변동성으로 나눠 크기 맞춤
    for i, bars in zip([14, 15, 16, 17], [1, 12, 48, 288]):
        n = 200 * bars
        sma = c.rolling(n, min_periods=n // 2).mean()
        X[i] = np.log(c / sma) / (vol * np.sqrt(min(n, 2016)))
    # RSI14 (5분봉 / 1시간봉 / 4시간봉) → -1~+1 로 (50 = 0)
    for i, bars in zip([18, 19, 20], [1, 12, 48]):
        ch = c.diff(bars)
        up = ch.clip(lower=0).rolling(14 * bars).mean()
        dn = (-ch).clip(lower=0).rolling(14 * bars).mean()
        X[i] = (up / (up + dn + 1e-12) - 0.5) * 4
    # 매수세: 캔들 안에서 종가가 고가 쪽이면 매수 우세 (+1), 저가 쪽이면 매도 우세 (-1) — 거래량 가중
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((2 * c - df["high"] - df["low"]) / rng).fillna(0)
    for i, w in zip([21, 22], [12, 48]):
        X[i] = (clv * v).rolling(w).sum() / (v.rolling(w).sum() + 1e-9) * 3
    # 최근 고가/저가 범위 안에서 지금 위치 (0 = 바닥, 1 = 꼭대기 → -1~+1)
    for i, w in zip([23, 24], [288, 2016]):
        hi, lo = df["high"].rolling(w).max(), df["low"].rolling(w).min()
        X[i] = ((c - lo) / (hi - lo + 1e-9) - 0.5) * 4
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


# ---------------- 진화 (단계별 60일 블록) ----------------
def crossover(a, b):
    return np.where(np.random.rand(GENOME_LEN) < 0.5, a, b)


def mutate(g, rate=0.1, sigma=0.3):
    g = g.copy()
    m = np.random.rand(GENOME_LEN) < rate
    g[m] += np.random.normal(0, sigma, m.sum())
    return g


def fitness_chunks(g, chunks):
    s = np.array([window_score(g, C) for C in chunks])
    return float(s.mean() - 0.5 * s.std())


def fitness_and_daily(g, chunks):
    """(학습 점수, 60일 하루 평균 수익) 을 한 번에"""
    scores, lsum, days = [], 0.0, 0
    for C in chunks:
        logp, pen, *_ = run(g, C)
        scores.append(score_of(logp.sum(), pen.sum())[0])
        lsum += logp.sum(); days += n_days(C)
    s = np.array(scores)
    return float(s.mean() - 0.5 * s.std()), float(np.expm1(lsum / max(days, 1)))


def block_chunks(B):
    starts = np.concatenate([[0], np.nonzero(np.diff(B["day"]))[0] + 1, [len(B["day"])]])
    nd = len(starts) - 1
    cuts = np.linspace(0, nd, CHUNKS + 1).astype(int)
    return [seg(B, starts[cuts[k]], starts[cuts[k + 1]]) for k in range(CHUNKS)]


def day_label(D, i):
    return datetime.fromtimestamp(int(D["day"][i]) * 86400, KST).strftime("%Y-%m-%d")




def make_block(C, first_day):
    """C = 코인 하나의 데이터 묶음, first_day = 그 코인 데이터에서 몇 번째 날부터"""
    D, starts = C["D"], C["starts"]
    a, b = starts[first_day], starts[first_day + BLOCK_DAYS]
    B = seg(D, a, b)
    return B, block_chunks(B), day_label(D, a), day_label(D, b - 1)


def pick_new_block(coins, used):
    """랜덤 코인 x 랜덤 30일. 같은 코인에서 이미 학습한 기간과 겹치는 곳은 피함"""
    best, best_ov = None, 1e9
    for _ in range(80):
        coin = random.choice(list(coins))
        C = coins[coin]
        f = random.randint(0, C["tr_days"] - BLOCK_DAYS)
        dn = int(C["udays"][f])
        ov = max([max(0, min(dn + BLOCK_DAYS, u + BLOCK_DAYS) - max(dn, u)) for (uc, u) in used if uc == coin] or [0])
        if ov < best_ov:
            best, best_ov = (coin, f), ov
        if ov == 0:
            break
    return best


def locate(C, day_num):
    f = int(np.searchsorted(C["udays"], day_num))
    return min(max(f, 0), C["tr_days"] - BLOCK_DAYS)


# ---------------- 저장/불러오기 ----------------
def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def mean_eval(g, sets):
    """여러 코인 구간에서 평가해 평균 (점수, 하루 평균 수익)"""
    es = [evaluate(g, S) for S in sets]
    return float(np.mean([e["score"] for e in es])), float(np.mean([e["avg_daily"] for e in es])), es


# ---------------- 메인 ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(int(time.time()) % 2**31); random.seed(int(time.time()))
    t_start = time.time()

    # 코인별 데이터 준비
    coins = {}
    for coin in COINS:
        try:
            df = fake(HISTORY_DAYS) if "--sim" in sys.argv else load_data(coin)
            if len(df) < BPD * 120:
                print(f"[{coin}] 데이터 부족 → 제외", flush=True)
                continue
            D = make_features(df)
        except Exception as ex:
            print(f"[{coin}] 데이터 오류 → 제외: {ex}", flush=True)
            continue
        day = D["day"]
        starts = np.concatenate([[0], np.nonzero(np.diff(day))[0] + 1, [len(day)]])
        nd = len(starts) - 1
        tr_days = nd - VAL_DAYS - TEST_DAYS
        coins[coin] = {"D": D, "starts": starts, "tr_days": tr_days, "udays": day[starts[:-1]],
                       "VA": seg(D, starts[tr_days], starts[tr_days + VAL_DAYS]),
                       "TE": seg(D, starts[tr_days + VAL_DAYS], starts[nd])}
        print(f"[{coin}] 학습 가능 {tr_days}일", flush=True)
    VAs = [c["VA"] for c in coins.values()]
    TEs = {k: c["TE"] for k, c in coins.items()}

    state = load_json(f"{OUT}/state.json", {})
    state.setdefault("run", 0); state.setdefault("total_gens", 0)
    state.setdefault("champion", None); state.setdefault("champion_id", None); state.setdefault("champion_born", None)
    state.setdefault("stages", [])
    if state["champion"] is not None and len(state["champion"]) != GENOME_LEN:
        state["champion"] = None
    pop = [np.array(g) for g in load_json(f"{OUT}/population.json", []) if len(g) == GENOME_LEN]
    if len(pop) < POP:
        pop += [random_genome() for _ in range(POP - len(pop))]
    state["run"] += 1
    run_no = state["run"]

    def new_stage(n):
        used = [(s_["coin"], s_["first_day_num"]) for s_ in state["stages"] if "coin" in s_]
        coin, f = pick_new_block(coins, used)
        C = coins[coin]
        _, _, a_lbl, b_lbl = make_block(C, f)
        return {"n": n, "coin": coin, "first_day_num": int(C["udays"][f]), "from": a_lbl, "to": b_lbl,
                "gens": 0, "best_avg_daily": None, "started": datetime.now(KST).strftime("%Y-%m-%d %H:%M")}

    cur = state.get("current")
    if cur is None or cur.get("coin") not in coins:
        cur = new_stage(cur["n"] if cur else 1)
    C = coins[cur["coin"]]
    B, chunks, a_lbl, b_lbl = make_block(C, locate(C, cur["first_day_num"]))
    print(f"=== 실행 #{run_no} · {cur['n']}단계 [{cur['coin']}] {a_lbl} ~ {b_lbl} · 이 단계 누적 {cur['gens']}세대 ===", flush=True)

    gen_log = []
    graduations = []
    budget = RUN_MINUTES * 60 - (time.time() - t_start) - 300
    t0 = time.time()
    gen = 0
    while time.time() - t0 < budget:
        res = [fitness_and_daily(g, chunks) for g in pop]
        scores = np.array([r[0] for r in res]); daily = np.array([r[1] for r in res])
        order = np.argsort(-scores)
        pop = [pop[i] for i in order]; scores = scores[order]; daily = daily[order]
        alive = [i for i in range(len(pop)) if daily[i] >= SURVIVE_DAILY][:MAX_SURVIVORS]
        n_alive = len(alive)
        gen += 1; cur["gens"] += 1; state["total_gens"] += 1
        if gen % CHECK_EVERY == 0 or gen == 1:
            e = evaluate(pop[0], B)
            cur["best_avg_daily"] = e["avg_daily"]
            gen_log.append({"gen": state["total_gens"], "stage": cur["n"], "coin": cur["coin"], "best": float(scores[0]),
                            "top20": float(scores[:ELITE].mean()), "mean": float(np.mean(np.maximum(scores, -50))),
                            "avg_daily": e["avg_daily"], "alive": n_alive})
            if gen % (CHECK_EVERY * 4) == 0 or gen == 1:
                print(f"  {cur['n']}단계[{cur['coin']}] {cur['gens']}세대: 1등 하루 평균 {e['avg_daily']*100:+.3f}% "
                      f"(목표 +{GOAL_DAILY*100:.0f}%), 거래 {e['trades_per_day']:.1f}회/일, 생존 {n_alive}/{POP}", flush=True)
            done = e["avg_daily"] >= GOAL_DAILY
            if done or cur["gens"] >= STAGE_MAX_GENS:
                rec = {**cur, "ended": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                       "result": "졸업" if done else "미달", "final_avg_daily": e["avg_daily"]}
                state["stages"].append(rec)
                graduations.append(rec)
                print(f"  ★ {cur['n']}단계[{cur['coin']}] {'졸업! (목표 달성)' if done else '미달 (세대 한도)'} "
                      f"하루 평균 {e['avg_daily']*100:+.3f}% → 다음 랜덤 코인·30일로", flush=True)
                cur = new_stage(cur["n"] + 1)
                C = coins[cur["coin"]]
                B, chunks, a_lbl, b_lbl = make_block(C, locate(C, cur["first_day_num"]))
        # 생존 규칙: 하루 평균 +0.1% 이상만 생존
        if n_alive >= 2:
            parents = [pop[i] for i in alive]
            nxt = list(parents)
        else:
            parents = pop[:5]
            nxt = [pop[i] for i in alive] + [random_genome() for _ in range(POP // 2)]
        while len(nxt) < POP:
            a, b = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
            nxt.append(mutate(crossover(a, b)))
        pop = nxt
    state["current"] = cur
    print(f"이번 실행 {gen}세대 진화 완료 ({(time.time()-t0)/60:.1f}분)", flush=True)

    scores = np.array([fitness_chunks(g, chunks) for g in pop])
    order = np.argsort(-scores); pop = [pop[i] for i in order]; final_scores = scores[order]

    # 도전자: 상위 10마리 중 모든 코인 검증 구간 평균이 가장 좋은 개체
    val = [mean_eval(g, VAs)[0] for g in pop[:10]]
    ch = pop[int(np.argmax(val))]
    ch_sc, ch_avg, ch_es = mean_eval(ch, list(TEs.values()))
    champ = np.array(state["champion"]) if state["champion"] else None
    if champ is not None:
        cp_sc, cp_avg, cp_es = mean_eval(champ, list(TEs.values()))
    replaced = champ is None or ch_sc > cp_sc
    if replaced:
        state["champion"] = ch.tolist()
        state["champion_id"] = f"G{state['total_gens']}-S{cur['n']}"
        state["champion_born"] = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
        cp_sc, cp_avg, cp_es = ch_sc, ch_avg, ch_es
    champ = np.array(state["champion"])
    coin_tests = {k: e for k, e in zip(TEs.keys(), cp_es)}
    main_coin = "BTC" if "BTC" in coin_tests else list(coin_tests)[0]
    cp_te = coin_tests[main_coin]
    TE = TEs[main_coin]
    ch_te = dict(zip(TEs.keys(), ch_es))[main_coin]

    # 현재 1등: 이번 단계 + 지난 단계 복습
    leader = pop[0]
    lead_block = evaluate(leader, B)
    stage_rows = []
    for s_ in state["stages"][-12:]:
        if s_.get("coin") not in coins:
            continue
        Cs = coins[s_["coin"]]
        Bs, _, _, _ = make_block(Cs, locate(Cs, s_["first_day_num"]))
        e_ = evaluate(leader, Bs)
        stage_rows.append({"n": s_["n"], "coin": s_["coin"], "from": s_["from"], "to": s_["to"], "gens": s_["gens"],
                           "result": s_["result"], "avg_daily_at_end": s_["final_avg_daily"],
                           "avg_daily_now": e_["avg_daily"], "ended": s_["ended"]})

    # 월별 성적표 (챔피언, 비트코인 2년)
    months = []
    D0 = coins[main_coin]["D"]
    mk = pd.Index(D0["ts"]).strftime("%Y-%m")
    for m in sorted(set(mk)):
        idx = np.nonzero(mk == m)[0]
        if len(idx) < BPD * 5:
            continue
        S = seg(D0, idx[0], idx[-1] + 1)
        e = evaluate(champ, S)
        months.append({"month": m, "score": e["score"], "ret": e["ret"], "hit_days": e["hit_days"],
                       "days": e["days"], "liq": 0, "bankrupt": False, "avg_daily": e["avg_daily"],
                       "btc": float(S["close"][-1] / S["close"][0] - 1)})

    mask, W1, b1, w2, b2, th, _, _ = decode(champ)
    lev, frac = 1.0, 1.0
    imp = np.abs(W1).sum(axis=0) * mask
    imp = imp / imp.max() if imp.max() > 0 else imp
    feats = [FEATURES[i] for i in np.argsort(-imp) if mask[i] > 0][:5]
    secs = time.time() - t_start
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    bh = float(TE["close"][-1] / TE["close"][0] - 1)

    hist = load_json(f"{OUT}/history.json", [])
    hist.append({"day": run_no, "date": now, "total_gens": state["total_gens"], "stage": cur["n"], "coin": cur["coin"],
                 "stage_avg_daily": lead_block["avg_daily"], "graduated": len(graduations),
                 "champion_id": state["champion_id"], "champion_score": cp_te["score"],
                 "champion_avg_daily": cp_te["avg_daily"], "champion_all_coins_avg_daily": cp_avg,
                 "champion_test": cp_te["ret"], "champion_trades": cp_te["trades"],
                 "champion_hit_days": cp_te["hit_days"], "champion_loss_days": cp_te["loss_days"],
                 "champion_liquidations": 0, "test_days": cp_te["days"], "lev": lev, "frac": frac,
                 "challenger_score": ch_sc, "months_positive": sum(1 for m in months if m["score"] > 0),
                 "months_total": len(months), "replaced": replaced, "buyhold_test": bh, "features": feats})
    hist = hist[-500:]
    save_json(f"{OUT}/history.json", hist)
    save_json(f"{OUT}/population.json", [np.round(g, 4).tolist() for g in pop])
    save_json(f"{OUT}/state.json", state)
    glog = (load_json(f"{OUT}/gen_log.json", []) + gen_log)[-2000:]
    save_json(f"{OUT}/gen_log.json", glog)

    D_last = coins[main_coin]["D"]
    base_now = (D_last["X"][-1:] * mask[:FM]) @ W1[:, :FM].T + b1
    s_now = np.tanh(np.tanh(base_now[0]) @ w2 + b2)
    now_pos = 1.0 if s_now > th else (-1.0 if s_now < -th else 0.0)
    curve = cp_te["curve"]
    idx = np.linspace(0, len(curve) - 1, min(240, len(curve))).astype(int)
    bh_curve = TE["close"] / TE["close"][0]
    labels = [datetime.fromtimestamp(int(x) * 86400 + 9 * 3600, KST).strftime("%m-%d") for x in np.unique(TE["day"])]
    lb_curve = lead_block["curve"]; li = np.linspace(0, len(lb_curve) - 1, min(240, len(lb_curve))).astype(int)
    keys = ["score", "ret", "avg_daily", "days", "hit_days", "loss_days", "best_streak", "trades", "trades_per_day",
            "wins", "small_losses", "big_losses", "liquidations", "overnights", "dd_days", "fee_pct", "bankrupt",
            "lev", "frac", "cats", "max_underwater"]
    latest = {
        "mode": "curriculum-multicoin", "updated": now, "day": run_no, "total_gens": state["total_gens"],
        "generations_today": gen, "pop": POP, "seconds": round(secs, 1),
        "coins": {k: COIN_NAMES.get(k, k) for k in coins},
        "data": {"from": str(D0["ts"][0].date()), "to": str(D0["ts"][-1].date()), "train_days": coins[main_coin]["tr_days"],
                 "win_days": BLOCK_DAYS, "n_windows": 1, "val_days": VAL_DAYS, "test_days": TEST_DAYS},
        "settings": {"fee": FEE, "daily_target": GOAL_DAILY, "max_lev": 1, "rules": RULES, "mode": "classic",
                     "profit_bonus": PROFIT_BONUS, "penalty_w": PENALTY_W, "stage_max_gens": STAGE_MAX_GENS,
                     "block_days": BLOCK_DAYS, "run_minutes": RUN_MINUTES, "survive_daily": SURVIVE_DAILY},
        "curriculum": {"current": cur,
                       "leader_on_block": {k: lead_block[k] for k in ["avg_daily", "ret", "trades_per_day", "hit_days",
                                                                      "days", "score", "wins", "small_losses"]},
                       "leader_block_curve": [float(lb_curve[i]) for i in li],
                       "stages": stage_rows,
                       "graduated_total": sum(1 for s_ in state["stages"] if s_["result"] == "졸업"),
                       "failed_total": sum(1 for s_ in state["stages"] if s_["result"] == "미달")},
        "coin_tests": {k: {"avg_daily": e["avg_daily"], "ret": e["ret"], "hit_days": e["hit_days"], "days": e["days"],
                           "trades_per_day": e["trades_per_day"],
                           "buyhold": float(TEs[k]["close"][-1] / TEs[k]["close"][0] - 1)} for k, e in coin_tests.items()},
        "champion_all_coins_avg_daily": cp_avg,
        "replaced": bool(replaced),
        "champion": {"id": state["champion_id"], "born": state["champion_born"],
                     **{k: cp_te[k] for k in keys}, "threshold": th, "now_position": now_pos,
                     "features": [{"name": FEATURES[i], "used": bool(mask[i] > 0), "weight": float(imp[i]),
                                   "state": i >= FM} for i in range(F)]},
        "challenger": {k: ch_te[k] for k in keys},
        "buyhold": bh, "last_price": float(D_last["close"][-1]),
        "test_daily": {"labels": labels, "ai": [float(x) for x in cp_te["daily"]],
                       "pts": [float(x) for x in cp_te["d_pts"]]},
        "test_curve": {"t": [TE["ts"][i].tz_convert(KST).strftime("%m-%d %H:%M") for i in idx],
                       "ai": [float(curve[i]) for i in idx], "buyhold": [float(bh_curve[i]) for i in idx]},
        "months": months,
        "gen_log": glog,
        "last_windows": [f"{COIN_NAMES.get(cur['coin'], cur['coin'])} {cur['from']} ~ {cur['to']}"],
        "population_scores": [round(float(x), 3) for x in final_scores],
    }
    save_json(f"{OUT}/latest.json", latest)

    def pct(x, d=2): return f"{x*100:+.{d}f}%"
    c = cp_te
    st_rows = "\n".join(f"| {r['n']} | {r['coin']} | {r['from']} ~ {r['to']} | {r['gens']} | {r['result']} | "
                        f"{pct(r['avg_daily_at_end'], 3)} | {pct(r['avg_daily_now'], 3)} |" for r in stage_rows[::-1])
    ct_rows = "\n".join(f"| {COIN_NAMES.get(k, k)} | {pct(v['avg_daily'], 3)} | {pct(v['ret'])} | {pct(v['buyhold'])} | "
                        f"{v['hit_days']}/{v['days']} | {v['trades_per_day']:.1f} |" for k, v in latest["coin_tests"].items())
    month_rows = "\n".join(f"| {m['month']} | {m['score']:+.2f} | {pct(m['ret'])} | {pct(m['btc'])} | "
                           f"{m['hit_days']}/{m['days']} |" for m in months[::-1])
    hist_rows = "\n".join(
        f"| {h['day']} | {h['date']} | {h['total_gens']} | {h.get('stage', '-')} {h.get('coin', '')} | "
        f"{pct(h.get('stage_avg_daily', 0), 3)} | {pct(h.get('champion_all_coins_avg_daily', h['champion_avg_daily']), 3)} | "
        f"{'교체' if h['replaced'] else '유지'} |" for h in hist[-30:][::-1])
    md = f"""# 🧬 단계별 진화 트레이딩 AI 리포트 (여러 코인)

_마지막 실행: {now} KST · 실행 #{run_no} · 누적 {state['total_gens']}세대 · 이번 실행 {gen}세대 ({secs/60:.0f}분)_

> 모의투자 실험입니다. 실제 주문은 하지 않습니다. **최종 목표: 하루 +1% 수익을 내는 AI**
> 학습: 랜덤 코인({', '.join(COIN_NAMES.get(k, k) for k in coins)}) x 랜덤 30일을 반복 학습 → 1등이 하루 평균 +1% 달성하면 졸업 → 다음 랜덤 코인·30일
> 생존: 매 세대 하루 평균 +0.1% 이상만 생존 · 점수 = 수익 + 수익 보너스(2배) − 손실 보유 감점 · 레버리지 없음
> 📊 실시간 대시보드: https://ksuin39.github.io/24h_bot/dashboard/

## 지금 단계: {cur['n']}단계 — {COIN_NAMES.get(cur['coin'], cur['coin'])} ({cur['from']} ~ {cur['to']})
- 이 단계 누적 {cur['gens']}세대 · 1등의 이 30일 하루 평균 **{pct(lead_block['avg_daily'], 3)}** (목표 +1%)
- 지금까지 졸업 {latest['curriculum']['graduated_total']}단계 · 미달 {latest['curriculum']['failed_total']}단계

| 단계 | 코인 | 기간 | 세대 | 결과 | 당시 하루 평균 | 지금 1등의 복습 |
|---|---|---|---|---|---|---|
{st_rows}

## 👑 챔피언 `{state['champion_id']}` — 학습에 안 쓴 최근 {c['days']}일 실전 시험
- **5개 코인 평균 하루 수익 {pct(cp_avg, 3)}**

| 코인 | 하루 평균 | 9일 합계 | 그냥 보유 | +1% 달성일 | 하루 거래 |
|---|---|---|---|---|---|
{ct_rows}

- 주로 보는 정보: {', '.join(feats) if feats else '없음'}

### 월별 성적표 (챔피언, 비트코인 2년)
| 월 | 점수 | AI 수익 | 비트코인 | +1% 달성일 |
|---|---|---|---|---|
{month_rows}

## 실행 기록
| 실행 | 시각 | 누적 세대 | 단계 | 1등의 단계 하루 평균 | 챔피언 코인 평균 하루 | 결과 |
|---|---|---|---|---|---|---|
{hist_rows}
"""
    with open(f"{OUT}/README.md", "w", encoding="utf-8") as f:
        f.write(md)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write(md)
    print(md[:3000])


if __name__ == "__main__":
    main()
