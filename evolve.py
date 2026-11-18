"""
evolve.py — 매일 스스로 진화하는 '데이트레이딩' AI (모의투자, 실제 주문 없음)
=========================================================================
목표: 하루에 계좌의 1% 수익을 꾸준히.

사람이 정한 매매 규칙 없음. AI 200마리가 각자
  - 어떤 정보를 볼지 (14가지 중 선택 = 유전자)
  - 그 정보를 어떻게 조합해 판단할지 (작은 신경망 가중치 = 유전자)
  - 얼마나 확신할 때 진입할지 (문턱값 = 유전자)
를 갖고 태어나서, 과거 데이터 시합 → 상위만 생존 → 교배/돌연변이 를 반복한다.

데이트레이딩 규칙
  - 매일 오전 9시(KST, = OKX 일봉 마감) 직전에 포지션 강제 청산 → 다음 날로 못 넘김
  - 점수는 '하루 단위'로 계산:
      하루 점수 = 하루 평균 수익
                + 수익 보너스 (평균 수익이 플러스면 2배로 인정)
                + 목표 달성 보너스 (하루 +1% 이상 달성한 날 비율 x TARGET_BONUS)
                - 손실 보유 감점 (손실 상태로 버틴 깊이 x 시간)
  - 학습 점수는 기간을 4등분해서 '꾸준히' 잘해야 높음

매일 실행될 때마다:
  1) 최근 60일 BTC 5분봉 수집 (Coinbase 공개 API)
  2) 어제까지 살아남은 개체군을 이어받아 N세대 더 진화 (학습 구간)
  3) 상위 10마리를 '검증 구간'에서 시험 → 오늘의 도전자 선발
  4) 도전자 vs 기존 챔피언을 '시험 구간(가장 최근, 학습에 안 쓴 데이터)'에서 대결
  5) 이기면 챔피언 교체 → 기록/대시보드 데이터를 evolution/ 폴더에 저장

실행:  python evolve.py            (실제 데이터)
       python evolve.py --sim      (가짜 데이터로 테스트)
"""
import json, os, sys, time, random, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

# ---------------- 설정 ----------------
DAYS         = 60      # 몇 일치 데이터를 쓸지
POP          = 200     # 개체 수
ELITE        = 20      # 매 세대 그대로 살아남는 수
GENERATIONS  = 300     # 하루 실행당 진화 세대 수
IMMIGRANTS   = 0.15    # 매일 새로 섞어 넣는 랜덤 개체 비율 (다양성 유지)
HIDDEN       = 4       # 신경망 은닉 뉴런 수
FEE          = 0.0005  # 포지션 변경 수수료 0.05%
DAILY_TARGET = 0.01    # 하루 목표 수익 1%
TARGET_BONUS = 0.005   # 목표 달성 보너스: 모든 날 달성 시 하루 +0.5점
PROFIT_BONUS = 1.0     # 하루 평균 수익이 플러스면 (1+PROFIT_BONUS)배로 인정 (1.0 = 2배)
PENALTY_W    = 1.0     # 손실 보유 감점: 손실 깊이 x 버틴 시간(일) 만큼
OUT          = "evolution"
KST          = timezone(timedelta(hours=9))
BARS_PER_DAY = 288

FEATURES = ["5분 변화", "15분 변화", "1시간 변화", "4시간 변화", "1일 변화",
            "1시간 평균 대비 위치", "4시간 평균 대비 위치", "1일 평균 대비 위치",
            "거래량 (1시간 대비)", "거래량 (1일 대비)", "변동성 수준",
            "시간대 (sin)", "시간대 (cos)", "오늘 시작가 대비 위치"]
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
    n = int(days * BARS_PER_DAY)
    p = 60000 * np.exp(np.cumsum(np.random.normal(0, 0.0015, n)))
    v = np.abs(np.random.normal(10, 3, n))
    idx = pd.date_range(end=datetime.now(timezone.utc).replace(second=0, microsecond=0),
                        periods=n, freq="5min")
    return pd.DataFrame({"close": p, "volume": v}, index=idx)


def make_features(df):
    c, v = df["close"], df["volume"]
    lr = np.log(c).diff()
    vol = lr.rolling(BARS_PER_DAY).std()             # 1일 변동성
    X = pd.DataFrame(index=df.index)
    for i, k in enumerate([1, 3, 12, 48, 288]):
        X[i] = np.log(c / c.shift(k)) / (vol * np.sqrt(k))
    for i, w in zip([5, 6, 7], [12, 48, 288]):
        X[i] = np.log(c / c.rolling(w).mean()) / (vol * np.sqrt(w))
    X[8] = np.log((v + 1e-9) / (v.rolling(12).mean() + 1e-9))
    X[9] = np.log((v + 1e-9) / (v.rolling(288).mean() + 1e-9))
    X[10] = np.log(vol / lr.rolling(2016).std())     # 1일 vs 1주 변동성
    hour = df.index.hour + df.index.minute / 60       # UTC 시각 = 하루 시작(09시 KST)부터 경과
    X[11] = np.sin(2 * np.pi * hour / 24)
    X[12] = np.cos(2 * np.pi * hour / 24)
    day_key = df.index.floor("D")
    day_open = c.groupby(day_key).transform("first")
    X[13] = np.log(c / day_open) / vol
    X = X.clip(-5, 5)
    ret = c.shift(-1) / c - 1                         # 다음 5분 수익률
    ok = X.notna().all(axis=1) & ret.notna()
    day = np.asarray((df.index.normalize() - pd.Timestamp("1970-01-01", tz="UTC")).days, dtype=np.int64)  # UTC 날짜 번호
    return (X[ok].to_numpy(np.float32), ret[ok].to_numpy(np.float64), c[ok].to_numpy(),
            day[ok.to_numpy()], df.index[ok.to_numpy()])


# ---------------- AI (유전자 → 행동) ----------------
def random_genome():
    g = np.random.normal(0, 1, GENOME_LEN)
    g[-1] = np.random.uniform(0.1, 0.9)              # 문턱값
    return g


def decode(g):
    i = 0
    mask = (g[i:i + F] > 0).astype(np.float32); i += F
    W1 = g[i:i + HIDDEN * F].reshape(HIDDEN, F); i += HIDDEN * F
    b1 = g[i:i + HIDDEN]; i += HIDDEN
    w2 = g[i:i + HIDDEN]; i += HIDDEN
    b2 = g[i]; th = float(np.clip(g[i + 1], 0.0, 0.98))
    return mask, W1, b1, w2, b2, th


def raw_positions(g, X):
    mask, W1, b1, w2, b2, th = decode(g)
    h = np.tanh((X * mask) @ W1.T + b1)
    s = np.tanh(h @ w2 + b2)
    return np.where(s > th, 1.0, np.where(s < -th, -1.0, 0.0))


def day_end_mask(day):
    """그날의 마지막 5분봉 = 다음 봉이 다른 날이면 True (여기서 포지션을 들고 있으면 날을 넘김)"""
    return np.concatenate([day[1:] != day[:-1], [True]])


def positions(g, X, day):
    pos = raw_positions(g, X)
    pos[day_end_mask(day)] = 0.0                       # 오전 9시 직전 강제 청산
    return pos


def daily_score(d_ret, d_pen):
    """하루 단위 배열 → 하루 평균 점수"""
    if len(d_ret) == 0:
        return 0.0
    r = d_ret.mean()
    hit = (d_ret >= np.log1p(DAILY_TARGET)).mean()
    return float(r + PROFIT_BONUS * max(0.0, r) + TARGET_BONUS * hit - d_pen.mean())


def evaluate(g, X, ret, day, curve=False):
    pos = positions(g, X, day)
    change = np.abs(np.diff(np.concatenate([[0.0], pos])))
    pnl = pos * ret - FEE * change
    logp = np.log1p(pnl)

    # 손실 보유 감점: 진입 후 누적 손익이 마이너스인 동안 (손실 깊이 / 하루 봉 수)
    c = np.cumsum(logp)
    c_prev = np.concatenate([[0.0], c[:-1]])
    start = np.maximum.accumulate(np.where(change > 0, np.arange(len(pos)), 0))
    unreal = c - c_prev[start]
    underwater = np.where(pos != 0, np.maximum(0.0, -unreal), 0.0)
    pen = PENALTY_W * underwater / BARS_PER_DAY

    # 하루 단위 집계
    _, d_idx = np.unique(day, return_inverse=True)
    d_ret = np.bincount(d_idx, weights=logp)
    d_pen = np.bincount(d_idx, weights=pen)

    # 거래 통계
    entries = int(((pos != 0) & (change > 0)).sum())
    seg = np.cumsum(change > 0)
    seg_total = np.bincount(seg, weights=logp)
    ends = np.concatenate([np.nonzero(np.diff(seg))[0], [len(pos) - 1]]) if len(pos) else np.array([], int)
    held = [e_ for e_ in ends if pos[e_] != 0]
    wins = int(sum(1 for e_ in held if seg_total[seg[e_]] > 0))
    hold_bars = int((pos != 0).sum())

    r_mean = float(d_ret.mean()) if len(d_ret) else 0.0
    out = {
        "ret": float(np.expm1(logp.sum())), "score": daily_score(d_ret, d_pen),
        "avg_daily": float(np.expm1(r_mean)),
        "bonus": PROFIT_BONUS * max(0.0, r_mean),
        "target_bonus": TARGET_BONUS * float((d_ret >= np.log1p(DAILY_TARGET)).mean()) if len(d_ret) else 0.0,
        "penalty": float(d_pen.mean()) if len(d_pen) else 0.0,
        "days": len(d_ret), "hit_days": int((d_ret >= np.log1p(DAILY_TARGET)).sum()),
        "loss_days": int((d_ret < 0).sum()), "daily": np.expm1(d_ret),
        "trades": entries, "trades_per_day": entries / max(1, len(d_ret)),
        "wins": wins, "positions_taken": len(held),
        "avg_hold_min": (hold_bars / entries * 5) if entries else 0.0,
        "max_underwater": float(underwater.max()) if len(underwater) else 0.0,
        "logp": logp, "pen": pen, "d_ret": d_ret, "d_pen": d_pen,
    }
    if curve:
        out["curve"] = np.exp(c)
    return out


def fitness(g, X, ret, day):
    """학습 점수: 기간을 4등분, 각 구간의 '하루 평균 점수'가 꾸준히 높아야 함"""
    e = evaluate(g, X, ret, day)
    parts = [daily_score(a, b) for a, b in zip(np.array_split(e["d_ret"], 4), np.array_split(e["d_pen"], 4))]
    parts = np.array(parts)
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


def evolve(pop, X, ret, day, gens, log):
    for gen in range(gens):
        scores = np.array([fitness(g, X, ret, day) for g in pop])
        order = np.argsort(-scores)
        pop = [pop[i] for i in order]
        scores = scores[order]
        log.append({"gen": gen + 1, "best": float(scores[0]), "mean": float(scores.mean()),
                    "top20": float(scores[:ELITE].mean())})
        if gen % 30 == 0 or gen == gens - 1:
            print(f"  세대 {gen+1:3d}/{gens}: 최고 {scores[0]*100:+.3f}점/일, 평균 {scores.mean()*100:+.3f}", flush=True)
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


def describe(g):
    mask, W1, *_ = decode(g)
    imp = np.abs(W1).sum(axis=0) * mask
    used = [FEATURES[i] for i in np.argsort(-imp) if mask[i] > 0]
    return used[:5], int(mask.sum())


# ---------------- 메인 ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(int(time.time()) % 2**31)
    df = fake(DAYS) if "--sim" in sys.argv else fetch(DAYS)
    X, ret, close, day, ts = make_features(df)

    # 날짜 경계로 학습/검증/시험 분할 (하루가 쪼개지지 않게)
    udays = np.unique(day)
    nd = len(udays)
    d_va, d_te = udays[int(nd * 0.70)], udays[int(nd * 0.85)]
    tr = day < d_va
    va = (day >= d_va) & (day < d_te)
    te = day >= d_te
    n_days = lambda m: len(np.unique(day[m]))
    print(f"학습 {n_days(tr)}일 / 검증 {n_days(va)}일 / 시험 {n_days(te)}일", flush=True)

    state = load_json(f"{OUT}/state.json", {"day": 0, "total_gens": 0, "champion": None,
                                             "champion_id": None, "champion_born": None})
    pop_saved = load_json(f"{OUT}/population.json", [])
    pop = [np.array(g) for g in pop_saved if len(g) == GENOME_LEN]
    if state["champion"] is not None and len(state["champion"]) != GENOME_LEN:
        state["champion"] = None
    n_imm = int(POP * IMMIGRANTS) if pop else POP
    pop = pop[:POP - n_imm] + [random_genome() for _ in range(n_imm)]
    state["day"] += 1
    dnum = state["day"]
    print(f"\n=== {dnum}일차 진화 시작 (이어받은 개체 {POP - n_imm}, 새 개체 {n_imm}) ===", flush=True)

    t0 = time.time()
    gen_log = []
    pop = evolve(pop, X[tr], ret[tr], day[tr], GENERATIONS, gen_log)
    state["total_gens"] += GENERATIONS

    # 도전자 선발: 학습 상위 10마리 중 검증 구간 점수 최고
    top = pop[:10]
    val_scores = [evaluate(g, X[va], ret[va], day[va])["score"] for g in top]
    ch = top[int(np.argmax(val_scores))]

    # 시험 구간 대결
    ch_te = evaluate(ch, X[te], ret[te], day[te], curve=True)
    champ = np.array(state["champion"]) if state["champion"] else None
    cp_te = evaluate(champ, X[te], ret[te], day[te], curve=True) if champ is not None else None
    close_te = close[te]
    bh = float(close_te[-1] / close_te[0] - 1)

    replaced = cp_te is None or ch_te["score"] > cp_te["score"]
    if replaced:
        state["champion"] = ch.tolist()
        state["champion_id"] = f"G{state['total_gens']}-D{dnum}"
        state["champion_born"] = datetime.now(KST).strftime("%Y-%m-%d")
        cp_te = ch_te
    champ = np.array(state["champion"])
    feats, nfeat = describe(champ)
    secs = time.time() - t0

    # ---- 기록 ----
    hist = load_json(f"{OUT}/history.json", [])
    hist.append({"day": dnum, "date": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
                 "total_gens": state["total_gens"], "champion_id": state["champion_id"],
                 "champion_score": cp_te["score"], "champion_avg_daily": cp_te["avg_daily"],
                 "champion_test": cp_te["ret"], "champion_trades": cp_te["trades"],
                 "champion_hit_days": cp_te["hit_days"], "champion_loss_days": cp_te["loss_days"],
                 "test_days": cp_te["days"], "champion_bonus": cp_te["bonus"],
                 "champion_target_bonus": cp_te["target_bonus"], "champion_penalty": cp_te["penalty"],
                 "challenger_score": ch_te["score"], "replaced": replaced,
                 "buyhold_test": bh, "features": feats})
    save_json(f"{OUT}/history.json", hist)
    save_json(f"{OUT}/population.json", [np.round(g, 4).tolist() for g in pop])
    save_json(f"{OUT}/state.json", state)

    # ---- 대시보드 데이터 ----
    mask, W1, b1, w2, b2, th = decode(champ)
    imp = np.abs(W1).sum(axis=0) * mask
    imp = imp / imp.max() if imp.max() > 0 else imp
    now_raw = float(raw_positions(champ, X[-1:])[0])
    ts_te = ts[te]
    idx = np.linspace(0, len(cp_te["curve"]) - 1, min(240, len(cp_te["curve"]))).astype(int)
    bh_curve = close_te / close_te[0]
    test_days_lbl = [datetime.fromtimestamp(int(d) * 86400 + 9 * 3600, KST).strftime("%m-%d")
                     for d in np.unique(day[te])]
    final_scores = sorted([fitness(g, X[tr], ret[tr], day[tr]) for g in pop], reverse=True)
    pick = lambda e: {k: e[k] for k in ["score", "ret", "avg_daily", "bonus", "target_bonus", "penalty",
                                        "days", "hit_days", "loss_days", "trades", "trades_per_day",
                                        "wins", "positions_taken", "avg_hold_min", "max_underwater"]}
    latest = {
        "mode": "daytrading", "updated": datetime.now(KST).strftime("%Y-%m-%d %H:%M"), "day": dnum,
        "total_gens": state["total_gens"], "generations_today": GENERATIONS, "pop": POP,
        "seconds": round(secs, 1),
        "split_days": {"train": n_days(tr), "val": n_days(va), "test": n_days(te)},
        "settings": {"fee": FEE, "penalty_w": PENALTY_W, "profit_bonus": PROFIT_BONUS,
                     "daily_target": DAILY_TARGET, "target_bonus": TARGET_BONUS},
        "replaced": bool(replaced),
        "champion": {"id": state["champion_id"], "born": state["champion_born"], **pick(cp_te),
                     "threshold": th, "now_position": now_raw,
                     "features": [{"name": FEATURES[i], "used": bool(mask[i] > 0), "weight": float(imp[i])}
                                  for i in range(F)]},
        "challenger": pick(ch_te),
        "buyhold": bh, "last_price": float(close[-1]),
        "test_daily": {"labels": test_days_lbl, "ai": [float(x) for x in cp_te["daily"]]},
        "test_curve": {"t": [ts_te[i].tz_convert(KST).strftime("%m-%d %H:%M") for i in idx],
                       "ai": [float(cp_te["curve"][i]) for i in idx],
                       "buyhold": [float(bh_curve[i]) for i in idx]},
        "gen_log": gen_log,
        "population_scores": [round(x, 6) for x in final_scores],
    }
    save_json(f"{OUT}/latest.json", latest)

    # ---- 리포트 (README) ----
    def pct(x, d=2): return f"{x*100:+.{d}f}%"
    def pt(x): return f"{x*100:+.3f}"
    hist_rows = "\n".join(
        f"| {h['day']} | {h['date'][:10]} | {h['total_gens']} | {pt(h['champion_score'])} | "
        f"{pct(h['champion_avg_daily'], 3)} | {h['champion_hit_days']}/{h['test_days']} | "
        f"{h['champion_loss_days']} | {h['champion_trades']} | {'교체' if h['replaced'] else '유지'} |"
        for h in hist[-30:][::-1])
    daily_rows = " ".join(f"`{l} {x*100:+.2f}%`" for l, x in zip(test_days_lbl, cp_te["daily"]))
    md = f"""# 🧬 진화형 데이트레이딩 AI 리포트

_마지막 실행: {datetime.now(KST):%Y-%m-%d %H:%M} KST · {dnum}일차 · 누적 {state['total_gens']}세대 · 계산 {secs:.0f}초_

> 모의투자 실험입니다. 실제 주문은 하지 않습니다. **목표: 하루 +1%**
> 매일 오전 9시(KST) 전에 포지션을 강제 청산합니다(날을 넘겨 들고 있을 수 없음).
> 모든 성적은 **AI가 학습에 쓰지 않은 가장 최근 {cp_te['days']}일(시험 구간)** 기준입니다.
> 📊 실시간 대시보드: https://ksuin39.github.io/24h_bot/dashboard/

## 오늘의 결과

| | **하루 점수** | 하루 평균 수익 | 목표(+1%) 달성일 | 손실일 | 하루 거래 | 평균 보유 |
|---|---|---|---|---|---|---|
| 👑 챔피언 `{state['champion_id']}` | **{pt(cp_te['score'])}** | {pct(cp_te['avg_daily'], 3)} | {cp_te['hit_days']}/{cp_te['days']} | {cp_te['loss_days']} | {cp_te['trades_per_day']:.1f}회 | {cp_te['avg_hold_min']:.0f}분 |
| 🥊 오늘의 도전자 | {pt(ch_te['score'])} | {pct(ch_te['avg_daily'], 3)} | {ch_te['hit_days']}/{ch_te['days']} | {ch_te['loss_days']} | {ch_te['trades_per_day']:.1f}회 | {ch_te['avg_hold_min']:.0f}분 |

- 그냥 보유했을 때 시험 구간 전체: {pct(bh)} · 챔피언 시험 구간 전체: {pct(cp_te['ret'])}
- 결과: **{'첫 챔피언 등장' if dnum == 1 else '도전자가 이겨서 챔피언 교체!' if replaced else '챔피언 방어 성공'}**
- 챔피언이 보는 정보 {nfeat}개 중 가장 중요하게 쓰는 것: {', '.join(feats) if feats else '없음'}
- 챔피언의 날짜별 수익: {daily_rows}

> **하루 점수 = 하루 평균 수익 + 수익 보너스(플러스면 2배) + 목표 달성 보너스(달성일 비율 × 0.5) − 손실 보유 감점**

## 일차별 기록

| 일차 | 날짜 | 누적 세대 | 하루 점수 | 하루 평균 수익 | 목표 달성일 | 손실일 | 거래 | 결과 |
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
