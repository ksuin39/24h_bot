"""
evolve3.py — 진화형 트레이딩 AI v3 : '제대로 된 실험' 버전 (모의투자, 실제 주문 없음)
=======================================================================================
v2 까지의 실험 설계 문제를 고친 버전.

1) 검증 재사용 금지 → '걸어가며 검증(워크포워드)'
   - 최근 8개월을 각각 '시험 달'로 쓰는 시험 구간 8개(fold)를 만든다.
   - fold k 의 AI 는 '시험 달 이전 데이터'로만 학습한다. 시험 달은 학습에도, AI 선발에도 쓰지 않고 성적 보고에만 쓴다.
   - 앙상블(상위 5마리) 선발은 '학습 데이터 전체'에서의 점수로만 한다 (별도 검증 기간을 반복 사용하지 않음).
   - 섬끼리 개체 교환은 같은 fold 안에서만 (미래 데이터를 본 개체가 과거 fold 로 넘어가는 누수 방지).
   → 최종 성적 = 8개 시험 달 x 5개 코인 = 40개 '코인-달' (상승·하락·횡보가 섞임, 각 달은 한 번씩만 평가)

2) 규칙 충돌 제거
   - '거래한 날 70%' 생존 조건 삭제 (확신 없으면 쉬어도 됨)
   - 비트코인 레버리지 최대 5배 → 2배, 나머지 코인 1배
   - 생존: 학습 구간 하루 평균 +0.1% 이상 (유지)

3) 기준선 비교 (같은 달, 같은 수수료)
   아무것도 안 함 / 그냥 보유 / 랜덤 매매(AI와 같은 거래 빈도) / 추세 추종(24시간 vs 1주 평균) / 돌파(3일 고저)
   → AI 가 랜덤과 단순 전략을 이겨야 '진짜 뭔가 찾았다'

유지: 1시간봉, 초반 점수 방식(수익 + 보너스 − 손실 보유 감점), 최대 보유 24시간, 지정가 수수료 0.03%, 16대 병렬

실행:
  python evolve3.py island --id 5 --minutes 50
  python evolve3.py report
  python evolve3.py selftest
"""
import glob, json, os, sys, time, random
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import evolve2 as E

COINS = E.COINS
COIN_NAMES = E.COIN_NAMES
COIN_MAX_LEV = {"BTC": 2}
OUT = "v3"
ISLAND_OUT = "v3_out"
KST = timezone(timedelta(hours=9))
N_FOLDS = 8
ISLANDS_PER_FOLD = 2
POP = 120
ELITE = 12
BATCH = 16
MIGRANTS = 10
ENS_K = 5
SURVIVE_DAILY = 0.001
MAX_SURVIVORS = 60
FEE = E.FEE
E.QFRAC = 0.5
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "24"))


# ---------------- 데이터 / fold ----------------
def load_coins(sim=False):
    coins = {}
    for i, coin in enumerate(COINS):
        df5 = E.fake_5m(seed=i) if sim else E.read_coin_5m(coin)
        if df5 is None or len(df5) < 288 * 200:
            continue
        D = E.make_features(E.to_hourly(df5))
        D["max_lev"] = COIN_MAX_LEV.get(coin, 1)
        D["coin"] = coin
        coins[coin] = D
    return coins


def month_segments(D):
    mk = pd.Index(D["ts"]).strftime("%Y-%m")
    out = []
    for m in pd.unique(mk):
        ix = np.nonzero(mk == m)[0]
        if len(ix) >= E.BPD * 10:
            S = E.seg(D, ix[0], ix[-1] + 1, {"month": m, "a": int(ix[0]), "b": int(ix[-1] + 1)})
            out.append(S)
    return out


def build_folds(coins):
    """fold k: 시험 달 = 최근 8개월 중 k번째, 학습 = 그 달 이전의 모든 코인-달"""
    segs = {c: month_segments(D) for c, D in coins.items()}
    all_months = sorted(set(S["month"] for ss in segs.values() for S in ss))
    test_months = all_months[-N_FOLDS:]
    folds = []
    for k, tm in enumerate(test_months):
        train = [S for ss in segs.values() for S in ss if S["month"] < tm]
        test = {c: S for c, ss in segs.items() for S in ss if S["month"] == tm}
        folds.append({"k": k, "test_month": tm, "train": train, "test": test,
                      "train_months": sorted(set(S["month"] for S in train))})
    return folds


# ---------------- 학습 점수 ----------------
def fitness(g, batch):
    sc, lsum, days = [], 0.0, 0
    for S in batch:
        logp, pen, eq, st = E.run([g], S)
        tot = logp.sum(); nd = E.ndays(S)
        sc.append((tot + E.PROFIT_BONUS * max(0.0, tot) - pen.sum()) * 100 / nd * 30)
        lsum += tot; days += nd
    sc = np.array(sc)
    return float(sc.mean() - 0.5 * sc.std()), float(np.expm1(lsum / days))


# ---------------- 섬 (fold 하나를 맡음) ----------------
def island(iid, minutes, sim=False):
    t0 = time.time()
    np.random.seed((int(time.time()) + iid * 7919) % 2**31); random.seed(time.time() + iid)
    k = iid // ISLANDS_PER_FOLD
    coins = load_coins(sim)
    folds = build_folds(coins)
    if k >= len(folds):
        print(f"[섬 {iid}] fold {k} 없음 → 종료"); return
    F_ = folds[k]
    train = F_["train"]
    print(f"[섬 {iid}] fold {k} · 시험 달 {F_['test_month']} · 학습 코인-달 {len(train)}개 "
          f"({F_['train_months'][0]} ~ {F_['train_months'][-1]}) · 준비 {time.time()-t0:.0f}초", flush=True)
    pop = [np.array(g) for g in E.load_json(f"{OUT}/islands/{iid}.json", []) if len(g) == E.GENOME_LEN]
    migr = [np.array(g) for g in E.load_json(f"{OUT}/migrants_f{k}.json", []) if len(g) == E.GENOME_LEN]
    random.shuffle(migr)
    pop = pop[:POP - MIGRANTS] + migr[:MIGRANTS]
    if len(pop) < POP:
        pop += [E.random_genome() for _ in range(POP - len(pop))]
    sigma = [0.2, 0.4][iid % 2]
    log, gen = [], 0
    budget = minutes * 60 - (time.time() - t0)
    t1 = time.time()
    while time.time() - t1 < budget:
        batch = random.sample(train, min(BATCH, len(train)))
        res = [fitness(g, batch) for g in pop]
        sc = np.array([r[0] for r in res]); dl = np.array([r[1] for r in res])
        o = np.argsort(-sc); pop = [pop[i] for i in o]; sc, dl = sc[o], dl[o]
        alive = [i for i in range(len(pop)) if dl[i] >= SURVIVE_DAILY][:MAX_SURVIVORS]
        gen += 1
        if gen % 25 == 1:
            log.append({"gen": gen, "best": float(sc[0]), "top": float(sc[:ELITE].mean()),
                        "best_daily": float(dl[0]), "alive": len(alive)})
        if gen % 200 == 1:
            print(f"  [섬 {iid}] {gen}세대: 1등 {sc[0]:+.2f}, 하루 {dl[0]*100:+.3f}%, 생존 {len(alive)}/{POP}", flush=True)
        if len(alive) >= 2:
            parents = [pop[i] for i in alive]; nxt = list(parents)
        else:
            parents = pop[:5]; nxt = [pop[i] for i in alive] + [E.random_genome() for _ in range(POP // 2)]
        while len(nxt) < POP:
            a, b = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0])
            nxt.append(E.mutate(E.crossover(a, b), sigma=sigma))
        pop = nxt
    E.save_json(f"{ISLAND_OUT}/island_{iid}.json",
                {"id": iid, "fold": k, "gens": gen, "pop": [np.round(g, 4).tolist() for g in pop[:POP]],
                 "log": log, "minutes": (time.time() - t0) / 60})
    print(f"[섬 {iid}] 완료 {gen}세대", flush=True)


# ---------------- 기준선 ----------------
def bt(pos, close, fee):
    """포지션 배열(-1/0/1, 레버리지 1배) → 구간 수익률"""
    r = pos[:-1] * (close[1:] / close[:-1] - 1)
    ch = np.abs(np.diff(np.concatenate([[0.0], pos])))[:-1]
    eq = np.prod(1 + r - fee * ch) * (1 - fee * abs(pos[-1]))
    return float(eq - 1)


def baseline_signals(D):
    c = pd.Series(D["close"]); hi = pd.Series(D["high"]); lo = pd.Series(D["low"])
    trend = np.where(c.rolling(24).mean() > c.rolling(168).mean(), 1.0, -1.0)
    up, dn = hi.rolling(72).max().shift(1), lo.rolling(72).min().shift(1)
    br = pd.Series(np.where(c > up, 1.0, np.where(c < dn, -1.0, np.nan))).ffill().fillna(0).to_numpy()
    return {"trend": trend, "breakout": br}


def random_baseline(close, tpd, fee, seeds=30):
    n = len(close); p = min(0.9, max(tpd, 0.05) / 24)
    outs = []
    rng = np.random.default_rng(12345)
    for _ in range(seeds):
        pos = np.zeros(n); cur = 0.0
        for t in range(n):
            if rng.random() < p:
                cur = float(rng.choice([-1.0, 0.0, 1.0]))
            pos[t] = cur
        outs.append(bt(pos, close, fee))
    return float(np.mean(outs))


def daily_of(ret, days):
    return float(np.expm1(np.log1p(max(ret, -0.99)) / max(days, 1)))


# ---------------- 리포트 ----------------
def report(sim=False):
    t0 = time.time()
    if not sim:
        try:
            import evolve as v1
            for coin in COINS:
                v1.load_data(coin)
        except Exception as e:
            print("데이터 갱신 실패(이전 데이터로 진행):", e, flush=True)
    coins = load_coins(sim)
    folds = build_folds(coins)
    sigs = {c: baseline_signals(D) for c, D in coins.items()}
    state = E.load_json(f"{OUT}/state.json", {"round": 0, "total_gens": 0,
                                               "started": datetime.now(KST).strftime("%Y-%m-%d %H:%M")})
    state["round"] += 1
    by_fold, islands, round_gens = {}, [], 0
    for f in sorted(glob.glob(f"{ISLAND_OUT}/island_*.json")):
        r = E.load_json(f, {})
        pop = [np.array(g) for g in r.get("pop", []) if len(g) == E.GENOME_LEN]
        if not pop:
            continue
        E.save_json(f"{OUT}/islands/{r['id']}.json", [g.tolist() for g in pop])
        by_fold.setdefault(r["fold"], []).extend(pop[:10])
        islands.append({"id": r["id"], "fold": r["fold"], "gens": r["gens"], "log": r["log"][-30:]})
        round_gens += r["gens"]
    state["total_gens"] += round_gens

    fold_rows, cm_rows = [], []
    ens_store = E.load_json(f"{OUT}/ensembles.json", {})
    for F_ in folds:
        k = F_["k"]
        cands = by_fold.get(k, []) + [np.array(g) for g in ens_store.get(str(k), [])]
        if not cands:
            continue
        E.save_json(f"{OUT}/migrants_f{k}.json", [g.tolist() for g in by_fold.get(k, [])])
        # 앙상블 선발: 학습 데이터 전체(시험 달 이전 모든 코인-달)에서의 점수로만
        train = F_["train"]
        sc = [fitness(g, train)[0] for g in cands]
        order = np.argsort(sc)[::-1]
        ens = [cands[i] for i in order[:ENS_K]]
        ens_store[str(k)] = [g.tolist() for g in ens]
        row = {"fold": k, "test_month": F_["test_month"], "train_from": F_["train_months"][0],
               "train_to": F_["train_months"][-1], "train_score": float(sc[order[0]]), "coins": {}}
        for coin, S in F_["test"].items():
            e = E.evaluate(ens, S)
            nd = e["days"]; cl = S["close"]; a, b = S["a"], S["b"]
            res = {"ai": e["ret"], "flat": 0.0, "hold": float(cl[-1] / cl[0] - 1) - 2 * FEE,
                   "random": random_baseline(cl, e["trades_per_day"], FEE),
                   "trend": bt(sigs[coin]["trend"][a:b], cl, FEE),
                   "breakout": bt(sigs[coin]["breakout"][a:b], cl, FEE)}
            row["coins"][coin] = {**{kk: v for kk, v in res.items()}, "days": nd,
                                  "ai_tpd": e["trades_per_day"], "ai_daily": e["avg_daily"]}
            cm_rows.append({"fold": k, "month": F_["test_month"], "coin": coin, "days": nd,
                            **{kk: v for kk, v in res.items()}, "market": "up" if res["hold"] > 0 else "down"})
        fold_rows.append(row)
    E.save_json(f"{OUT}/ensembles.json", ens_store)

    names = ["ai", "flat", "hold", "random", "trend", "breakout"]
    agg = {}
    for nm in names:
        rets = np.array([r[nm] for r in cm_rows]); days = np.array([r["days"] for r in cm_rows])
        agg[nm] = {"mean_month_ret": float(rets.mean()) if len(rets) else 0.0,
                   "pos_ratio": float((rets > 0).mean()) if len(rets) else 0.0,
                   "avg_daily": float(np.mean([daily_of(r, d) for r, d in zip(rets, days)])) if len(rets) else 0.0,
                   "up_mean": float(np.mean([r[nm] for r in cm_rows if r["market"] == "up"] or [0])),
                   "down_mean": float(np.mean([r[nm] for r in cm_rows if r["market"] == "down"] or [0]))}
    beats = {nm: float(np.mean([r["ai"] > r[nm] for r in cm_rows])) for nm in names if nm != "ai"}
    goal = {"avg_daily": agg["ai"]["avg_daily"], "beats_all_baselines":
            bool(all(agg["ai"]["mean_month_ret"] > agg[nm]["mean_month_ret"] for nm in names if nm != "ai"))}

    # 지금 쓸 AI = 가장 최근 fold 의 앙상블 (가장 최신 데이터까지 학습)
    live = None
    if fold_rows:
        lk = str(fold_rows[-1]["fold"])
        ens = [np.array(g) for g in ens_store[lk]]
        E.save_json(f"{OUT}/live_ensemble.json", ens_store[lk])
        imp = np.zeros(E.F)
        for g in ens:
            mask, W1, *_ = E.decode(g)
            x = np.abs(W1).sum(axis=0) * mask
            imp += x / (x.max() + 1e-9)
        imp /= len(ens)
        live = {"fold": int(lk), "features": [{"name": E.FEATURES[i], "weight": float(imp[i]), "state": bool(i >= E.FM)}
                                              for i in np.argsort(-imp)[:12]],
                "btc_lev": float(np.median([E.coin_lev(g, 2) for g in ens]))}

    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    hist = E.load_json(f"{OUT}/history.json", [])
    hist.append({"round": state["round"], "time": now, "total_gens": state["total_gens"], "round_gens": round_gens,
                 **{f"{nm}_mean_month_ret": agg[nm]["mean_month_ret"] for nm in names},
                 "ai_avg_daily": agg["ai"]["avg_daily"], "ai_pos_ratio": agg["ai"]["pos_ratio"],
                 "beats_random": beats.get("random", 0), "beats_trend": beats.get("trend", 0)})
    E.save_json(f"{OUT}/history.json", hist[-500:])
    E.save_json(f"{OUT}/state.json", state)
    latest = {"updated": now, "round": state["round"], "max_rounds": MAX_ROUNDS, "started": state["started"],
              "total_gens": state["total_gens"], "round_gens": round_gens, "islands": islands,
              "settings": {"fee": FEE, "timeframe": "1시간봉", "folds": N_FOLDS, "ens_k": ENS_K,
                           "btc_max_lev": 2, "survive_daily": SURVIVE_DAILY},
              "folds": fold_rows, "coin_months": cm_rows, "agg": agg, "beats": beats, "goal": goal, "live": live,
              "coin_names": COIN_NAMES}
    E.save_json(f"{OUT}/latest.json", latest)

    def pct(x, d=2): return f"{x*100:+.{d}f}%"
    label = {"ai": "🧬 AI 앙상블", "flat": "아무것도 안 함", "hold": "그냥 보유", "random": "랜덤 매매",
             "trend": "추세 추종 (24h vs 1주)", "breakout": "돌파 (3일 고저)"}
    arows = "\n".join(f"| {label[nm]} | {pct(agg[nm]['mean_month_ret'])} | {pct(agg[nm]['avg_daily'], 3)} | "
                      f"{agg[nm]['pos_ratio']*100:.0f}% | {pct(agg[nm]['up_mean'])} | {pct(agg[nm]['down_mean'])} |"
                      f"{' - |' if nm == 'ai' else ' ' + str(round(beats[nm]*100)) + '% |'}" for nm in names)
    frows = "\n".join(f"| {r['test_month']} | {r['train_from']} ~ {r['train_to']} | " +
                      " | ".join(f"{pct(v['ai'])}" for v in r["coins"].values()) + " |" for r in fold_rows)
    md = f"""# 🧬 진화형 트레이딩 AI v3 — 걸어가며 검증 리포트

_마지막: {now} KST · 라운드 {state['round']}/{MAX_ROUNDS} · 누적 {state['total_gens']:,}세대 (이번 {round_gens:,}세대)_

> 모의투자 실험 (실제 주문 없음). 최근 8개월을 각각 '시험 달'로, 그 이전 데이터로만 학습 → 시험 달은 학습·선발에 전혀 안 씀
> 규칙: 1시간봉 · 생존 하루 +0.1% (거래 강제 없음) · BTC 레버리지 최대 2배 · 수수료 0.03% · 최대 보유 24시간
> 📊 대시보드: https://ksuin39.github.io/24h_bot/dashboard3/

## AI vs 기준선 (8개월 x 5코인 = {len(cm_rows)}개 코인-달, 각 달 한 번씩만 평가)
| 전략 | 월평균 수익 | 하루 평균 | 플러스인 달 | 오른 달 평균 | 내린 달 평균 | AI가 이긴 비율 |
|---|---|---|---|---|---|---|
{arows}

- **AI가 모든 기준선보다 월평균이 높은가: {'✅ 예' if goal['beats_all_baselines'] else '❌ 아니오'}**

## 시험 달별 AI 수익 ({' / '.join(COIN_NAMES[c] for c in coins)})
| 시험 달 | 학습 기간 | {' | '.join(COIN_NAMES[c] for c in coins)} |
|---|---|{'---|' * len(coins)}
{frows}
"""
    with open(f"{OUT}/README.md", "w", encoding="utf-8") as f:
        f.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)
    print(md)
    print(f"리포트 {time.time()-t0:.0f}초", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    if mode == "island":
        st = E.load_json(f"{OUT}/state.json", {})
        if st.get("round", 0) >= MAX_ROUNDS:
            print("목표 라운드 완료 → 종료"); sys.exit(0)
        island(int(arg("--id", "0")), float(arg("--minutes", "50")), sim="--sim" in sys.argv)
    elif mode == "report":
        report(sim="--sim" in sys.argv)
    else:
        for i in [0, 1, 14, 15]:
            island(i, float(arg("--minutes", "1")), sim=True)
        report(sim=True)
