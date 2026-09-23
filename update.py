"""Falling Knives daily updater.

Runs in GitHub Actions after the US market close. Pulls daily prices for every
S&P 500 member from Yahoo Finance (via yfinance), ranks the 50 worst performers
month-to-date, rebuilds the blue-chip section, and writes data.js for the site.

Reasons come from reasons.json, which this script only READS. Stocks without a
reason are shown as "Reason pending" with their latest headlines.
"""
import calendar
import datetime as dt
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import yfinance as yf

CONSTITUENTS = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
SKIP_BIG = {"GOOG", "FOX", "NWS"}          # second share classes
TOP_N, BIG_N = 50, 50


def r2(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), 2)


def week_key(d):
    return (d - dt.timedelta(days=d.weekday())).isoformat()


def load_universe():
    df = pd.read_csv(io.StringIO(requests.get(CONSTITUENTS, timeout=30).text))
    df["Y"] = df["Symbol"].str.replace(".", "-", regex=False)
    return {r.Y: {"name": r.Security, "sector": r["GICS Sector"]} for _, r in df.iterrows()}


def download(symbols):
    frames = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period="2y", interval="1d", auto_adjust=False,
                                  group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:  # noqa
                print("download retry", e, file=sys.stderr); time.sleep(5)
        for s in chunk:
            try:
                sub = raw[s] if len(chunk) > 1 else raw
                sub = sub[["Close", "High", "Low", "Volume"]].dropna(subset=["Close"])
                if len(sub) > 60:
                    frames[s] = sub
            except Exception:
                pass
    return frames


def metrics(s, df, month_start):
    dates = [d.date() for d in df.index]
    C = [float(x) for x in df["Close"]]
    H = [float(x) if not pd.isna(x) else c for x, c in zip(df["High"], C)]
    V = [float(x) if not pd.isna(x) else 0.0 for x in df["Volume"]]
    n = len(C)
    last, last_d = C[-1], dates[-1]
    pre = [c for c, d in zip(C, dates) if d < month_start]
    base = pre[-1] if pre else C[0]
    ye = [c for c, d in zip(C, dates) if d.year < last_d.year]
    ytdb = ye[-1] if ye else None
    y1 = list(range(max(0, n - 252), n))
    hi_i = max(y1, key=lambda i: C[i]); lo_i = min(y1, key=lambda i: C[i])
    hi2_i = max(range(n), key=lambda i: C[i])
    m3 = list(range(max(0, n - 63), n)); p3_i = max(m3, key=lambda i: C[i])
    # weeks (Monday-based), last close of each week
    W = []
    for c, d in zip(C, dates):
        k = week_key(d)
        if W and W[-1]["k"] == k:
            W[-1]["c"], W[-1]["e"] = c, d
        else:
            W.append({"k": k, "c": c, "e": d})
    for i in range(1, len(W)):
        W[i]["r"] = W[i]["c"] / W[i - 1]["c"] - 1
    cur_k = week_key(last_d)
    done = [w for w in W if w["k"] != cur_k]
    cur = W[-1]
    last_pos = next((done[i] for i in range(len(done) - 1, 0, -1) if done[i].get("r", 0) > 0), None)
    dw = 0
    for i in range(len(done) - 1, 0, -1):
        if done[i].get("r", 0) < 0: dw += 1
        else: break
    dd = 0
    for i in range(n - 1, 0, -1):
        if C[i] < C[i - 1]: dd += 1
        else: break
    sma = lambda k: sum(C[-k:]) / min(k, n)
    g = l = 0.0
    start = max(1, n - 200)
    for i in range(start, n):
        ch = C[i] - C[i - 1]; G, L = max(ch, 0), max(-ch, 0)
        if i < start + 14: g += G / 14; l += L / 14
        else: g = (g * 13 + G) / 14; l = (l * 13 + L) / 14
    rsi = 100 - 100 / (1 + g / l) if l else 100.0
    vol = lambda k: sum(V[-k:]) / k
    worst = None
    for i in range(1, n):
        if dates[i] >= month_start:
            r = C[i] / C[i - 1] - 1
            if worst is None or r < worst[0]: worst = (r, dates[i])
    ret = lambda k: (C[-1] / C[max(0, n - 1 - k)] - 1) * 100
    iso = lambda d: d.isoformat() if d else None
    return {
        "s": s, "last": r2(last), "prev": r2(C[-2]), "day": r2((C[-1] / C[-2] - 1) * 100), "asof": iso(last_d),
        "base": round(base, 4), "mtd": r2((last / base - 1) * 100),
        "w1": r2(ret(5)), "m1": r2(ret(21)), "m3": r2(ret(63)),
        "ytd": r2((last / ytdb - 1) * 100) if ytdb else None, "y1": r2(ret(min(251, n - 1))),
        "hi52": r2(C[hi_i]), "hi52d": iso(dates[hi_i]), "dd52": r2((last / C[hi_i] - 1) * 100),
        "tdSinceHi": n - 1 - hi_i, "hi2y": r2(C[hi2_i]), "hi2yd": iso(dates[hi2_i]),
        "lo52": r2(C[lo_i]), "lo52d": iso(dates[lo_i]), "offLo": r2((last / C[lo_i] - 1) * 100),
        "p3": r2(C[p3_i]), "p3d": iso(dates[p3_i]), "ddp3": r2((last / C[p3_i] - 1) * 100),
        "lastPosWk": iso(last_pos["e"]) if last_pos else None,
        "lastPosWkRet": r2(last_pos["r"] * 100) if last_pos else None,
        "downWks": dw, "curWk": r2(cur.get("r", 0) * 100), "downDays": dd,
        "rsi": r2(rsi), "v50": r2((last / sma(50) - 1) * 100), "v200": r2((last / sma(200) - 1) * 100),
        "volx": r2(vol(5) / vol(50)) if vol(50) else None,
        "worstDay": r2(worst[0] * 100) if worst else None, "worstDayD": iso(worst[1]) if worst else None,
        "wks": [r2(w.get("r", 0) * 100) for w in W[-10:]],
        "wkends": [w["e"].strftime("%b %-d") for w in W[-10:]],
    }


def market_caps(symbols):
    def one(s):
        for _ in range(2):
            try:
                v = yf.Ticker(s).fast_info.get("marketCap")
                if v: return s, float(v)
            except Exception:
                time.sleep(1)
        return s, None
    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, symbols))


def details(s):
    out = {}
    try:
        t = yf.Ticker(s)
        info = t.info or {}
        out.update({
            "tgt": r2(info.get("targetMeanPrice")), "rec": info.get("recommendationKey"),
            "nA": info.get("numberOfAnalystOpinions"), "ind": info.get("industry"),
            "fpe": r2(info.get("forwardPE")),
            "dy": r2((info.get("trailingAnnualDividendYield") or 0) * 100) or None,
        })
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        out["ed"] = dt.datetime.utcfromtimestamp(ts).date().isoformat() if ts else None
        news = []
        for item in (t.news or [])[:6]:
            c = item.get("content", item)
            title = c.get("title")
            when = c.get("pubDate") or c.get("displayTime")
            if not when and item.get("providerPublishTime"):
                when = dt.datetime.utcfromtimestamp(item["providerPublishTime"]).isoformat()
            if title:
                d = (when or "")[:10]
                try: d = dt.date.fromisoformat(d).strftime("%b %-d")
                except Exception: pass
                news.append([d, title])
        out["news"] = news[:3]
    except Exception as e:
        print("details fail", s, e, file=sys.stderr)
    return s, out


def auto_stories(D, month_name):
    themes = {}
    for r in D:
        if r["theme"] == "Reason pending": continue
        themes.setdefault(r["theme"], []).append(r)
    top = sorted(themes.items(), key=lambda kv: -len(kv[1]))[:3]
    out = []
    for th, rows in top:
        rows = sorted(rows, key=lambda r: r["mtd"])
        w = rows[0]
        out.append({"k": f"{len(rows)} of the 50", "h": f"{th}.",
                    "p": f"The biggest single theme behind this {month_name} list. Hardest hit: {w['name']} ({w['s']}), down {abs(w['mtd']):.1f}% this month.",
                    "t": " · ".join(r["s"] for r in rows[:8])})
    return out


def main():
    uni = load_universe()
    syms = sorted(uni)
    frames = download(syms + ["^GSPC", "SPY"])
    last_date = max(f.index[-1].date() for s, f in frames.items() if s in uni)
    month_start = last_date.replace(day=1)
    M = {}
    for s in syms:
        if s in frames and frames[s].index[-1].date() == last_date:
            try:
                M[s] = metrics(s, frames[s], month_start)
            except Exception as e:
                print("metrics fail", s, e, file=sys.stderr)
    print(f"computed {len(M)} of {len(syms)} as of {last_date}")
    if len(M) < 450:
        sys.exit("Too few symbols computed; not publishing.")
    spx = metrics("^GSPC", frames["^GSPC"], month_start)
    spy = metrics("SPY", frames["SPY"], month_start) if "SPY" in frames else None

    caps = market_caps(syms)
    ranked = sorted(M.values(), key=lambda r: r["mtd"])
    top = ranked[:TOP_N]
    big = sorted([s for s in syms if caps.get(s) and s not in SKIP_BIG and s in M],
                 key=lambda s: -caps[s])[:BIG_N]
    bc = sorted([M[s] for s in big if M[s]["mtd"] < 0], key=lambda r: r["mtd"])
    card_syms = sorted({r["s"] for r in top} | {r["s"] for r in bc})
    with ThreadPoolExecutor(6) as ex:
        det = dict(ex.map(details, card_syms))

    try:
        reasons = json.load(open("reasons.json"))
    except FileNotFoundError:
        reasons = {}

    def card(r, rank):
        s = r["s"]; o = {k: v for k, v in r.items() if k not in ("base", "wkends")}
        o.update(det.get(s, {}))
        o["name"] = uni[s]["name"]; o["sector"] = uni[s]["sector"]; o["rank"] = rank
        o["mc"] = r2(caps[s] / 1e9) if caps.get(s) else None
        rs = reasons.get(s)
        if rs and isinstance(rs, list) and len(rs) == 3:
            o["theme"], o["conf"], o["why"] = rs
            o.pop("news", None)
        else:
            o["theme"], o["conf"], o["why"] = "Reason pending", "Pending", ""
        return o

    D = [card(r, i + 1) for i, r in enumerate(top)]
    capRank = {s: i + 1 for i, s in enumerate(big)}
    BC = [card(r, i + 1) for i, r in enumerate(bc)]
    for o in BC: o["capRank"] = capRank[o["s"]]
    BIG = [[s, round(caps[s] / 1e9), M[s]["mtd"]] for s in big]
    BASE = {s: M[s]["base"] for s in set(card_syms) | set(big)}
    if spy: BASE["SPY"] = spy["base"]

    month_name = calendar.month_name[last_date.month]
    stories_cfg = reasons.get("_stories", {})
    stories = stories_cfg.get("tiles") if stories_cfg.get("month") == last_date.strftime("%Y-%m") else None
    all_mtd = sorted(r["mtd"] for r in M.values())
    meta = {
        "asof": last_date.isoformat(), "monthName": month_name, "year": last_date.year,
        "spx_mtd": spx["mtd"], "ndown": sum(1 for x in all_mtd if x < 0), "n": len(M),
        "median": all_mtd[len(all_mtd) // 2],
        "wkdates": ["Wk of " + x for x in top[0]["wkends"][:-1]] + ["This week (so far)"],
        "stories": stories or auto_stories(D, month_name),
        "storySub": stories_cfg.get("sub") if stories else "The biggest themes behind this list.",
        "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    payload = {"D": D, "BC": BC, "BIG": BIG, "M": meta, "BASE": BASE}
    with open("data.js", "w") as f:
        f.write("window.FK=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    pend = [o["s"] for o in D + BC if o["conf"] == "Pending"]
    print(f"wrote data.js: {len(D)} knives, {len(BC)} falling blue chips; pending reasons: {pend}")


if __name__ == "__main__":
    main()
