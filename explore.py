"""
Game State Effect - Cycle 9, Day 3 project
Does a team really change how it plays when it leads or trails? Every event
in Copa America 2024 is tagged with the score at that moment, from each
team's point of view: leading, level or trailing. Then each team is compared
with ITSELF when the score was level, so a strong team that leads often is
not mistaken for a "leading style".

Method:
  1. Periods 1 and 2 only (90 minutes plus stoppage, extra time and shootouts
     are left out). Match time is continuous: second-half time starts where
     the first half ended.
  2. Score state = goals for minus goals against before each event. Goals are
     shots with outcome Goal (penalties in play included) plus own goals.
  3. Time in a state = the gap to the next event (capped at 60 seconds), so
     minutes are approximate but the same for numerator and denominator.
  4. Six measures per team and state:
       share of passes (a possession proxy: team passes / both teams' passes)
       passes per minute (tempo)
       average pass position, yards from the team's own goal line
       average position of defensive actions (pressure, duel, interception,
         block, clearance, ball recovery, foul), the height of its defending
       non-penalty xG for and against per 10 minutes
  5. For every team with at least 15 minutes both level and in the state,
     change = value in the state minus value when level. The tournament
     result is the mean change across teams with a 5th to 95th percentile
     range from 5,000 bootstrap resamples of the teams.
  6. A second chart repeats this by exact goal difference (down 2 or more to
     up 2 or more), using teams with at least 8 minutes in each.

Limitations: only 32 matches, so many teams have little time trailing and the
ranges are wide. Effects mix tactics with fatigue, substitutions and red
cards, none of which are separated here. Passes stand in for possession time.
Level includes early 0-0 minutes, when teams are cautious. The score is
tagged event by event, so very fast goals can shift a few seconds of state.

Outputs: game_state_effect.png, game_state_by_goal_difference.png,
game_state_replay.mp4 (match replay plus the two charts as end cards),
game_state_metrics.csv, game_state_deltas.csv

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

import os
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Circle
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

FOCUS_TEAM = "Argentina"
MIN_MIN = 15
MIN_MIN_BIN = 8
N_BOOT = 5000
SEED = 11
DOT_WINDOW = 180
LINE_WINDOW = 300
N_FRAMES = 300
HOLD_FRAMES = 30
END_CARDS = ["game_state_effect.png", "game_state_by_goal_difference.png"]
END_CARD_SECONDS = 4

BG = "#0d1b2a"
GRID = "#3a4a5a"
BLUE = "#4a90d9"
ORANGE = "#f4a300"
MUTED = "#a8b8c8"
GREEN = "#6fcf97"
RED = "#e76f51"

DEF_TYPES = ["Pressure", "Duel", "Interception", "Block", "Clearance", "Ball Recovery", "Foul Committed"]
METRICS = [("pass_share", "Share of passes (%)"),
           ("tempo", "Passes per minute"),
           ("pass_x", "Pass position (yards from own goal)"),
           ("def_x", "Defending height (yards from own goal)"),
           ("xg_for", "xG for per 10 min"),
           ("xg_against", "xG against per 10 min")]

matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)
for col in ["shot_type", "shot_outcome", "shot_statsbomb_xg", "location", "minute", "second", "index"]:
    if col not in events.columns:
        events[col] = np.nan

xy = events["location"].astype(str).str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")
events["lx"] = pd.to_numeric(xy[0], errors="coerce")
events["ly"] = pd.to_numeric(xy[1], errors="coerce")

# ---------- match clock and score state ----------
ev = events[events["period"].isin([1, 2])].copy()
ev["minute"] = ev["minute"].fillna(0)
ev["second"] = ev["second"].fillna(0)
ev["t"] = ev["minute"] * 60 + ev["second"]
p1_len = ev[ev["period"] == 1].groupby("match_id")["t"].max()
ev["t"] = np.where(ev["period"] == 2,
                   ev["match_id"].map(p1_len).fillna(2700) + (ev["t"] - 2700), ev["t"])
sort_cols = ["match_id", "index"] if ev["index"].notna().any() else ["match_id", "t"]
ev = ev.sort_values(sort_cols).reset_index(drop=True)
nxt = ev.groupby(["match_id", "period"])["t"].shift(-1)
ev["dt"] = (nxt - ev["t"]).clip(0, 60).fillna(0)

home_of = matches.set_index("match_id")["home_team"]
away_of = matches.set_index("match_id")["away_team"]
ev["home"] = ev["match_id"].map(home_of)
ev["away"] = ev["match_id"].map(away_of)
ev = ev[ev["home"].notna()].reset_index(drop=True)

is_goal = ((ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")) | (ev["type"] == "Own Goal For")
ev["g_h"] = (is_goal & (ev["team"] == ev["home"])).astype(int)
ev["g_a"] = (is_goal & (ev["team"] == ev["away"])).astype(int)
cum_h = ev.groupby("match_id")["g_h"].cumsum() - ev["g_h"]
cum_a = ev.groupby("match_id")["g_a"].cumsum() - ev["g_a"]
ev["gd_home"] = cum_h - cum_a
ev["xg_np"] = np.where((ev["type"] == "Shot") & (ev["shot_type"].fillna("") != "Penalty"),
                       ev["shot_statsbomb_xg"].fillna(0.0), 0.0)

# sanity check against the official scores (extra time goals will differ)
tot = ev.groupby("match_id")[["g_h", "g_a"]].sum()
chk = matches.set_index("match_id")[["home_score", "away_score"]].join(tot, how="inner")
bad = chk[(chk["home_score"] != chk["g_h"]) | (chk["away_score"] != chk["g_a"])]
print(f"\nScore check: {len(chk) - len(bad)} of {len(chk)} matches match the official score "
      f"({len(bad)} differ, expected for extra-time matches)")

# one copy of every event per team's point of view
keep = ["match_id", "team", "type", "lx", "xg_np", "dt"]
parts = []
for side in ("home", "away"):
    f = ev[keep].copy()
    f["persp"] = ev["home"] if side == "home" else ev["away"]
    gd = ev["gd_home"] if side == "home" else -ev["gd_home"]
    f["gd"] = gd.clip(-2, 2).astype(int)
    f["state"] = np.sign(gd).astype(int)
    f["own"] = f["team"] == f["persp"]
    parts.append(f)
pv = pd.concat(parts, ignore_index=True)


# ---------- metrics per team and state ----------
def team_metrics(df, key):
    idx = ["persp", key]
    own = df[df["own"]]
    opp = df[~df["own"]]
    tm = pd.DataFrame({"minutes": df.groupby(idx)["dt"].sum() / 60})
    own_p = own[own["type"] == "Pass"]
    opp_p = opp[opp["type"] == "Pass"]
    pf = own_p.groupby(idx).size().reindex(tm.index, fill_value=0)
    pa = opp_p.groupby(idx).size().reindex(tm.index, fill_value=0)
    mins = tm["minutes"].replace(0, np.nan)
    tm["pass_share"] = pf / (pf + pa).replace(0, np.nan) * 100
    tm["tempo"] = pf / mins
    tm["pass_x"] = own_p.groupby(idx)["lx"].mean().reindex(tm.index)
    tm["def_x"] = own[own["type"].isin(DEF_TYPES)].groupby(idx)["lx"].mean().reindex(tm.index)
    tm["xg_for"] = own.groupby(idx)["xg_np"].sum().reindex(tm.index, fill_value=0) / mins * 10
    tm["xg_against"] = opp.groupby(idx)["xg_np"].sum().reindex(tm.index, fill_value=0) / mins * 10
    return tm


def deltas(tm, key, levels, min_min):
    tm = tm[tm["minutes"] >= min_min]
    rng = np.random.default_rng(SEED)
    base = tm.xs(0, level=key)
    out = []
    for lvl in levels:
        if lvl not in tm.index.get_level_values(key):
            continue
        cur = tm.xs(lvl, level=key)
        common = cur.index.intersection(base.index)
        for mk, _ in METRICS:
            d = (cur.loc[common, mk] - base.loc[common, mk]).dropna().to_numpy()
            n = len(d)
            if n == 0:
                continue
            if n >= 3:
                bm = rng.choice(d, (N_BOOT, n)).mean(axis=1)
                lo, hi = np.percentile(bm, [5, 95])
            else:
                lo = hi = np.nan
            out.append(dict(level=lvl, metric=mk, n_teams=n, mean=float(d.mean()),
                            lo=float(lo), hi=float(hi), share_positive=float((d > 0).mean())))
    return pd.DataFrame(out)


tm_state = team_metrics(pv, "state")
tm_gd = team_metrics(pv, "gd")
d_state = deltas(tm_state, "state", [1, -1], MIN_MIN)
d_gd = deltas(tm_gd, "gd", [-2, -1, 1, 2], MIN_MIN_BIN)

tm_state.reset_index().assign(table="state").to_csv("game_state_metrics.csv", index=False)
pd.concat([d_state.assign(table="state"), d_gd.assign(table="goal_difference")]).to_csv(
    "game_state_deltas.csv", index=False)

state_name = {1: "Leading", 0: "Level", -1: "Trailing"}
tot_min = pv.groupby("state")["dt"].sum() / 60
print("\nTeam-minutes per state (all teams, all matches):")
for s in (1, 0, -1):
    print(f"  {state_name[s]:<9}{tot_min.get(s, 0):8.0f} minutes")

label = dict(METRICS)
print("\nChange vs the same team when level (mean across teams, 5th to 95th percentile):")
for lvl in (1, -1):
    print(f"\n  {state_name[lvl]}:")
    for mk, _ in METRICS:
        r = d_state[(d_state["level"] == lvl) & (d_state["metric"] == mk)]
        if r.empty:
            continue
        r = r.iloc[0]
        print(f"    {label[mk]:<42} {r['mean']:+7.2f}  ({r['lo']:+.2f} to {r['hi']:+.2f})  "
              f"n={int(r['n_teams'])}  teams higher: {r['share_positive']:.0%}")

print(f"\n{FOCUS_TEAM} by state:")
if FOCUS_TEAM in tm_state.index.get_level_values("persp"):
    print(tm_state.loc[FOCUS_TEAM].round(2).rename(index=state_name).to_string())

# ---------- chart 1: change by state ----------
fig1, axes = plt.subplots(2, 3, figsize=(11, 8))
fig1.patch.set_facecolor(BG)
for ax, (mk, title) in zip(axes.ravel(), METRICS):
    ax.set_facecolor(BG)
    vals = []
    for xi, (lvl, col, nm) in enumerate(((1, GREEN, "Leading"), (-1, RED, "Trailing"))):
        r = d_state[(d_state["level"] == lvl) & (d_state["metric"] == mk)]
        if r.empty:
            continue
        r = r.iloc[0]
        lo = r["lo"] if np.isfinite(r["lo"]) else r["mean"]
        hi = r["hi"] if np.isfinite(r["hi"]) else r["mean"]
        ax.bar(xi, r["mean"], width=0.6, color=col)
        ax.errorbar(xi, r["mean"], yerr=[[r["mean"] - lo], [hi - r["mean"]]], fmt="none",
                    ecolor="white", capsize=4, linewidth=1.3)
        vals += [lo, hi, r["mean"], 0.0]
    if not vals:
        ax.axis("off")
        continue
    span = max(max(vals) - min(vals), 1e-6)
    ymin, ymax = min(vals) - span * 0.22, max(vals) + span * 0.22
    ax.set_ylim(ymin, ymax)
    for xi, lvl in enumerate((1, -1)):
        r = d_state[(d_state["level"] == lvl) & (d_state["metric"] == mk)]
        if r.empty:
            continue
        r = r.iloc[0]
        lo = r["lo"] if np.isfinite(r["lo"]) else r["mean"]
        hi = r["hi"] if np.isfinite(r["hi"]) else r["mean"]
        yy = hi + span * 0.04 if r["mean"] >= 0 else lo - span * 0.04
        ax.text(xi, yy, f"{r['mean']:+.2f}", ha="center", va="bottom" if r["mean"] >= 0 else "top",
                color="white", fontsize=10.5)
    ns = [d_state[(d_state["level"] == l) & (d_state["metric"] == mk)]["n_teams"] for l in (1, -1)]
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Leading\n(n={int(ns[0].iloc[0]) if len(ns[0]) else 0})",
                        f"Trailing\n(n={int(ns[1].iloc[0]) if len(ns[1]) else 0})"],
                       color="white", fontsize=9.5)
    ax.axhline(0, color="white", linewidth=0.9)
    ax.set_title(title, color="white", fontsize=10.5, pad=6)
    ax.tick_params(axis="y", colors="white", labelsize=9)
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
fig1.suptitle("How Teams Change When the Score Changes", color="white", fontsize=17,
              fontweight="bold", y=0.98)
fig1.text(0.5, 0.925, "Change versus the same team when the score was level. Copa America 2024, first 90 minutes",
          ha="center", color=MUTED, fontsize=10.5)
fig1.text(0.5, 0.012, f"Bars: mean change across teams. Whiskers: 5th to 95th percentile. "
          f"Teams with {MIN_MIN}+ minutes in both states.", ha="center", color=MUTED, fontsize=9.5)
plt.tight_layout(rect=[0, 0.03, 1, 0.91])
fig1.savefig("game_state_effect.png", dpi=150, facecolor=BG)
plt.close(fig1)
print("\nSaved chart: game_state_effect.png")

# ---------- chart 2: by exact goal difference ----------
BIN_LABELS = {-2: "Down 2+", -1: "Down 1", 1: "Up 1", 2: "Up 2+"}
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 7.5))
fig2.patch.set_facecolor(BG)
for ax, (mk, title) in zip(axes2, [("def_x", "Defending height (yards from own goal)"),
                                    ("pass_share", "Share of passes (%)")]):
    ax.set_facecolor(BG)
    xs, vals, labels = [], [], []
    for xi, lvl in enumerate([-2, -1, 1, 2]):
        r = d_gd[(d_gd["level"] == lvl) & (d_gd["metric"] == mk)]
        if r.empty:
            labels.append(f"{BIN_LABELS[lvl]}\n(n=0)")
            continue
        r = r.iloc[0]
        lo = r["lo"] if np.isfinite(r["lo"]) else r["mean"]
        hi = r["hi"] if np.isfinite(r["hi"]) else r["mean"]
        ax.bar(xi, r["mean"], width=0.6, color=RED if lvl < 0 else GREEN)
        ax.errorbar(xi, r["mean"], yerr=[[r["mean"] - lo], [hi - r["mean"]]], fmt="none",
                    ecolor="white", capsize=4, linewidth=1.3)
        vals += [lo, hi, r["mean"], 0.0]
        labels.append(f"{BIN_LABELS[lvl]}\n(n={int(r['n_teams'])})")
    if vals:
        span = max(max(vals) - min(vals), 1e-6)
        ax.set_ylim(min(vals) - span * 0.22, max(vals) + span * 0.22)
        for xi, lvl in enumerate([-2, -1, 1, 2]):
            r = d_gd[(d_gd["level"] == lvl) & (d_gd["metric"] == mk)]
            if r.empty:
                continue
            r = r.iloc[0]
            lo = r["lo"] if np.isfinite(r["lo"]) else r["mean"]
            hi = r["hi"] if np.isfinite(r["hi"]) else r["mean"]
            yy = hi + span * 0.04 if r["mean"] >= 0 else lo - span * 0.04
            ax.text(xi, yy, f"{r['mean']:+.2f}", ha="center",
                    va="bottom" if r["mean"] >= 0 else "top", color="white", fontsize=11)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, color="white", fontsize=10.5)
    ax.axhline(0, color="white", linewidth=0.9)
    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.tick_params(axis="y", colors="white")
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_color(GRID)
fig2.suptitle("Does a Bigger Lead Change More?", color="white", fontsize=17, fontweight="bold", y=0.97)
fig2.text(0.5, 0.905, "Change versus the same team when level, by goal difference. Copa America 2024",
          ha="center", color=MUTED, fontsize=10.5)
fig2.text(0.5, 0.02, f"Whiskers: 5th to 95th percentile across teams. Teams with {MIN_MIN_BIN}+ minutes in the state.",
          ha="center", color=MUTED, fontsize=9.5)
plt.tight_layout(rect=[0, 0.05, 1, 0.9])
fig2.savefig("game_state_by_goal_difference.png", dpi=150, facecolor=BG)
plt.close(fig2)
print("Saved chart: game_state_by_goal_difference.png")

# ---------- choose a match to replay ----------
cand_ids = matches[(matches["home_team"] == FOCUS_TEAM) | (matches["away_team"] == FOCUS_TEAM)]["match_id"]
best_mid, best_secs = None, -1.0
for mid in cand_ids:
    sub = pv[(pv["match_id"] == mid) & (pv["persp"] == FOCUS_TEAM)]
    secs = sub[sub["state"] != 0]["dt"].sum()
    if secs > best_secs:
        best_mid, best_secs = mid, secs
if best_mid is None or best_secs <= 0:
    raise ValueError(f"No {FOCUS_TEAM} match with a lead or a deficit found to replay")
mev = ev[ev["match_id"] == best_mid].reset_index(drop=True)
home, away = mev["home"].iloc[0], mev["away"].iloc[0]
F = FOCUS_TEAM
O = away if home == F else home
t_end = float(mev["t"].max())
print(f"\nReplaying {F} vs {O}: {best_secs / 60:.0f} minutes not level")

gl = mev[(mev["g_h"] + mev["g_a"]) > 0][["t", "g_h", "g_a"]]
goals = [(float(r.t), home if r.g_h else away) for r in gl.itertuples()]
P1 = float(p1_len.get(best_mid, 2700.0))


def passes_of(team, flip):
    d = mev[(mev["team"] == team) & (mev["type"] == "Pass") & mev["lx"].notna() & mev["ly"].notna()]
    out = pd.DataFrame({"t": d["t"].to_numpy(), "x": d["lx"].to_numpy(), "y": d["ly"].to_numpy()})
    if flip:
        out["x"] = 120 - out["x"]
        out["y"] = 80 - out["y"]
    return out


pf_df = passes_of(F, False)
po_df = passes_of(O, True)
T_grid = np.linspace(0, t_end, N_FRAMES)


def roll_avg(df, T, flip):
    w = df[(df["t"] > T - LINE_WINDOW) & (df["t"] <= T)]
    if len(w) < 5:
        return np.nan
    m = float(w["x"].mean())
    return 120 - m if flip else m


f_roll = np.array([roll_avg(pf_df, T, False) for T in T_grid])
o_roll = np.array([roll_avg(po_df, T, True) for T in T_grid])
all_roll = np.concatenate([f_roll[np.isfinite(f_roll)], o_roll[np.isfinite(o_roll)]])
y_lo, y_hi = (all_roll.min() - 4, all_roll.max() + 4) if len(all_roll) else (30, 80)


def disp_min(T):
    return int(T / 60) if T <= P1 else int(45 + (T - P1) / 60)


def score_at(T):
    fs = sum(1 for g in goals if g[0] <= T and g[1] == F)
    os_ = sum(1 for g in goals if g[0] <= T and g[1] == O)
    return fs, os_


# state segments from the focal team's view
segs = []
cur_t, cur_gd = 0.0, 0
for gt, gteam in goals:
    segs.append((cur_t, gt, int(np.sign(cur_gd))))
    cur_gd += 1 if gteam == F else -1
    cur_t = gt
segs.append((cur_t, t_end, int(np.sign(cur_gd))))

tick_t = [0, 900, 1800, P1, P1 + 900, P1 + 1800, P1 + 2700]
tick_l = ["0'", "15'", "30'", "45'", "60'", "75'", "90'"]


def draw_pitch(ax):
    ax.set_facecolor("#10253a")
    kw = dict(fill=False, edgecolor="#35506b", linewidth=1.6)
    ax.add_patch(Rectangle((0, 0), 120, 80, **kw))
    ax.plot([60, 60], [0, 80], color="#35506b", linewidth=1.6)
    ax.add_patch(Circle((60, 40), 10, **kw))
    ax.add_patch(Rectangle((0, 18), 18, 44, **kw))
    ax.add_patch(Rectangle((102, 18), 18, 44, **kw))
    ax.add_patch(Rectangle((0, 30), 6, 20, **kw))
    ax.add_patch(Rectangle((114, 30), 6, 20, **kw))
    ax.set_xlim(-2, 122)
    ax.set_ylim(82, -2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


fig = plt.figure(figsize=(10, 10), dpi=108)  # 1080 x 1080 pixels
fig.patch.set_facecolor(BG)
axp = fig.add_axes([0.12, 0.33, 0.76, 0.50])
axt = fig.add_axes([0.10, 0.09, 0.82, 0.17])
fig.text(0.5, 0.965, "How the Scoreline Changes a Team's Shape", ha="center",
         color="white", fontsize=23, fontweight="bold")
fig.text(0.5, 0.928, f"{F} vs {O}, Copa America 2024", ha="center", color=MUTED, fontsize=14)
left_t = fig.text(0.47, 0.882, "", ha="right", color=BLUE, fontsize=26, fontweight="bold")
fig.text(0.5, 0.882, "-", ha="center", color="white", fontsize=26, fontweight="bold")
right_t = fig.text(0.53, 0.882, "", ha="left", color=ORANGE, fontsize=26, fontweight="bold")
state_t = fig.text(0.5, 0.845, "", ha="center", color="white", fontsize=17, fontweight="bold")
fig.text(0.5, 0.303, "Dots: passes in the last 3 minutes. Dashed lines: average pass position over the last 5 minutes.",
         ha="center", color=MUTED, fontsize=10.5)
fig.text(0.5, 0.272, f"{F} attacks right, {O} attacks left", ha="center", color=MUTED, fontsize=10.5)
fig.text(0.5, 0.012, "StatsBomb open data, Copa America 2024.", ha="center", color=MUTED, fontsize=9)


def dots(ax, df, T, col):
    w = df[(df["t"] > T - DOT_WINDOW) & (df["t"] <= T)]
    if w.empty:
        return
    age = (T - w["t"].to_numpy()) / DOT_WINDOW
    cols = [to_rgba(col, float(0.95 - 0.8 * a)) for a in age]
    ax.scatter(w["x"], w["y"], s=46, c=cols, edgecolors="none", zorder=3)


def draw(i):
    k = min(i, N_FRAMES - 1)
    T = T_grid[k]
    axp.clear()
    axt.clear()
    draw_pitch(axp)
    dots(axp, pf_df, T, BLUE)
    dots(axp, po_df, T, ORANGE)
    if np.isfinite(f_roll[k]):
        axp.axvline(f_roll[k], color=BLUE, linewidth=3, linestyle="--", zorder=4)
    if np.isfinite(o_roll[k]):
        axp.axvline(120 - o_roll[k], color=ORANGE, linewidth=3, linestyle="--", zorder=4)
    fs, os_ = score_at(T)
    left_t.set_text(f"{F} {fs}")
    right_t.set_text(f"{os_} {O}")
    diff = fs - os_
    word, col = (("leading", GREEN) if diff > 0 else (("trailing", RED) if diff < 0 else ("level", MUTED)))
    state_t.set_text(f"{disp_min(T)}'   |   {F} {word}")
    state_t.set_color(col)
    # trace of the average pass position
    axt.set_facecolor(BG)
    for (a, b, s) in segs:
        if a >= T:
            break
        if s != 0:
            axt.axvspan(a, min(b, T), color=GREEN if s > 0 else RED, alpha=0.16, linewidth=0)
    axt.plot(T_grid[: k + 1], f_roll[: k + 1], color=BLUE, linewidth=2.6)
    axt.plot(T_grid[: k + 1], o_roll[: k + 1], color=ORANGE, linewidth=2.6)
    cg_f = cg_o = 0
    for gt, gteam in goals:
        if gt <= T:
            if gteam == F:
                cg_f += 1
            else:
                cg_o += 1
            axt.axvline(gt, color="white", linewidth=1, linestyle=":")
            axt.text(gt, y_hi, f" {cg_f}-{cg_o}", color="white", fontsize=10, va="top", ha="left")
    axt.set_xlim(0, t_end)
    axt.set_ylim(y_lo, y_hi)
    axt.set_xticks(tick_t)
    axt.set_xticklabels(tick_l)
    axt.tick_params(colors="white", labelsize=9)
    for sp in axt.spines.values():
        sp.set_color(GRID)
    axt.set_ylabel("Average pass\nposition (yards)", color=MUTED, fontsize=9)


frames = list(range(N_FRAMES)) + [N_FRAMES - 1] * HOLD_FRAMES
FPS = 20
RAW_VIDEO = "game_state_replay_raw.mp4"
FINAL_VIDEO = "game_state_replay.mp4"
anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames), interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
print(f"Rendering {len(frames)} frames ({len(frames) / FPS:.0f} seconds), this takes a minute...")
anim.save(RAW_VIDEO, writer=writer, dpi=108, savefig_kwargs={"facecolor": BG})
print("Saved replay:", RAW_VIDEO)


# ---------- append the charts to the video as end cards ----------
def append_end_cards(video_in, video_out, cards, seconds, size=1080, fps=20, bg="0d1b2a"):
    cards = [c for c in cards if os.path.exists(c)]
    if not cards:
        print("No chart PNGs found, keeping the replay without end cards.")
        os.replace(video_in, video_out)
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", video_in]
    for c in cards:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", c]
    norm = (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=0x{bg},"
            f"setsar=1,fps={fps},format=yuv420p")
    n = len(cards) + 1
    parts_f = [f"[{i}:v]{norm}[v{i}]" for i in range(n)]
    labels = "".join(f"[v{i}]" for i in range(n))
    fc = ";".join(parts_f) + f";{labels}concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", video_out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        print("End cards failed, keeping the replay without them.")
        os.replace(video_in, video_out)
        return
    os.remove(video_in)
    print("Added end cards:", ", ".join(cards))


append_end_cards(RAW_VIDEO, FINAL_VIDEO, END_CARDS, END_CARD_SECONDS, fps=FPS)
print("Saved video:", FINAL_VIDEO)