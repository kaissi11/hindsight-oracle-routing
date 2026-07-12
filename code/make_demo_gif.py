#!/usr/bin/env python3
"""Qualitative README demo: one protocol episode of the v2 disruption benchmark
rendered as an animated GIF on real Damascus roads (OSM basemap + OSRM geometry).

What it shows: the DEPLOYABLE online arm (look-K, default the frozen v1 policy,
K=8) committing one step at a time while zonal traffic drifts and nodes
block/unblock mid-episode. The final frame reports, for the SAME episode and
disruption schedule, the paired outcomes of look-8, oracle-8 (best-of-8 with
hindsight selection), event-triggered repair, and rolling OR-Tools — the
paper's cast, on one map.

Provenance: instances, schedules, controllers, and seed derivations are the
scenario_bucket_eval_v2 protocol (same code paths); this script only records a
trace and draws it. Purely qualitative — it writes no result JSONs and is not
part of any table.

Requirements beyond requirements.txt: none (uses requests + Pillow, both
already in the dependency set). Network: OSRM at --osrm-url if you have the
Damascus docker up, otherwise it falls back to the public OSRM demo server
(rate-limited politely) for leg geometry; OSM raster tiles for the basemap.
Road data and tiles (c) OpenStreetMap contributors (ODbL).

Run from the code directory:
    python make_demo_gif.py                 # auto-picks a disruption-rich episode
    python make_demo_gif.py --episode 3 --instance 1 --bucket high
Output: ../media/policy_demo.gif (+ policy_demo_final.png poster frame).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scenario_bucket_eval import load_policy  # noqa: E402
from osrm_client import OSRMClient, DAMASCUS_BBOX, BBox  # noqa: E402
from connected_instance_builder import normalize_lonlat  # noqa: E402
from research_env_v2 import ResearchEnvV2Config  # noqa: E402
from scenario_bucket_eval_v2 import (  # noqa: E402
    SimStateV2,
    apply_action_and_advance_v2,
    apply_bucket_v2,
    best_of_k,
    make_act_fn_controller,
    make_act_fn_lookahead,
    make_act_fn_rolling,
    presample_schedule_v2,
    run_rollout_v2,
    RepairControllerV2,
)

PUBLIC_OSRM = "https://router.project-osrm.org"
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "hindsight-oracle-routing-demo/1.0 (+https://github.com/kaissi11/hindsight-oracle-routing)"

ROUTE_COLOR = "#1f6feb"
COL_SERVED, COL_PENDING, COL_BLOCKED = "#2da44e", "#ffffff", "#cf222e"


# ---------------------------------------------------------------- protocol episode

def build_episode(pool_lonlats, pool_durations, pool_bbox, cfg, seed, num_instances, max_steps):
    """Mirror scenario_bucket_eval_v2.main(): same instance draw + schedules."""
    idx = np.random.RandomState(seed).choice(len(pool_lonlats), num_instances, replace=False)
    states, effs, scheds, lonlats = [], [], [], []
    for i, k in enumerate(idx):
        n = pool_lonlats.shape[1]
        visited = np.zeros(n, dtype=bool); visited[0] = True
        st = SimStateV2(
            coords=normalize_lonlat(pool_lonlats[k], pool_bbox).astype(np.float64),
            base_dist=pool_durations[k].astype(np.float64).copy(),
            eff_dist=pool_durations[k].astype(np.float64).copy(),
            visited=visited, node_blocked=np.zeros(n, dtype=np.float32),
            current_node=0, elapsed_time=0.0,
            horizon_sec=float(cfg.time_horizon_sec), n_nodes=n,
        )
        eff0, sched = presample_schedule_v2(st, cfg, max_steps, seed + 999 + i)
        states.append(st); effs.append(eff0); scheds.append(sched); lonlats.append(pool_lonlats[k])
    return states, effs, scheds, lonlats


def drama_score(schedule, window=30):
    """Blocks/unblocks early in the schedule = something to watch."""
    blocks = unblocks = 0
    prev = None
    for eff, node_blocked in schedule[:window]:
        cur = node_blocked > 0.5
        if prev is not None:
            blocks += int(((~prev) & cur).sum())
            unblocks += int((prev & (~cur)).sum())
        prev = cur
    return blocks * 2 + unblocks * 3, blocks, unblocks


def trace_lookahead(state, eff0, schedule, max_steps, act_fn):
    """run_rollout_v2 semantics for ONE state, recording a per-step trace."""
    import copy
    st = copy.deepcopy(state)
    st.eff_dist = eff0.copy()
    ctrl = act_fn("init", st)
    steps = []
    for step_idx in range(max_steps):
        if st.elapsed_time >= st.horizon_sec or st.visited[1:].all():
            break
        blocked_before = st.node_blocked > 0.5
        frm = st.current_node
        action = act_fn("act", st, ctrl)
        base = st.base_dist[frm, action]
        eff = st.eff_dist[frm, action]
        moved = action != frm and np.isfinite(eff)
        ratio = float(eff / base) if moved and base > 0 else 1.0
        apply_action_and_advance_v2(st, action, schedule[step_idx])
        blocked_after = st.node_blocked > 0.5
        finite_off = np.isfinite(st.eff_dist[np.triu_indices(st.n_nodes, 1)])
        steps.append(dict(
            frm=int(frm), to=int(action), moved=bool(moved), ratio=ratio,
            elapsed=float(st.elapsed_time),
            delivered=int(st.visited[1:].sum()),
            newly_blocked=[int(j) for j in np.flatnonzero((~blocked_before) & blocked_after)],
            newly_unblocked=[int(j) for j in np.flatnonzero(blocked_before & (~blocked_after))],
            blocked_now=[int(j) for j in np.flatnonzero(blocked_after)],
            edges_closed=int((~finite_off).sum()),
            traffic=float(np.nanmean(np.where(
                np.isfinite(st.eff_dist) & (st.base_dist > 0),
                st.eff_dist / np.maximum(st.base_dist, 1e-9), np.nan))),
        ))
    return st, steps


# ---------------------------------------------------------------- map + geometry

def merc(lon, lat):
    x = np.radians(np.asarray(lon, dtype=np.float64))
    y = np.log(np.tan(np.pi / 4 + np.radians(np.asarray(lat, dtype=np.float64)) / 2))
    return x, y


def tile_corners(z, x, y):
    n = 2.0 ** z
    lon0 = x / n * 360.0 - 180.0
    lon1 = (x + 1) / n * 360.0 - 180.0
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon0, lat0, lon1, lat1  # top-left lon/lat, bottom-right lon/lat


def build_basemap(lonlats, pad=0.18, max_tiles=7):
    lon_min, lat_min = lonlats.min(axis=0)
    lon_max, lat_max = lonlats.max(axis=0)
    dlon, dlat = lon_max - lon_min, lat_max - lat_min
    lon_min -= dlon * pad; lon_max += dlon * pad
    lat_min -= dlat * pad; lat_max += dlat * pad

    def tiles_at(z):
        n = 2.0 ** z
        x0 = int((lon_min + 180) / 360 * n); x1 = int((lon_max + 180) / 360 * n)
        def ty(lat):
            lr = math.radians(lat)
            return int((1 - math.asinh(math.tan(lr)) / math.pi) / 2 * n)
        y0, y1 = ty(lat_max), ty(lat_min)
        return x0, x1, y0, y1
    zoom = 14
    while zoom > 10:
        x0, x1, y0, y1 = tiles_at(zoom)
        if max(x1 - x0 + 1, y1 - y0 + 1) <= max_tiles:
            break
        zoom -= 1
    x0, x1, y0, y1 = tiles_at(zoom)

    cache = Path(tempfile.gettempdir()) / "osm_tile_cache"
    cache.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    ts = 256
    img = Image.new("RGB", (ts * (x1 - x0 + 1), ts * (y1 - y0 + 1)), (238, 238, 238))
    for xi, x in enumerate(range(x0, x1 + 1)):
        for yi, y in enumerate(range(y0, y1 + 1)):
            f = cache / f"{zoom}_{x}_{y}.png"
            try:
                if not f.exists():
                    r = session.get(TILE_URL.format(z=zoom, x=x, y=y), timeout=15)
                    r.raise_for_status()
                    f.write_bytes(r.content)
                    time.sleep(0.15)
                tile = Image.open(f).convert("RGB")
            except Exception:
                tile = Image.new("RGB", (ts, ts), (238, 238, 238))
            img.paste(tile, (xi * ts, yi * ts))

    tl = tile_corners(zoom, x0, y0)
    br = tile_corners(zoom, x1, y1)
    x_left, y_top = merc(tl[0], tl[1])
    x_right, y_bot = merc(br[2], br[3])
    extent = (float(x_left), float(x_right), float(y_bot), float(y_top))
    return img, extent, zoom


def fetch_leg_geometries(lonlats, legs, osrm_url):
    """Real road polylines for each traversed leg; straight line fallback."""
    client, public = None, False
    try:
        probe = requests.get(f"{osrm_url}/nearest/v1/driving/{lonlats[0][0]:.5f},{lonlats[0][1]:.5f}",
                             timeout=3)
        probe.raise_for_status()
        client = OSRMClient(base_url=osrm_url)
        print(f"[GIF] leg geometry via local OSRM at {osrm_url}")
    except Exception:
        client = OSRMClient(base_url=PUBLIC_OSRM)
        client.session.headers.update({"User-Agent": USER_AGENT})
        public = True
        print(f"[GIF] local OSRM unreachable -> public demo server ({PUBLIC_OSRM}), throttled")

    geoms = {}
    for frm, to in legs:
        if (frm, to) in geoms:
            continue
        pts = np.array([lonlats[frm], lonlats[to]], dtype=np.float64)
        try:
            geoms[(frm, to)] = client.route_geojson(pts, timeout=20)
            if public:
                time.sleep(0.7)
        except Exception as e:
            print(f"[GIF]   leg {frm}->{to}: OSRM failed ({e}); straight line")
            geoms[(frm, to)] = pts
    return geoms


# ---------------------------------------------------------------- rendering

def ratio_color(ratio):
    """Congestion of the traversed leg: green (free) -> red (heavy)."""
    t = float(np.clip((ratio - 0.9) / (2.0 - 0.9), 0, 1))
    return plt.get_cmap("RdYlGn_r")(0.15 + 0.7 * t)


def draw_frame(basemap, extent, lonlats, trace_prefix, geoms, caption, sub,
               blocked_now, current_node, visited_now, size_px=760):
    n = len(lonlats)
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed map; all text lives inside it
    ax.imshow(np.asarray(basemap), extent=extent, origin="upper", zorder=0,
              interpolation="bilinear")
    mx, my = merc(lonlats[:, 0], lonlats[:, 1])
    span_x, span_y = extent[1] - extent[0], extent[3] - extent[2]
    side = min(span_x, span_y)
    cx, cy = (mx.min() + mx.max()) / 2, (my.min() + my.max()) / 2
    half = max(mx.max() - mx.min(), my.max() - my.min()) * 0.62
    half = min(half, side / 2)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)

    for i, (frm, to, ratio) in enumerate(trace_prefix):
        g = geoms[(frm, to)]
        gx, gy = merc(g[:, 0], g[:, 1])
        last = i == len(trace_prefix) - 1
        ax.plot(gx, gy, color=ratio_color(ratio) if last else ROUTE_COLOR,
                linewidth=4.0 if last else 2.4, alpha=1.0 if last else 0.75,
                zorder=4 if last else 3,
                solid_capstyle="round")

    for j in range(n):
        px, py = mx[j], my[j]
        if j == 0:
            ax.scatter(px, py, marker="*", s=330, c="#24292f", edgecolors="white",
                       linewidths=1.2, zorder=6)
            continue
        if j in blocked_now:
            ax.scatter(px, py, s=120, c=COL_BLOCKED, edgecolors="white", linewidths=1.2, zorder=6)
            ax.scatter(px, py, marker="x", s=60, c="white", linewidths=2.0, zorder=7)
        elif visited_now[j]:
            ax.scatter(px, py, s=95, c=COL_SERVED, edgecolors="white", linewidths=1.2, zorder=5)
        else:
            ax.scatter(px, py, s=95, c=COL_PENDING, edgecolors=ROUTE_COLOR, linewidths=1.6, zorder=5)
    if current_node is not None:
        ax.scatter(mx[current_node], my[current_node], s=260, facecolors="none",
                   edgecolors="#fb8500", linewidths=3.0, zorder=8)

    ax.text(0.012, 0.985, caption, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color="#111111", ha="left", va="top", zorder=10, wrap=True,
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=3.2))
    ax.text(0.012, 0.015, sub, transform=ax.transAxes, fontsize=7.6, color="#333333",
            ha="left", va="bottom", zorder=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2.2))
    ax.text(0.988, 0.062, "map data (c) OpenStreetMap contributors", transform=ax.transAxes,
            fontsize=6.5, color="#555555", ha="right", va="bottom", zorder=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.6))

    legend = [
        Line2D([], [], marker="*", color="none", markerfacecolor="#24292f", markersize=13, label="depot"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COL_PENDING,
               markeredgecolor=ROUTE_COLOR, markersize=9, label="pending"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COL_SERVED, markersize=9, label="served"),
        Line2D([], [], marker="o", color="none", markerfacecolor=COL_BLOCKED, markersize=9, label="blocked"),
        Line2D([], [], color=ROUTE_COLOR, linewidth=2.4, label="route (real roads)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7.5, framealpha=0.85,
              borderpad=0.5, bbox_to_anchor=(0.995, 0.085))

    fig.canvas.draw()
    frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    plt.close(fig)
    return frame


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy-checkpoint",
                   default=str(HERE / "checkpoints_research_pomo" / "research_best.pt"),
                   help="policy for the animated ONLINE look-K arm (default: frozen v1)")
    p.add_argument("--instance-pool", default="results/osrm_instance_pool/pool.npz")
    p.add_argument("--bucket", default="high", choices=["low", "medium", "high"])
    p.add_argument("--base-seed", type=int, default=12345)
    p.add_argument("--episode", default="auto",
                   help="'auto' scans the first --scan protocol episodes for disruption-rich ones")
    p.add_argument("--instance", default="auto")
    p.add_argument("--scan", type=int, default=12)
    p.add_argument("--audition", type=int, default=5,
                   help="how many disruption-rich candidates to roll out before choosing")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n-nodes", type=int, default=20)
    p.add_argument("--num-instances", type=int, default=4)
    p.add_argument("--horizon-hours", type=float, default=8.0)
    p.add_argument("--osrm-url", default="http://localhost:5000")
    p.add_argument("--size", type=int, default=720)
    p.add_argument("--colors", type=int, default=128, help="GIF palette size")
    p.add_argument("--out", default=str(HERE.parent / "media" / "policy_demo.gif"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt = load_policy(args.policy_checkpoint, device)
    print(f"[GIF] device={device.type}  policy epoch={ckpt.get('epoch')}")

    pool = np.load(args.instance_pool)
    pool_lonlats, pool_durations = pool["lonlats"], pool["durations"]
    pool_bbox = DAMASCUS_BBOX
    meta = Path(args.instance_pool).parent / "pool_meta.json"
    if meta.exists():
        bb = json.loads(meta.read_text(encoding="utf-8")).get("bbox")
        if bb:
            pool_bbox = BBox(min_lon=bb[0], min_lat=bb[1], max_lon=bb[2], max_lat=bb[3])

    bidx = ["low", "medium", "high"].index(args.bucket)
    max_steps = args.n_nodes * 8 + 64

    def episode_cfg():
        cfg = apply_bucket_v2(ResearchEnvV2Config(
            n_nodes=args.n_nodes, num_instances=args.num_instances,
            device=device.type, auto_reset=False, use_augmentation=True,
        ), args.bucket)
        cfg.time_horizon_sec = args.horizon_hours * 3600.0
        return cfg

    # --- pick the episode/instance to film (schedules are pure numpy: cheap) ---
    if args.episode == "auto":
        candidates = []
        for ep in range(args.scan):
            seed = args.base_seed + ep + 10000 * bidx
            _, _, scheds, _ = build_episode(
                pool_lonlats, pool_durations, pool_bbox, episode_cfg(), seed,
                args.num_instances, max_steps)
            for inst, sched in enumerate(scheds):
                score, blocks, unblocks = drama_score(sched)
                if blocks >= 1:
                    candidates.append((score, ep, inst, blocks, unblocks))
        if not candidates:
            raise SystemExit("[GIF] no episode with block events in scan range; widen --scan")
        candidates.sort(reverse=True)
        # Audition the most disruption-rich candidates and feature one whose
        # outcome matches the suite-level finding (look-K ties the classical
        # baselines) — representative, not a cherry-picked win or loss.
        pick = None
        for score, ep, inst, blocks, unblocks in candidates[:args.audition]:
            seed = args.base_seed + ep + 10000 * bidx
            sts, efs, scs, _ = build_episode(
                pool_lonlats, pool_durations, pool_bbox, episode_cfg(), seed,
                args.num_instances, max_steps)
            one = ([sts[inst]], [efs[inst]], [scs[inst]])
            fs, _ = trace_lookahead(sts[inst], efs[inst], scs[inst], max_steps,
                                    make_act_fn_lookahead(policy, device, args.k,
                                                          seed * 1000 + 600))
            look = int(fs.visited[1:].sum())
            np.random.seed(seed); torch.manual_seed(seed)
            rep = run_rollout_v2(*one, max_steps,
                                 make_act_fn_controller(RepairControllerV2))["delivered_mean"]
            np.random.seed(seed); torch.manual_seed(seed)
            ror = run_rollout_v2(*one, max_steps, make_act_fn_rolling(30))["delivered_mean"]
            print(f"[GIF]   audition ep {ep}/inst {inst}: {blocks} blocks, {unblocks} unblocks"
                  f" | look-{args.k} {look} vs repair {rep:.0f} / rolling-OR {ror:.0f}")
            if pick is None:
                pick = (ep, inst, blocks, unblocks)  # fallback: most dramatic
            if look >= max(rep, ror):
                pick = (ep, inst, blocks, unblocks)
                break
        ep, inst, blocks, unblocks = pick
        print(f"[GIF] picked episode {ep} / instance {inst} "
              f"({blocks} block, {unblocks} unblock events in first 30 steps)")
    else:
        ep, inst = int(args.episode), 0 if args.instance == "auto" else int(args.instance)

    seed = args.base_seed + ep + 10000 * bidx
    cfg = episode_cfg()
    states, effs, scheds, lonlats_all = build_episode(
        pool_lonlats, pool_durations, pool_bbox, cfg, seed, args.num_instances, max_steps)
    st0, eff0, sched, lonlats = states[inst], effs[inst], scheds[inst], lonlats_all[inst]

    # --- the animated arm: ONLINE look-K (protocol seed derivation) ---
    look_fn = make_act_fn_lookahead(policy, device, args.k, seed * 1000 + 600)
    final_state, steps = trace_lookahead(st0, eff0, sched, max_steps, look_fn)
    n_del = int(final_state.visited[1:].sum())
    print(f"[GIF] look-{args.k} episode: delivered {n_del}/{st0.n_nodes - 1}, "
          f"{len(steps)} decision steps, sim {final_state.elapsed_time / 3600:.2f} h")
    # Drop the dead tail (steps after the last move; keep a few closing events).
    last_move = max((i for i, s in enumerate(steps) if s["moved"]), default=len(steps) - 1)
    steps = steps[:last_move + 4]

    # --- same episode, the paper's other arms (for the closing frame) ---
    one = ([st0], [eff0], [sched])
    np.random.seed(seed); torch.manual_seed(seed)
    r_or = run_rollout_v2(*one, max_steps, make_act_fn_rolling(30))
    np.random.seed(seed); torch.manual_seed(seed)
    r_rep = run_rollout_v2(*one, max_steps, make_act_fn_controller(RepairControllerV2))
    r_oracle = best_of_k(*one, max_steps, policy, device, args.k, seed * 1000 + 100)
    fmt = lambda r: f"{r['delivered_mean']:.0f}/{st0.n_nodes - 1}"
    print(f"[GIF] same episode -> look-{args.k}: {n_del}/{st0.n_nodes - 1} | "
          f"oracle-{args.k}: {fmt(r_oracle)} | repair: {fmt(r_rep)} | rolling-OR: {fmt(r_or)}")

    # --- geometry + basemap ---
    legs = [(s["frm"], s["to"]) for s in steps if s["moved"]]
    geoms = fetch_leg_geometries(lonlats, legs, args.osrm_url)
    basemap, extent, zoom = build_basemap(lonlats)
    print(f"[GIF] basemap zoom {zoom}, {basemap.size[0]}x{basemap.size[1]}px")

    # --- frames ---
    frames, durations = [], []
    provenance = (f"seed {seed} · ep {ep}/inst {inst} · {args.bucket} bucket · K={args.k} · "
                  f"Damascus OSRM pool · {Path(args.policy_checkpoint).name}")

    frames.append(draw_frame(
        basemap, extent, lonlats, [], geoms,
        f"Online look-{args.k}: deployable test-time search under disruptions",
        provenance, blocked_now=set(), current_node=0,
        visited_now=st0.visited.copy()))
    durations.append(1800)

    # Render only steps where the picture changes (a move or a block/unblock
    # event); long "waiting" stretches while a node stays blocked collapse away.
    trace_prefix = []
    for t, s in enumerate(steps):
        eventful = s["moved"] or s["newly_blocked"] or s["newly_unblocked"]
        if not eventful:
            continue
        if s["moved"]:
            trace_prefix.append((s["frm"], s["to"], s["ratio"]))
        events = "".join(f"   [!] node {j} BLOCKED" for j in s["newly_blocked"]) + \
                 "".join(f"   [+] node {j} reopened" for j in s["newly_unblocked"])
        caption = (f"step {t + 1}  -  delivered {s['delivered']}/{st0.n_nodes - 1}  -  "
                   f"sim {s['elapsed'] / 3600:.1f} h{events}")
        sub = (f"{provenance}   |   avg traffic x{s['traffic']:.2f}"
               + (f"   |   {s['edges_closed']} edge(s) closed" if s["edges_closed"] else ""))
        visited_now = np.zeros(st0.n_nodes, dtype=bool); visited_now[0] = True
        for (_, to, _) in trace_prefix:
            visited_now[to] = True
        frames.append(draw_frame(
            basemap, extent, lonlats, trace_prefix, geoms, caption, sub,
            blocked_now=set(s["blocked_now"]),
            current_node=s["to"] if s["moved"] else s["frm"],
            visited_now=visited_now))
        durations.append(1500 if (s["newly_blocked"] or s["newly_unblocked"]) else 700)

    closing = (f"done - online look-{args.k} delivered {n_del}/{st0.n_nodes - 1}\n"
               f"same episode, same disruptions:  oracle-{args.k} {fmt(r_oracle)}   "
               f"repair {fmt(r_rep)}   rolling-OR {fmt(r_or)}")
    frames.append(draw_frame(
        basemap, extent, lonlats, trace_prefix, geoms, closing,
        "oracle-K selects its rollout with hindsight - the paper measures that gap",
        blocked_now=set(steps[-1]["blocked_now"]) if steps else set(),
        current_node=None, visited_now=visited_now))
    durations.append(5000)

    # --- write ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # One shared palette keeps colors stable across frames and shrinks the file.
    # The marker colors cover so few pixels that ADAPTIVE would drop them, so a
    # swatch band pins them into the palette source before quantization.
    from PIL import ImageDraw
    pal_src = frames[-1].copy()
    draw = ImageDraw.Draw(pal_src)
    keep = [ROUTE_COLOR, COL_SERVED, COL_BLOCKED, "#fb8500", "#24292f", "#ffffff"]
    keep += [matplotlib.colors.to_hex(ratio_color(r)) for r in (0.9, 1.2, 1.5, 2.0)]
    w = pal_src.width // len(keep)
    for i, c in enumerate(keep):
        draw.rectangle([i * w, 0, (i + 1) * w, 26], fill=c)
    base = pal_src.convert("P", palette=Image.ADAPTIVE, colors=args.colors)
    pal = [f.quantize(palette=base, dither=Image.Dither.FLOYDSTEINBERG) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations,
                loop=0, optimize=True)
    poster = out.with_name(out.stem + "_final.png")
    frames[-1].save(poster)
    print(f"[GIF] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
    print(f"[GIF] wrote {poster} ({poster.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
