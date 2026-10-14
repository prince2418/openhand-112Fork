"""Live state of the city for the dispatch app: units, incidents, recommendations, green corridors.

Time runs DEMO_SPEED times faster than the wall clock so a 6-minute drive plays in ~36 s on stage.
Travel times use the same road graph and calibration as the benchmark (scale fitted to the published 4:32).
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..demand import HOURLY_PROFILE, estimate_weights, generate, scenario_weights, tod_multiplier
from ..network import RoadNet
from ..placement import mexclp
from ..triage.repeat import RepeatTracker
from ..triage.rules import decide
from ..triage.schema import Extract

ROOT = Path(__file__).resolve().parents[3]
DEMO_SPEED = float(os.environ.get("DEMO_SPEED", "10"))
# Simulated time on scene. Compressed for the live demo (30-60 s on the 10x clock) so an incident closes on stage;
# the benchmark uses its own, longer service times.
ON_SCENE_S = {"police": 300, "ambulance": 360, "fire": 600}
CAMERA_SITES = 285      # ICCC live-monitored locations (Tribune, Aug 2024); positions here are SIMULATED
FOG_SLOWDOWN = 1.5      # winter fog: drives take 50% longer (ASSUMPTION; visibility-limited speeds)
CLOSED_S = 1e7          # travel time on a closed road link (effectively impassable)
WINDOW_BEFORE_S = 120   # a junction alert goes out this long (simulated time) before the unit gets there
WINDOW_AFTER_S = 20
TARGET_S = {"police": 300, "ambulance": 480, "fire": 300}        # response targets used in the benchmark
SCENE_WORK = {"police": "secured the scene", "ambulance": "treated and loaded the patient", "fire": "put out the fire"}
KIND_LABEL = {"police": "PCR", "ambulance": "Ambulance", "fire": "Fire tender"}
FLEET = {  # (station kind in stations.csv, units). Published / assumed as in run_benchmark.py
    "police": ("police", 31),
    "fire": ("fire", 14),
    "ambulance": ("ambulance_base", 6),
}
MAJOR = {"trunk", "primary", "secondary"}


def _speed_scale() -> float:
    p = ROOT / "results/benchmark.json"
    try:
        return float(json.loads(p.read_text())["speed_scale"])
    except Exception:  # noqa: BLE001
        return 0.912


@dataclass
class Unit:
    id: str
    kind: str
    base_name: str
    home: int
    status: str = "available"           # available | en_route | on_scene | returning
    incident: str | None = None
    route: list = field(default_factory=list)     # [[lng, lat], ...]
    route_t: list = field(default_factory=list)   # sim seconds at each route point (absolute)
    t_arrive: float = 0.0
    t_clear: float = 0.0
    t_home: float = 0.0
    t_dispatch: float = 0.0
    off_duty: bool = False                        # night-shift what-if: van not on duty
    route_m: list = field(default_factory=list)   # metres from the start at each route point


@dataclass
class Incident:
    id: str
    t: float
    call: dict
    triage: dict
    lat: float | None
    lng: float | None
    node: int | None
    status: str = "new"                 # new | dispatched | resolved
    units: list = field(default_factory=list)
    recommendation: dict | None = None
    corridor: dict | None = None
    calls: int = 1
    merged_last: bool = False
    events: list = field(default_factory=list)    # [{t, kind, text}] for the incident timeline
    outcome: dict | None = None
    t_resolved: float | None = None
    arrivals: dict = field(default_factory=dict)  # unit id -> seconds from call to on scene
    gemini: dict | None = None                    # background Gemini status when the keyword fallback answered first
    fast: str | None = None                       # words that triggered the sexual-violence fast path
    traffic: list = field(default_factory=list)   # junction alerts, one per junction with its own time window
    cctv: list = field(default_factory=list)      # camera checks: what a CCTV frame showed vs the call
    extra: dict | None = None                     # more units suggested after the first dispatch (e.g. ambulance)


def _dist_m(la1, lo1, la2, lo2) -> float:
    return float(np.hypot((la2 - la1) * 111_320, (lo2 - lo1) * 111_320 * np.cos(np.radians(la1))))


def _mmss(s: float) -> str:
    s = max(0, round(s))
    return f"{s // 60}:{s % 60:02d}"


class World:
    def __init__(self, graph=ROOT / "data/chandigarh_enriched.graphml", stations=ROOT / "data/stations.csv"):
        self.net = RoadNet.from_graphml(str(graph))
        self.scale = _speed_scale()
        self.t0_wall = time.time()
        self.lock = threading.RLock()
        self.incidents: dict[str, Incident] = {}
        self.tracker = RepeatTracker()
        self._ids = itertools.count(1)
        self.alerts: list[dict] = []
        st = pd.read_csv(stations)
        self.stations = [dict(kind=r.kind, name=r.name, lat=float(r.lat), lng=float(r.lng)) for r in st.itertuples()]
        self.units: dict[str, Unit] = {}
        for kind, (skind, n) in FLEET.items():
            rows = st[st["kind"] == skind].reset_index(drop=True)
            nodes = self.net.snap(rows["lat"].to_numpy(), rows["lng"].to_numpy())
            for i in range(n):
                j = i % len(rows)
                uid = f"{KIND_LABEL[kind].split()[0][0]}{i + 1:02d}"
                self.units[uid] = Unit(uid, kind, rows.loc[j, "name"], int(nodes[j]))
        self._standby = None
        self.hotspots = pd.read_csv(ROOT / "data/hotspots_geocoded.csv").to_dict("records")
        # what-if controls
        self.fog = False
        self.night = False
        self.closures: list[dict] = []
        self._closure_ids = itertools.count(1)
        self._A0 = self.net.A.copy()
        self._rng = np.random.default_rng(7)
        self.cameras = self._camera_sites()
        from .flood_ops import FloodOps
        self.flood = FloodOps(self)
        from .dpi import Outbox
        from .watch import Watch
        self.outbox = Outbox()
        self.demo_speed = DEMO_SPEED
        self.ROOT_HOTSPOTS = ROOT / "data/hotspots_geocoded.csv"
        self.watcher = Watch(self)
        # warm the slow parts (standby plan, forecast skill) so the first look at the Predict tab is instant
        # threading.Thread(target=self._warm, daemon=True).start()
    if os.environ.get("WARM_CACHE", "0") == "1":
    threading.Thread(target=self._warm, daemon=True).start()

    def _warm(self):
        try:
            self.standby_plan()
            self.watcher.skill()
            self.watcher._candidates()
        except Exception as e:  # noqa: BLE001
            print(f"[watch] warm-up skipped: {e}")

    # ---------- clock ----------
    def now(self) -> float:
        """Simulated seconds since the app started."""
        return (time.time() - self.t0_wall) * DEMO_SPEED

    def hour(self) -> int:
        return time.localtime().tm_hour

    def travel_scale(self) -> float:
        return self.scale * float(tod_multiplier(self.hour())) * (FOG_SLOWDOWN if self.fog else 1.0)

    # ---------- CCTV ----------
    def _camera_sites(self, n: int = CAMERA_SITES, min_gap_m: float = 160.0) -> list[dict]:
        """Chandigarh's ICCC watches about 285 locations live (2024), mostly junctions on main roads. The real
        list is not public, so we place n SIMULATED sites at major-road junctions, busiest road classes first."""
        from ..triage.location import ROADS
        und = self.net.A + self.net.A.T
        deg = np.diff(und.indptr)
        rank = {"trunk": 0, "primary": 1, "secondary": 2}
        cand = [i for i in range(self.net.n) if self.net.node_class[i] in rank and deg[i] >= 3]
        cand.sort(key=lambda i: (rank[self.net.node_class[i]], i))
        xy = self.net._xy(self.net.lat, self.net.lon)
        chosen: list[int] = []
        for i in cand:
            if all(np.hypot(*(xy[i] - xy[j])) >= min_gap_m for j in chosen[-400:]):
                chosen.append(i)
            if len(chosen) >= n:
                break
        road_pts = [(r, np.array(p)) for r, p in ROADS.items()]
        cams = []
        for k, i in enumerate(chosen):
            la, lo = float(self.net.lat[i]), float(self.net.lon[i])
            near = [(float(np.hypot((pts[:, 0] - la) * 111_320, (pts[:, 1] - lo) * 95_700).min()), r) for r, pts in road_pts if len(pts)]
            dmin, road = min(near) if near else (1e9, None)
            road = road if dmin < 80 else None
            cams.append({"id": f"CAM-{k + 1:03d}", "lat": la, "lng": lo, "road": road or f"{self.net.node_class[i]} road junction"})
        return cams

    def cameras_near(self, lat: float, lng: float, k: int = 3, max_m: float = 700) -> list[dict]:
        d = [(_dist_m(lat, lng, c["lat"], c["lng"]), c) for c in self.cameras]
        return [dict(c, m=round(m)) for m, c in sorted(d, key=lambda x: x[0])[:k] if m <= max_m]

    def camera(self, cid: str) -> dict:
        c = next((c for c in self.cameras if c["id"] == cid), None)
        if c is None:
            raise KeyError(cid)
        return c

    def apply_cctv(self, iid: str, cam_id: str, check: dict, meta: dict) -> "Incident":
        """Record what a camera frame shows and act on it. CCTV can only RAISE urgency or add units
        (a camera that sees nothing may just be pointing the wrong way); it never lowers priority."""
        with self.lock:
            inc = self.incidents[iid]
            notes, add = [], []
            d = inc.triage["decision"]
            if check["shows_incident"] == "no" or check["looks_stale_or_unrelated"]:
                notes.append("camera does not show it: possible hoax, old image or wrong place; check with the caller")
            if check["weapon_visible"] or check["violence_visible"]:
                add.append("police")
                if d["priority"] != "P1":
                    notes.append("weapon or violence visible: raised to P1")
                    d["priority"] = "P1"
                    d["reasons"] = d["reasons"] + ["CCTV: weapon or violence visible"]
            if check["fire_or_smoke"]:
                add.append("fire")
            if check["injured_visible"]:
                add.append("ambulance")
            if check["crowd_building"]:
                add.append("police")
                notes.append("crowd building: watch for crush risk")
            add = [u for u in dict.fromkeys(add) if u not in d["units"]]
            if add:
                notes.append("camera suggests adding " + ", ".join(KIND_LABEL[u] for u in add))
                if inc.status == "new":
                    d["units"] = sorted(set(d["units"]) | set(add))
                    if inc.node is not None:
                        inc.recommendation = self.recommend(inc)
                elif inc.status == "dispatched":
                    self._offer_extra(inc, add)
            inc.cctv.append({"camera": cam_id, "t": round(self.now()), "check": check, "model": meta.get("model"),
                             "latency_s": meta.get("latency_s"), "notes": notes})
            self.outbox.send("in", f"ICCC camera {cam_id}", "CCTV frame checked", "JPEG frame + camera ID (frame not stored)",
                             f"{cam_id} on {iid}: shows incident = {check['shows_incident']}",
                             dict(incident=iid, camera=cam_id, shows_incident=check["shows_incident"], seen=check["what_is_seen"],
                                  read_by=meta.get("model"), actions=notes))
            verdict = {"yes": "confirms the call", "no": "does NOT show the incident", "unclear": "is unclear"}[check["shows_incident"]]
            self._log(inc, "note", f"CCTV {cam_id} {verdict}: {check['what_is_seen']}" + (f" ({'; '.join(notes)})" if notes else ""))
            return inc

    # ---------- what-if controls ----------
    def set_conditions(self, fog: bool | None = None, night: bool | None = None) -> dict:
        """Fog: every drive slower (FOG_SLOWDOWN). Night shift: 11 of 31 PCR vans off duty (20 left)."""
        with self.lock:
            if fog is not None:
                self.fog = bool(fog)
            if night is not None:
                self.night = bool(night)
                police = [u for u in self.units.values() if u.kind == "police"]
                for i, u in enumerate(police):
                    u.off_duty = self.night and i % 3 == 0
            self._refresh_recommendations()
            return self.conditions()

    def close_road(self, lat: float, lng: float, radius_m: float = 35.0) -> dict:
        """Close the road at a point: every link touching road nodes within radius_m becomes impassable,
        so routes and ETAs go around it (flooded underpass, VIP movement, protest, crash)."""
        with self.lock:
            xy = self.net._xy([lat], [lng])[0]
            nodes = set(self.net._tree.query_ball_point(xy, radius_m)) or {int(self.net.snap(lat, lng)[0])}
            c = {"id": next(self._closure_ids), "lat": lat, "lng": lng, "nodes": nodes}
            self.closures.append(c)
            self._apply_closures()
            self._refresh_recommendations()
            return self.conditions()

    def reopen_roads(self) -> dict:
        with self.lock:
            self.closures = []
            self._apply_closures()
            self._refresh_recommendations()
            return self.conditions()

    def _apply_closures(self):
        """Road links touching closed nodes become impassable. Waterlogging (flood plan) closes spots at >= 30 cm
        and makes links through 15-30 cm of water DIVERT_SLOW times slower; a manual closure wins over both."""
        A = self._A0.tocoo(copy=True)
        fl = getattr(self, "flood", None)
        lv = fl.node_level if fl is not None else {}
        if lv:
            from ..flood import DIVERT_SLOW
            slow = np.array([n for n, v in lv.items() if v == 2])
            shut = np.array([n for n, v in lv.items() if v >= 3])
            if len(slow):
                A.data[np.isin(A.row, slow) | np.isin(A.col, slow)] *= DIVERT_SLOW
            if len(shut):
                A.data[np.isin(A.row, shut) | np.isin(A.col, shut)] = CLOSED_S
            self._flood_segs = {}
            for key, idx in (("closed", shut), ("divert", slow)):
                m = (np.isin(A.row, idx) & np.isin(A.col, idx)) if len(idx) else np.zeros(len(A.row), bool)
                self._flood_segs[key] = [[[round(float(self.net.lon[a]), 5), round(float(self.net.lat[a]), 5)], [round(float(self.net.lon[b]), 5), round(float(self.net.lat[b]), 5)]]
                                         for a, b in zip(A.row[m], A.col[m]) if a < b]
        else:
            self._flood_segs = {"closed": [], "divert": []}
        closed = set().union(*(c["nodes"] for c in self.closures)) if self.closures else set()
        if closed:
            idx = np.array(sorted(closed))
            mask = np.isin(A.row, idx) | np.isin(A.col, idx)
            A.data[mask] = CLOSED_S
            for c in self.closures:
                ci = np.array(sorted(c["nodes"]))
                m = np.isin(A.row, ci) | np.isin(A.col, ci)
                c["segments"] = [[[float(self.net.lon[a]), float(self.net.lat[a])], [float(self.net.lon[b]), float(self.net.lat[b])]]
                                 for a, b in zip(A.row[m], A.col[m])]
        self.net.A = A.tocsr()
        if hasattr(self.net, "_AT"):
            del self.net._AT

    def _target(self, node: int) -> int:
        """Where units drive to. If the incident is inside a closed area (a flooded underpass, a closed road),
        they stop at the nearest open road point: the water's edge or the barricade."""
        shut = {n for n, v in (self.flood.node_level.items() if getattr(self, "flood", None) else []) if v >= 3}
        for c in self.closures:
            shut |= c["nodes"]
        if node not in shut:
            return node
        xy = self.net._xy(self.net.lat, self.net.lon)
        d = np.linalg.norm(xy - xy[node], axis=1)
        for j in np.argsort(d)[:400]:
            if int(j) not in shut:
                return int(j)
        return node

    def _refresh_recommendations(self):
        for inc in self.incidents.values():
            if inc.status == "new" and inc.node is not None:
                inc.recommendation = self.recommend(inc)

    def conditions(self) -> dict:
        on = sum(1 for u in self.units.values() if u.kind == "police" and not u.off_duty)
        return {"fog": self.fog, "night": self.night, "pcr_on_duty": on, "fog_slowdown": FOG_SLOWDOWN,
                "closures": [{"id": c["id"], "lat": c["lat"], "lng": c["lng"], "segments": c.get("segments", [])} for c in self.closures],
                "flood": self._flood_state()}

    def _flood_state(self) -> dict | None:
        fl = getattr(self, "flood", None)
        if fl is None or fl.run is None:
            return None
        segs = getattr(self, "_flood_segs", {"closed": [], "divert": []})
        lv = [r["level"][fl.step] for r in fl.run["spots"]]
        return {"label": fl.meta["label"], "at": fl.run["labels"][fl.step], "closed": sum(1 for x in lv if x == 3),
                "diverted": sum(1 for x in lv if x == 2), "seg_closed": segs["closed"], "seg_divert": segs["divert"]}

    # ---------- unit lifecycle ----------
    def _log(self, inc: "Incident", kind: str, text: str, t: float | None = None):
        inc.events.append({"t": round(self.now() if t is None else t, 1), "kind": kind, "text": text})

    def _advance(self):
        """Move every unit along en_route -> on_scene -> returning -> available, logging each step on its incident."""
        t = self.now()
        for u in sorted(self.units.values(), key=lambda x: x.t_arrive):
            inc = self.incidents.get(u.incident) if u.incident else None
            if u.status == "en_route" and t >= u.t_arrive:
                u.status = "on_scene"
                if inc:
                    # response time = 60 s call handling + drive, as in the benchmark. The time the presenter
                    # spends reading the card is not counted (on the 10x clock it would add minutes).
                    rt = u.t_arrive - u.t_dispatch
                    ok = rt <= TARGET_S[u.kind]
                    self._log(inc, "arrive", f"{u.id} ({KIND_LABEL[u.kind]}) on scene: response {_mmss(rt)} "
                              f"({'within' if ok else 'over'} the {_mmss(TARGET_S[u.kind])} target)", u.t_arrive)
                    inc.arrivals[u.id] = round(rt)
            if u.status == "on_scene" and t >= u.t_clear:
                u.status = "returning"
                if inc:
                    self._log(inc, "clear", f"{u.id} {SCENE_WORK[u.kind]}, returning to {u.base_name}", u.t_clear)
                    if inc.status != "resolved" and all(self.units[x].status not in ("en_route", "on_scene") for x in inc.units):
                        self._resolve(inc, u.t_clear)
                self._send_home(u, u.t_clear)
            if u.status == "returning" and t >= u.t_home:
                u.status, u.incident, u.route, u.route_t, u.route_m = "available", None, [], [], []

    def _send_home(self, u: "Unit", t_start: float):
        """Route home = reverse of the way in."""
        dur = u.t_arrive - u.route_t[0] if u.route_t else 0
        if u.route_t:
            u.route, u.route_t = u.route[::-1], [t_start + (u.t_arrive - x) for x in u.route_t[::-1]]
            if u.route_m:
                u.route_m = [u.route_m[-1] - x for x in u.route_m[::-1]]
        u.t_home = t_start + dur

    def _resolve(self, inc: "Incident", t: float, by: str = "units"):
        arr = inc.arrivals
        first = min(arr.values()) if arr else None
        kinds = {self.units[x].kind for x in inc.units}
        met = {x: v <= TARGET_S[self.units[x].kind] for x, v in arr.items()}
        saved = sum(max(0, r.get("saved_s") or 0) for r in (inc.recommendation or {}).values())
        inc.status, inc.t_resolved = "resolved", t
        inc.outcome = {
            "first_on_scene_s": first, "per_unit_s": arr, "targets_met": met,
            "all_targets_met": bool(met) and all(met.values()),
            "target_s": min((TARGET_S[k] for k in kinds), default=None),
            "saved_vs_nearest_s": saved, "junctions_alerted": sum(a["junctions"] for a in self.alerts if a["incident"] == inc.id),
            "duration_s": round(t - inc.t), "closed_by": by, "calls": inc.calls,
        }
        self._log(inc, "resolved", "Incident closed" + (" by the dispatcher" if by == "dispatcher" else ": all units cleared"), t)
        if hasattr(self, "outbox"):
            self.outbox.send("out", "ERSS-112", "Incident closed", "JSON over HTTPS",
                             f"{inc.id} closed" + (f": first unit on scene in {_mmss(first)}" if first is not None else ""),
                             dict(incident=inc.id, first_on_scene_s=first, per_unit_s=arr, targets_met=met, calls=inc.calls, closed_by=by))

    def close(self, iid: str) -> "Incident":
        """Dispatcher closes the incident: units still on scene clear now; a unit still driving can't be closed."""
        with self.lock:
            self._advance()
            inc = self.incidents[iid]
            if inc.status == "resolved":
                return inc
            if any(self.units[x].status == "en_route" for x in inc.units):
                raise ValueError("a unit is still on the way: wait until it arrives")
            t = self.now()
            for x in inc.units:
                u = self.units[x]
                if u.status == "on_scene":
                    u.t_clear, u.status = t, "returning"
                    self._log(inc, "clear", f"{u.id} {SCENE_WORK[u.kind]}, returning to {u.base_name}", t)
                    self._send_home(u, t)
            if not inc.units:
                self._log(inc, "note", "No unit needed: caller referred to the right service", t)
            self._resolve(inc, t, by="dispatcher")
            return inc

    def telemetry(self, u: Unit) -> dict:
        """Where a moving unit is on its route: metres and (real-world) seconds left, current speed, heading."""
        if u.status not in ("en_route", "returning") or len(u.route) < 2 or not u.route_m:
            return {}
        t = self.now()
        ts = np.array(u.route_t)
        if t <= ts[0]:
            i, f, rolling = 0, 0.0, False
        elif t >= ts[-1]:
            i, f, rolling = len(ts) - 2, 1.0, True
        else:
            i = int(np.searchsorted(ts, t)) - 1
            f, rolling = (t - ts[i]) / max(1e-9, ts[i + 1] - ts[i]), True
        # speed over a short window around now (single geometry segments are too jumpy)
        j0, j1 = max(0, i - 3), min(len(ts) - 1, i + 4)
        dt = ts[j1] - ts[j0]
        speed = 3.6 * (u.route_m[j1] - u.route_m[j0]) / dt if dt > 0 and rolling else 0.0
        done_m = u.route_m[i] + f * (u.route_m[i + 1] - u.route_m[i])
        a, b = u.route[i], u.route[i + 1]
        heading = float(np.degrees(np.arctan2((b[0] - a[0]) * np.cos(np.radians(a[1])), b[1] - a[1]))) % 360
        end = u.t_arrive if u.status == "en_route" else u.t_home
        return {"left_km": round(float(max(0.0, u.route_m[-1] - done_m)) / 1000, 2), "route_km": round(float(u.route_m[-1]) / 1000, 2),
                "speed_kmh": int(round(float(speed))), "left_s": int(max(0, round(float(end - t)))), "heading": int(round(heading)),
                "rolling": bool(rolling)}

    def unit_position(self, u: Unit) -> tuple[float, float]:
        t = self.now()
        if u.status in ("en_route", "returning") and u.route:
            ts = np.array(u.route_t)
            if t <= ts[0]:
                return u.route[0][1], u.route[0][0]
            if t >= ts[-1]:
                return u.route[-1][1], u.route[-1][0]
            i = int(np.searchsorted(ts, t)) - 1
            f = (t - ts[i]) / max(1e-9, ts[i + 1] - ts[i])
            a, b = u.route[i], u.route[i + 1]
            return a[1] + f * (b[1] - a[1]), a[0] + f * (b[0] - a[0])
        if u.status == "on_scene" and u.route:
            return u.route[-1][1], u.route[-1][0]
        return float(self.net.lat[u.home]), float(self.net.lon[u.home])

    # ---------- incidents ----------
    def add_incident(self, call: dict, triage: dict) -> Incident:
        with self.lock:
            loc = triage["location"]
            node = int(self.net.snap(loc["lat"], loc["lng"])[0]) if loc["resolved"] else None
            iid = f"INC-{next(self._ids):04d}"
            x = triage["extract"]
            key = (x.get("landmark") or "").lower() or (f"sector {x['sector']}" if x.get("sector") else None)
            n = self.tracker.add(iid, self.now(), x["incident_type"], loc.get("lat"), loc.get("lng"), key)
            if n > 1:  # another call about an incident we already have: merge and escalate, don't open a new card
                first = next(c.call_ids[0] for c in self.tracker.clusters if iid in c.call_ids)
                orig = self.incidents.get(first)
                if orig is not None and orig.status != "resolved":
                    orig.calls = n
                    orig.call.setdefault("more_calls", []).append(call.get("transcript"))
                    ox = Extract(**orig.triage["extract"])
                    orig.triage["decision"] = decide(ox, repeat_calls=n, location_resolved=orig.node is not None).model_dump()
                    orig.triage["repeat_calls"] = n
                    if orig.node is not None and orig.status == "new":
                        orig.recommendation = self.recommend(orig)
                    orig.merged_last = True
                    self._log(orig, "call", f"Another call about this incident ({n} calls): priority now {orig.triage['decision']['priority']}")
                    self.outbox.send("out", "ERSS-112", "Repeat call merged", "JSON over HTTPS",
                                     f"{orig.id}: call {n} merged, priority {orig.triage['decision']['priority']}",
                                     dict(incident=orig.id, calls=n, priority=orig.triage["decision"]["priority"]))
                    return orig
            inc = Incident(iid, self.now(), call, triage, loc.get("lat"), loc.get("lng"), node, calls=n)
            self.incidents[iid] = inc
            m = triage.get("meta") or {}
            lang = {"en": "English", "hi": "Hindi", "pa": "Punjabi", "hi-latn": "Hinglish", "pa-latn": "Punjabi (Roman)", "mixed": "mixed language"}.get(x.get("caller_language") or "")
            self._log(inc, "call", "112 call received" + (f" in {lang}" if lang else ""))
            gem = str(m.get("model") or "").startswith(("gemini", "gemma"))
            self._log(inc, "triage", (f"Read by {m['model']} in {m['latency_s']} s" if gem and m.get("latency_s") is not None else "Read by the keyword fallback")
                      + f": {triage['decision']['priority']}, " + (", ".join(KIND_LABEL[k] for k in triage["decision"]["units"]) or "no unit needed"))
            if node is not None:
                inc.recommendation = self.recommend(inc)
            d = triage["decision"]
            self.outbox.send("out", "ERSS-112", "Incident record", "JSON over HTTPS",
                             f"{iid}: {x['incident_type'].replace('_', ' ')}, {d['priority']}, " + (", ".join(KIND_LABEL[k] for k in d["units"]) or "no unit"),
                             dict(incident=iid, type=x["incident_type"], priority=d["priority"], units=d["units"],
                                  location=dict(resolved=bool(loc["resolved"]), lat=round(loc["lat"], 5) if loc.get("lat") is not None else None,
                                                lng=round(loc["lng"], 5) if loc.get("lng") is not None else None, text=loc.get("label")),
                                  summary=x.get("summary_en"), read_by=m.get("model") or "keyword fallback",
                                  note="caller name and number are never sent; personal details are removed before the AI reads the call"))
            return inc

    def gemini_progress(self, iid: str, state: str, attempt: int, error: str | None):
        with self.lock:
            inc = self.incidents.get(iid)
            if inc is not None:
                inc.gemini = {"state": state, "attempt": attempt, "error": error}
                if state == "gave_up":
                    self._log(inc, "note", f"Gemini could not read this call ({error}); the keyword reading stands")

    def late_triage(self, iid: str, result: dict):
        """Gemini answered after the keyword fallback had already triaged the call: upgrade the card."""
        with self.lock:
            inc = self.incidents.get(iid)
            if inc is None:
                return
            m = result.get("meta") or {}
            d = result["decision"]
            old = inc.triage["decision"]["priority"]
            if inc.status == "new":
                # keep the old place if the dispatcher set it, or if Gemini couldn't place the call
                kept_loc = inc.triage["location"] if (inc.triage["location"].get("method") == "dispatcher"
                                                      or not result["location"]["resolved"]) else None
                inc.triage = result
                if kept_loc:
                    inc.triage["location"] = kept_loc
                elif result["location"]["resolved"]:
                    inc.lat, inc.lng = result["location"]["lat"], result["location"]["lng"]
                    inc.node = int(self.net.snap(inc.lat, inc.lng)[0])
                if inc.node is not None:
                    inc.triage["decision"] = decide(Extract(**result["extract"]), repeat_calls=inc.calls, location_resolved=True).model_dump()
                    inc.recommendation = self.recommend(inc)
                inc.triage["note"] = None
                self._log(inc, "triage", f"Gemini answered after {m.get('latency_s')} s and replaced the keyword reading: "
                          f"{old} -> {inc.triage['decision']['priority']}")
            else:
                inc.triage["late"] = result
                self._log(inc, "triage", f"Gemini answered after {m.get('latency_s')} s (units already sent): "
                          f"it rates this {d['priority']}, the keyword reading said {old}")
                self._offer_extra(inc, d["units"])
            if inc.fast and inc.status == "new":   # fast path still waiting for a place: Gemini may have found one
                inc.triage["held_units"] = [k for k in inc.triage["decision"]["units"] if k != "police"]
                inc.triage["decision"]["units"] = ["police"]
                if inc.node is not None:
                    self._fast_dispatch(inc)

    def set_location(self, iid: str, lat: float, lng: float, label: str | None = None) -> Incident:
        with self.lock:
            inc = self.incidents[iid]
            inc.lat, inc.lng = lat, lng
            inc.node = int(self.net.snap(lat, lng)[0])
            inc.triage["location"] |= {"lat": lat, "lng": lng, "resolved": True, "method": "dispatcher", "label": label or "set on map by dispatcher"}
            inc.triage["decision"] = decide(Extract(**inc.triage["extract"]), repeat_calls=inc.calls, location_resolved=True).model_dump()
            inc.recommendation = self.recommend(inc)
            self._log(inc, "location", f"Place set by the dispatcher: {label or 'marked on the map'}")
            if inc.fast:
                inc.triage["decision"]["priority"], inc.triage["decision"]["units"] = "P1", ["police"]
                self._fast_dispatch(inc)
            return inc

    # ---------- sexual-violence fast path ----------
    def fast_track(self, iid: str, trigger: str) -> Incident:
        """Police now, without waiting for Gemini; any other unit (ambulance) is offered for the dispatcher to add."""
        with self.lock:
            inc = self.incidents[iid]
            inc.fast = trigger
            d = inc.triage["decision"]
            held = [k for k in d["units"] if k != "police"]
            d["priority"] = "P1"
            d["units"] = ["police"]
            d["reasons"] = [f"sexual violence or harassment words heard (\"{trigger}\"): nearest PCR sent at once"] + \
                [r for r in d["reasons"] if "sexual assault" not in r]
            inc.triage["held_units"] = held
            self._log(inc, "call", f"Fast path: \"{trigger}\" heard, police sent without waiting for Gemini")
            if inc.node is not None:
                self._fast_dispatch(inc)
            return inc

    def _fast_dispatch(self, inc: Incident):
        if inc.status != "new":
            return
        inc.recommendation = self.recommend(inc, ["police"])
        best = (inc.recommendation.get("police") or {}).get("best")
        if best:
            self.dispatch(inc.id, [best["unit"]])
        self._offer_extra(inc, inc.triage.get("held_units") or [])

    def _offer_extra(self, inc: Incident, kinds: list[str]):
        sent = {self.units[u].kind for u in inc.units}
        want = [k for k in kinds if k not in sent]
        inc.extra = self.recommend(inc, want) if want and inc.node is not None and inc.status != "resolved" else None
        if inc.extra:
            self._log(inc, "note", "Suggested next: " + ", ".join(KIND_LABEL[k] for k in want))

    def recommend(self, inc: Incident, kinds: list[str] | None = None) -> dict:
        """For each unit type the rules asked for: the free unit that arrives first by road, vs the nearest
        free unit in a straight line (what CAD does today), with ETAs."""
        self._advance()
        to_inc = self.net.times_to(self._target(inc.node)) * self.travel_scale()
        out = {}
        for kind in (kinds if kinds is not None else inc.triage["decision"]["units"]):
            free = [u for u in self.units.values() if u.kind == kind and u.status == "available" and not u.off_duty]
            if not free:
                out[kind] = {"available": 0, "best": None, "note": f"no {KIND_LABEL[kind].lower()} free: request mutual aid or wait"}
                continue
            homes = np.array([u.home for u in free])
            eta = to_inc[homes]
            straight = self.net.straight_line_from(homes)[:, inc.node]
            order = np.argsort(eta)
            nearest = int(np.argmin(straight))
            cand = [dict(unit=free[i].id, base=free[i].base_name, eta_s=round(float(eta[i])), km=round(float(straight[i]) / 1000, 2))
                    for i in order[:3] if np.isfinite(eta[i]) and eta[i] < CLOSED_S / 2]
            if not cand:
                out[kind] = {"available": len(free), "best": None, "note": "no open road to this place: reopen a road or send on foot"}
                continue
            best = cand[0] if cand else None
            out[kind] = {"available": len(free), "best": best, "options": cand,
                         "nearest_by_distance": dict(unit=free[nearest].id, base=free[nearest].base_name,
                                                     eta_s=round(float(eta[nearest])), km=round(float(straight[nearest]) / 1000, 2)),
                         "saved_s": round(float(eta[nearest] - eta[order[0]])) if best else 0}
        return out

    def dispatch(self, iid: str, unit_ids: list[str]) -> Incident:
        with self.lock:
            self._advance()
            inc = self.incidents[iid]
            if inc.node is None:
                raise ValueError("set the incident location first")
            if inc.status == "resolved":
                raise ValueError("this incident is closed")
            corridor_all = []
            for uid in unit_ids:
                u = self.units[uid]
                if u.status != "available":
                    raise ValueError(f"{uid} is not available")
                path = self.net.route(u.home, self._target(inc.node))
                _, cum = self.net.path_geometry(path)
                coords, secs, mets = self.net.path_timed(path)
                k = self.travel_scale()
                t0 = self.now() + 60  # call handling before the unit rolls (same assumption as the benchmark)
                total = secs[-1] * k if secs else 0.0
                u.route, u.route_t, u.route_m = coords, [t0 + x * k for x in secs], mets
                u.t_arrive = t0 + total
                u.t_clear = u.t_arrive + ON_SCENE_S[u.kind]
                u.status, u.incident, u.t_dispatch = "en_route", iid, self.now()
                inc.units.append(uid)
                corridor_all.append(self._corridor(u, path, cum, k, t0))
            inc.status = "dispatched"
            inc.corridor = {"routes": ((inc.corridor or {}).get("routes", []) + corridor_all)}
            if inc.extra:   # units just sent are no longer "suggested"
                inc.extra = {k: v for k, v in inc.extra.items() if k not in {self.units[u].kind for u in unit_ids}} or None
            x = inc.triage["extract"]
            for r in corridor_all:
                self.outbox.send("out", f"{r['unit']} mobile data terminal", "Dispatch order", "JSON over the CAD link",
                                 f"{r['unit']} to {iid} ({inc.triage['decision']['priority']}), {_mmss(r['eta_s'])} drive",
                                 dict(incident=iid, unit=r["unit"], priority=inc.triage["decision"]["priority"], type=x["incident_type"],
                                      to=dict(lat=round(inc.lat, 5), lng=round(inc.lng, 5)), eta_s=r["eta_s"], km=r["km"],
                                      junctions=len(r["junctions"]), brief=x.get("summary_en"), hazards=x.get("hazards")))
                if r["kind"] == "ambulance":
                    hosp = min(self.flood.hospitals, key=lambda h: _dist_m(inc.lat, inc.lng, float(self.net.lat[h[1]]), float(self.net.lon[h[1]])))
                    self.outbox.send("out", f"{hosp[0]} emergency department", "Hospital pre-alert", "JSON over HTTPS (no caller identity)",
                                     f"{r['unit']} heading to {iid}; likely patient for {hosp[0].split(' (')[0]}",
                                     dict(incident=iid, ambulance=r["unit"], scene_eta_s=r["eta_s"], type=x["incident_type"],
                                          injured=x.get("people_injured"), unconscious_or_not_breathing=x.get("unconscious_or_not_breathing"),
                                          trapped=x.get("people_trapped"), priority=inc.triage["decision"]["priority"]))
            for r in corridor_all:
                self._log(inc, "dispatch", f"{r['unit']} ({KIND_LABEL[r['kind']]}) dispatched from {self.units[r['unit']].base_name}, "
                          f"{_mmss(r['eta_s'])} drive, {len(r['junctions'])} main-road junctions")
            return inc

    def _corridor(self, u: Unit, path, cum, k, t0) -> dict:
        """Junctions on major roads along the route, with when the unit will reach each: what traffic
        police need to hold green or clear."""
        und = self.net.A + self.net.A.T
        deg = np.diff(und.indptr)
        junctions = []
        for idx, (node, c) in enumerate(zip(path, cum)):
            if 0 < idx < len(path) - 1 and deg[node] >= 3 and self.net.node_class[node] in MAJOR:
                # one alert per real junction: the nodes of a roundabout or a split carriageway are merged (150 m)
                if junctions and _dist_m(junctions[-1]["lat"], junctions[-1]["lng"], self.net.lat[node], self.net.lon[node]) < 150:
                    continue
                junctions.append(dict(lat=float(self.net.lat[node]), lng=float(self.net.lon[node]),
                                      road=self.net.node_class[node], at_s=round(c * k), at_wall=round((t0 + c * k - self.now()) / DEMO_SPEED)))
        km = round(u.route_m[-1] / 1000, 2) if u.route_m else None
        eta = round(cum[-1] * k) if cum else 0
        return {"unit": u.id, "kind": u.kind, "eta_s": eta, "route": u.route, "junctions": junctions,
                "km": km, "avg_kmh": round(3.6 * u.route_m[-1] / eta) if km and eta else None}

    def notify_traffic(self, iid: str) -> dict:
        """One alert per junction, each with its own window: it opens WINDOW_BEFORE_S before the unit reaches
        that junction and closes WINDOW_AFTER_S after. The traffic post acknowledges (SIMULATED: ~85% reply
        within 10-40 s; there is no live link to Chandigarh traffic police)."""
        with self.lock:
            inc = self.incidents[iid]
            now = self.now()
            posts = []
            for r in (inc.corridor or {}).get("routes", []):
                u = self.units.get(r["unit"])
                if u is None or u.status != "en_route" or u.incident != iid:
                    continue
                t0 = u.t_dispatch + 60
                for j in r["junctions"]:
                    arrive = t0 + j["at_s"]
                    if arrive < now:
                        continue
                    open_t = max(now, arrive - WINDOW_BEFORE_S)
                    ack = open_t + float(self._rng.uniform(10, 40)) if self._rng.random() < 0.85 else None
                    posts.append({"unit": u.id, "road": j["road"], "lat": j["lat"], "lng": j["lng"], "arrive": arrive,
                                  "open": open_t, "close": arrive + WINDOW_AFTER_S, "ack": ack if ack is None or ack < arrive else None})
            posts.sort(key=lambda p: p["arrive"])
            inc.traffic = posts
            alert = {"id": f"TP-{len(self.alerts) + 1:03d}", "incident": iid, "junctions": len(posts), "t": now,
                     "message": f"{len(posts)} junction alerts scheduled for {', '.join(inc.units)} heading to {iid}, "
                                f"each sent {WINDOW_BEFORE_S // 60} min before the unit reaches it"}
            self.alerts.append(alert)
            self.outbox.send("out", "Traffic police / ICCC signal control", "Green-corridor windows", "JSON, one message per junction",
                             f"{len(posts)} junction windows for {', '.join(inc.units)} to {iid}",
                             dict(incident=iid, junctions=[dict(unit=p["unit"], lat=round(p["lat"], 5), lng=round(p["lng"], 5), road=p["road"],
                                                                open_in_s=round(p["open"] - now), unit_due_in_s=round(p["arrive"] - now),
                                                                hold_s=round(p["close"] - p["open"])) for p in posts]))
            self._log(inc, "traffic", f"Traffic police: {len(posts)} junction alerts scheduled, each with its own time window ({alert['id']})")
            return alert

    def _traffic_json(self, inc: Incident) -> list[dict]:
        now, out = self.now(), []
        for i, p in enumerate(inc.traffic):
            if now < p["open"]:
                st = "scheduled"
            elif p["ack"] is not None and now >= p["ack"]:
                st = "cleared" if now < p["close"] else "done"
            elif now < p["close"]:
                st = "sent"
            else:
                st = "no_reply"
            out.append({"i": i, "unit": p["unit"], "road": p["road"], "lat": p["lat"], "lng": p["lng"], "status": st,
                        "opens_in_s": max(0, round((p["open"] - now) / DEMO_SPEED)),
                        "unit_in_s": max(0, round((p["arrive"] - now) / DEMO_SPEED)),
                        "ack_after_s": round(p["ack"] - p["open"]) if p["ack"] is not None else None})
        return out

    # ---------- prediction layers ----------
    def risk_grid(self, kind: str = "crash", cell_m: float = 450) -> dict:
        """Where incidents are most likely this hour, as grid cells (synthetic demand model, labelled)."""
        w = scenario_weights(self.net, kind, str(ROOT / "data/hotspots_geocoded.csv"))
        w = w * HOURLY_PROFILE[self.hour()] * 24
        xy = self.net._xy(self.net.lat, self.net.lon)
        gx, gy = np.floor(xy[:, 0] / cell_m).astype(int), np.floor(xy[:, 1] / cell_m).astype(int)
        df = pd.DataFrame({"gx": gx, "gy": gy, "w": w, "lat": self.net.lat, "lng": self.net.lon})
        g = df.groupby(["gx", "gy"]).agg(w=("w", "sum"), lat=("lat", "mean"), lng=("lng", "mean")).reset_index()
        g["level"] = g["w"] / g["w"].max()
        top = g.sort_values("w", ascending=False).head(120)
        return {"hour": self.hour(), "cell_m": cell_m, "cells": top[["lat", "lng", "level"]].round(5).to_dict("records")}

    def standby_plan(self) -> dict:
        """Where free PCR vans should wait (risk-based standby points from the benchmark method)."""
        if self._standby is None:
            w = scenario_weights(self.net, "pcr")
            hist = generate(w, 300, 90, seed=5000, mean_service_s=1500).node
            fc = estimate_weights(self.net, hist, prior=np.ones(self.net.n) / self.net.n)
            typical = float((HOURLY_PROFILE * tod_multiplier(np.arange(24))).sum())
            # MEXCLP (Daskin): beat our own greedy planner by ~4-5 points in the benchmark, so it is what we use
            k = FLEET["police"][1]
            pts = mexclp(self.net, fc, k, (300 - 60) / (self.scale * typical), busy_p=min(0.9, 300 * (1500 + 600) / (k * 86400)))
            self._standby = [dict(lat=float(self.net.lat[p]), lng=float(self.net.lon[p])) for p in pts]
        return {"points": self._standby, "units": FLEET["police"][1]}

    # ---------- serialisation ----------
    def snapshot(self) -> dict:
        with self.lock:
            self._advance()
            units = []
            for u in self.units.values():
                lat, lng = self.unit_position(u)
                units.append(dict(id=u.id, kind=u.kind, base=u.base_name, status="off_duty" if u.off_duty and u.status == "available" else u.status, incident=u.incident,
                                  lat=lat, lng=lng, eta_wall_s=max(0, round((u.t_arrive - self.now()) / DEMO_SPEED)) if u.status == "en_route" else None,
                                  scene_left_wall_s=max(0, round((u.t_clear - self.now()) / DEMO_SPEED)) if u.status == "on_scene" else None,
                                  home_wall_s=max(0, round((u.t_home - self.now()) / DEMO_SPEED)) if u.status == "returning" else None,
                                  tele=self.telemetry(u) or None,
                                  # full timed path for moving units, so the browser can animate them smoothly
                                  path={"c": [[round(x, 5), round(y, 5)] for x, y in u.route], "t": [round(x, 1) for x in u.route_t],
                                        "m": [round(float(x)) for x in u.route_m]}
                                  if u.status in ("en_route", "returning") and u.route_m else None))
            incs = [self.incident_json(i) for i in sorted(self.incidents.values(), key=lambda i: -i.t)]
            free = {k: sum(1 for u in self.units.values() if u.kind == k and u.status == "available" and not u.off_duty) for k in FLEET}
            return {"now": self.now(), "demo_speed": DEMO_SPEED, "units": units, "incidents": incs,
                    "free": free, "alerts": self.alerts[-10:], "conditions": self.conditions()}

    def incident_json(self, i: Incident) -> dict:
        return dict(id=i.id, t=i.t, age_s=round((self.now() - i.t) / DEMO_SPEED), status=i.status, lat=i.lat, lng=i.lng,
                    calls=i.calls, more_calls=i.call.get("more_calls", []), units=i.units, triage=i.triage, recommendation=i.recommendation,
                    corridor=i.corridor, transcript=i.call.get("transcript"), outcome=i.outcome, gemini=i.gemini, fast=i.fast, extra=i.extra, traffic=self._traffic_json(i),
                    cctv=i.cctv, cameras=self.cameras_near(i.lat, i.lng) if i.lat is not None else [],
                    events=[e | {"at_s": round(e["t"] - i.t)} for e in i.events])
