"""HTTP API + the built dashboard.  Run:  python run_app.py   then open http://localhost:8001

Gemini is used when GEMINI_API_KEY is set (.env); otherwise, or if Gemini is down, the keyword baseline
triages calls and the dashboard says so. The app never pretends a fallback answer came from Gemini.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..triage.baseline import keyword_extract
from ..triage.client import FakeTriage, GeminiTriage, load_env
from ..triage.local import LocalGemmaTriage
from ..triage.fastpath import sexual_violence_alert
from ..triage.pipeline import triage_call
from .world import DEMO_SPEED, KIND_LABEL, ROOT, World

load_env(str(ROOT / ".env"))
app = FastAPI(title="OpenHand 112", version="1.0")
world = World()
# threading.Thread(target=world.flood.risk, daemon=True).start()   # 30+ monsoons through the water model: warm the cache
baseline = FakeTriage(keyword_extract)
try:
    # Each request is cut off after 15 s (Google rejects deadlines under 10 s); at most ~40 s in all, then give up.
    gemini = GeminiTriage(max_retries=2, budget_s=float(os.environ.get("GEMINI_BUDGET_S", "40")), request_timeout_s=15) \
        if os.environ.get("GEMINI_API_KEY") else None
except Exception:  # noqa: BLE001
    gemini = None
# Two AI engines, one switch. `gemini` below always means "the active AI engine":
#   cloud: Google Gemini through the Gemini API (fallbacks: other Gemini models, Gemma on Google's servers)
#   local: Gemma (open weights) on this computer through Ollama: sovereign mode, no call data leaves the machine
cloud_ai = gemini
local_ai = LocalGemmaTriage()
ENGINE = "gemini" if gemini else "none"
if os.environ.get("TRIAGE_ENGINE", "").lower() in ("local", "gemma", "sovereign"):
    _st = local_ai.status()
    if _st["ok"]:
        gemini, ENGINE = local_ai, "local"
        print(f"[ai] Sovereign mode: calls are read by {local_ai.model} on this computer (Ollama)", flush=True)
    else:
        print(f"[ai] TRIAGE_ENGINE=local, but local Gemma isn't ready: {_st['reason']}", flush=True)


def ai_name() -> str:
    return "local Gemma" if ENGINE == "local" else "Gemini"


GEMINI_WAIT_S = float(os.environ.get("GEMINI_WAIT_S", "20"))  # longest a dispatcher waits before the fallback answers
GEMINI_RETRY_FOR_S = float(os.environ.get("GEMINI_RETRY_FOR_S", "300"))  # keep asking Gemini in the background this long
RETRY_BASE_S = 5.0
POOL = ThreadPoolExecutor(max_workers=8)


def _log_triage(result: dict, note: str | None):
    m = result.get("meta") or {}
    print(f"[triage] {m.get('model')} {m.get('latency_s')} s, attempts {m.get('attempts')}"
          + (f", errors: {' | '.join(e[:300] for e in m.get('errors', []))}" if m.get("errors") else "")
          + (f"  -> {note}" if note else ""), flush=True)


def _follow_up(iid: str, call: dict, fut, attempt: int, started: float):
    """Keep asking Gemini about a call the keyword fallback answered, until it answers, the incident closes, or
    GEMINI_RETRY_FOR_S passes. Waits 5, 10, 20, 30 s between attempts so a busy server isn't hammered."""
    def done(f):
        try:
            r = f.result()
        except Exception as e:  # noqa: BLE001
            err = str(e)
            busy, quota = "503" in err or "high demand" in err.lower(), "429" in err
            short = ("Google's servers are busy" if busy else "") + (" and " if busy and quota else "") + \
                ("a backup model's free quota is used up" if quota else "")
            short = short or ("took too long" if "longer than" in err else err[:120])
            inc = world.incidents.get(iid)
            if inc is None or inc.status == "resolved" or time.time() - started > GEMINI_RETRY_FOR_S:
                world.gemini_progress(iid, "gave_up", attempt, short)
                print(f"[triage] {ai_name()} gave up on {iid} after {attempt} attempts: {err[:200]}", flush=True)
                return
            world.gemini_progress(iid, "trying", attempt + 1, short)
            delay = min(30, RETRY_BASE_S * 2 ** (attempt - 1))
            print(f"[triage] {iid}: Gemini attempt {attempt} failed ({short}); retrying in {delay} s", flush=True)
            tm = threading.Timer(delay, lambda: _follow_up(iid, call, POOL.submit(triage_call, gemini, call), attempt + 1, started))
            tm.daemon = True
            tm.start()
            return
        r["note"] = None
        _log_triage(r, f"late answer for {iid}")
        world.late_triage(iid, r)
        world.gemini_progress(iid, "done", attempt, None)
    fut.add_done_callback(done)


def _gemini_self_test():
    """One tiny call at startup, so the Terminal says straight away whether Gemini works and how fast."""
    import threading
    import time as _t

    def run():
        t0 = _t.time()
        try:
            _, m = gemini.extract("Tribune Chowk pe accident, ek aadmi behosh hai")
            print(f"[{ENGINE}] OK: {m['model']} answered in {_t.time() - t0:.1f} s. Backups if it fails: {', '.join(gemini.fallbacks)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{ENGINE}] NOT WORKING right now (backups tried: {', '.join(gemini.fallbacks)}). Reason: {str(e)[:600]}", flush=True)
    if gemini is not None and os.environ.get("GEMINI_SELF_TEST", "0") != "0":
        threading.Thread(target=run, daemon=True).start()


_gemini_self_test()


class LatLng(BaseModel):
    lat: float
    lng: float
    label: str | None = None


class DispatchBody(BaseModel):
    units: list[str] | None = None


@app.get("/api/health")
def health():
    return {"triage": ENGINE if gemini else "keyword_baseline", "model": getattr(gemini, "model", None), "ai_name": ai_name(),
            "fallbacks": getattr(gemini, "fallbacks", []), "demo_speed": DEMO_SPEED, "speed_scale": world.scale}


@app.get("/api/meta")
def meta():
    from ..triage import fastpath
    from ..triage.location import ROADS, SECTOR_POINTS
    return {"stations": world.stations, "hotspots": world.hotspots, "unit_labels": KIND_LABEL,
            "sectors": {int(k): [v[1][0], v[1][1]] for k, v in sorted(SECTOR_POINTS.items())},
            "cameras": world.cameras,
            "flood_spots": [dict(id=f.id, name=f.name, kind=f.kind, lat=f.lat, lng=f.lng, radius_m=f.radius_m, sources=f.sources,
                                 accuracy=f.accuracy, region=f.region) for f in world.flood.spots],
            "alert_words": {"direct": fastpath.DIRECT, "female": fastpath.FEMALE, "threat": fastpath.THREAT},
            "roads": {k: [[p[1], p[0]] for p in v] for k, v in ROADS.items()},
            "bounds": [[76.7076, 30.6697], [76.8478, 30.7912]], "center": [76.779, 30.733]}


@app.get("/api/state")
def state():
    return world.snapshot()


@app.post("/api/calls")
async def new_call(transcript: str = Form(""), lat: float | None = Form(None), lng: float | None = Form(None),
                   image: UploadFile | None = File(None), audio: UploadFile | None = File(None)):
    if not transcript.strip() and image is None and audio is None:
        raise HTTPException(400, "Add a transcript, a photo or an audio clip.")
    call = {"transcript": transcript.strip() or None, "phone_lat": lat, "phone_lng": lng}
    if image is not None:
        call["image"] = (await image.read(), image.content_type or "image/jpeg")
    if audio is not None:
        call["audio"] = (await audio.read(), audio.content_type or "audio/wav")
    note = None
    from starlette.concurrency import run_in_threadpool
    pending = None
    alert = sexual_violence_alert(call["transcript"])
    if alert:
        # Sexual violence or harassment: no waiting. Keyword reading now, police sent now, Gemini in the background.
        result = await run_in_threadpool(triage_call, baseline, call)
        note = (f"Fast path: \"{alert}\" heard, so the nearest PCR was sent at once without waiting for the AI."
                + (f" {ai_name()} is reading the call in the background and may suggest more units." if gemini else ""))
        if gemini is not None:
            pending = POOL.submit(triage_call, gemini, call)
    elif gemini is None:
        result = await run_in_threadpool(triage_call, baseline, call)
        note = "Triaged by the keyword fallback: no AI engine available (no Gemini key, local Gemma off)"
    else:
        # Race Gemini against the clock: a 112 dispatcher can't wait for an overloaded server. If Gemini hasn't
        # answered within GEMINI_WAIT_S, the keyword fallback answers now and Gemini's reading replaces it when it lands.
        fut = POOL.submit(triage_call, gemini, call)
        try:
            result = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(fut)), timeout=GEMINI_WAIT_S)
        except asyncio.TimeoutError:
            if (image is not None or audio is not None) and not transcript.strip():
                result = await asyncio.wrap_future(fut)   # photo/audio only: nothing else can read it, so wait
            else:
                result = await run_in_threadpool(triage_call, baseline, call)
                note = (f"{ai_name()} hadn't answered after {GEMINI_WAIT_S:.0f} s"
                        + (" (Google's servers are busy)" if ENGINE == "gemini" else "") + ", so the keyword fallback triaged this call. "
                        f"{ai_name()} keeps trying in the background and the card updates when it answers.")
                pending = fut
        except Exception as e:  # noqa: BLE001
            if (image is not None or audio is not None) and not transcript.strip():
                raise HTTPException(503, f"{ai_name()} could not read this photo or audio: {str(e)[:200]}")
            result = await run_in_threadpool(triage_call, baseline, call)
            note = f"Triaged by the keyword fallback: {ai_name()} failed ({str(e)[:120]})"
    result["note"] = note
    _log_triage(result, note)
    stored = {k: v for k, v in call.items() if k not in ("image", "audio")}
    stored["had_image"], stored["had_audio"] = image is not None, audio is not None
    inc = world.add_incident(stored, result)
    merged = getattr(inc, "merged_last", False)
    inc.merged_last = False
    if alert and inc.status == "new":
        world.fast_track(inc.id, alert)
    if pending is not None:
        world.gemini_progress(inc.id, "trying", 1, None)
        _follow_up(inc.id, call, pending, 1, time.time())
    return {"incident": world.incident_json(inc), "merged": merged}


@app.post("/api/incidents/{iid}/location")
def set_location(iid: str, body: LatLng):
    _exists(iid)
    return world.incident_json(world.set_location(iid, body.lat, body.lng, body.label))


@app.post("/api/incidents/{iid}/dispatch")
def dispatch(iid: str, body: DispatchBody):
    inc = _exists(iid)
    units = body.units or [r["best"]["unit"] for r in (inc.recommendation or {}).values() if r.get("best")]
    if not units:
        raise HTTPException(409, "No free unit to send. Set the location first, or wait for a unit to clear.")
    try:
        return world.incident_json(world.dispatch(iid, units))
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/incidents/{iid}/notify-traffic")
def notify(iid: str):
    _exists(iid)
    return world.notify_traffic(iid)


@app.post("/api/incidents/{iid}/close")
def close(iid: str):
    _exists(iid)
    try:
        return world.incident_json(world.close(iid))
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


class Conditions(BaseModel):
    fog: bool | None = None
    night: bool | None = None


# ---------- CCTV (Gemini reads the frame; frames are not stored, only what was seen) ----------
def _check_frame(image_bytes: bytes, mime: str, report: str | None):
    if gemini is None:
        raise HTTPException(503, "Reading camera frames needs an AI engine (Gemini key or local Gemma).")
    from ..triage.redact import redact
    report, _ = redact(report)
    try:
        x, m = gemini.verify_frame((image_bytes, mime), report)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"{ai_name()} could not read the frame: {str(e)[:200]}") from e
    return x.model_dump(), m


@app.post("/api/incidents/{iid}/cctv")
async def cctv_check(iid: str, camera_id: str = Form(...), image: UploadFile = File(...)):
    inc = _exists(iid)
    try:
        world.camera(camera_id)
    except KeyError:
        raise HTTPException(404, f"No camera {camera_id}") from None
    from starlette.concurrency import run_in_threadpool
    report = f"{inc.triage['extract'].get('summary_en', '')}. Caller said: {inc.call.get('transcript') or ''}"
    check, m = await run_in_threadpool(_check_frame, await image.read(), image.content_type or "image/jpeg", report)
    return world.incident_json(world.apply_cctv(iid, camera_id, check, m))


@app.post("/api/cctv/alert")
async def cctv_alert(camera_id: str = Form(...), image: UploadFile = File(...)):
    """A camera (or ICCC video analytics) flags something: Gemini reads the frame and, if it shows an
    emergency, an incident is opened at the camera's position. No caller needed."""
    try:
        cam = world.camera(camera_id)
    except KeyError:
        raise HTTPException(404, f"No camera {camera_id}") from None
    from starlette.concurrency import run_in_threadpool
    from ..triage.location import Place
    from ..triage.rules import decide
    from ..triage.schema import Extract
    check, m = await run_in_threadpool(_check_frame, await image.read(), image.content_type or "image/jpeg", None)
    if check["seen_type"] == "non_emergency" and check["shows_incident"] != "yes":
        return {"incident": None, "check": check, "message": f"{camera_id}: nothing that needs a response ({check['what_is_seen']})"}
    x = Extract(incident_type=check["seen_type"], people_injured=1 if check["injured_visible"] else None,
                unconscious_or_not_breathing=False, violence_ongoing=check["weapon_visible"] or check["violence_visible"],
                fire_or_smoke=check["fire_or_smoke"], hazards=check["hazards"],
                summary_en=f"CCTV {camera_id}: {check['what_is_seen']}", location_text=cam["road"],
                missing_info=["Is anyone calling 112 about this?", "Check the next camera along the road"], confidence=check["confidence"])
    place = Place(cam["lat"], cam["lng"], "camera", f"camera {camera_id} ({cam['road']})")
    result = {"extract": x.model_dump(), "location": place.__dict__ | {"resolved": True}, "repeat_calls": 1,
              "decision": decide(x).model_dump(), "meta": dict(m, redacted=[]), "note": None}
    inc = world.add_incident({"transcript": None, "camera": camera_id}, result)
    merged = getattr(inc, "merged_last", False)
    inc.merged_last = False
    world.apply_cctv(inc.id, camera_id, check, m)
    if merged:   # the camera is looking at an incident someone already called in
        world._log(inc, "call", f"Camera alert from {camera_id} matches this incident: merged, not opened twice")
    else:
        world._log(inc, "call", f"Opened from a camera alert ({camera_id}), no caller")
    return {"incident": world.incident_json(inc), "check": check, "merged": merged,
            "message": f"{camera_id} alert matches {inc.id}: merged into it" if merged else None}


@app.post("/api/conditions")
def set_conditions(body: Conditions):
    return world.set_conditions(body.fog, body.night)


@app.post("/api/closures")
def close_road(body: LatLng):
    return world.close_road(body.lat, body.lng)


@app.delete("/api/closures")
def reopen_roads():
    return world.reopen_roads()


# ---------- rain and waterlogging ----------
class FloodStart(BaseModel):
    scenario: str
    mm_h: float | None = None
    hours: int | None = None


class FloodStep(BaseModel):
    step: int | None = None
    pumps: bool | None = None


@app.get("/api/flood/scenarios")
def flood_scenarios():
    return world.flood.scenarios()


@app.post("/api/flood/start")
async def flood_start(body: FloodStart):
    from starlette.concurrency import run_in_threadpool
    from .. import flood as F
    live = None
    if body.scenario == "live":
        try:
            live = await run_in_threadpool(F.fetch_live)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"Couldn't reach Open-Meteo for live rain ({type(e).__name__}). Check the internet "
                                     "connection, or pick a past storm to replay.") from e
    elif body.scenario == "custom":
        if not body.mm_h or not body.hours or not (0 < body.mm_h <= 200 and 0 < body.hours <= 24):
            raise HTTPException(400, "Give a rain rate (1-200 mm/h) and a number of hours (1-24).")
    elif body.scenario not in F.SCENARIOS:
        raise HTTPException(404, f"No scenario {body.scenario}")
    custom = {"mm_h": body.mm_h, "hours": body.hours} if body.scenario == "custom" else None
    return await run_in_threadpool(world.flood.start, body.scenario, custom, live)


@app.post("/api/flood/apply")
def flood_apply(body: FloodStep):
    if world.flood.run is None:
        raise HTTPException(400, "Start a rain scenario first.")
    if body.pumps is not None and body.pumps != world.flood.pumps:
        world.flood.pumps = body.pumps
    return world.flood.apply(world.flood.step if body.step is None else body.step)


@app.get("/api/flood/plan")
def flood_plan():
    if world.flood.run is None:
        return {"active": False}
    with world.lock:
        return world.flood.plan()


@app.delete("/api/flood")
def flood_clear():
    return world.flood.clear()


@app.get("/api/flood/risk")
async def flood_risk():
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(world.flood.risk)


@app.post("/api/flood/advisory")
async def flood_advisory():
    """Public advisory: Gemini rewrites the plan's facts in English, Hindi and Punjabi; a fixed template if it can't."""
    if world.flood.run is None:
        raise HTTPException(400, "Start a rain scenario first.")
    from starlette.concurrency import run_in_threadpool
    with world.lock:
        facts = world.flood.advisory_facts()
    if gemini is not None:
        try:
            adv, m = await asyncio.wait_for(run_in_threadpool(gemini.write_advisory, facts), timeout=GEMINI_WAIT_S)
            world.flood.last_advisory = adv.model_dump()
            return {"advisory": adv.model_dump(), "by": m.get("model", "gemini"), "latency_s": m.get("latency_s"), "facts": facts}
        except Exception as e:  # noqa: BLE001
            note = f"{ai_name()} didn't answer ({type(e).__name__}); this is the fixed template."
    else:
        note = "No AI engine: this is the fixed template."
    return {"advisory": world.flood.template(facts), "by": "template", "note": note, "facts": facts}


# ---------- predict and monitor ----------
@app.get("/api/predict/outlook")
async def predict_outlook(hours: int = 6):
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(world.watcher.outlook, max(1, min(12, hours)))


@app.get("/api/watch")
async def live_watch():
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(world.watcher.watch)


@app.post("/api/watch/{aid}/dismiss")
def watch_dismiss(aid: str):
    world.watcher.dismiss(aid)
    return {"ok": True}


@app.post("/api/units/{uid}/reposition")
def unit_reposition(uid: str, body: LatLng):
    if uid not in world.units:
        raise HTTPException(404, f"No unit {uid}")
    try:
        return world.watcher.reposition(uid, body.lat, body.lng)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


# ---------- digital public infrastructure ----------
@app.get("/api/dpi/catalogue")
def dpi_catalogue():
    from .dpi import CATALOGUE
    return {"interfaces": CATALOGUE, "api_docs": "/docs", "openapi": "/openapi.json", "licence": "MIT"}


@app.get("/api/dpi/outbox")
def dpi_outbox(since: int = 0, limit: int = 60):
    return {"messages": world.outbox.list(since, limit), "total": len(world.outbox.items)}


@app.get("/api/dpi/cap/flood.xml")
def dpi_cap_flood():
    from fastapi.responses import Response
    from .dpi import flood_cap
    with world.lock:
        xml = flood_cap(world, getattr(world.flood, "last_advisory", None))
    if xml is None:
        raise HTTPException(404, "No rain plan with closed or diverted roads right now. Start one on the Rain tab.")
    return Response(xml, media_type="application/xml", headers={"Content-Disposition": "inline; filename=openhand112-waterlogging-cap.xml"})


@app.post("/api/dpi/cap/flood/send")
def dpi_cap_send():
    from .dpi import flood_cap
    with world.lock:
        xml = flood_cap(world, getattr(world.flood, "last_advisory", None))
        if xml is None:
            raise HTTPException(404, "No rain plan with closed or diverted roads right now. Start one on the Rain tab.")
        p = world.flood.plan()
        n = sum(1 for a in p["actions"] if a["kind"] in ("close", "preclose"))
        m = world.outbox.send("out", "NDMA SACHET (CAP gateway)", "Public alert", "OASIS CAP 1.2 XML, status Exercise",
                              f"Waterlogging alert at {p['now']}: {n} road{'s' if n != 1 else ''} closed or closing, English, Hindi and Punjabi",
                              xml, fmt="xml")
    return m


@app.get("/api/open/incidents.geojson")
def open_geojson():
    from .dpi import open_geojson as g
    with world.lock:
        return g(world)


@app.get("/api/open/incidents.csv")
def open_csv():
    from fastapi.responses import Response
    from .dpi import open_csv as c
    with world.lock:
        return Response(c(world), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=openhand112_incidents.csv"})


@app.get("/api/risk")
def risk(kind: str = "crash"):
    if kind not in ("crash", "pcr", "ambulance", "fire"):
        raise HTTPException(400, "kind must be crash, pcr, ambulance or fire")
    return world.risk_grid(kind)


@app.get("/api/standby")
def standby():
    return world.standby_plan()


@app.get("/api/samples")
def samples():
    rows = []
    for f in ("data/triage_eval/calls.jsonl", "data/triage_eval/calls_india.jsonl"):
        p = ROOT / f
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [{"id": r["id"], "lang": r["lang"], "transcript": r["transcript"]} for r in rows]


class EngineBody(BaseModel):
    engine: str


@app.get("/api/ai")
def ai_engines():
    """Which AI reads calls, and what is available: cloud Gemini and/or Gemma on this computer."""
    return {"active": ENGINE if gemini else "none", "ai_name": ai_name(),
            "gemini": {"available": cloud_ai is not None, "model": getattr(cloud_ai, "model", None),
                       "fallbacks": getattr(cloud_ai, "fallbacks", []), "where": "Google Cloud (Gemini API)"},
            "local": local_ai.status() | {"where": "this computer (Ollama)", "audio": False}}


@app.post("/api/ai")
def ai_switch(body: EngineBody):
    global gemini, ENGINE
    if body.engine == "local":
        st = local_ai.status()
        if not st["ok"]:
            raise HTTPException(400, st["reason"])
        gemini, ENGINE = local_ai, "local"
    elif body.engine == "gemini":
        if cloud_ai is None:
            raise HTTPException(400, "No Gemini key in .env, so cloud Gemini isn't available.")
        gemini, ENGINE = cloud_ai, "gemini"
    else:
        raise HTTPException(400, "engine must be 'gemini' or 'local'")
    print(f"[ai] switched to {ENGINE}: {getattr(gemini, 'model', '')}", flush=True)
    return ai_engines()


# ---------- any city: the engine run on other cities (scripts/add_city.py) ----------
CITY_KEYS = ("slug", "name", "state", "place", "source", "generated", "junctions", "road_km", "police_stations", "fire_stations",
             "hospitals", "vans", "seconds")


def _city_files():
    return sorted((ROOT / "cities").glob("*/summary.json"))


@app.get("/api/cities")
def cities():
    out = []
    for f in _city_files():
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        out.append({k: d.get(k) for k in CITY_KEYS} | {"pcr": {k: v for k, v in d["pcr"].items() if k != "standby_points"},
                                                       "fire": {k: v for k, v in (d.get("fire") or {}).items() if k != "new_sites"} or None,
                                                       "hospital": d.get("hospital")})
    out.sort(key=lambda c: (c["slug"] != "chandigarh", c["name"]))
    return {"cities": out, "how": 'python3 scripts/add_city.py "Lucknow, Uttar Pradesh, India"'}


@app.get("/api/cities/{slug}")
def city(slug: str):
    f = ROOT / "cities" / slug / "summary.json"
    if not slug.replace("-", "").isalnum() or not f.exists():
        raise HTTPException(404, f"No city {slug}: run scripts/add_city.py")
    return json.loads(f.read_text())


@app.get("/api/evidence")
def evidence():
    out = {}
    for key, f in (("benchmark", "results/benchmark.json"), ("fire_siting", "results/fire_siting.json"),
                   ("triage_gemini", "results/triage_gemini.json"), ("triage_baseline", "results/triage_keyword_baseline.json"),
                   ("triage_local", "results/triage_local.json"),
                   ("india_gemini", "results/triage_india_gemini.json"), ("india_local", "results/triage_india_local.json"),
                   ("india_baseline", "results/triage_india_keyword_baseline.json")):
        p = ROOT / f
        if p.exists():
            d = json.loads(p.read_text())
            if key.startswith(("triage", "india")):
                d = {"summary": d["summary"], "system": d.get("system"), "model": d.get("model")}
            if key == "benchmark":
                for s in d.get("scenarios", {}).values():
                    s.pop("sweep", None)
                d.pop("robustness_raw", None)
            out[key] = d
    return out


@app.post("/api/reset")
def reset():
    global world
    world = World()
    return {"ok": True}


def _exists(iid: str):
    inc = world.incidents.get(iid)
    if inc is None:
        raise HTTPException(404, f"No incident {iid}")
    return inc


DIST = ROOT / "web/dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        # never cache the page itself, so a new version shows on a normal reload (assets have hashed names)
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-store"})
else:
    @app.get("/")
    def no_ui():
        return JSONResponse({"message": "API running. Build the dashboard in web/ (npm run build) to see the UI."})
