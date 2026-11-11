from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT / "src"))

from mushroom_monitor.config import AppPaths, DEFAULT_INFERENCE_SIDE
from mushroom_monitor.db import Database, utc_now
from mushroom_monitor.environment import growth_environment_join, parse_environment_csv
from mushroom_monitor.image_io import store_capture
from mushroom_monitor.maturity import rebuild_flush_groups
from mushroom_monitor.model_registry import install_model_bundle
from mushroom_monitor.learning import export_reviewed_yolo_dataset
from mushroom_monitor.review import export_reviewed_samples, review_prediction, replace_prediction_mask
from mushroom_monitor.service import ModelService, add_environment_rows
from mushroom_monitor.tracking import rebuild_sequence_tracks

st.set_page_config(page_title="Mushroom Growth Monitor", page_icon="🍄", layout="wide")
paths = AppPaths.from_env()
db = Database(paths.database)
db.ensure_defaults()

@st.cache_resource
def get_model_service(_db_path: str, _app_root: str):
    return ModelService(Database(Path(_db_path)), AppPaths.from_env())

service = get_model_service(str(paths.database), str(paths.root))


def data_frame(sql: str, params=()):
    return pd.DataFrame(db.query(sql, params))


def unit_options():
    return db.query("SELECT u.*,s.name AS site_name FROM grow_units u JOIN sites s ON s.id=u.site_id ORDER BY s.name,u.code")


def camera_options(unit_id: int):
    return db.query("SELECT * FROM cameras WHERE unit_id=? ORDER BY code", (unit_id,))


def model_options(key: str | None = None):
    sql = "SELECT * FROM model_versions WHERE status='active'"
    params = ()
    if key:
        sql += " AND model_key=?"
        params = (key,)
    sql += " ORDER BY model_key,installed_at DESC"
    return db.query(sql, params)


def show_status():
    registered = db.query("SELECT model_key,version,architecture,source_species,status FROM model_versions ORDER BY model_key,installed_at DESC")
    if not registered:
        st.warning("No model weights are installed. Import the M18K YOLO model bundle under Models before running inference.")
    return registered


def image_with_mask(capture_path: str, mask_path: str, caption: str):
    original = Image.open(capture_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(original.size, Image.Resampling.NEAREST)
    import numpy as np
    arr = np.asarray(original).copy()
    m = np.asarray(mask) > 0
    if m.any():
        # Red overlay; this is a visualization only and does not alter the stored source image.
        arr[m] = (0.65 * arr[m] + 0.35 * np.array([255, 30, 30])).astype("uint8")
    st.image(arr, caption=caption, width="stretch")


def refresh_after_review(prediction_id: int):
    row = db.one("""SELECT p.capture_id,p.model_version_id,m.model_key,c.unit_id,c.camera_id,c.series_key
                   FROM predictions p JOIN model_versions m ON m.id=p.model_version_id
                   JOIN captures c ON c.id=p.capture_id WHERE p.id=?""", (prediction_id,))
    if not row:
        return
    rebuild_flush_groups(db,row["capture_id"],model_version_id=int(row["model_version_id"]))
    if row["model_key"] in {"charisis_cluster_tracking", "m18k_yolo_seg"}:
        rebuild_sequence_tracks(db,row["unit_id"],row["camera_id"],row["series_key"],row["model_version_id"])


st.title("FungiCirno")
st.caption("Local mushroom monitoring: segmentation, tracking, growth measurements, environmental records, human review and model lifecycle management.")
page = st.sidebar.radio("Workspace", ["Overview", "Cultivation", "Capture images", "Run inference", "Growth tracking", "Maturity & grouping", "Environment", "Review & learning", "Learning lab", "Models"])

if page == "Overview":
    models = show_status()
    captures = db.one("SELECT COUNT(*) AS n FROM captures") or {"n": 0}
    tracks = db.one("SELECT COUNT(*) AS n FROM tracks") or {"n": 0}
    active_tracks = db.one("SELECT COUNT(*) AS n FROM tracks WHERE status='active'") or {"n": 0}
    pending = db.one("SELECT COUNT(*) AS n FROM predictions WHERE review_status='pending'") or {"n": 0}
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Captured images", captures["n"])
    c2.metric("Tracked clusters", tracks["n"])
    c3.metric("Active tracks", active_tracks["n"])
    c4.metric("Awaiting review", pending["n"])
    st.subheader("Latest captures")
    recent = data_frame("""SELECT c.id,c.original_name,c.captured_at,c.status,u.code AS unit,c.series_key,ca.code AS camera
                          FROM captures c JOIN grow_units u ON u.id=c.unit_id
                          JOIN cameras ca ON ca.id=c.camera_id ORDER BY c.imported_at DESC LIMIT 20""")
    st.dataframe(recent, width="stretch", hide_index=True)
    st.subheader("Installed models")
    if models:
        st.dataframe(pd.DataFrame(models)[["model_key","version","architecture","source_species","status"]], width="stretch", hide_index=True)
    else:
        st.info("Model weights are not included in the submitted result ZIPs. Install the exported model bundles from Colab first.")
    st.markdown("**Current direction:** M18K-trained YOLOv8-seg is the primary detector. Charisis contributes the mask-based temporal-tracking methodology; its original Mask R-CNN is not required. The WB-only M18K checkpoint is not a maturity classifier and has not yet been validated on side-view oyster images.")

elif page == "Cultivation":
    st.subheader("Sites, growing units and cameras")
    sites = db.query("SELECT * FROM sites ORDER BY name")
    st.subheader("Sites")
    st.dataframe(pd.DataFrame(sites), width="stretch", hide_index=True)

    # The previous UI only rendered the sites table. Units and cameras were
    # stored in SQLite and usable by downstream selectors, but never listed here.
    st.subheader("Growing units")
    units = unit_options()
    if units:
        unit_columns = ["id", "site_name", "code", "species", "cultivar", "substrate", "batch_code", "notes"]
        unit_frame = pd.DataFrame(units)
        st.dataframe(unit_frame[[c for c in unit_columns if c in unit_frame.columns]], width="stretch", hide_index=True)
    else:
        st.info("No growing units have been registered yet.")

    st.subheader("Cameras / views")
    camera_rows = db.query("""SELECT ca.id, s.name AS site, u.code AS growing_unit,
                             ca.code AS camera, ca.view_name AS view, ca.mm_per_px
                             FROM cameras ca JOIN grow_units u ON u.id=ca.unit_id
                             JOIN sites s ON s.id=u.site_id
                             ORDER BY s.name,u.code,ca.code""")
    if camera_rows:
        st.dataframe(pd.DataFrame(camera_rows), width="stretch", hide_index=True)
    else:
        st.info("No cameras have been registered yet.")

    with st.expander("Add site"):
        with st.form("add_site"):
            name = st.text_input("Site name", "Grow room 1")
            location = st.text_input("Location")
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Save site")
        if submitted:
            name = name.strip()
            if not name:
                st.error("Enter a site name.")
            else:
                try:
                    db.execute("INSERT INTO sites(name,location,notes,created_at) VALUES(?,?,?,?)", (name,location,notes,utc_now()))
                except sqlite3.IntegrityError:
                    st.error("A site with this name already exists. Choose a different name.")
                else:
                    st.success("Site added.")
                    st.rerun()
    sites = db.query("SELECT * FROM sites ORDER BY name")
    with st.expander("Add growing unit"):
        with st.form("add_unit"):
            site_map = {int(x["id"]): x for x in sites}
            site_id = st.selectbox("Site", list(site_map), format_func=lambda x: site_map[x]["name"], key="new_unit_site")
            code = st.text_input("Unit / column / tray code", "COLUMN-02")
            species = st.text_input("Species", "Pleurotus ostreatus")
            cultivar = st.text_input("Cultivar / strain")
            substrate = st.text_input("Substrate")
            batch = st.text_input("Batch code", "batch-01")
            notes = st.text_area("Notes", key="unit_notes")
            submitted = st.form_submit_button("Save growing unit")
        if submitted:
            code = code.strip()
            if not code:
                st.error("Enter a growing-unit code.")
            else:
                try:
                    db.execute("INSERT INTO grow_units(site_id,code,species,cultivar,substrate,batch_code,notes) VALUES(?,?,?,?,?,?,?)",
                               (site_id,code,species,cultivar,substrate,batch,notes))
                except sqlite3.IntegrityError:
                    st.error("That growing-unit code already exists at this site. Use a different code or select the existing unit.")
                else:
                    st.success("Growing unit added.")
                    st.rerun()
    units = unit_options()
    with st.expander("Add camera / view"):
        with st.form("add_camera"):
            unit_map = {int(x["id"]): x for x in units}
            unit = st.selectbox("Growing unit", list(unit_map), format_func=lambda x: f"{unit_map[x]['site_name']} / {unit_map[x]['code']}", key="new_cam_unit")
            cam_code = st.text_input("Camera code", "CAM-02")
            view = st.selectbox("View", ["side", "front", "top", "other"])
            mm_per_px = st.number_input("Calibration (mm per source-image pixel; leave 0 if unknown)", min_value=0.0, value=0.0, format="%.8f")
            submitted = st.form_submit_button("Save camera")
        if submitted:
            cam_code = cam_code.strip()
            if not cam_code:
                st.error("Enter a camera code.")
            else:
                try:
                    db.execute("INSERT INTO cameras(unit_id,code,view_name,mm_per_px) VALUES(?,?,?,?)",
                               (unit,cam_code,view,mm_per_px if mm_per_px > 0 else None))
                except sqlite3.IntegrityError:
                    st.error("That camera code already exists for this growing unit. Use a different code or select the existing camera.")
                else:
                    st.success("Camera added.")
                    st.rerun()
    st.subheader("Camera calibration")
    cams = db.query("""SELECT ca.*,u.code AS unit_code FROM cameras ca JOIN grow_units u ON u.id=ca.unit_id ORDER BY u.code,ca.code""")
    if cams:
        camera_map = {int(x["id"]): x for x in cams}
        camera_id = st.selectbox("Camera", list(camera_map),
                                 format_func=lambda x: f"{camera_map[x]['unit_code']} / {camera_map[x]['code']} ({camera_map[x]['view_name']})")
        camera = camera_map[camera_id]
        st.caption(f"Current scale: {camera['mm_per_px'] or 'not calibrated'} mm/source pixel")
        with st.form("calibrate_reference"):
            known_mm = st.number_input("Known reference dimension (mm)",min_value=0.0,value=float(camera["reference_width_mm"] or 0.0),format="%.3f")
            measured_px = st.number_input("Same reference dimension in image (pixels)",min_value=0.0,value=0.0,format="%.3f")
            submitted = st.form_submit_button("Calculate and save mm/pixel")
        if submitted:
            if known_mm<=0 or measured_px<=0:
                st.error("Enter a positive physical length and its measured pixel length.")
            else:
                scale=known_mm/measured_px
                db.execute("UPDATE cameras SET mm_per_px=?,reference_width_mm=? WHERE id=?",(scale,known_mm,camera_id))
                st.success(f"Saved {scale:.8f} mm/pixel.")
        with st.expander("Enter a previously measured scale"):
            value = st.number_input("mm per source-image pixel", min_value=0.0, value=float(camera["mm_per_px"] or 0.0), format="%.8f",key="direct_mm_per_px")
            if st.button("Update direct calibration"):
                db.execute("UPDATE cameras SET mm_per_px=? WHERE id=?", (value if value > 0 else None, camera_id))
                st.success("Calibration updated. Physical area is an approximation valid only for a fixed camera/reference plane.")
        st.caption("Use a known reference in the same plane as the measured mushrooms. A single mm/pixel scale is not perspective correction and may be invalid at different depths.")

elif page == "Capture images":
    st.subheader("Import capture images")
    units = unit_options()
    if not units:
        st.warning("Create a growing unit first.")
    else:
        unit_map = {int(x["id"]): x for x in units}
        unit_id = st.selectbox("Growing unit", list(unit_map), format_func=lambda x: f"{unit_map[x]['site_name']} / {unit_map[x]['code']} — {unit_map[x]['species']}")
        unit = unit_map[unit_id]
        cameras = camera_options(unit_id)
        camera_map = {int(x["id"]): x for x in cameras}
        camera_id = st.selectbox("Camera / view", list(camera_map), format_func=lambda x: f"{camera_map[x]['code']} ({camera_map[x]['view_name']})")
        camera = camera_map[camera_id]
        series = st.text_input("Capture series / batch sequence", unit["batch_code"] or "default")
        use_manual_time = st.checkbox("Use the same capture timestamp for these images")
        if use_manual_time:
            manual_date = st.date_input("Capture date", value=date.today())
            manual_time = st.time_input("Capture time", value=time(9, 0))
            manual_dt = datetime.combine(manual_date, manual_time).astimezone().isoformat()
        else:
            manual_dt = None
        uploads = st.file_uploader("Image files", type=["jpg","jpeg","png","tif","tiff","webp","bmp"], accept_multiple_files=True)
        if st.button("Import images", disabled=not uploads):
            added, skipped, errors = 0, 0, []
            for upload in uploads:
                try:
                    raw = upload.getvalue()
                    file_path, digest, width, height, exif_dt = store_capture(raw, upload.name, paths.captures)
                    existing = db.one("SELECT id FROM captures WHERE sha256=?", (digest,))
                    if existing:
                        skipped += 1
                        continue
                    captured_at = manual_dt or (exif_dt.isoformat() if exif_dt else None)
                    time_source = "manual" if manual_dt else ("exif" if exif_dt else "unknown")
                    capture_id = db.create_capture(
                        unit_id=unit["id"],camera_id=camera["id"],series_key=series.strip() or "default",
                        original_name=upload.name,file_path=str(file_path),sha256=digest,captured_at=captured_at,
                        time_source=time_source,imported_at=utc_now(),width=width,height=height,source="upload",
                        metadata_json=json.dumps({"content_type": upload.type}),status="pending",
                    )
                    added += 1
                except Exception as exc:
                    errors.append(f"{upload.name}: {type(exc).__name__}: {exc}")
            st.success(f"Imported {added}; skipped duplicates {skipped}; errors {len(errors)}.")
            for error in errors:
                st.error(error)
        st.caption("Capture timestamps use your supplied time first, then EXIF. Images without either remain undated; import time is not treated as capture time.")
    recent = data_frame("""SELECT c.id,c.original_name,c.captured_at,c.time_source,c.width,c.height,c.status,u.code AS unit,ca.code AS camera
                          FROM captures c JOIN grow_units u ON u.id=c.unit_id JOIN cameras ca ON ca.id=c.camera_id
                          ORDER BY c.imported_at DESC LIMIT 40""")
    st.dataframe(recent, width="stretch", hide_index=True)

elif page == "Run inference":
    st.subheader("Run an installed model")
    models = model_options()
    if not models:
        show_status()
    else:
        model_map = {int(x["id"]): x for x in models}
        model_id = st.selectbox("Model version", list(model_map), format_func=lambda x: f"{model_map[x]['model_key']} / {model_map[x]['version']} ({model_map[x]['architecture']})")
        model = model_map[model_id]
        config = json.loads(model["config_json"])
        threshold_default = float(config.get("confidence_threshold", 0.30 if model["model_key"] == "charisis_cluster_tracking" else 0.50))
        units = unit_options()
        unit_map = {int(x["id"]): x for x in units}
        unit_id = st.selectbox("Growing unit", list(unit_map), format_func=lambda x: f"{unit_map[x]['site_name']} / {unit_map[x]['code']}")
        unit = unit_map[unit_id]
        cameras = camera_options(unit_id)
        camera_map = {int(x["id"]): x for x in cameras}
        camera_id = st.selectbox("Camera", list(camera_map), format_func=lambda x: camera_map[x]["code"])
        camera = camera_map[camera_id]
        series = st.text_input("Series key", unit["batch_code"] or "default")
        pending_captures = db.query("""SELECT * FROM captures WHERE unit_id=? AND camera_id=? AND series_key=?
                                       ORDER BY COALESCE(captured_at,imported_at),id""",
                                    (unit_id,camera_id,series))
        if pending_captures:
            capture_map = {int(r["id"]): r for r in pending_captures}
            default_ids = [int(r["id"]) for r in pending_captures if r["status"] != "processed" or not db.one("SELECT id FROM predictions WHERE capture_id=? AND model_version_id=?", (r["id"],model["id"]))]
            choice_ids = st.multiselect("Captures to process", list(capture_map), default=default_ids,
                                     format_func=lambda x: f"#{x} {capture_map[x]['original_name']} — {capture_map[x]['captured_at'] or 'time unknown'}")
            choices = [capture_map[x] for x in choice_ids]
            side = st.select_slider("Inference image long side", options=[512,640,768,896,1024], value=min(768, DEFAULT_INFERENCE_SIDE))
            threshold = st.slider("Minimum confidence used in application outputs", 0.05, 0.95, threshold_default, 0.01)
            device = st.selectbox("Compute backend", ["amd", "cpu"], index=0,
                                  format_func=lambda x: "AMD Radeon RX 5500 XT (DirectML)" if x == "amd" else "CPU (compatibility fallback)",
                                  help="Choose AMD for acceleration or CPU when DirectML cannot execute an operator. The backend is recorded for every inference run.")
            if st.button("Process selected captures", disabled=not choices):
                progress = st.progress(0.0)
                messages = st.empty()
                successes, failures = [], []
                for index, capture in enumerate(choices, start=1):
                    try:
                        # Every inference attempt is retained as an immutable run record.
                        result = service.run_capture(capture["id"],model["id"],inference_side=side,
                                                     device_preference=device,final_threshold=threshold,
                                                     rebuild_derived=False)
                        successes.append((capture["id"],result))
                        messages.write(f"Processed {index}/{len(choices)}: capture #{capture['id']} — {result['detection_count']} candidate masks")
                    except Exception as exc:
                        failures.append((capture["id"],str(exc)))
                        messages.error(f"Capture #{capture['id']}: {type(exc).__name__}: {exc}")
                    progress.progress(index / len(choices))
                if model["model_key"] in {"charisis_cluster_tracking", "m18k_yolo_seg"}:
                    tracked = rebuild_sequence_tracks(db,unit["id"],camera["id"],series,model["id"])
                    st.write("Tracking summary", tracked)
                for capture in choices:
                    rebuild_flush_groups(db,int(capture["id"]),model_version_id=int(model["id"]))
                st.success(f"Completed {len(successes)} captures; failed {len(failures)}.")
                if failures:
                    st.warning("Failures are preserved below and in SQLite. Fix the reported issue, then rerun only failed captures.")
        else:
            st.info("No captures found for this unit/camera/series. Import images first.")

    st.subheader("Recent inference runs and retained errors")
    run_history = data_frame("""SELECT r.id AS run_id,c.original_name,m.model_key,m.version,r.started_at,r.status,
                                  r.detection_count,r.device,r.duration_seconds,r.error
                           FROM inference_runs r JOIN captures c ON c.id=r.capture_id
                           JOIN model_versions m ON m.id=r.model_version_id
                           ORDER BY r.id DESC LIMIT 30""")
    st.dataframe(run_history, width="stretch", hide_index=True)

elif page == "Growth tracking":
    st.subheader("Cluster growth")
    units = unit_options()
    if units:
        unit_map = {int(x["id"]): x for x in units}
        unit_id = st.selectbox("Growing unit", list(unit_map), format_func=lambda x: f"{unit_map[x]['site_name']} / {unit_map[x]['code']}")
        cameras = camera_options(unit_id)
        camera_map = {int(x["id"]): x for x in cameras}
        camera_id = st.selectbox("Camera", list(camera_map), format_func=lambda x: camera_map[x]["code"])
        series_rows = db.query("SELECT DISTINCT series_key FROM captures WHERE unit_id=? AND camera_id=? ORDER BY series_key", (unit_id,camera_id))
        if series_rows:
            series = st.selectbox("Series", [r["series_key"] for r in series_rows])
            track_models = db.query("""SELECT DISTINCT t.model_version_id,m.model_key,m.version FROM tracks t
                                       JOIN model_versions m ON m.id=t.model_version_id
                                       WHERE t.unit_id=? AND t.camera_id=? AND t.series_key=? ORDER BY m.installed_at DESC""",
                                    (unit_id,camera_id,series))
            if track_models:
                tm = {int(r["model_version_id"]):r for r in track_models}
                track_model_id = st.selectbox("Tracking model version", list(tm),
                    format_func=lambda x:f"{tm[x]['model_key']} / {tm[x]['version']}")
                track_rows = db.query("""SELECT t.id,t.track_key,t.class_name,t.observations,t.growth_pct,t.status,
                                       MIN(o.frame_index) AS first_frame,MAX(o.frame_index) AS last_frame
                                       FROM tracks t JOIN track_observations o ON o.track_id=t.id
                                       WHERE t.unit_id=? AND t.camera_id=? AND t.series_key=? AND t.model_version_id=?
                                       GROUP BY t.id ORDER BY t.observations DESC""", (unit_id,camera_id,series,track_model_id))
            else:
                track_rows=[]
            if track_rows:
                st.dataframe(pd.DataFrame(track_rows), width="stretch", hide_index=True)
                track_map = {int(x["id"]): x for x in track_rows}
                selected_id = st.selectbox("Track", list(track_map),
                    format_func=lambda x: f"{track_map[x]['track_key']} — {track_map[x]['observations']} observations / {float(track_map[x]['growth_pct'] or 0):.1f}% area change")
                selected = track_map[selected_id]
                observations = db.query("""SELECT o.frame_index,o.captured_at,o.area_px2,o.area_mm2,o.growth_vs_first_pct,
                                       c.original_name FROM track_observations o JOIN captures c ON c.id=o.capture_id
                                       WHERE o.track_id=? ORDER BY o.frame_index""", (selected_id,))
                chart = pd.DataFrame(observations)
                st.line_chart(chart.set_index("frame_index")[["area_px2"]])
                st.line_chart(chart.set_index("frame_index")[["growth_vs_first_pct"]])
                if chart["area_mm2"].notna().any():
                    st.line_chart(chart.set_index("frame_index")[["area_mm2"]])
                st.dataframe(chart, width="stretch", hide_index=True)
                st.download_button("Download track observations CSV",data=chart.to_csv(index=False),
                                   file_name=f"track_{selected_id}_observations.csv",mime="text/csv")
                st.caption("Area is an image-pixel estimate. mm² is shown only when a camera calibration has been entered. Growth is compared within one camera/view/series.")
            else:
                st.info("No tracks yet. Process captures with the active M18K or supported cluster-segmentation model.")

elif page == "Maturity & grouping":
    st.subheader("Spatial groups and maturity outputs")
    units = unit_options()
    if units:
        unit_map = {int(x["id"]): x for x in units}
        unit_id = st.selectbox("Growing unit", list(unit_map), format_func=lambda x: f"{unit_map[x]['site_name']} / {unit_map[x]['code']}")
        rows = db.query("""SELECT f.*,c.original_name,c.captured_at,u.code AS unit,m.model_key,m.version
                          FROM flush_groups f JOIN captures c ON c.id=f.capture_id
                          JOIN grow_units u ON u.id=c.unit_id LEFT JOIN model_versions m ON m.id=f.model_version_id
                          WHERE c.unit_id=? ORDER BY COALESCE(c.captured_at,c.imported_at) DESC""", (unit_id,))
        if rows:
            frame = pd.DataFrame(rows)
            frame["class_area_json"] = frame["class_area_json"].apply(json.loads)
            st.dataframe(frame.drop(columns=["class_area_json"]), width="stretch", hide_index=True)
            st.caption("Groups describe spatially adjacent detections within one capture. They are not persistent track IDs. A WB-only M18K model detects White Button instances; it does not infer developmental maturity stages. Maturity labels appear only when the active model has stage-labelled classes.")
        else:
            st.info("No spatial groups yet. Process captures with an active segmentation model.")

elif page == "Environment":
    st.subheader("Environmental readings")
    units = unit_options()
    if units:
        unit_map = {int(x["id"]): x for x in units}
        unit_id = st.selectbox("Growing unit", list(unit_map), format_func=lambda x: f"{unit_map[x]['site_name']} / {unit_map[x]['code']}")
        uploaded = st.file_uploader("Import CSV", type=["csv"])
        if uploaded and st.button("Import readings"):
            temp_path = paths.exports / "environment_upload.csv"
            temp_path.write_bytes(uploaded.getvalue())
            try:
                parsed = parse_environment_csv(temp_path)
                inserted = add_environment_rows(db,unit_id,parsed,"csv")
                st.success(f"Imported {inserted} readings.")
            except Exception as exc:
                st.error(str(exc))
        readings = data_frame("SELECT measured_at,temperature_c,humidity_pct,co2_ppm,notes FROM environmental_readings WHERE unit_id=? ORDER BY measured_at", (unit_id,))
        if not readings.empty:
            st.dataframe(readings, width="stretch", hide_index=True)
            st.download_button("Download environmental readings CSV",data=readings.to_csv(index=False),file_name=f"environment_unit_{unit_id}.csv",mime="text/csv")
            if readings[["temperature_c","humidity_pct","co2_ppm"]].notna().any().any():
                plot = readings.melt(id_vars=["measured_at"], var_name="metric", value_name="value").dropna()
                st.line_chart(plot.pivot(index="measured_at",columns="metric",values="value"))
        joined = pd.DataFrame(growth_environment_join(db,unit_id))
        if not joined.empty:
            st.subheader("Nearest environmental reading vs tracked growth")
            st.dataframe(joined, width="stretch", hide_index=True)
            st.caption("This is a time-aligned descriptive comparison, not a causal analysis.")
            metric_cols=[c for c in ["temperature_c","humidity_pct","co2_ppm","growth_vs_first_pct"] if c in joined.columns]
            if len(joined)>=3 and len(metric_cols)>=2:
                st.subheader("Exploratory correlations")
                st.dataframe(joined[metric_cols].corr(numeric_only=True),width="stretch")
                st.caption("Exploratory Pearson correlations only. Small samples, temporal autocorrelation and confounding prevent causal interpretation.")

elif page == "Review & learning":
    st.subheader("Human review queue")
    queue = db.query("""SELECT p.*,c.file_path,c.original_name,c.width,c.height,c.captured_at,m.model_key,m.classes_json
                       FROM predictions p JOIN captures c ON c.id=p.capture_id
                       JOIN model_versions m ON m.id=p.model_version_id
                       WHERE p.review_status='pending' ORDER BY p.confidence DESC,p.id DESC LIMIT 300""")
    if queue:
        st.caption("Candidates are ordered from highest to lowest confidence so the strongest current detections are inspected first. Lower-confidence candidates remain in the queue.")
        queue_map = {int(x["id"]): x for x in queue}
        selected_id = st.selectbox("Prediction", list(queue_map), format_func=lambda x: f"#{x} {queue_map[x]['model_key']} — {queue_map[x]['class_name']} {queue_map[x]['confidence']:.3f} — {queue_map[x]['original_name']}")
        selected = queue_map[selected_id]
        c1,c2 = st.columns(2)
        with c1:
            st.image(selected["file_path"], caption=selected["original_name"], width="stretch")
        with c2:
            image_with_mask(selected["file_path"],selected["mask_path"],f"Prediction #{selected['id']} — {selected['class_name']} ({selected['confidence']:.3f})")
        class_map = json.loads(selected["classes_json"])
        choices = list(class_map.values())
        corrected = st.selectbox("Corrected class (used only if you choose Correct)", choices)
        reviewer = st.text_input("Reviewer", "local_user")
        notes = st.text_area("Review notes")
        corrected_mask_upload = st.file_uploader("Optional corrected binary mask (black background, white foreground)",
                                                   type=["png","tif","tiff"], key=f"maskfix_{selected_id}")
        if corrected_mask_upload and st.button("Save corrected mask", key=f"save_mask_{selected_id}"):
            try:
                path = replace_prediction_mask(db,selected_id,corrected_mask_upload.getvalue(),paths.masks,reviewer,notes)
                refresh_after_review(selected_id)
                st.success(f"Corrected mask saved: {path.name}")
                st.rerun()
            except Exception as exc: st.error(f"Mask correction failed: {exc}")
        a,b,c = st.columns(3)
        if a.button("Accept mask and class"):
            review_prediction(db,selected["id"],"accepted",reviewer=reviewer,notes=notes)
            refresh_after_review(selected["id"])
            st.success("Accepted.")
            st.rerun()
        if b.button("Correct class"):
            review_prediction(db,selected["id"],"corrected",reviewer=reviewer,corrected_class=corrected,notes=notes)
            refresh_after_review(selected["id"])
            st.success("Class corrected; predicted mask retained.")
            st.rerun()
        if c.button("Reject prediction"):
            review_prediction(db,selected["id"],"rejected",reviewer=reviewer,notes=notes)
            refresh_after_review(selected["id"])
            st.success("Rejected.")
            st.rerun()
    else:
        st.info("No pending predictions. Low-confidence detections appear here for human review.")
    st.divider()
    st.subheader("Export reviewed samples for Colab")
    model_keys = db.query("SELECT DISTINCT model_key FROM model_versions ORDER BY model_key")
    if model_keys:
        model_key = st.selectbox("Model family", [r["model_key"] for r in model_keys])
        if st.button("Export accepted / corrected samples"):
            try:
                destination = paths.exports / f"reviewed_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                path, count = export_reviewed_samples(db,destination,model_key)
                st.success(f"Exported {count} reviewed annotations.")
                st.download_button("Download reviewed dataset", data=path.read_bytes(), file_name=path.name, mime="application/zip")
            except Exception as exc:
                st.error(str(exc))

elif page == "Learning lab":
    st.subheader("Human-reviewed incremental learning")
    st.caption("This workflow exports reviewed predictions for offline Colab fine-tuning. It does not train inside the local app or automatically replace the active model.")
    summary=data_frame("""SELECT m.model_key,COUNT(*) AS predictions,
                       SUM(CASE WHEN p.review_status='pending' THEN 1 ELSE 0 END) AS pending,
                       SUM(CASE WHEN p.review_status='accepted' THEN 1 ELSE 0 END) AS accepted,
                       SUM(CASE WHEN p.review_status='corrected' THEN 1 ELSE 0 END) AS corrected,
                       SUM(CASE WHEN p.review_status='rejected' THEN 1 ELSE 0 END) AS rejected
                       FROM predictions p JOIN model_versions m ON m.id=p.model_version_id GROUP BY m.model_key""")
    st.dataframe(summary,width="stretch",hide_index=True)
    model_keys=db.query("SELECT DISTINCT model_key FROM model_versions ORDER BY model_key")
    if model_keys:
        key=st.selectbox("Model family to export",[r["model_key"] for r in model_keys],key="learning_model_key")
        reviewer_note=st.text_area("Dataset notes",placeholder="Side-view target domain, capture source, annotation review notes...")
        if st.button("Build reviewed YOLO-seg dataset ZIP"):
            try:
                out=paths.exports/f"reviewed_yolo_{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                exported,count=export_reviewed_yolo_dataset(db,out,key,reviewer_note)
                st.success(f"Prepared {count} reviewed instance masks.")
                st.download_button("Download training dataset",data=exported.read_bytes(),file_name=exported.name,mime="application/zip")
            except Exception as exc: st.error(f"Dataset export failed: {type(exc).__name__}: {exc}")
    st.markdown("### Candidate training (Colab)")
    st.code("""# Learning Lab ZIP is already YOLO-seg format:
python -m zipfile -e reviewed_yolo_dataset.zip reviewed_dataset
python tools/train_candidate.py --weights m18k_wb_best.pt \
  --data reviewed_dataset/data.yaml --holdout-data sideview_holdout.yaml \
  --source-holdout-data m18k_source_holdout.yaml --epochs 50 --imgsz 640 --batch 8 --device 0 \
  --species 'Pleurotus pulmonarius' --taxonomy sideview_mushroom_instance \
  --out candidate_bundle.zip

# Alternative: convert the COCO ZIP exported from Review & Learning:
python tools/prepare_reviewed_dataset.py reviewed_coco.zip --out reviewed_dataset""",language="bash")
    st.warning("Do not use reviewed predictions as unquestioned ground truth. Inspect/correct masks, keep a separate sequence- or farm-level holdout, and evaluate baseline and candidate on exactly the same holdout before activation.")
    jobs=data_frame("SELECT id,model_key,created_at,finished_at,status,dataset_path,candidate_path,metrics_json,error FROM learning_jobs ORDER BY id DESC LIMIT 20")
    if not jobs.empty:
        st.subheader("Learning job history"); st.dataframe(jobs,width="stretch",hide_index=True)
    exports=data_frame("SELECT id,created_at,path,sample_count,model_key,notes FROM learning_exports ORDER BY id DESC LIMIT 20")
    if not exports.empty:
        st.subheader("Reviewed dataset exports"); st.dataframe(exports,width="stretch",hide_index=True)

elif page == "Models":
    st.subheader("Model registry")
    st.write("Import a ZIP containing an Ultralytics YOLO segmentation checkpoint (`best.pt`), `model_config.json`, and `class_mapping.json`. Legacy TorchVision bundles remain supported for research comparison.")
    uploaded_bundle = st.file_uploader("Model bundle ZIP", type=["zip"])
    if uploaded_bundle and st.button("Install model bundle"):
        incoming = paths.incoming_models / uploaded_bundle.name
        incoming.write_bytes(uploaded_bundle.getvalue())
        try:
            model_id = install_model_bundle(incoming,paths.model_registry,db)
            service.clear_model_cache()
            st.success(f"Installed as model version #{model_id}. It is a candidate; activate it after checking its metrics.")
        except Exception as exc:
            st.error(f"Bundle rejected: {type(exc).__name__}: {exc}")
    registry = db.query("SELECT * FROM model_versions ORDER BY model_key,installed_at DESC")
    if registry:
        st.dataframe(pd.DataFrame(registry)[["id","model_key","version","architecture","source_species","status","installed_at","checkpoint_sha256"]], width="stretch", hide_index=True)
        registry_map = {int(x["id"]): x for x in registry}
        candidate_id = st.selectbox("Model version to activate / archive", list(registry_map),
                                 format_func=lambda x: f"#{x} {registry_map[x]['model_key']} {registry_map[x]['version']} [{registry_map[x]['status']}]")
        candidate = registry_map[candidate_id]
        candidate_metrics = json.loads(candidate.get("metrics_json") or "{}")
        if candidate_metrics:
            st.subheader("Selected model evaluation")
            st.json(candidate_metrics)
        force_activate = st.checkbox("Allow activation without a passing independent holdout report", value=False,
                                     help="Use only for an initial prototype or an explicitly reviewed exception.")
        if st.button("Set active"):
            try:
                db.set_model_status(candidate["id"],"active",force=force_activate)
                service.clear_model_cache()
                st.success("Model activated. The previous active version remains installed as a rollback candidate.")
                st.rerun()
            except Exception as exc:
                st.error(f"Activation blocked: {exc}")
        if st.button("Archive version"):
            db.set_model_status(candidate["id"],"archived")
            service.clear_model_cache()
            st.success("Model archived.")
            st.rerun()
    st.info("Candidate models are not used until activated. Keep the previous active version installed for rollback. Promotion evidence should include an independent holdout evaluation, not training metrics alone.")
