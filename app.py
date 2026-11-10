from __future__ import annotations

import json
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
from mushroom_monitor.review import export_reviewed_samples, review_prediction
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
    registered = db.query("SELECT model_key,version,status FROM model_versions ORDER BY model_key,installed_at DESC")
    if not registered:
        st.warning("No trained model bundles are installed. Import the members' model_bundle.zip files under Models before running inference.")
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
    st.image(arr, caption=caption, use_container_width=True)


def refresh_after_review(prediction_id: int):
    row = db.one("""SELECT p.capture_id,p.model_version_id,m.model_key,c.unit_id,c.camera_id,c.series_key
                   FROM predictions p JOIN model_versions m ON m.id=p.model_version_id
                   JOIN captures c ON c.id=p.capture_id WHERE p.id=?""", (prediction_id,))
    if not row:
        return
    if row["model_key"] == "charisis_cluster_tracking":
        rebuild_sequence_tracks(db,row["unit_id"],row["camera_id"],row["series_key"],row["model_version_id"])
    elif row["model_key"] == "mushr_maturity":
        rebuild_flush_groups(db,row["capture_id"])


st.title("Mushroom Growth Monitor")
st.caption("Local capture records, model inference, cluster tracking, growth measurements, maturity grouping and human-reviewed retraining data.")
page = st.sidebar.radio("Workspace", ["Overview", "Cultivation", "Capture images", "Run inference", "Growth tracking", "Maturity", "Environment", "Review & learning", "Models"])

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
    st.dataframe(recent, use_container_width=True, hide_index=True)
    st.subheader("Installed models")
    if models:
        st.dataframe(pd.DataFrame(models)[["model_key","version","architecture","source_species","status"]], use_container_width=True, hide_index=True)
    else:
        st.info("Model weights are not included in the submitted result ZIPs. Install the exported model bundles from Colab first.")
    st.markdown("**Supported experiments:** Charisis oyster-cluster segmentation/tracking and MushR oyster maturity segmentation. Their class taxonomies remain separate. Other mushroom species can be recorded, but these two models have not been trained to recognize them.")

elif page == "Cultivation":
    st.subheader("Sites, growing units and cameras")
    sites = db.query("SELECT * FROM sites ORDER BY name")
    st.dataframe(pd.DataFrame(sites), use_container_width=True, hide_index=True)
    with st.expander("Add site"):
        with st.form("add_site"):
            name = st.text_input("Site name", "Grow room 1")
            location = st.text_input("Location")
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Save site")
        if submitted:
            db.execute("INSERT INTO sites(name,location,notes,created_at) VALUES(?,?,?,?)", (name,location,notes,utc_now()))
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
            db.execute("INSERT INTO grow_units(site_id,code,species,cultivar,substrate,batch_code,notes) VALUES(?,?,?,?,?,?,?)",
                       (site_id,code,species,cultivar,substrate,batch,notes))
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
            db.execute("INSERT INTO cameras(unit_id,code,view_name,mm_per_px) VALUES(?,?,?,?)",
                       (unit,cam_code,view,mm_per_px if mm_per_px > 0 else None))
            st.success("Camera added.")
            st.rerun()
    st.subheader("Camera calibration")
    cams = db.query("""SELECT ca.*,u.code AS unit_code FROM cameras ca JOIN grow_units u ON u.id=ca.unit_id ORDER BY u.code,ca.code""")
    if cams:
        camera_map = {int(x["id"]): x for x in cams}
        camera_id = st.selectbox("Camera", list(camera_map),
                                 format_func=lambda x: f"{camera_map[x]['unit_code']} / {camera_map[x]['code']} ({camera_map[x]['view_name']})")
        camera = camera_map[camera_id]
        value = st.number_input("mm per pixel", min_value=0.0, value=float(camera["mm_per_px"] or 0.0), format="%.8f")
        if st.button("Update calibration"):
            db.execute("UPDATE cameras SET mm_per_px=? WHERE id=?", (value if value > 0 else None, camera_id))
            st.success("Calibration updated. Physical area will remain blank until a valid scale is entered.")

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
    st.dataframe(recent, use_container_width=True, hide_index=True)

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
            device = st.selectbox("Compute backend", ["auto", "amd", "cpu"], index=0,
                                  help="Auto tries PyTorch GPU/ROCm, then DirectML on Windows, then CPU. The first DirectML inference tests operator support and falls back to CPU if needed.")
            if st.button("Process selected captures", disabled=not choices):
                progress = st.progress(0.0)
                messages = st.empty()
                successes, failures = [], []
                for index, capture in enumerate(choices, start=1):
                    try:
                        # Remove previous predictions for this model/capture before an explicit rerun.
                        old_runs = db.query("SELECT id FROM inference_runs WHERE capture_id=? AND model_version_id=?", (capture["id"],model["id"]))
                        for old in old_runs:
                            old_mask_dir = paths.masks / f"capture_{capture['id']}" / f"run_{old['id']}"
                            db.execute("DELETE FROM inference_runs WHERE id=?", (old["id"],))
                            if old_mask_dir.exists():
                                import shutil
                                shutil.rmtree(old_mask_dir, ignore_errors=True)
                        result = service.run_capture(capture["id"],model["id"],inference_side=side,
                                                     device_preference=device,final_threshold=threshold,
                                                     rebuild_derived=False)
                        successes.append((capture["id"],result))
                        messages.write(f"Processed {index}/{len(choices)}: capture #{capture['id']} — {result['detection_count']} candidate masks")
                    except Exception as exc:
                        failures.append((capture["id"],str(exc)))
                        messages.error(f"Capture #{capture['id']}: {type(exc).__name__}: {exc}")
                    progress.progress(index / len(choices))
                if model["model_key"] == "charisis_cluster_tracking":
                    tracked = rebuild_sequence_tracks(db,unit["id"],camera["id"],series,model["id"])
                    st.write("Tracking summary", tracked)
                else:
                    for capture in choices:
                        rebuild_flush_groups(db,int(capture["id"]))
                st.success(f"Completed {len(successes)} captures; failed {len(failures)}.")
                if failures:
                    st.warning("Some captures failed. Their error details are shown above; other captures were still processed.")
                st.rerun()
        else:
            st.info("No captures found for this unit/camera/series. Import images first.")

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
            track_rows = db.query("""SELECT t.id,t.track_key,t.class_name,t.observations,t.growth_pct,t.status,
                                   MIN(o.frame_index) AS first_frame,MAX(o.frame_index) AS last_frame
                                   FROM tracks t JOIN track_observations o ON o.track_id=t.id
                                   WHERE t.unit_id=? AND t.camera_id=? AND t.series_key=?
                                   GROUP BY t.id ORDER BY t.observations DESC""", (unit_id,camera_id,series))
            if track_rows:
                st.dataframe(pd.DataFrame(track_rows), use_container_width=True, hide_index=True)
                track_map = {int(x["id"]): x for x in track_rows}
                selected_id = st.selectbox("Track", list(track_map),
                    format_func=lambda x: f"{track_map[x]['track_key']} — {track_map[x]['observations']} observations / {float(track_map[x]['growth_pct'] or 0):.1f}% area change")
                selected = track_map[selected_id]
                observations = db.query("""SELECT o.frame_index,o.captured_at,o.area_px2,o.area_mm2,o.growth_vs_first_pct,
                                       c.original_name FROM track_observations o JOIN captures c ON c.id=o.capture_id
                                       WHERE o.track_id=? ORDER BY o.frame_index""", (selected_id,))
                chart = pd.DataFrame(observations)
                st.line_chart(chart.set_index("frame_index")[["area_px2"]])
                if chart["area_mm2"].notna().any():
                    st.line_chart(chart.set_index("frame_index")[["area_mm2"]])
                st.dataframe(chart, use_container_width=True, hide_index=True)
                st.caption("Area is an image-pixel estimate. mm² is shown only when a camera calibration has been entered. Growth is compared within one camera/view/series.")
            else:
                st.info("No tracks yet. Process captures with the Charisis cluster model.")

elif page == "Maturity":
    st.subheader("Maturity and spatial flushes")
    units = unit_options()
    if units:
        unit_map = {int(x["id"]): x for x in units}
        unit_id = st.selectbox("Growing unit", list(unit_map), format_func=lambda x: f"{unit_map[x]['site_name']} / {unit_map[x]['code']}")
        rows = db.query("""SELECT f.*,c.original_name,c.captured_at,u.code AS unit FROM flush_groups f
                          JOIN captures c ON c.id=f.capture_id JOIN grow_units u ON u.id=c.unit_id
                          WHERE c.unit_id=? ORDER BY COALESCE(c.captured_at,c.imported_at) DESC""", (unit_id,))
        if rows:
            frame = pd.DataFrame(rows)
            frame["class_area_json"] = frame["class_area_json"].apply(json.loads)
            st.dataframe(frame.drop(columns=["class_area_json"]), use_container_width=True, hide_index=True)
            st.caption("MushR maturity classes are specific to the oyster-mushroom dataset. Flush groups are spatial groups within one image, not persistent temporal IDs.")
        else:
            st.info("No maturity outputs yet. Process captures with the MushR maturity model.")

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
            st.dataframe(readings, use_container_width=True, hide_index=True)
            if readings[["temperature_c","humidity_pct","co2_ppm"]].notna().any().any():
                plot = readings.melt(id_vars=["measured_at"], var_name="metric", value_name="value").dropna()
                st.line_chart(plot.pivot(index="measured_at",columns="metric",values="value"))
        joined = pd.DataFrame(growth_environment_join(db,unit_id))
        if not joined.empty:
            st.subheader("Nearest environmental reading vs tracked growth")
            st.dataframe(joined, use_container_width=True, hide_index=True)
            st.caption("This is a time-aligned descriptive comparison, not a causal analysis.")

elif page == "Review & learning":
    st.subheader("Human review queue")
    queue = db.query("""SELECT p.*,c.file_path,c.original_name,c.width,c.height,c.captured_at,m.model_key,m.classes_json
                       FROM predictions p JOIN captures c ON c.id=p.capture_id
                       JOIN model_versions m ON m.id=p.model_version_id
                       WHERE p.review_status='pending' ORDER BY p.confidence ASC LIMIT 300""")
    if queue:
        queue_map = {int(x["id"]): x for x in queue}
        selected_id = st.selectbox("Prediction", list(queue_map), format_func=lambda x: f"#{x} {queue_map[x]['model_key']} — {queue_map[x]['class_name']} {queue_map[x]['confidence']:.3f} — {queue_map[x]['original_name']}")
        selected = queue_map[selected_id]
        c1,c2 = st.columns(2)
        with c1:
            st.image(selected["file_path"], caption=selected["original_name"], use_container_width=True)
        with c2:
            image_with_mask(selected["file_path"],selected["mask_path"],f"Prediction #{selected['id']} — {selected['class_name']} ({selected['confidence']:.3f})")
        class_map = json.loads(selected["classes_json"])
        choices = list(class_map.values())
        corrected = st.selectbox("Corrected class (used only if you choose Correct)", choices)
        reviewer = st.text_input("Reviewer", "local_user")
        notes = st.text_area("Review notes")
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

elif page == "Models":
    st.subheader("Model registry")
    st.write("Import the `model_bundle.zip` exported by each Colab notebook. Keep the Charisis and MushR model families separate.")
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
        st.dataframe(pd.DataFrame(registry)[["id","model_key","version","architecture","source_species","status","installed_at","checkpoint_sha256"]], use_container_width=True, hide_index=True)
        registry_map = {int(x["id"]): x for x in registry}
        candidate_id = st.selectbox("Model version to activate / archive", list(registry_map),
                                 format_func=lambda x: f"#{x} {registry_map[x]['model_key']} {registry_map[x]['version']} [{registry_map[x]['status']}]")
        candidate = registry_map[candidate_id]
        if st.button("Set active"):
            db.set_model_status(candidate["id"],"active")
            service.clear_model_cache()
            st.success("Model activated.")
            st.rerun()
        if st.button("Archive version"):
            db.set_model_status(candidate["id"],"archived")
            service.clear_model_cache()
            st.success("Model archived.")
            st.rerun()
    st.warning("The latest uploaded result ZIPs do not include `.pth` checkpoints. Import the actual model bundles from Colab Drive before expecting predictions. The current smoke-run weights are not production-accurate models.")
