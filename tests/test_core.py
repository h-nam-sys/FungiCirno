import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mushroom_monitor.db import Database, utc_now
from mushroom_monitor.image_io import store_capture
from mushroom_monitor.maturity import rebuild_flush_groups
from mushroom_monitor.model_registry import read_model_package, install_model_bundle
from mushroom_monitor.tracking import rebuild_sequence_tracks


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = Database(self.root / "test.sqlite3")
        self.db.ensure_defaults()
        self.site = self.db.one("SELECT * FROM sites LIMIT 1")
        self.unit = self.db.one("SELECT * FROM grow_units LIMIT 1")
        self.camera = self.db.one("SELECT * FROM cameras LIMIT 1")
        self.model_id = self.db.execute(
            """INSERT INTO model_versions(model_key,version,architecture,source_species,taxonomy,
               classes_json,config_json,bundle_path,checkpoint_path,checkpoint_sha256,status,installed_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("charisis_cluster_tracking","test-v1","TorchVision Mask R-CNN with ConvNeXt-Tiny-FPN",
             "Pleurotus ostreatus","charisis_oyster_cluster_stage",
             json.dumps({"1":"immature_cluster","2":"mature_cluster","3":"overstayed_cluster"}),
             json.dumps({"confidence_threshold":0.2}),str(self.root),str(self.root/"weights.pth"),"abc","candidate",utc_now()),
        )

    def tearDown(self):
        self.temp.cleanup()

    def add_capture(self, name: str, when: str):
        image_path = self.root / name
        Image.new("RGB", (320, 240), "white").save(image_path)
        return self.db.create_capture(
            unit_id=self.unit["id"], camera_id=self.camera["id"], series_key="series-A",
            original_name=name, file_path=str(image_path), sha256=name,
            captured_at=when, time_source="manual", imported_at=utc_now(),
            width=320, height=240, source="test", metadata_json="{}", status="processed",
        )

    def add_prediction(self, capture_id: int, mask_path: Path, det_idx: int, x_offset: float = 0.0):
        run = self.db.one("SELECT id FROM inference_runs WHERE capture_id=? AND model_version_id=?", (capture_id,self.model_id))
        run_id = int(run["id"]) if run else self.db.execute(
            "INSERT INTO inference_runs(capture_id,model_version_id,started_at,status) VALUES(?,?,?,'completed')",
            (capture_id,self.model_id,utc_now()),
        )
        self.db.add_prediction(
            run_id=run_id, capture_id=capture_id, model_version_id=self.model_id,
            detection_index=det_idx, class_id=1, class_name="immature_cluster", confidence=0.8,
            x1=50+x_offset, y1=40, x2=100+x_offset, y2=90, area_px2=2500,
            area_mm2=None, centroid_x=75+x_offset, centroid_y=65,
            mask_path=str(mask_path), mask_width=100, mask_height=100,
            review_status="accepted", final_for_analysis=1, metadata_json="{}",
        )

    def test_default_database_records(self):
        self.assertEqual(self.db.one("SELECT COUNT(*) n FROM sites")["n"], 1)
        self.assertEqual(self.db.one("SELECT COUNT(*) n FROM grow_units")["n"], 1)
        self.assertEqual(self.db.one("SELECT COUNT(*) n FROM cameras")["n"], 1)

    def test_capture_store_hashes_and_deduplicates(self):
        import io
        data = io.BytesIO()
        Image.new("RGB", (33, 20), "green").save(data, format="JPEG")
        path, digest, width, height, exif = store_capture(data.getvalue(), "test.jpg", self.root / "captures")
        path2, digest2, *_ = store_capture(data.getvalue(), "test-again.jpg", self.root / "captures")
        self.assertEqual(path, path2)
        self.assertEqual(digest, digest2)
        self.assertEqual((width, height), (33, 20))

    def test_maturity_grouping(self):
        capture_id = self.add_capture("maturity.jpg", "2026-11-10T08:00:00")
        mask = np.zeros((100, 100), np.uint8)
        mask[20:40, 20:40] = 255
        path1 = self.root / "mask1.png"; Image.fromarray(mask).save(path1)
        mask2 = np.zeros((100, 100), np.uint8)
        mask2[22:42, 45:65] = 255
        path2 = self.root / "mask2.png"; Image.fromarray(mask2).save(path2)
        self.add_prediction(capture_id,path1,0,0)
        self.add_prediction(capture_id,path2,1,25)
        count = rebuild_flush_groups(self.db,capture_id,multiplier=5.0)
        self.assertEqual(count,1)
        group = self.db.one("SELECT * FROM flush_groups WHERE capture_id=?",(capture_id,))
        self.assertEqual(group["fruitbody_count"],2)

    def test_tracking_connects_two_frames_and_persists_track(self):
        c1 = self.add_capture("frame1.jpg", "2026-11-10T08:00:00")
        c2 = self.add_capture("frame2.jpg", "2026-11-10T08:15:00")
        mask = np.zeros((100,100),np.uint8); mask[20:60,20:60] = 255
        p1=self.root/"frame1_mask.png"; Image.fromarray(mask).save(p1)
        mask2=np.zeros((100,100),np.uint8); mask2[21:61,21:61] = 255
        p2=self.root/"frame2_mask.png"; Image.fromarray(mask2).save(p2)
        self.add_prediction(c1,p1,0)
        self.add_prediction(c2,p2,0,1)
        result = rebuild_sequence_tracks(self.db,self.unit["id"],self.camera["id"],"series-A",self.model_id)
        self.assertEqual(result["tracks"],1)
        self.assertEqual(result["observations"],2)
        track = self.db.one("SELECT * FROM tracks WHERE unit_id=?",(self.unit["id"],))
        self.assertEqual(track["observations"],2)

    def test_tracking_counts_empty_capture_as_missed_frame(self):
        c1 = self.add_capture("gap_frame1.jpg", "2026-11-10T08:00:00")
        self.add_capture("gap_frame2.jpg", "2026-11-10T08:15:00")  # no prediction in this frame
        c3 = self.add_capture("gap_frame3.jpg", "2026-11-10T08:30:00")
        mask = np.zeros((100,100),np.uint8); mask[20:60,20:60] = 255
        p1=self.root/"gap_frame1_mask.png"; Image.fromarray(mask).save(p1)
        p3=self.root/"gap_frame3_mask.png"; Image.fromarray(mask).save(p3)
        self.add_prediction(c1,p1,0)
        self.add_prediction(c3,p3,0,1)
        result = rebuild_sequence_tracks(self.db,self.unit["id"],self.camera["id"],"series-A",self.model_id)
        self.assertEqual(result["captures"],3)
        self.assertEqual(result["tracks"],1)
        self.assertEqual(result["observations"],2)
        track = self.db.one("SELECT * FROM tracks WHERE unit_id=?",(self.unit["id"],))
        frames = self.db.query("SELECT frame_index FROM track_observations WHERE track_id=? ORDER BY frame_index",(track["id"],))
        self.assertEqual([x["frame_index"] for x in frames],[0,2])

    def test_model_bundle_zip_installs_into_registry(self):
        import zipfile
        root = self.root / "bundle_source"
        root.mkdir()
        (root / "model_config.json").write_text(json.dumps({"architecture":"TorchVision Mask R-CNN with ConvNeXt-Tiny-FPN"}))
        (root / "class_mapping.json").write_text(json.dumps({"label_to_name":{"1":"immature_cluster","2":"mature_cluster","3":"overstayed_cluster"}}))
        (root / "model_checkpoint.pth").write_bytes(b"checkpoint-stub")
        bundle = self.root / "model_bundle.zip"
        with zipfile.ZipFile(bundle,"w") as zf:
            for file in root.iterdir():
                zf.write(file,file.name)
        model_id = install_model_bundle(bundle,self.root/"registry",self.db)
        record = self.db.one("SELECT * FROM model_versions WHERE id=?",(model_id,))
        self.assertTrue(Path(record["checkpoint_path"]).exists())
        self.assertEqual(record["status"],"candidate")
        self.assertEqual(record["model_key"],"charisis_cluster_tracking")

    def test_model_bundle_metadata_is_read(self):
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "model_config.json").write_text(json.dumps({"architecture":"TorchVision Mask R-CNN with ConvNeXt-Tiny-FPN","confidence_threshold":0.2}))
        (bundle / "class_mapping.json").write_text(json.dumps({"label_to_name":{"1":"immature_cluster","2":"mature_cluster","3":"overstayed_cluster"}}))
        (bundle / "model_checkpoint.pth").write_bytes(b"checkpoint-stub")
        package = read_model_package(bundle)
        self.assertEqual(package.model_key,"charisis_cluster_tracking")
        self.assertEqual(len(package.classes),3)
        self.assertTrue(package.checkpoint_sha256)


if __name__ == "__main__":
    unittest.main()
