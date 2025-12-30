
import os, json, time, argparse
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import requests
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(ROOT, "api_client_config.json")


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ["token", "face_api_url", "course_api_url", "attendance_api_url"]:
        if not cfg.get(k):
            raise ValueError(f"api_client_config.json 缺少必要欄位：{k}")
    cfg.setdefault("timeout_sec", 20)
    cfg.setdefault("verify_ssl", True) 
    return cfg


def l2norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).flatten()
    return v / (float(np.linalg.norm(v)) + 1e-12)


class ActionModel:

    def __init__(self, wts_path: str, map_path: str, device: torch.device):
        self.ok = False
        self.device = device
        self.idx_to_class: Dict[int, str] = {}

        if not (os.path.exists(wts_path) and os.path.exists(map_path)):
            print(f"[Action] 行為模型停用（缺檔）：{wts_path} / {map_path}")
            return

        with open(map_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        if not isinstance(class_to_idx, dict) or not class_to_idx:
            print(f"[Action] 行為模型停用（action_map 格式不正確）：{map_path}")
            return

        self.idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
        num_classes = len(self.idx_to_class)

        state = torch.load(wts_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            print(f"[Action] 行為模型停用（pth 內容非 state_dict）：{wts_path}")
            return

        state = {k.replace("module.", ""): v for k, v in state.items()}

        m = models.resnet18(weights=None)

        has_fc1 = any(k.startswith("fc.1.") for k in state.keys())
        if has_fc1:
            m.fc = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(m.fc.in_features, num_classes),
            )
        else:
            m.fc = torch.nn.Linear(m.fc.in_features, num_classes)

        try:
            m.load_state_dict(state, strict=True)
        except Exception:
            m.load_state_dict(state, strict=False)

        m.eval().to(device)
        self.model = m

        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.ok = True
        print("[Action] 行為分類模型載入成功")

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> str:
        if not self.ok:
            return "unknown"
        if bgr is None or bgr.size == 0:
            return "unknown"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        y = self.model(x)
        idx = int(torch.argmax(y, dim=1).item())
        return self.idx_to_class.get(idx, "unknown")


class MonitorUploader:
    def __init__(
        self,
        cfg: Dict[str, Any],
        det_weights: str,
        tracker: str,
        action_wts: str,
        action_map: str,
        source: str,
        arc_sim_th: float,
        upload_interval: float,
        system_sync_interval: float,
        face_interval: float,
        status_interval: float,
        print_speed: bool,
        overlay_speed: bool,
    ):
        self.cfg = cfg
        self.tracker = tracker
        self.source = int(source) if str(source).isdigit() else source

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Device] 使用裝置:", self.device)

        # YOLO
        print("[YOLO] 載入偵測權重：", det_weights)
        self.yolo = YOLO(det_weights)

        # InsightFace
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
        print("[Face] 初始化 InsightFace FaceAnalysis (buffalo_l)...")
        self.face_app = FaceAnalysis(name="buffalo_l", root=".", providers=providers)
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("[Face] 初始化完成")

        # Action
        self.action_model = ActionModel(action_wts, action_map, self.device)

        # thresholds & intervals
        self.arc_sim_th = float(arc_sim_th)
        self.upload_interval = float(upload_interval)
        self.system_sync_interval = float(system_sync_interval)

        self.face_interval = max(0.1, float(face_interval))       # 每人最少 0.1s
        self.status_interval = max(0.5, float(status_interval))   # 每人最少 0.5s

        self.print_speed = bool(print_speed)
        self.overlay_speed = bool(overlay_speed)

        # runtime state
        self.known_embs: List[Dict[str, Any]] = []   # [{sid, emb(np)}]
        self.current_course: Optional[str] = None
        self.last_system_sync = 0.0

        self.last_upload: Dict[Tuple[str, str, str], float] = {}  # (sid, course, status) -> ts
        self.track_sid: Dict[int, str] = {}                       # track_id -> sid

        # 人臉 / 姿態 節流
        self.last_face_t: Dict[int, float] = {}                   # track_id -> last time
        self.last_status_t: Dict[int, float] = {}                 # track_id -> last time
        self.track_status: Dict[int, str] = {}                    # track_id -> cached status

        # speed
        self._sp_last = time.perf_counter()
        self._sp_frames = 0
        self._fps = 0.0
        self._yolo_ms = 0.0
        self._yolo_parts = None  # (pre, inf, post)

    # API
    def _api_get_system_data(self) -> List[Dict[str, Any]]:
        r = requests.get(
            self.cfg["face_api_url"],
            params={"token": self.cfg["token"], "action": "get_system_data"},
            timeout=self.cfg["timeout_sec"],
            verify=self.cfg["verify_ssl"],
        )
        r.raise_for_status()
        j = r.json()
        return j.get("embs", []) if isinstance(j, dict) else []

    def _api_get_current_course(self) -> Optional[str]:
        r = requests.get(
            self.cfg["course_api_url"],
            params={"token": self.cfg["token"], "action": "current_course"},
            timeout=self.cfg["timeout_sec"],
            verify=self.cfg["verify_ssl"],
        )
        r.raise_for_status()
        j = r.json()
        if isinstance(j, dict):
            c = j.get("course")
            return str(c) if c else None
        return None

    def _sync_system_if_needed(self):
        now = time.time()
        if now - self.last_system_sync < self.system_sync_interval:
            return
        self.last_system_sync = now

        # embeddings
        known = []
        try:
            embs_raw = self._api_get_system_data()
            for r in embs_raw:
                sid = str(r.get("student_no", "")).strip()
                emb_s = r.get("face_embedding", None)
                if not sid or not emb_s:
                    continue
                try:
                    emb = l2norm(np.array(json.loads(emb_s), dtype=np.float32))
                    known.append({"sid": sid, "emb": emb})
                except Exception:
                    continue
        except Exception as e:
            print("[WARN] 同步 embeddings 失敗：", e)

        self.known_embs = known

        # course
        try:
            self.current_course = self._api_get_current_course()
        except Exception as e:
            print("[WARN] 取得 current_course 失敗：", e)
            self.current_course = None

        print(f"[Sync] embeddings={len(self.known_embs)} course={self.current_course}")


    # Face identify
    def _identify_sid_from_crop(self, crop_bgr: np.ndarray) -> Optional[str]:
        if not self.known_embs:
            return None
        faces = self.face_app.get(crop_bgr)
        if not faces:
            return None

        best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        feat = getattr(best, "normed_embedding", None)
        if feat is None:
            return None
        feat = l2norm(np.asarray(feat, dtype=np.float32))

        best_sid, best_sim = None, -1.0
        for k in self.known_embs:
            sim = float(np.dot(feat, k["emb"]))
            if sim > best_sim:
                best_sim = sim
                best_sid = k["sid"]

        if best_sid is not None and best_sim >= self.arc_sim_th:
            return best_sid
        return None

    def _should_upload(self, sid: str, course: str, status: str) -> bool:
        key = (sid, course, status)
        now = time.time()
        last = self.last_upload.get(key, 0.0)
        if now - last >= self.upload_interval:
            self.last_upload[key] = now
            return True
        return False

    def _upload_snapshot(self, sid: str, course: str, status: str, crop_bgr: np.ndarray):
        ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        files = {"image": ("snap.jpg", buf.tobytes(), "image/jpeg")}
        data = {
            "token": self.cfg["token"],
            "sid": sid,
            "course": course,
            "status": status,
            "save_only": "0",
        }
        try:
            r = requests.post(
                self.cfg["attendance_api_url"],
                data=data,
                files=files,
                timeout=self.cfg["timeout_sec"],
                verify=self.cfg["verify_ssl"],
            )
            if r.status_code != 200:
                print("[Upload] FAIL", r.status_code, r.text[:200])
                return
            j = r.json() if r.text.strip() else {}
            
            if isinstance(j, dict) and j.get("ok"):
                resp_data = j.get("data", {})
                snap_path = resp_data.get("snapshot") if isinstance(resp_data, dict) else None
                print("[Upload] OK", snap_path)
            else:
                print("[Upload] OK", r.text[:120])
            
        except Exception as e:
            print("[Upload] EXC", e)

    def _update_speed(self, results, yolo_ms: float):
        self._sp_frames += 1
        self._yolo_ms = float(yolo_ms)

        self._yolo_parts = None
        try:
            if results and hasattr(results[0], "speed") and isinstance(results[0].speed, dict):
                sp = results[0].speed
                self._yolo_parts = (sp.get("preprocess"), sp.get("inference"), sp.get("postprocess"))
        except Exception:
            self._yolo_parts = None

        nowp = time.perf_counter()
        dt = nowp - self._sp_last
        if dt >= 1.0:
            self._fps = self._sp_frames / max(dt, 1e-6)
            self._sp_frames = 0
            self._sp_last = nowp

            if self.print_speed:
                if self._yolo_parts and all(v is not None for v in self._yolo_parts):
                    pre, inf, post = self._yolo_parts
                    total = float((pre or 0) + (inf or 0) + (post or 0))
                    print(f"[Speed] FPS={self._fps:.1f} | YOLO(pre/inf/post)={pre:.1f}/{inf:.1f}/{post:.1f} ms | total={total:.1f} ms")
                else:
                    print(f"[Speed] FPS={self._fps:.1f} | YOLO={self._yolo_ms:.1f} ms")

    # Main loop
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影像來源：{self.source}")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            now = time.time()

            self._sync_system_if_needed()

            t0 = time.perf_counter()
            results = self.yolo.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                conf=0.25,
                iou=0.45,
                verbose=False,
                device=0 if self.device.type == "cuda" else "cpu",
            )
            yolo_ms = (time.perf_counter() - t0) * 1000.0
            self._update_speed(results, yolo_ms)

            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    cls = int(b.cls.item())
                    if cls != 0:  
                        continue

                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    if crop.size <= 10:
                        continue

                    track_id = None
                    if getattr(b, "id", None) is not None and len(b.id) > 0:
                        track_id = int(b.id.item())

                    if track_id is not None:
                        sid = self.track_sid.get(track_id, "unknown")
                        if sid == "unknown":
                            last_t = self.last_face_t.get(track_id, 0.0)
                            if (now - last_t) >= self.face_interval:
                                self.last_face_t[track_id] = now
                                got = self._identify_sid_from_crop(crop)
                                if got:
                                    self.track_sid[track_id] = got
                                    sid = got

                    status = "unknown"
                    did_status_update = False
                    if track_id is not None:
                        last_st = self.last_status_t.get(track_id, 0.0)
                        if (now - last_st) >= self.status_interval:
                            self.last_status_t[track_id] = now
                            status = self.action_model.predict(crop)
                            self.track_status[track_id] = status
                            did_status_update = True
                        else:
                            status = self.track_status.get(track_id, "unknown")
                    else:
                        status = "unknown"

                    if did_status_update and sid != "unknown" and self.current_course:
                        if self._should_upload(sid, self.current_course, status):
                            self._upload_snapshot(sid, self.current_course, status, crop)

                    # UI
                    color = (0, 255, 0) if sid != "unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{sid} | {status}", (x1, max(20, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.overlay_speed:
                cv2.putText(
                    frame,
                    f"FPS {self._fps:.1f} | YOLO {self._yolo_ms:.1f} ms",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Pose Monitor Uploader", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--source", default="0")
    ap.add_argument("--det-weights", default=os.path.join(ROOT, "yolov5su.pt"))
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--action-wts", default=os.path.join(ROOT, "best_action_model.pth"))
    ap.add_argument("--action-map", default=os.path.join(ROOT, "action_class_to_idx.json"))
    ap.add_argument("--arc-sim-th", type=float, default=0.60)
    ap.add_argument("--face-interval", type=float, default=1.0)     # 每個人 1 秒最多辨識一次
    ap.add_argument("--status-interval", type=float, default=60.0)  # 每個人 60 秒最多分類一次（做完就上傳）
    ap.add_argument("--upload-interval", type=float, default=60.0)  # 同狀態同課程至少隔 60 秒

    ap.add_argument("--system-sync-interval", type=float, default=60.0)

    # 速度資訊
    ap.add_argument("--print-speed", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")

    args = ap.parse_args()

    cfg = load_cfg(args.config)

    m = MonitorUploader(
        cfg=cfg,
        det_weights=args.det_weights,
        tracker=args.tracker,
        action_wts=args.action_wts,
        action_map=args.action_map,
        source=args.source,
        arc_sim_th=args.arc_sim_th,
        upload_interval=args.upload_interval,
        system_sync_interval=args.system_sync_interval,
        face_interval=args.face_interval,
        status_interval=args.status_interval,
        print_speed=args.print_speed,
        overlay_speed=(not args.no_overlay),
    )
    m.run()


if __name__ == "__main__":
    main()
