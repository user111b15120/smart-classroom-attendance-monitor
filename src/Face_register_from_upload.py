
import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import requests

import onnxruntime as ort
from insightface.app import FaceAnalysis


ROOT = os.path.dirname(os.path.abspath(__file__))

def load_cfg() -> Dict[str, Any]:
    cfg_path = os.path.join(ROOT, "api_client_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到 api_client_config.json：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for k in ["token", "face_api_url", "uploads_base_url"]:
        if not cfg.get(k):
            raise ValueError(f"api_client_config.json 缺少必要欄位：{k}")

    cfg["face_api_url"] = str(cfg["face_api_url"]).strip()
    cfg["uploads_base_url"] = str(cfg["uploads_base_url"]).strip().rstrip("/") + "/"
    cfg.setdefault("timeout_sec", 20)
    cfg.setdefault("verify_ssl", True)
    return cfg


def init_face_app() -> FaceAnalysis:
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
    else:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
    app = FaceAnalysis(name="buffalo_l", root=".", providers=providers)
    
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print(f"[Face] providers={providers}")
    return app


def pick_largest_face(faces) -> Optional[Any]:
    if not faces:
        return None
    def key_fn(f):
        x1, y1, x2, y2 = map(float, f.bbox)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        score = float(getattr(f, "det_score", 0.0))
        return (area, score)
    return max(faces, key=key_fn)


def get_embedding_from_image(face_app: FaceAnalysis, img_bgr: np.ndarray) -> Optional[np.ndarray]:
    if img_bgr is None or img_bgr.size == 0:
        return None
    faces = face_app.get(img_bgr)
    best = pick_largest_face(faces)
    if best is None:
        return None
    emb = getattr(best, "normed_embedding", None)
    if emb is None:
        return None
    emb = np.asarray(emb, dtype=np.float32).flatten()
    n = float(np.linalg.norm(emb) + 1e-12)
    return emb / n

def api_get_pending(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    r = requests.get(
        cfg["face_api_url"],
        params={"token": cfg["token"], "action": "get_pending"},
        timeout=cfg["timeout_sec"],
        verify=cfg["verify_ssl"],
    )
    r.raise_for_status()
    j = r.json()
    if isinstance(j, dict) and j.get("error"):
        raise RuntimeError(j)
    if not isinstance(j, list):
        raise RuntimeError(f"get_pending 回傳格式非 list：{type(j)} {j}")
    return j


def api_update_embedding(cfg: Dict[str, Any], student_no: str, emb: np.ndarray) -> None:
    emb_json = json.dumps(emb.tolist(), ensure_ascii=False)
    r = requests.post(
        cfg["face_api_url"],
        data={
            "token": cfg["token"],
            "action": "update_embedding",
            "student_no": student_no,
            "embedding": emb_json,
        },
        timeout=cfg["timeout_sec"],
        verify=cfg["verify_ssl"],
    )
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(f"update_embedding 失敗：{j}")


def normalize_photo_path(photo_name: str) -> str:
    s = (photo_name or "").strip()
    if not s:
        return s
    if s.startswith("http://") or s.startswith("https://"):
        return s
    s = s.lstrip("/")
    if s.startswith("uploads/"):
        s = s[len("uploads/"):]
    elif "/uploads/" in s:
        s = s.split("/uploads/", 1)[1]
    return s


def download_pending_photo(cfg: Dict[str, Any], photo_name: str) -> np.ndarray:
    norm = normalize_photo_path(photo_name)
    if not norm:
        raise RuntimeError("photo_name 空值")

    if norm.startswith("http://") or norm.startswith("https://"):
        url = norm
    else:
        url = cfg["uploads_base_url"] + norm

    r = requests.get(url, timeout=cfg["timeout_sec"], verify=cfg["verify_ssl"])
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"下載或解碼失敗：{url}")
    return img


def sync_pending(cfg: Dict[str, Any], face_app: FaceAnalysis, sleep_sec: float = 0.1) -> Dict[str, int]:
    pending = api_get_pending(cfg)
    if not pending:
        return {"ok": 0, "fail": 0, "total": 0}

    ok_cnt = 0
    fail_cnt = 0

    for row in pending:
        sid = str(row.get("student_no", "")).strip()
        photo_name = str(row.get("photo_name", "")).strip()
        if not sid or not photo_name:
            fail_cnt += 1
            continue

        try:
            img = download_pending_photo(cfg, photo_name)
            emb = get_embedding_from_image(face_app, img)
            if emb is None:
                raise RuntimeError("找不到臉或無法取得 embedding（照片太糊/角度太大/臉太小）")
            api_update_embedding(cfg, sid, emb)
            ok_cnt += 1
            print(f"[OK] {sid} embedding 已回寫")
        except Exception as e:
            fail_cnt += 1
            print(f"[FAIL] {sid}: {e}")

        time.sleep(sleep_sec)

    return {"ok": ok_cnt, "fail": fail_cnt, "total": len(pending)}


def watch_pending(cfg: Dict[str, Any], face_app: FaceAnalysis, interval_sec: float) -> None:
    print(f"[Watch] 每 {interval_sec:.1f}s 檢查 pending...")
    while True:
        try:
            res = sync_pending(cfg, face_app)
            if res["total"] == 0:
                print("[Watch] no pending")
            else:
                print(f"[Watch] done: total={res['total']} ok={res['ok']} fail={res['fail']}")
        except Exception as e:
            print(f"[Watch] error: {e}")

        time.sleep(max(1.0, float(interval_sec)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-pending", action="store_true", help="單次處理 pending（網站照片→embedding→回寫）")
    ap.add_argument("--watch-pending", action="store_true", help="常駐輪詢 pending")
    ap.add_argument("--interval", type=float, default=10.0, help="常駐模式輪詢秒數")
    args = ap.parse_args()

    cfg = load_cfg()
    face_app = init_face_app()

    if args.watch_pending:
        watch_pending(cfg, face_app, args.interval)
        return

    if args.sync_pending:
        res = sync_pending(cfg, face_app)
        if res["total"] == 0:
            print("沒有 pending（都已有 face_embedding）")
        else:
            print(f"完成：total={res['total']} ok={res['ok']} fail={res['fail']}")
        return

    ap.error("請使用 --sync-pending 或 --watch-pending")


if __name__ == "__main__":
    main()
