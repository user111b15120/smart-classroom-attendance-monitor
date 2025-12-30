import tkinter as tk
from tkinter import scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import queue
import sys
import os
import time
import subprocess
import argparse
import re

try:
    import realtime_course_monitor_head as monitor_module
    HAS_MONITOR_MODULE = True
except ImportError:
    HAS_MONITOR_MODULE = False

class TextRedirector:
    def __init__(self, q, tag="INFO"):
        self.q = q
        self.tag = tag

    def write(self, msg):
        if msg.strip():
            self.q.put(("log", f"[{self.tag}] {msg}"))

    def flush(self):
        pass

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智慧教室監控儀表板 (Smart Dashboard)")
        self.root.geometry("1000x650")
        
        # 設定 Icon
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        # --- 變數初始化 ---
        self.stop_monitor_event = threading.Event()
        self.register_process = None
        self.queue = queue.Queue()
        self.current_image = None
        
        # --- UI 佈局 ---
        top_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
        top_frame.pack(fill="x")
        
        tk.Label(top_frame, text="功能控制:", bg="#f0f0f0", font=("微軟正黑體", 12, "bold")).pack(side="left", padx=20)
        
        self.btn_monitor = tk.Button(top_frame, text="▶ 啟動實時監控", bg="#ccffcc", font=("微軟正黑體", 10), command=self.toggle_monitor)
        self.btn_monitor.pack(side="left", padx=5)
        
        self.btn_register = tk.Button(top_frame, text="▶ 啟動人臉註冊", bg="#ccffcc", font=("微軟正黑體", 10), command=self.toggle_register)
        self.btn_register.pack(side="left", padx=5)

        tk.Button(top_frame, text="離開", bg="#ffcccc", font=("微軟正黑體", 10), command=self.on_closing).pack(side="right", padx=20)

        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # 左側：影像顯示區
        video_frame = tk.LabelFrame(main_frame, text="即時監控畫面", font=("微軟正黑體", 11))
        video_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        self.lbl_video = tk.Label(video_frame, bg="black", text="等待啟動...", fg="white", font=("Arial", 14))
        self.lbl_video.pack(fill="both", expand=True)

        # 右側：運行日誌區
        log_frame = tk.LabelFrame(main_frame, text="系統運行日誌", font=("微軟正黑體", 11))
        log_frame.pack(side="right", fill="y", padx=5)
        
        self.txt_log = scrolledtext.ScrolledText(log_frame, width=55, state="disabled", font=("Consolas", 9), bg="#1e1e1e", fg="#ffffff")
        self.txt_log.pack(fill="both", expand=True)
        
        # --- 設定顏色標籤 (支援 ANSI Color) ---
        self.txt_log.tag_config("green", foreground="#00ff00")
        self.txt_log.tag_config("red", foreground="#ff5555")
        self.txt_log.tag_config("yellow", foreground="#ffff00")
        self.txt_log.tag_config("blue", foreground="#5555ff")
        self.txt_log.tag_config("cyan", foreground="#00ffff")
        self.txt_log.tag_config("magenta", foreground="#ff00ff")
        self.txt_log.tag_config("white", foreground="#ffffff")
        
        # 預留原始標籤相容性
        self.txt_log.tag_config("INFO", foreground="#ffffff")
        self.txt_log.tag_config("ERR", foreground="#ff5555")
        self.txt_log.tag_config("SYS", foreground="#55aaff")

        self.update_ui_loop()

    def insert_colored_text(self, text):
        """解析 ANSI 代碼並插入顏色文字"""
        color_map = {
            '31': 'red', '32': 'green', '33': 'yellow',
            '34': 'blue', '35': 'magenta', '36': 'cyan', '37': 'white'
        }
        # 正規表達式切分 ANSI 代碼
        parts = re.split(r'\x1b\[(\d+)m', text)
        current_tag = None
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # 文字部分
                if part:
                    if current_tag:
                        self.txt_log.insert("end", part, current_tag)
                    else:
                        self.txt_log.insert("end", part)
            else:  # 代碼部分
                if part == '0':
                    current_tag = None
                else:
                    current_tag = color_map.get(part, current_tag)

    def mock_imshow(self, winname, frame):
        if frame is None: return
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape
        display_h = self.lbl_video.winfo_height()
        display_w = self.lbl_video.winfo_width()
        
        if display_h > 10 and display_w > 10:
            scale = min(display_w/w, display_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            if new_w > 0 and new_h > 0:
                pil_image = Image.fromarray(rgb_image).resize((new_w, new_h))
            else:
                pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = Image.fromarray(rgb_image)

        tk_image = ImageTk.PhotoImage(image=pil_image)
        self.queue.put(("frame", tk_image))

    def mock_waitkey(self, delay):
        if self.stop_monitor_event.is_set():
            return ord('q')
        time.sleep(delay / 1000.0)
        return -1

    def toggle_monitor(self):
        if not HAS_MONITOR_MODULE:
            self.log("SYS", "錯誤：找不到 realtime_course_monitor_head.py")
            return

        if self.stop_monitor_event.is_set():
            self.stop_monitor_event.clear()

        if self.btn_monitor["text"] == "▶ 啟動實時監控":
            self.btn_monitor.config(text="■ 停止監控", bg="#ffcccc")
            self.log("SYS", "正在啟動監控系統...")
            threading.Thread(target=self.run_monitor_thread, daemon=True).start()
        else:
            self.btn_monitor.config(text="停止中...", state="disabled")
            self.stop_monitor_event.set()

    def run_monitor_thread(self):
        original_imshow = cv2.imshow
        original_waitkey = cv2.waitKey
        cv2.imshow = self.mock_imshow
        cv2.waitKey = self.mock_waitkey
        
        original_stdout = sys.stdout
        sys.stdout = TextRedirector(self.queue, "MONITOR")

        try:
            cfg_path = os.path.join(os.getcwd(), "api_client_config.json")
            if not os.path.exists(cfg_path):
                self.log("ERR", f"找不到設定檔: {cfg_path}")
                return

            self.log("SYS", "初始化模型中，請稍候...")
            cfg = monitor_module.load_cfg(cfg_path)
            
            app = monitor_module.MonitorUploader(
                cfg=cfg,
                det_weights="yolov5su.pt",
                tracker="bytetrack.yaml",
                action_wts="best_action_model.pth",
                action_map="action_class_to_idx.json",
                source="0", 
                arc_sim_th=0.6,
                upload_interval=60.0,
                system_sync_interval=60.0,
                face_interval=1.0,
                status_interval=60.0,
                print_speed=True,
                overlay_speed=True
            )
            self.log("SYS", "模型載入完成，開始監控。")
            app.run() 
            
        except Exception as e:
            self.log("ERR", f"監控執行錯誤: {e}")
        finally:
            cv2.imshow = original_imshow
            cv2.waitKey = original_waitkey
            sys.stdout = original_stdout
            self.queue.put(("monitor_stopped", None))

    def toggle_register(self):
        if self.register_process is None:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.getcwd()

            script_path = os.path.join(base_dir, "Face_register_from_upload.py")
            if not os.path.exists(script_path):
                self.log("ERR", f"找不到檔案: {script_path}")
                return

            venv_python = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
            python_exe = venv_python if os.path.exists(venv_python) else sys.executable
            
            if os.path.exists(venv_python):
                self.log("SYS", "使用 .venv 環境啟動...")

            try:
                # smart_dashboard.py
                self.register_process = subprocess.Popen(
                    [python_exe, "-u", script_path, "--watch-pending", "--interval", "10"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=0x08000000, 
                    text=True,                
                    bufsize=1,                
                    encoding='cp950',        
                    errors='ignore',         
                    cwd=base_dir
                )
                self.btn_register.config(text="■ 停止註冊", bg="#ffcccc")
                threading.Thread(target=self.read_subprocess_output, args=(self.register_process,), daemon=True).start()
            except Exception as e:
                self.log("ERR", f"啟動失敗: {e}")
                self.register_process = None
        else:
            self.log("SYS", "停止人臉註冊程式...")
            if self.register_process:
                self.register_process.terminate()
                self.register_process = None
            self.btn_register.config(text="▶ 啟動人臉註冊", bg="#ccffcc")

    def read_subprocess_output(self, process):
        """讀取 subprocess 輸出並傳送到 Queue"""
        for line in iter(process.stdout.readline, ''):
            if line:
                self.queue.put(("log_raw", f"[REGISTER] {line}"))
        process.stdout.close()

    def update_ui_loop(self):
        try:
            while True:
                action, data = self.queue.get_nowait()
                
                if action == "frame":
                    self.lbl_video.config(image=data, text="")
                    self.current_image = data 
                
                elif action == "log":
                    self.txt_log.config(state="normal")
                    self.txt_log.insert("end", data + "\n")
                    self.txt_log.see("end") 
                    self.txt_log.config(state="disabled")

                elif action == "log_raw":
                    self.txt_log.config(state="normal")
                    self.insert_colored_text(data)
                    self.txt_log.see("end")
                    self.txt_log.config(state="disabled")
                
                elif action == "monitor_stopped":
                    self.btn_monitor.config(text="▶ 啟動實時監控", bg="#ccffcc", state="normal")
                    self.stop_monitor_event.clear()
                    self.lbl_video.config(image="", text="監控已停止", bg="black")
                    self.log("SYS", "監控系統已停止。")

        except queue.Empty:
            pass
        
        self.root.after(30, self.update_ui_loop)

    def log(self, tag, msg):
        self.queue.put(("log", f"[{tag}] {msg}"))

    def on_closing(self):
        if self.register_process:
            self.register_process.terminate()
        self.stop_monitor_event.set()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.mainloop()