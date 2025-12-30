# 安裝與執行說明（Installation & Usage）

> 本段落提供系統之安裝與基本操作說明，供成果展示與示範使用。  
> 為避免敏感資訊與大型檔案公開，本 Repository 未包含完整安裝檔與模型權重。

---

## 硬體需求
- **攝影機**：USB 攝影機或任何可被 OpenCV 讀取之影像來源（預設使用本機鏡頭 `index = 0`）
- **GPU（建議）**：NVIDIA CUDA 顯示卡（可加速推論）；無 GPU 亦可使用 CPU 執行但速度較慢
- **記憶體**：至少 8 GB（建議 16 GB 以上）
- **CPU**：Intel i5 / AMD Ryzen 5 以上（建議）

## 軟體需求
- **作業系統**：Windows 10 / Windows 11（64-bit）
- **Python**：Python 3.11（建議官方 Windows x64 版本）
- **顯示卡驅動（選用）**：NVIDIA Driver（若使用 GPU 推論）

---

## 安裝方式（展示用）
1. 執行安裝程式 `SmartClassroom_Setup.exe`
2. 選擇安裝路徑後點擊 **Next**
3. 選擇是否建立桌面捷徑
4. 確認設定後點擊 **Install**
5. 等待安裝完成
6. 安裝完成後執行 `SmartDashboard.exe` 主控制程式

### 注意事項
- 安裝完成後請勿任意移動資料夾或檔案
- 若系統未安裝 Python 3.11（或未加入 PATH），安裝程序可能失敗  
  可先安裝 Python 後重新執行，或手動執行 `install_env.bat`

---

## 系統執行方式
- **SmartDashboard.exe**
  - 啟動後點擊「啟動即時監控」  
    → 開啟預設鏡頭進行人臉辨識與狀態監測  
    → 已註冊學生的狀態將寫入對應課程與時間之資料庫
  - 點擊「啟動人臉註冊」  
    → 系統每 10 秒檢查一次資料庫中是否有未完成臉部特徵建檔之學生
  - 點擊右上角關閉按鈕可停止程式執行

---

## 系統展示網站（Demo）
- 系統網站首頁：`https://www.cjcu3402.oo.gd`  
  （展示期限至 **2026/12/22**）
- 教師帳號註冊預設密碼：`cjcu2025`

---

## 重要聲明
- 本 GitHub 專案僅作為**書面審查與成果展示用途**
- 未公開以下內容：
  - API token 與敏感設定檔
  - 學生個資與臉部特徵資料
  - 大型模型權重（`.pt` / `.pth`）
- 完整可執行版本與安裝檔，將於書審資料「其他補充說明」中另行提供下載方式

## 下載與展示連結（Download & Demo）

- **系統安裝檔 **：  
  https://（你的 Google Drive / OneDrive 連結）

- **成果展示 GitHub **：  
  https://github.com/user111b15120/smart-classroom-attendance-monitor

> ※ 本安裝檔僅供成果展示使用，未包含 API token、個資與模型權重。
