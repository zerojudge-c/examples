CARLA V6 使用說明
====================

檔案放置位置
------------
請把以下檔案全部放在：
CARLA_0.9.16\PythonAPI\examples\

1. carla_v6_oneclick.py
2. carla_cinematic_rear_10s_v6.py
3. CARLA_V6_START.bat
4. CARLA_V6_RESUME.bat
5. carla_integrated_monitor_v5_targetcheck.py（原本 V5，不需替換）
6. blind_zone_lidar_v2.py（V5 原本依賴的檔案）
7. chase_target.txt（可由程式自動建立）

啟動前
------
1. 先啟動 CARLA 0.9.16。
2. 確認 Ego #112 存在。
3. 確認 CARLA 模擬時間持續前進；此整合程式不會自行啟動 CARLA 或交通流。
4. 確認 Python 3.10 與 CARLA PythonAPI、OpenCV、NumPy 環境可正常使用。

啟動
----
直接雙擊 CARLA_V6_START.bat。

流程：
1. 清空舊的 chase_target.txt 目標值，避免讀到舊 NPC ID。
2. 啟動 V5 監測程式，等待 MONITOR READY。
3. 啟動 V6 cinematic rear-contact 演示。
4. V6 自動把實際選中的 NPC ID 寫入 chase_target.txt。
5. V5 熱更新 GT 目標 ID，持續記錄 LiDAR / GT / TTC / 盲區。
6. 演示結束後，V6 嘗試以 Ctrl+Break 正常停止 V5，讓 CSV 收尾。
7. 自動產生 carla_v6_report_*.txt。

NPC 選擇
--------
預設由 cinematic 程式從現有車輛中選擇一台 NPC。
若要指定 NPC，請編輯 CARLA_V6_START.bat，將啟動行改成：
py -3.10 carla_v6_oneclick.py --ego-id 112 --npc-id 24

請先確認該 NPC ID 當下仍存在，Actor ID 可能隨場景重建而改變。

恢復車輛
--------
追撞演示結束時，Ego 與 NPC 會被凍結。
執行 CARLA_V6_RESUME.bat 可分開車輛並恢復物理與 Autopilot。
在恢復前不要開始下一次測試。

重要限制
--------
目前 cinematic 程式是「程式控制位置形成幾何重疊」，不是自然物理碰撞。
因此 collision sensor 可能不觸發。報告會分開記錄碰撞 callback 與幾何接觸限制。
不可把幾何重疊直接宣稱為真實碰撞感測成功。
LiDAR warning 是未識別障礙物預警；GT / blind-zone 與 GT-assisted 配對使用 CARLA Actor 真值，
屬於評估資料，不等於 LiDAR 單獨辨識出 NPC 身分。

若啟動逾時
----------
若出現等待 MONITOR READY 逾時，請檢查：
- CARLA Server 是否在 127.0.0.1:2000 正常運作
- Ego ID 是否正確
- blind_zone_lidar_v2.py 是否在同一資料夾
- Python 3.10 是否能匯入 carla、cv2、numpy
- CARLA 模擬時間是否持續前進
