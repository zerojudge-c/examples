CARLA V6.1 - GT position warning diagnostic

把這些新檔案放到 CARLA_0.9.16\PythonAPI\examples：
- carla_integrated_monitor_v61_gtposition.py
- carla_v61_oneclick.py
- CARLA_V61_START.bat（可不用）
- CARLA_V61_RESUME.bat（可不用）

沿用舊檔，不必刪除：
- carla_cinematic_rear_10s_v6.py
- blind_zone_lidar_v2.py

啟動前：CARLA、generate_traffic.py、ego_vehicle.py 正常執行；關閉舊的 V5/V6 監測視窗，避免重複 LiDAR 感測器。

不使用 BAT 的操作：
cd C:\Users\user\Desktop\CARLA_0.9.16\PythonAPI\examples
py -3.10 carla_v61_oneclick.py --ego-id 112

追撞結束後解除凍結：
py -3.10 carla_cinematic_rear_10s_v6.py --ego-id 112 --resume

測試結果：
- integrated_carla_v61_gtposition_*.csv
- accident_events_v61_gtposition_*.csv
- carla_v61_report_*.txt

新增兩欄（只用於 GT 評估；不改 LiDAR）：
- gt_position_warning：最近前方 GT Actor 依位置差分計算 TTC 後的警報
- selected_gt_position_warning：指定 NPC 的 GT 位置差分警報
原本的 gt_warning 保留原始 Actor 速度向量估計方式；不會強制改成 1。

限制：cinematic 使用位置控制形成幾何重疊，不是自然物理碰撞。
GT position warning 在這個情境下是診斷訊號，而非獨立的真實事故證據。
現有 LiDAR 演算法、盲區、事故感測器、CARLA 世界同步模式均不修改。
本檔只做語法檢查，尚未連線使用者的 CARLA 實測。
