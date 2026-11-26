CARLA V6.2 Shadow 診斷版
=======================

用途：不破壞 V6.1 原始 LiDAR/GT/盲區/事故功能，在 CSV 中新增「最新兩幀距離差分接近速度」和「候選 Shadow 警報」。

放置：把 carla_integrated_monitor_v62.py 和 carla_v62_compare.py 放進 CARLA_0.9.16\PythonAPI\examples。
此程式依賴原本的 blind_zone_lidar_v2.py，不能刪掉。

測試時，不要同時執行其他 LiDAR 監測器，也不要執行追撞程式。
先確認 CARLA、交通流與 Ego 正常運行，Ego actor ID 可能改變，請確認後再指定。

啟動獨立鳥瞰圖監測：
py -3.10 carla_integrated_monitor_v62.py --ego-id 112

按 Q / Esc 或 CMD 的 Ctrl+C 停止，以儲存 CSV。
輸出：integrated_carla_v62_shadow_YYYYMMDD_HHMMSS.csv

離線分析：
py -3.10 carla_v62_compare.py

觀察重點：
- lidar_warning：V6.1 原有正式警報，完全不變
- lidar_instant_closing_mps：同一個 LiDAR Track 最近兩個有效距離量測計算的接近速度
- lidar_shadow_warning：候選警報（僅 CSV，不控制畫面/UDP）
- lidar_shadow_mode：候選判定原因
- lidar_shadow_differs：與原版警報是否不同

注意：V6.1 原本已使用最近兩幀速度檢查，當前 Shadow 只是探索更保守的 1.25 m/s 快速解除閾值。
這個候選值可能抑制真實緩慢接近風險，不能未驗證就套到正式預警。
不能只用 GT 與 LiDAR 警報不同來判定誤報，需確認是否為相同障礙物並完成獨立場景驗證。
此診斷版不會重新載入地圖、不改世界設定、不呼叫 world.tick()，但會在 Ego 上掛載自己的 LiDAR/碰撞感測器，退出時正常清理。
