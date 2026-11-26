#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA V6.1 one-click runner (Windows / CARLA 0.9.16).

Starts the existing V5 observation monitor, waits until its sensors are ready,
runs the scripted cinematic rear-contact demo, then stops the monitor cleanly
and writes a text summary from the monitor CSV files.

Important:
- CARLA must already be running and its simulation clock must be advancing.
- The cinematic script creates a KINEMATIC GEOMETRIC OVERLAP, not a natural
  physics crash. The collision sensor may not fire; the report keeps these
  outcomes separate.
- This script does not start CARLA, spawn traffic, or alter world settings.
"""

import argparse
import csv
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def read_number(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv_rows(path):
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_report(folder, monitor_csv, event_csv, cinematic_code, started_at):
    rows = read_csv_rows(monitor_csv)
    events = read_csv_rows(event_csv)

    sim_times = [read_number(r.get("sim_time_s")) for r in rows]
    sim_times = [x for x in sim_times if x is not None]
    lidar_warn_rows = [r for r in rows if r.get("lidar_warning") == "1"]
    gt_warn_rows = [r for r in rows if r.get("gt_warning") == "1"]
    gt_pos_warn_rows = [r for r in rows if r.get("gt_position_warning") == "1"]

    lidar_ttc = [read_number(r.get("lidar_ttc_s_est")) for r in rows]
    lidar_ttc = [x for x in lidar_ttc if x is not None and x >= 0]
    gt_ttc = [read_number(r.get("gt_ttc_s")) for r in rows]
    gt_ttc = [x for x in gt_ttc if x is not None and x >= 0]

    match_errors = []
    for row in rows:
        status = (row.get("match_status") or "")
        err = read_number(row.get("matched_error_m"))
        if status.startswith("YES") and err is not None:
            match_errors.append(err)

    target_ids = [r.get("selected_gt_target_id", "").strip()
                  for r in rows if r.get("selected_gt_target_id", "").strip()]
    target_id = target_ids[-1] if target_ids else "未取得"
    target_status = Counter(
        r.get("selected_gt_status", "UNKNOWN") or "UNKNOWN" for r in rows
    )
    crash_count_values = [read_number(r.get("unique_crash_count")) for r in rows]
    crash_count_values = [int(x) for x in crash_count_values if x is not None]
    crash_count = max(crash_count_values) if crash_count_values else len(events)

    first_lidar_warn = next(
        (read_number(r.get("sim_time_s")) for r in lidar_warn_rows
         if read_number(r.get("sim_time_s")) is not None), None
    )
    first_gt_warn = next(
        (read_number(r.get("sim_time_s")) for r in gt_warn_rows
         if read_number(r.get("sim_time_s")) is not None), None
    )

    lines = [
        "CARLA V6 測試報告",
        "=" * 48,
        f"報告建立時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"測試啟動時間：{started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"追撞程式結束碼：{cinematic_code}",
        "",
        "[監測資料]",
        f"監測 CSV：{Path(monitor_csv).name if monitor_csv else '未找到'}",
        f"事故事件 CSV：{Path(event_csv).name if event_csv else '未找到'}",
        f"有效監測筆數：{len(rows)}",
        f"模擬時間跨度：{(max(sim_times)-min(sim_times)):.2f} 秒" if len(sim_times) >= 2 else "模擬時間跨度：資料不足",
        f"自動同步的 GT 目標 ID：{target_id}",
        "",
        "[預警統計]",
        f"LiDAR warning 記錄筆數：{len(lidar_warn_rows)}",
        f"GT velocity warning 記錄筆數：{len(gt_warn_rows)}",
        f"GT position warning 記錄筆數：{len(gt_pos_warn_rows)}",
        f"首次 LiDAR warning 模擬時間：{first_lidar_warn:.2f} 秒" if first_lidar_warn is not None else "首次 LiDAR warning 模擬時間：無",
        f"首次 GT warning 模擬時間：{first_gt_warn:.2f} 秒" if first_gt_warn is not None else "首次 GT warning 模擬時間：無",
        f"最小 LiDAR TTC：{min(lidar_ttc):.3f} 秒" if lidar_ttc else "最小 LiDAR TTC：無有效估值",
        f"最小 GT TTC：{min(gt_ttc):.3f} 秒" if gt_ttc else "最小 GT TTC：無有效估值",
        "",
        "[LiDAR / GT 配對評估]",
        f"成功標記為 YES (GT-assisted) 的配對樣本：{len(match_errors)}",
        f"配對距離誤差平均值：{sum(match_errors)/len(match_errors):.3f} m" if match_errors else "配對距離誤差平均值：無有效配對",
        f"配對距離誤差中位數：{sorted(match_errors)[len(match_errors)//2]:.3f} m" if match_errors else "配對距離誤差中位數：無有效配對",
        "",
        "[GT 目標位置狀態]",
    ]
    for status, count in target_status.most_common():
        lines.append(f"{status}：{count}")

    lines += [
        "",
        "[碰撞事件]",
        f"V5 碰撞感測器事件筆數：{len(events)}",
        f"V5 去重後事故計數：{crash_count}",
    ]

    if events:
        for i, event in enumerate(events, 1):
            lines.append(
                f"事件 {i}：target={event.get('target_id','')}，"
                f"LiDAR lead={event.get('lidar_lead_s_unclassified','') or '無'} 秒，"
                f"GT lead={event.get('gt_lead_s_target_matched','') or '無'} 秒"
            )
    else:
        lines.append("本次沒有收到 V5 collision sensor 的碰撞事件回呼。")

    lines += [
        "",
        "[重要限制]",
        "本次 rear-contact cinematic 是程式控制車輛位置形成的幾何重疊，",
        "不是自然物理碰撞；因此不能以畫面接觸或幾何重疊宣稱碰撞感測器成功。",
        "只有事故事件 CSV 中實際記錄的 collision callback，才計入 V5 碰撞事件。",
        "LiDAR warning 是未識別障礙物的預警；GT / blind-zone 使用 CARLA Actor 真值，",
        "GT-assisted 配對屬於評估用途，不等於 LiDAR 單獨辨識出車輛 ID。",
        "",
        "GT position warning 是基於 Actor 位移的診斷警報，和速度向量 GT 分開。",
        "程式控制車輛的測試中，GT position warning 不可當成物理事故真值。",
        "本報告為本次模擬測試摘要，不代表道路實車驗證結果。",
    ]

    out = folder / f"carla_v61_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def pump_output(pipe, line_queue):
    try:
        for line in iter(pipe.readline, ""):
            print("[V5] " + line, end="", flush=True)
            line_queue.put(line)
    finally:
        pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego-id", type=int, required=True)
    parser.add_argument("--npc-id", type=int, default=None,
                        help="Optional fixed NPC ID; if omitted, cinematic script selects an existing NPC")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--gap", type=float, default=20.0)
    parser.add_argument("--warn-ttc", type=float, default=4.0)
    parser.add_argument("--post-roll", type=float, default=3.0,
                        help="Keep the monitor running this many seconds after the cinematic script ends")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--python-launcher", default="py",
                        help="Windows Python launcher command (default: py)")
    args = parser.parse_args()

    folder = Path(__file__).resolve().parent
    monitor_script = folder / "carla_integrated_monitor_v61_gtposition.py"
    cinematic_script = folder / "carla_cinematic_rear_10s_v6.py"
    target_file = folder / "chase_target.txt"
    frozen_state = folder / "carla_cinematic_rear_10s_state.json"
    required = [monitor_script, cinematic_script, folder / "blind_zone_lidar_v2.py"]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        print("缺少必要檔案：", ", ".join(missing))
        print("請將 V5 監測程式、V6 追撞程式及 blind_zone_lidar_v2.py 放在同一個 examples 資料夾。")
        return 2
    if frozen_state.exists():
        print("偵測到上一次追撞測試留下的凍結狀態。")
        print("請先執行 CARLA_V6_RESUME.bat，恢復 Ego / NPC 後再開始新測試。")
        return 2

    before_monitor = set(folder.glob("integrated_carla_v61_gtposition_*.csv"))
    before_events = set(folder.glob("accident_events_v61_gtposition_*.csv"))
    started_at = datetime.now()

    # Empty the previous target so the monitor cannot accidentally evaluate a stale ID.
    target_file.write_text("", encoding="ascii")

    monitor_cmd = [
        args.python_launcher, "-3.10", str(monitor_script),
        "--ego-id", str(args.ego_id),
        "--gt-target-file", target_file.name,
        "--warn-ttc", str(args.warn_ttc),
    ]
    if args.headless:
        monitor_cmd.append("--headless")

    cinematic_cmd = [
        args.python_launcher, "-3.10", str(cinematic_script),
        "--ego-id", str(args.ego_id),
        "--seconds", str(args.seconds),
        "--gap", str(args.gap),
        "--cinematic",
        "--diagnose",
        "--gt-target-file", target_file.name,
    ]
    if args.npc_id is not None:
        cinematic_cmd += ["--npc-id", str(args.npc_id)]

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    print("CARLA V6.1 ONE-CLICK START")
    print("Ego ID:", args.ego_id)
    print("NPC ID:", args.npc_id if args.npc_id is not None else "自動選擇既有 NPC")
    print("提醒：CARLA 必須已啟動，而且模擬時間必須持續前進。")
    print("啟動 V5 LiDAR / GT / 盲區 / 事故監測……")

    monitor = subprocess.Popen(
        monitor_cmd, cwd=str(folder), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1, creationflags=creationflags
    )
    output_queue = queue.Queue()
    reader = threading.Thread(
        target=pump_output, args=(monitor.stdout, output_queue), daemon=True
    )
    reader.start()

    ready = False
    deadline = time.monotonic() + 60.0
    try:
        while time.monotonic() < deadline:
            if monitor.poll() is not None:
                print("V5 監測程式提前結束，請檢查上方錯誤訊息。")
                return monitor.returncode or 2
            try:
                line = output_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if "MONITOR READY." in line:
                ready = True
                break

        if not ready:
            print("等待 V5 MONITOR READY 逾時，停止本次啟動。")
            return 2

        print("\nV5 感測器已就緒，開始追撞演示……")
        cinematic_result = subprocess.run(cinematic_cmd, cwd=str(folder)).returncode
        print("\n追撞演示結束，保留監測資料收尾……")
        time.sleep(max(0.0, args.post_roll))
        return_code = cinematic_result
    finally:
        if monitor.poll() is None:
            try:
                if os.name == "nt":
                    monitor.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    monitor.send_signal(signal.SIGINT)
                monitor.wait(timeout=15)
            except Exception:
                monitor.terminate()
                try:
                    monitor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
        reader.join(timeout=3)

    after_monitor = set(folder.glob("integrated_carla_v61_gtposition_*.csv"))
    after_events = set(folder.glob("accident_events_v61_gtposition_*.csv"))
    new_monitor = list(after_monitor - before_monitor)
    new_events = list(after_events - before_events)
    monitor_csv = max(new_monitor, key=lambda p: p.stat().st_mtime) if new_monitor else None
    event_csv = max(new_events, key=lambda p: p.stat().st_mtime) if new_events else None

    report = write_report(folder, monitor_csv, event_csv, return_code, started_at)
    print("\nCARLA V6 測試流程結束。")
    print("報告：", report)
    if monitor_csv:
        print("監測 CSV：", monitor_csv)
    if event_csv:
        print("事故事件 CSV：", event_csv)
    print("車輛若仍處於凍結狀態，請執行 CARLA_V6_RESUME.bat。")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
