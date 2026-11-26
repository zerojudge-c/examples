#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CARLA V6.2 shadow warning comparison. Offline, read-only; never starts CARLA.

Usage: py -3.10 carla_v62_compare.py
       py -3.10 carla_v62_compare.py --csv integrated_carla_v62_shadow_....csv
"""
import argparse
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter


def numeric(v):
    try:
        return float(v) if v not in (None, '') else None
    except ValueError:
        return None


def flag(row, name):
    return str(row.get(name, '')).strip().lower() in ('1', 'true')


def first_time(rows, key):
    for row in rows:
        if flag(row, key):
            t = numeric(row.get('sim_time_s'))
            if t is not None:
                return t
    return None


def episodes(rows, key):
    """Count warning rising edges in CSV order; not independently verified crashes."""
    count = 0
    previous = False
    for row in rows:
        active = flag(row, key)
        if active and not previous:
            count += 1
        previous = active
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path)
    args = ap.parse_args()
    folder = Path(__file__).resolve().parent
    path = args.csv
    if path is None:
        files = sorted(folder.glob('integrated_carla_v62_shadow_*.csv'), key=lambda p:p.stat().st_mtime, reverse=True)
        if not files:
            print('找不到 integrated_carla_v62_shadow_*.csv；請先執行 V6.2 監測器並正常關閉儲存 CSV。')
            return 2
        path = files[0]
    if not path.is_file():
        print('找不到 CSV:', path)
        return 2
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'sim_time_s','lidar_warning','lidar_shadow_warning','lidar_instant_closing_mps','gt_position_warning'}
        missing = required-set(reader.fieldnames or [])
        if missing:
            print('CSV 缺少 V6.2 欄位：', ', '.join(sorted(missing)))
            return 2
        rows = list(reader)
    if not rows:
        print('CSV 沒有資料。')
        return 2
    orig=sum(flag(r,'lidar_warning') for r in rows)
    shadow=sum(flag(r,'lidar_shadow_warning') for r in rows)
    gt=sum(flag(r,'gt_position_warning') for r in rows)
    disagree=[r for r in rows if flag(r,'lidar_warning')!=flag(r,'lidar_shadow_warning')]
    only_orig=[r for r in disagree if flag(r,'lidar_warning')]
    only_shadow=[r for r in disagree if flag(r,'lidar_shadow_warning')]
    statuses=Counter(r.get('lidar_shadow_mode','') for r in rows)
    first_orig=first_time(rows,'lidar_warning')
    first_shadow=first_time(rows,'lidar_shadow_warning')
    first_gt=first_time(rows,'gt_position_warning')
    lines=[
       'CARLA V6.2 LiDAR Shadow 對照分析（離線）',
       '='*50,
       '輸入 CSV：'+path.name,
       '樣本筆數：'+str(len(rows)),
       f'原版 LiDAR WARNING 取樣：{orig}；警報上升沿：{episodes(rows,"lidar_warning")}',
       f'候選 Shadow WARNING 取樣：{shadow}；警報上升沿：{episodes(rows,"lidar_shadow_warning")}',
       f'GT Position WARNING 取樣：{gt}',
       f'兩版本不同取樣：{len(disagree)}（原版有／候選無：{len(only_orig)}；原版無／候選有：{len(only_shadow)}）',
       f'原版首次警報模擬時刻：{first_orig if first_orig is not None else "無"}',
       f'候選首次警報模擬時刻：{first_shadow if first_shadow is not None else "無"}',
       f'GT 首次警報模擬時刻：{first_gt if first_gt is not None else "無"}',
       '',
       '候選模式樣本數：'+str(dict(statuses)),
       '',
       '警報不同的取樣（最多 30 筆）：',
       'wall_s | sim_s | track_id | original | shadow | instant_mps | GT_pos_warn | shadow_mode',
    ]
    for r in disagree[:30]:
        lines.append(' | '.join(str(r.get(k,'')) for k in (
           'wall_elapsed_s','sim_time_s','lidar_track_id','lidar_warning','lidar_shadow_warning',
           'lidar_instant_closing_mps','gt_position_warning','lidar_shadow_mode')))
    lines += [
      '',
      '解讀限制：',
      '1. Shadow 是離線候選策略，不影響實際 LiDAR WARNING、介面警報或 UDP。',
      '2. 與 GT 警報不一致不等於誤報；LiDAR Track 尚未必獨立識別對應的 NPC。',
      '3. 警報上升沿是依 CSV 記錄切分的候選事件，不等於物理事故。',
      '4. V6.1 原版已經用最近感測間隔排除 <=0.7m/s 接近速度；本版僅評估較保守的快速解除門檻。',
      '5. 不同場景、不同時間的 CSV 不能直接比較準確率。',
    ]
    output=folder / ('carla_v62_shadow_report_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt')
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    print('\n'.join(lines))
    print('\n報告已輸出：', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
