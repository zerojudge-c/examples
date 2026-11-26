#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline V6.1 warning analysis. Reads CSV only; NEVER connects to or controls CARLA."""
import argparse
import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


def number(x):
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def flag(row, name):
    return str(row.get(name, '')).strip().lower() in ('1', 'true', 'yes')


def fmt(x, decimals=3):
    return '--' if x is None else f'{x:.{decimals}f}'


def main():
    p = argparse.ArgumentParser(description='Analyze V6.1 LiDAR / GT Position warnings offline')
    p.add_argument('csvfile', nargs='?', help='V6.1 monitor CSV (default: newest in this directory)')
    p.add_argument('--out', default=None, help='Output report file')
    args = p.parse_args()
    folder = Path(__file__).resolve().parent
    if args.csvfile:
        source = Path(args.csvfile)
    else:
        candidates = sorted(folder.glob('integrated_carla_v61_gtposition_*.csv'),
                            key=lambda q: q.stat().st_mtime, reverse=True)
        if not candidates:
            p.error('找不到 integrated_carla_v61_gtposition_*.csv，請把本程式放在 examples 資料夾')
        source = candidates[0]
    if not source.exists():
        p.error(f'找不到檔案：{source}')
    with source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        need = {'sim_time_s', 'lidar_warning', 'gt_position_warning'}
        if not need.issubset(fields):
            p.error('CSV 缺少欄位：' + ', '.join(sorted(need-fields)))
        rows = list(reader)
    valid = [(number(r.get('sim_time_s')), r) for r in rows]
    valid = [(t, r) for t, r in valid if t is not None]
    valid.sort(key=lambda pair: pair[0])
    # Deduplicate exact timestamps; preserve latest row on duplicate times.
    by_t = {t: r for t, r in valid}
    samples = sorted(by_t.items())
    gt = [(t, r) for t, r in samples if flag(r, 'gt_position_warning')]
    ld = [(t, r) for t, r in samples if flag(r, 'lidar_warning')]
    both = [(t, r) for t, r in samples if flag(r, 'lidar_warning') and flag(r, 'gt_position_warning')]
    gt_only = [(t, r) for t, r in samples if flag(r, 'gt_position_warning') and not flag(r, 'lidar_warning')]
    ld_only = [(t, r) for t, r in samples if flag(r, 'lidar_warning') and not flag(r, 'gt_position_warning')]
    first_gt = gt[0][0] if gt else None
    first_ld = ld[0][0] if ld else None
    delta = first_ld-first_gt if first_ld is not None and first_gt is not None else None
    target_pairs = Counter((r.get('gt_vehicle_id') or '--', r.get('selected_gt_target_id') or '--') for _, r in samples)
    matched = [(t,r) for t,r in samples if (r.get('match_status') or '').startswith('YES')]
    numeric_ttc = lambda f: [number(r.get(f)) for _, r in samples if number(r.get(f)) is not None]
    lttc = numeric_ttc('lidar_ttc_s_est')
    gttc = numeric_ttc('gt_position_ttc_s')
    min_lidar = min(lttc) if lttc else None
    min_gt = min(gttc) if gttc else None
    lines = [
        'CARLA V6.1：LiDAR / GT Position 預警分析（離線）',
        '='*58,
        f'分析檔案：{source.name}',
        f'分析時間：{datetime.now():%Y-%m-%d %H:%M:%S}',
        f'CSV 原始筆數：{len(rows)}；有效且不同模擬時間點：{len(samples)}',
        f'LiDAR WARNING 取樣筆數：{len(ld)}',
        f'GT Position WARNING 取樣筆數：{len(gt)}',
        f'同筆兩者都警報：{len(both)}',
        f'同筆只有 GT Position 警報：{len(gt_only)}',
        f'同筆只有 LiDAR 警報：{len(ld_only)}',
        f'首次 LiDAR WARNING 模擬時間：{fmt(first_ld)} s',
        f'首次 GT Position WARNING 模擬時間：{fmt(first_gt)} s',
        f'首次警報差 (LiDAR - GT)：{fmt(delta)} s',
        ('判讀：GT Position 較早。' if delta is not None and delta>0 else
         '判讀：LiDAR 較早。' if delta is not None and delta<0 else
         '判讀：首次時間相同。' if delta is not None else '判讀：無法比較，至少一套沒有警報。'),
        f'最小 LiDAR TTC：{fmt(min_lidar)} s',
        f'最小 GT Position TTC：{fmt(min_gt)} s',
        f'GT-assisted 配對樣本：{len(matched)}',
        '', '[GT 最近前車, 指定目標：筆數]',
    ]
    for pair, count in target_pairs.most_common(8):
        lines.append(f'  {pair[0]} / {pair[1]}：{count}')
    lines += [
        '', '[每筆警報樣本（模擬秒）]',
        'time_s | GT_ID | SELECTED_ID | TRACK_ID | LiDAR_WARN | GT_POS_WARN | LiDAR_TTC | GT_POS_TTC',
    ]
    for t,r in samples:
        if flag(r, 'lidar_warning') or flag(r, 'gt_position_warning'):
            lines.append(' | '.join((fmt(t),r.get('gt_vehicle_id') or '--',
                    r.get('selected_gt_target_id') or '--',r.get('lidar_track_id') or '--',
                    '1' if flag(r,'lidar_warning') else '0',
                    '1' if flag(r,'gt_position_warning') else '0',
                    fmt(number(r.get('lidar_ttc_s_est'))),
                    fmt(number(r.get('gt_position_ttc_s'))))))
    lines += [
        '', '[研究限制]',
        '1. WARNING 筆數是取樣次數，並非獨立事故或獨立預警事件數。',
        '2. GT Position 是 CARLA Actor 位置差分之診斷參考；與 LiDAR 的資料來源獨立。',
        '3. GT Position 預警不等於碰撞真值；不能用它直接推算實際誤報率/漏報率。',
        '4. LiDAR Track 不帶車輛身分；即使最近前車 GT ID 相同也不能推論該 Track 已辨識該車。',
        '5. 電影式追撞是幾何接觸，沒有碰撞感測回呼時不能聲稱物理碰撞成功。',
        '6. 真正評估誤報/漏報需多次獨立場景、非碰撞負樣本、事件級對齊及真值標註。',
        '7. 第一筆預警的差值是 CSV 取樣時間差，不是精確警報產生延遲。',
    ]
    report = Path(args.out) if args.out else folder / f'carla_v61_warning_analysis_{datetime.now():%Y%m%d_%H%M%S}.txt'
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')
    print('\n'.join(lines))
    print('\n已輸出：', report)

if __name__ == '__main__':
    main()
