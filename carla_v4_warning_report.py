#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze CARLA V4 warning performance using saved CSV only (no CARLA connection).

Place this file next to integrated_carla_v4_*.csv and accident_events_v4_*.csv.
Run: py -3.10 carla_v4_warning_report.py
Optional: py -3.10 carla_v4_warning_report.py --csv integrated_carla_v4_YYYYMMDD_HHMMSS.csv
"""
import argparse
import csv
import statistics
from pathlib import Path

MONITOR_PREFIX = 'integrated_carla_v4_'
EVENT_PREFIX = 'accident_events_v4_'


def rows_from_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f'CSV is empty or lacks a header: {path.name}')
        return list(reader)


def number(value):
    try:
        return float(value) if value is not None and str(value).strip() != '' else None
    except (ValueError, TypeError):
        return None


def pretty(value, digits=2, suffix=''):
    return '--' if value is None else f'{value:.{digits}f}{suffix}'


def rate(count, total):
    return '--' if not total else f'{count / total * 100:.2f}%'


def describe_leads(events, key):
    leads = [number(ev.get(key)) for ev in events]
    leads = [v for v in leads if v is not None and v >= 0]
    return {
        'count': len(leads),
        'avg': statistics.mean(leads) if leads else None,
        'median': statistics.median(leads) if leads else None,
        'min': min(leads) if leads else None,
        'max': max(leads) if leads else None,
    }


def summarize(monitor_file, event_file):
    measurements = rows_from_csv(monitor_file)
    needed = {'match_gt_actor_id', 'match_status', 'matched_error_m', 'lidar_warning', 'gt_warning'}
    missing = needed - set(measurements[0]) if measurements else set()
    if missing:
        raise ValueError('Missing V4 monitor columns: ' + ', '.join(sorted(missing)))

    targets = [r for r in measurements if (r.get('match_gt_actor_id') or '').strip()]
    matched = [r for r in targets if (r.get('match_status') or '').strip().startswith('YES')]
    errors = [abs(v) for r in matched if (v := number(r.get('matched_error_m'))) is not None]
    episodes = rows_from_csv(event_file) if event_file.is_file() else None
    if episodes is not None:
        event_cols = {'event_number', 'target_id', 'collision_sim_time_s',
                      'lidar_lead_s_unclassified', 'gt_lead_s_target_matched'}
        if episodes and not event_cols.issubset(episodes[0]):
            raise ValueError('Event file does not have V4 event columns: ' + event_file.name)
    lid = describe_leads(episodes, 'lidar_lead_s_unclassified') if episodes is not None else None
    gt = describe_leads(episodes, 'gt_lead_s_target_matched') if episodes is not None else None

    lines = [
        'CARLA V4 | 事故預警效能報告',
        '=' * 60,
        f'監測資料：{monitor_file.name}',
        f'事故事件：{event_file.name if episodes is not None else "未找到相同場次的事件 CSV"}',
        f'監測取樣數：{len(measurements):,}',
        '',
        '一、GT 輔助空間配對與測距（非純 LiDAR 辨識準確率）',
        f'具 GT 目標的取樣數：{len(targets):,}',
        f'GT 輔助配對成功取樣：{len(matched):,}',
        f'配對比例：{rate(len(matched), len(targets))}',
        f'成功配對且有誤差數值的取樣：{len(errors):,}',
        f'配對樣本距離 MAE：{pretty(statistics.mean(errors) if errors else None, 3, " m")}',
        f'最小／最大絕對距離差：{pretty(min(errors) if errors else None, 3, " m")} / '
        f'{pretty(max(errors) if errors else None, 3, " m")}',
        '',
        '二、碰撞前預警（依 V4 事件檔記錄的回溯視窗，預設 10 秒）',
    ]

    if episodes is None:
        lines += [
            '找不到對應的 accident_events_v4_*.csv，無法計算事故前警報率。',
            '請確認 V4 監測器已正常停止並儲存事件 CSV。'
        ]
    else:
        count = len(episodes)
        lines += [
            f'去重後碰撞候選事件：{count}',
            f'LiDAR 碰撞前曾有 WARNING：{lid["count"]}/{count}（{rate(lid["count"], count)}）',
            f'LiDAR 有預警事件的平均提前量：{pretty(lid["avg"], 3, " s")}',
            f'LiDAR 提前量中位數：{pretty(lid["median"], 3, " s")}',
            f'GT（撞擊 NPC ID 相符）碰撞前曾有 WARNING：{gt["count"]}/{count}（{rate(gt["count"], count)}）',
            f'GT 有預警事件的平均提前量：{pretty(gt["avg"], 3, " s")}',
            f'GT 提前量中位數：{pretty(gt["median"], 3, " s")}',
            '',
            '三、各場碰撞事件',
        ]
        if not episodes:
            lines.append('這一場 V4 監測期間沒有已記錄的碰撞事件。請在 V4 開啟期間進行事故測試。')
        for ev in episodes:
            ev_num = (ev.get('event_number') or '?').strip()
            npc = (ev.get('target_id') or '?').strip()
            stamp = (ev.get('collision_sim_time_s') or '--').strip()
            lidar_lead = number(ev.get('lidar_lead_s_unclassified'))
            gt_lead = number(ev.get('gt_lead_s_target_matched'))
            lines.append(f'事件 #{ev_num}｜NPC {npc}｜模擬時間 {stamp} s｜'
                         f'LiDAR* {pretty(lidar_lead, 3, " s")}｜GT(ID) {pretty(gt_lead, 3, " s")}')

    lines += [
        '',
        '四、統計解讀與限制',
        '1. 碰撞數是監測器按接觸對象、分離和冷卻時間「去重」後的候選事件，不保證每筆都是獨立交通事故。',
        '2. LiDAR* 預警未綁定碰撞目標 ID；10 秒內發出過 WARNING 只表示時間上相關，不能宣稱辨識到被撞車。',
        '3. GT 警報以撞擊 NPC ID 相符做對照，但仍受取樣頻率與 TTC 定義影響。',
        '4. 平均預警提前量只對「有發出警報的事件」計算；沒有警報的事件不以 0 秒混入平均。',
        '5. GT 輔助配對率不是 LiDAR 自主辨識準確率；MAE 僅基於成功配對且有有效誤差的取樣。',
        '6. 本報告不計算誤報率：若要評估誤報，仍需定義非事故情境、警報事件去重與真實威脅標記。',
        '7. 此報告只讀取本機 CSV，不連線 CARLA，不移動 Ego/NPC，不會更改模擬設定。',
        '',
    ]
    return '\n'.join(lines), (None if episodes is None else len(episodes)), lid, gt, len(targets), len(matched), errors


def main():
    parser = argparse.ArgumentParser(description='CARLA V4 warning performance CSV report; no CARLA connection')
    parser.add_argument('--csv', help='V4 monitor CSV path (default: newest in script folder)')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    if args.csv:
        monitor = Path(args.csv).expanduser().resolve()
    else:
        options = sorted(base.glob(MONITOR_PREFIX + '*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not options:
            parser.error('No integrated_carla_v4_*.csv found. Put this script in PythonAPI/examples with the CSV files.')
        monitor = options[0]
    if not monitor.is_file():
        parser.error('Monitor CSV not found: ' + str(monitor))
    if not monitor.name.startswith(MONITOR_PREFIX) or not monitor.name.endswith('.csv'):
        parser.error('Expected a file named integrated_carla_v4_*.csv')

    tag = monitor.name[len(MONITOR_PREFIX):-4]
    event_file = monitor.with_name(f'{EVENT_PREFIX}{tag}.csv')
    try:
        report, n_events, lid, gt, n_targets, n_matched, errors = summarize(monitor, event_file)
    except (ValueError, OSError, csv.Error) as exc:
        parser.error(str(exc))

    result = monitor.with_name('carla_v4_warning_report_' + tag + '.txt')
    result.write_text(report, encoding='utf-8-sig')
    print('Report saved:', result)
    print('GT-assisted match:', f'{n_matched}/{n_targets}', '=', rate(n_matched, n_targets))
    print('Matched-sample MAE:', pretty(statistics.mean(errors) if errors else None, 3, ' m'))
    if n_events is None:
        print('Collision events: N/A (matching event CSV not found)')
    else:
        print('Candidate collision episodes:', n_events)
        print('LiDAR* warnings before collision:', f'{lid["count"]}/{n_events}', '=', rate(lid['count'], n_events))
        print('GT ID-matched warnings:', f'{gt["count"]}/{n_events}', '=', rate(gt['count'], n_events))
        print('Average lead: LiDAR*', pretty(lid['avg'], 3, ' s'), '| GT(ID)', pretty(gt['avg'], 3, ' s'))
    print('Note: LiDAR* warnings are NOT target-identified. Read the report limitations.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
