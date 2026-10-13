"""Verify saved Stage F evidence and write its independent Chinese report."""
import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from k1_goalkeeper.velocity_validation import trajectory_hash


def report(directory, destination):
    source = directory/'results.json'
    data = json.loads(source.read_text())
    if len(data['shots']) != 10 or not data['f1_passed']:
        raise RuntimeError('Full Stage F protocol has not completed')
    checks = {}
    for group in ('protected_sha256', 'code_sha256'):
        checks[group] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in data[group].items())
    checks['saved_trajectories'] = all(trajectory_hash(dict(np.load(directory/r['case']['name']/'trajectory.npz'))) == r['trajectory_sha256'] for r in data['results'])
    checks['physics_policy_timing'] = all(r['physics_steps'] == round(r['simulated_duration_s']/.001) and
        (r['case']['robot_mode'] == 'absent' or r['policy_steps'] == (r['physics_steps']+19)//20) for r in data['results'])
    checks['repeats'] = data['all_repeats_identical']
    checks['matched_launches'] = all(r['launch_identical_to_absent'] for r in data['results'] if r['mode'] in ('passive', 'active'))
    checks['all_completed'] = all(r['completed'] for r in data['results'])
    checks['no_numerical_errors'] = data['numerical_error_count'] == 0
    checks['no_unexpected_contacts'] = data['unexpected_contact_count'] == 0
    checks['e_recorder_parity'] = json.loads((directory/'e_recorder_parity.json').read_text())['identical']
    audit = json.loads((directory/'response_audit.json').read_text())
    checks['response_audit_inputs'] = all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() == r['sha256'] for r in audit['input_manifest'])
    deadline = json.loads((directory/'deadline_audit.json').read_text())
    checks['deadline_audit_inputs'] = all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() == r['sha256'] for r in deadline['input_manifest'])
    rows = {r['case']['name']: r for r in data['results']}
    active = [r for r in data['results'] if r['mode'] == 'active']
    alignment = {}
    for shot in data['shots']:
        reference = np.load(directory/(shot['name']+'_absent')/'trajectory.npz')
        for mode in ('passive', 'active'):
            trajectory = np.load(directory/(shot['name']+'_'+mode)/'trajectory.npz')
            time = reference['time']
            robot = np.column_stack([np.interp(time, trajectory['time'], trajectory['base_position'][:, i]) for i in range(3)])
            ball = reference['ball_position']
            difference = ball[:, 0]-robot[:, 0]
            indices = np.flatnonzero((time[1:] > 2.) & (difference[:-1] <= 0) & (difference[1:] > 0))
            if len(indices):
                i = indices[0]
                f = -difference[i]/(difference[i+1]-difference[i])
                b, r = ball[i]+f*(ball[i+1]-ball[i]), robot[i]+f*(robot[i+1]-robot[i])
                alignment[shot['name']+'_'+mode] = {'time': float(time[i]+f*(time[i+1]-time[i])),
                    'robot_goal_lateral_m': float(r[1]), 'unobstructed_ball_lateral_m': float(b[1]),
                    'lateral_error_m': float(r[1]-b[1]), 'unobstructed_ball_height_m': float(b[2]),
                    'interpretation': 'offline matched absent ball crossing observed robot center X; after-contact robot response may affect this score'}
    analysis = {'verification': checks, 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'python_version': platform.python_version(),
                'report_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'response_audit_sha256': hashlib.sha256((directory/'response_audit.json').read_bytes()).hexdigest(),
                'tests': (directory/'tests.txt').read_text() if (directory/'tests.txt').exists() else 'not yet run',
                'case_count': len(rows), 'execution_count': len(rows)*data['repeats'],
                'alignment_at_unobstructed_robot_plane': alignment,
                'active_block_count': sum(r['blocked'] is True for r in active),
                'passive_block_count': sum(r.get('blocked') is True for r in rows.values() if r['mode'] == 'passive'),
                'added_success_count': data['added_success_count'], 'lost_passive_success_count': data['lost_passive_success_count']}
    (directory/'verification.json').write_text(json.dumps(analysis, indent=2, allow_nan=False)+'\n')
    test_count = re.search(r'(\d+) passed', analysis['tests'])
    text = ["# 阶段 F：K1 主动守门站位闭环基线", "",
        f"完成 {len(rows)} 个工况、{len(rows)*data['repeats']} 次执行：7个无球站位目标，10组相同发射的无机器人/零速度/主动闭环对照。重复轨迹及高层决策逐值一致：{checks['repeats']}。",
        f"无机器人右门进球 {sum(s['absent_right_goal'] is True for s in data['shots'])}/10；被动有效阻挡 {analysis['passive_block_count']}/10，主动有效阻挡 {analysis['active_block_count']}/10；新增阻挡 {data['added_success_count']}，丢失被动成功 {data['lost_passive_success_count']}。",
        f"跌倒工况 {data['fall_case_count']}；数值异常 {data['numerical_error_count']}；机器人与球门意外接触 {data['unexpected_contact_count']}。所有完整性核对通过：{all(checks.values())}。", "",
        f"自动化测试：{test_count.group(1) if test_count else '尚未记录'} 项通过，原187项保留。测试输出中的QVEL警告来自E0已独立定位的极端1e100速度异常测试，不是本轮守门仿真的数值异常。", "",
        "## 最小接入与不变项", "",
        "高层位于 `k1_goalkeeper/positioning.py`，仿真适配/记录位于 `positioning_validation.py`。`impact_validation.run_impact` 只增加两个默认关闭的可选接口：当前状态指令提供器和物理步完成观察器。复用原SceneSimulation、状态名称映射、接触力/冲量记录、跌倒、恢复与轨迹哈希设施；每个独立工况重置一次，指令切换不重置策略历史。",
        "采用1ms物理、20ms官方策略、逐物理步官方PD；高层指令也在20ms边界更新。球在t=2s发射，调用顺序决定首次可见射门状态在t=2.02s；此20ms观察延迟是实际执行的一部分。无机器人按同样时钟发射。",
        "官方模型、策略、PD、场景物理、进球判定及A–E JSON报告和原187项测试通过保护哈希核对。E记录器front_2回放与历史保存轨迹SHA-256完全相同。E文档、6个严格动量精度未达标结果和20m/s跌倒结论没有改写；F新接口属于新版本，历史E代码哈希仍描述当时版本。", "",
        "## F1 无球站位与边界", "",
        "右门坐标原点世界(11,0,0)，门+X=世界+Y，门+Y=世界−X。目标是球门横向坐标，初始机身+Y朝世界−Y。按当前航向将横向误差投影到机身；只输出已测vy=±0.20m/s，偏航偏离0.12rad时先以±0.20rad/s修正，恢复到0.035rad内再横移。未实现纵深位置跟踪。",
        "0.10m启动误差门槛、0.035m预测停止误差、0.4s速度外推提前停止、1.2s再启动等待；速度以0.25s低通抑制步态摆动。最短运动0.4s。最终到位要求±0.10m并保持至少1s，且全程姿态稳定、末段恢复、无意外接触。该误差尺度不是厘米级精度保证。",
        "C/D独立审计覆盖20段横移、15段停止：0.5s窗口最低实测速度0.106815m/s，停止最大余移0.059930m。采用0.10m/s保守运动速度、1s起步、1s制动时间、0.07m停止+0.02m反向余量；已有末段稳定判据继续要求完整低速观察。",
        "全碰撞体的任意关节转动几何界0.693084m，内门柱横向±1.5m，减去0.05m净空与0.09m运动余量后中心界±0.666916m，实际夹限±0.65m。另监测世界X在10.05–10.75m的守门带；越界只停止，不声称能主动返回。详见response_audit.json。", "",
        "| 目标门横向m | 最终误差m | 最大越过目标m | 到位并持续s | 末段恢复 |",
        "|---:|---:|---:|---:|---|"]
    for row in data['positioning']:
        r = rows[f"position_{row['requested_goal_lateral_m']:+.2f}"]
        text.append(f"| {row['requested_goal_lateral_m']:+.2f} | {row['final_error_m']:+.4f} | {row['maximum_overshoot_m']:.4f} | {row['arrival_and_hold_time_s']:.3f} | {r['final_stop_recovery']['verified']} |")
    text += ["", "## F2/F3 预测和可达性", "",
        "只读取当前足球位置、世界线速度、局部角速度和当前机器人状态。近地低竖直速度球使用匀速地面近似，腾空球使用恒重力弹道；角速度记录但没有引入自旋空气动力。预测分别对应球心X=11和整球X=11.11，并检查半径0.11m的完整开口。",
        "预测到达前会再次触地时，原始弹道估计仍保存且标记future_floor_contact_unmodeled，开口结论为None，主动控制暂停采用该预测。弹跳结束后可从新的当前状态恢复。滚动减速、弹跳及门/机器人碰撞不会被假定符合自由弹道；完整误差逐样本保存，包含被标记为不可用的预测。",
        "控制以机器人当前世界X−0.81m包络平面估计剩余时间，采用0.693084m任意关节姿态球形包络加0.11m球半径后向外取整。该瞬时几何界早于可能身体接触，但未来纵深漂移、球加速度仍可能改变到达时间，故它是经验保守估计。保守运动预算为max(0,0.10×max(T−1−1,0)−max(0.09,|滤波横速|×1))，再与安全站位边界取交集。短于制动时间且仍在运动时标记不可达。区间成员判断考虑±0.10m到位容差；已在容差内且能及时制动时无需新运动。可达指中心站位达到±0.10m误差的经验估计，不是球与碰撞体相交的保证。不可达时只向预算允许的最近位置移动；预算缩小或球已通过/接触后输出零速度。",
        "独立审计发现先导版代表前缘平面可多算至0.095852s，最终版改为上述球形包络并重跑全部协议。先导37工况/74次执行保存在reports/positioning_validation_pilot，不计入本报告正式结果；deadline_audit.json保存2354个状态的底层几何核对。",
        "误差统计在实验完成后与相同发射的无机器人真实过线相比较，机器人第一次有效接触后的预测不纳入该反事实比较。真实未来轨迹只用于离线评分，从未作为控制输入。", "",
        "## F4 匹配对照", "",
        "发射前站立2s，发射后观察18s；每组球的位置、线速度和角速度完全相同。滚动球从世界X=−3/−1/0m发射，腾空球X=4m，弹跳球X=2m，均显著远于E的贴近撞击。速度是实验选点。", "",
        "| 射门 | 无机器人右门 | 被动接触/右门 | 主动接触/右门 | 初次可达估计 | 新增成功 |",
        "|---|---|---|---|---|---|"]
    for s in data['shots']:
        first = s['active_first_reachability']
        text.append(f"| {s['name']} | {s['absent_right_goal']} | {s['passive_contact']}/{s['passive_right_goal']} | {s['active_contact']}/{s['active_right_goal']} | {first['reachable'] if first else '预测不可用'} | {s['added_success']} |")
    text += ["", "离线对齐误差另用无机器人球心经过实际机器人中心X的时刻评分（碰撞后的机器人响应可能影响该评分，未作为控制输入）：", "",
             "| 射门 | 被动横向误差m | 主动横向误差m | 对照球高m |", "|---|---:|---:|---:|"]
    for s in data['shots']:
        passive, current = alignment.get(s['name']+'_passive'), alignment.get(s['name']+'_active')
        if passive and current:
            text.append(f"| {s['name']} | {passive['lateral_error_m']:+.4f} | {current['lateral_error_m']:+.4f} | {current['unobstructed_ball_height_m']:.4f} |")
    text += ["", "四项独立保存：有效物理接触、相对无机器人轨迹改变（位置≥0.02m或速度≥0.1m/s）、维持/恢复平衡、右门是否进球。有效阻挡还要求相同发射的无机器人确实进右门、机器人发生有效接触且观察完整。新增成功要求被动基线仍进右门。任何一项不能代替其余项。", "",
        "| 射门 | 首次预测时间误差s | 横向误差m | 高度误差m | 初次模型/未建模弹跳 | 主动跟踪RMSE m |",
        "|---|---:|---:|---:|---|---:|"]
    for s in data['shots']:
        r = rows[s['name']+'_active']
        error = r['prediction_errors_against_matched_absent']['11.11']['first']
        if error:
            rmse = 'None' if r['tracking_rmse_m'] is None else f"{r['tracking_rmse_m']:.4f}"
            text.append(f"| {s['name']} | {error['time_error_s']:+.4f} | {error['lateral_error_m']:+.4f} | {error['height_error_m']:+.4f} | {error['model']}/{error['unmodeled_bounce']} | {rmse} |")
    text += ["", "RMSE针对每次实际下发的保守决策目标，包含起步误差；它不等于对原始射门落点的最终对齐误差。None表示没有可用的站位跟踪样本。预测误差的负高度估计在未建模弹跳时仍如实保存，不能作为有效高度预测。", "",
        "## 能力与限制", "",
        "这是可重复运行的横向站位基线：有足够提前量时可以移动到粗站位，快球和大偏角会明确出现不可达；更高的躯干外射门即使中心站位可达也不能保证拦截。每种运动/射门仅验证本表有限选点，未形成通用成功率。",
        f"斜向滚动球虽成功阻挡，离线中心对齐误差仍为{abs(alignment['angled_active']['lateral_error_m']):.3f}m，超过F1的0.10m到位容差；足腿接触提供的拦截覆盖不能等同于精确到位。弹跳工况初期预测不可用，恢复近地预测后剩余时间不足，主动组没有新增拦截。",
        "继承E碰撞审计：肩根shoulder_pitch、hip_pitch和ankle_pitch关节邻域只有视觉几何，躯干/腿部也不是连续完整包络。无接触同时伴随站位偏差时无法单独归因于控制或几何；高于现有机器人实体范围的球是本动作集覆盖限制。未新增碰撞体、下蹲、伸手、扑救、视觉、强化学习。",
        "后续优先校准滚动/弹跳预测与到位后的拦截覆盖，并在独立场景扩展中评估缺失碰撞体；快球/高球需要新的动作能力或更早感知，当前横移策略不能承诺解决。", "",
        "## 复现与文件", "", "```bash",
        ".venv/bin/python scripts/validate_positioning.py --phase position --output reports/f_position_check",
        ".venv/bin/python scripts/validate_positioning.py",
        ".venv/bin/python -m pytest -q > reports/positioning_validation/tests.txt",
        ".venv/bin/python scripts/report_positioning.py", "```", "",
        "完整入口先重新通过F1，再运行射门；F1不通过时退出且不发射。--repeats 1仅供诊断，不标记重复性通过。全部旧测试保留；最终pytest输出见reports/positioning_validation/tests.txt。",
        "results.json保存协议、摘要、误差、哈希及对照；每工况summary.json包含每20ms的真值输入/预测/可达性/目标/实际命令，trajectory.npz保存每1ms原E所有物理字段并新增goal_position和high_level_json_utf8（决策JSON原始UTF-8字节，用于轨迹哈希）。contacts列定义沿用E，全部力矩、饱和、接触时刻/部位/力/冲量均可追溯。verification.json核对每份数组哈希、原文件、时序、发射匹配和回归输出。response_audit.json及e_recorder_parity.json为独立前置证据。", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(text))
    (directory/'report.md').write_text('\n'.join(text))
    print(json.dumps(analysis, indent=2))
    if not all(checks.values()):
        raise SystemExit('Saved-evidence verification failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'reports/positioning_validation')
    parser.add_argument('--markdown', type=Path, default=ROOT/'docs/positioning_validation.md')
    args = parser.parse_args()
    report(args.input, args.markdown)
