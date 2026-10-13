"""Render the independent Stage G report from completed, separately audited evidence."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'reports/prediction_validation'


def render():
    r = json.loads((DATA/'results.json').read_text())
    p = json.loads((DATA/'prediction_comparison.json').read_text())
    d = json.loads((DATA/'decision_audit.json').read_text())
    ablation = json.loads((DATA/'ablation.json').read_text())
    test_count = int(re.search(r'(\d+) passed', (DATA/'tests.txt').read_text()).group(1))
    if not (r.get('all_repeats_identical') and r.get('protected_unchanged') and
            p['complete'] and d['complete_protocol_observed']):
        raise RuntimeError('Complete physical validation, prediction comparison and independent audit required')
    lines = ['# 阶段 G：预测与可达性改进实验', '',
        '本阶段保留 F 默认控制器，仅新增显式 G 实验入口。官方模型、策略、PD、场景、球门、进球判定和 A–F 原始报告/轨迹均未修改。', '',
        f"原 10 球回归 F/G 均挡住 5 球；独立 8 球 F 挡住 1 球、G 挡住 2 球，新增 1 球且未丢失原成功。{test_count} 项自动化测试通过，保留原有 253 项。", '',
        f"正式比较包含 11 个无球目标 × F/G，以及 18 组射门 × 无机器人/被动/F/G，共 {len(r['results'])} 个物理工况。"
        f"其中 {r['new_cases']} 个新工况各执行两次，共 {r['new_cases']*2} 次；引用 {r['historical_cases']} 个已存档 F 工况的两次重复记录，未重写历史结果。",
        '另对原 F `center_active` 做独立全量重放，轨迹哈希与存档完全一致，见 `baseline_replay.json`。新实验每次重新初始化策略，连续控制过程中不重置策略历史。', '',
        f"重复一致：{r['all_repeats_identical']}；无球 G 门槛通过：{r['position_gate_passed']}；"
        f"跌倒工况 {len(r['fall_cases'])}，数值异常 {len(r['numerical_error_cases'])}，意外撞门 {len(r['unexpected_contact_cases'])}。", '',
        '## 1. 先审计 F 失败，不预设算法根因', '',
        '| F 工况 | 保存轨迹与原生几何证据 | 根因边界 |',
        '|---|---|---|',
        '| wide_left/right | 横向预测误差为零，到达时间误差不超过 20.842 ms；最大横移预算约 0.0281 m；全部零指令 | 剩余时间、死区及 ±0.65 m 安全中心范围限制；不能归因于预测不准或局部碰撞体缺失 |',
        '| air_wide | 可用时间约 0.6803 s，保守运动预算为零；高度预测误差约 4.3 mm | 现有横移来不及，需要更早响应或额外动作能力 |',
        '| air_high | 整段机器人通过区间内球底比所有原生碰撞体顶部至少高 0.0337 m；站位标签虽为真但未接触 | 现有站立/横移身体高度不覆盖；需要新的高球动作，不能把横向可达当可拦截 |',
        '| bouncing | F 对未来触地明确弃权，直到 t≈2.70 s 才恢复可用预测；之后预算约 0.0337 m，小于起步死区 | 存在可改善的预测空窗；预测改善后的运动收益另由 G 实验验证 |',
        '| F1 −0.4 m | 无转向指令；两方向首次运动均 2.44 s。负向停止后缓慢漂移，在 t≈10.591 s 越出容差，触发第二次启动 | 停止近似、死区与不对称站立漂移耦合；不是 yaw 纠正造成 |', '',
        '详细逐帧距离、碰撞体名单、动作时间与原始哈希见 `failure_audit.json` / `failure_audit.md`。审计使用保存的完整 qpos 重建真实碰撞几何，不把视觉网格当碰撞体，也不把无接触直接归因为控制失败。', '',
        '## 2. 最小改动与假设', '',
        '- `ball_prediction.py`：复制现有场景的球、地面、默认接触和求解参数，建立独立的单球预测模型。每次仅从当前 qpos/qvel（含姿态和角速度）开始推进，预测 11 m 中心门线、11.11 m 整球门线、机器人中心及保守前缘时刻。实际未来只供离线评分。',
        '- 模型选择前比较了线性/重力、历史速度估计减速度及同参数物理模型。已测滚动 X 减速度约 −0.01069 m/s²，斜向球 Y 不符合统一各向同性阻力近似；弹跳的水平损失与恢复比随触地变化，未硬拟合单一恢复系数。选择已知接触物理，避免为原十球逐例调参。',
        '- 物理预测保留触地事件及 invalid 原因；不接近、超预测时域、无效状态/数值异常均不能当作有效落点。它不包含未来机器人/球门接触，接触后不继续以自由球预测指挥。',
        '- 预测以仿真时间 10 Hz 更新，高层速度及官方策略 50 Hz，PD 每 1 ms 执行。独立预测数据不会推进或修改真实场景。计算成本随剩余飞行时间增加；这不是已验收的机器人实时计算预算。',
        '- 固定更新时序中首次发射后预测更新，G 在 t=2.10 s，F 为 t=2.02 s；F 对未建模触地仍弃权。G 付出约 80 ms 初始更新等待，该时序在所有冻结工况中相同，没有按射门结果修改。',
        '- `predictive_goalkeeper.py`：保持 F 的实测保守速度、起步/停止预算及安全范围。分别记录站位可达、当前身体覆盖、理想目标覆盖、实际保守决策站位覆盖及拦截可行。覆盖基于现有原生几何距离，要求至少 15 mm 重叠余量，并标明冻结当前姿态及缓存观察时刻。',
        '- 冻结姿态覆盖是几何假设，不是未来步态形状保证。覆盖只用 5 ms 采样路径与现有横向平移；无覆盖也不证明所有可能动作都无解。未新增下蹲、伸手、扑救或碰撞体。',
        '- 无球位置修正仅把停稳重启残余阈值改为 0.08 m，保留 1.2 s 停止等待、原运动指令与 PD。理想拦截横向目标取机器人中心平面的预测，运动截止仍取前缘保守平面。', '',
        '## 3. 无球位置闭环与代价', '',
        '| 目标 m | F/G 持续到位 s | F/G 最终误差 m | F/G 运动指令段数 | G 通过原门槛 |',
        '|---:|---:|---:|---:|---|']
    for pair in d['positioning'].values():
        f, g = pair['F'], pair['G']
        lines.append(f"| {g['requested_goal_lateral_m']:+.2f} | {f['arrival_and_hold_time_s']:.3f} / {g['arrival_and_hold_time_s']:.3f} | "
                     f"{f['final_error_m']:+.4f} / {g['final_error_m']:+.4f} | {len(f['delivered_control_bursts'])} / {len(g['delivered_control_bursts'])} | {g['passed']} |")
    lines += ['', '−0.4 m 的持续到位时间从 8.624 s 降至 2.974 s，但最终绝对误差从 0.0515 m 增至 0.0866 m；运动指令段从 2 增至 6，最后停止从 12.00 s 推迟到 13.76 s。改善的是进入并保持容差的时间，代价是更多短促修正，不能称为厘米级定位或全面更优。', '',
        '## 4. 同状态的新旧预测误差', '',
        '每隔 0.25 s，从匹配的无机器人轨迹取当前完整球状态，同时运行 F/G，随后才读取真实过线值评分。表中是 x=11 m 平面 RMSE；11.11 m 的独立结果、最大误差与全部采样保存在 `prediction_comparison.json`。', '',
        '| 集合/球型 | 采样数 | F/G 时间 RMSE s | F/G 横向 RMSE m | F/G 高度 RMSE m | F 未建模触地数量 / G invalid |',
        '|---|---:|---:|---:|---:|---:|']
    for split in ('regression', 'holdout'):
        for kind in ('rolling', 'airborne', 'bouncing'):
            modes = p['groups'][split][kind]['11.0']; f, g = modes['F'], modes['G']
            fe, ge = f['errors_all_valid']['rmse'], g['errors_all_valid']['rmse']
            lines.append(f"| {split}/{kind} | {f['sample_count']} | {fe[0]:.6g} / {ge[0]:.3g} | {fe[1]:.6g} / {ge[1]:.3g} | {fe[2]:.6g} / {ge[2]:.3g} | {f['unmodeled_floor_count']} / {g['invalid_count']} |")
    lines += ['', 'F 已明确标记未建模反弹的结果也列入上表原始误差，因此很大的高度误差不是 F 曾承诺有效的预测。机器结果另提供排除未建模触地后的误差和可用率。G 近浮点精度的结果来自使用相同已知仿真物理，并非真实足球预测精度；实球摩擦、碰撞和感知延迟尚未标定。', '',
        '## 5. 四方配对与独立泛化', '',
        '独立 8 球在 G 实验前写入 `protocol.json`，覆盖不同横向落点、速度、斜向滚动、不同高度空中球及两个首次落地条件；没有依据其结果修改参数。每组四方初始位置、线速度、角速度完全相同。', '',
        '| 集合/射门 | 无机器人右门进球 | 被动 / F / G 有效阻挡 | G 接触 | G 首次可行判断 | 新增 / 丢失 F 成功 |',
        '|---|---|---|---|---|---|']
    for shot in r['shots']:
        m = shot['modes']; first = m['g_active']['first_interception_decision']
        lines.append(f"| {shot['split']}/{shot['name']} | {shot['absent_right_goal']} | "
                     f"{m['passive']['blocked']} / {m['f_active']['blocked']} / {m['g_active']['blocked']} | "
                     f"{m['g_active']['contact']} | {first['intercept_feasible'] if first else None} | {shot['added_over_F']} / {shot['lost_F_success']} |")
    for split, s in r['summary'].items():
        lines += ['', f"{split}：被动 {s['passive']}，F {s['f_active']}，G {s['g_active']}；相对 F 新增 {s['added']}，丢失 {s['lost']}。"]
    lines += ['', '有效阻挡要求：同发射无机器人对照进右门、有机器人发生有效接触、完整观察期右门未进球。观察期为发射后 18 s；跌倒/截断不算成功。实际接触、轨迹变化、平衡、进球仍是独立量，详见各工况 summary/trajectory 及 physical_comparison。', '',
        '独立组新增成功为 bounce_early_right：F 第一次横移指令在 t≈2.481 s，G 在 t≈2.101 s；F 单次足部接触后仍进门，G 经足部/小腿多次接触后右门未进球。',
        f"事后单项消融把 G 定位器换回原 F 定位器，其余预测/覆盖/时序保持 G：有效阻挡 {ablation['blocked']}，重复一致 {ablation['reproducibility']['verified']}。这说明该收益不依赖 0.08 m 残差修正；尚未拆分预测、几何决策、目标平面和刷新时序各自的贡献。消融另计 1 工况/2 次，不纳入冻结独立组的新增成功数量。见 `ablation.json`。", '',
        '## 6. 可达性判定审计', '',
        '| 集合/算法 | 站位正判断工况 | 站位终点因接触截断 | 可观察站位样本 / 未到达 | 拦截正判断工况 | 有正判断却无接触 / 未挡住 |',
        '|---|---:|---:|---:|---:|---:|']
    for split in ('regression', 'holdout'):
        for mode in ('f_active', 'g_active'):
            a = d['aggregate'][split]['modes'][mode]
            lines.append(f"| {split}/{mode} | {a['station_positive_cases']} | {a['station_positive_censored_cases']} | "
                         f"{a['station_observable_cases']} / {a['declared_target_entry_misses_before_actual_plane']} | {a['interception_positive_cases'] if mode=='g_active' else '未定义'} | "
                         f"{str(a['operational_interception_contact_false_positives'])+' / '+str(a['operational_interception_block_false_positives']) if mode=='g_active' else '未定义'} |")
    lines += ['', 'F 原标签仅表示中心站位可达，本报告不把它重新解释为历史拦截承诺，因此不能用 F 的站位正判断直接与 G 的拦截假阳性做同名准确率比较。G 的 operational false positive 是“至少一次接触前可行判断，随后完整实验仍无接触/未挡住”；属于本协议经验观察，不是统计置信保证。',
        '站位审核用真实球跨越真实移动机器人中心平面的时刻；若接触先发生，终点评分标为 censored，只保留接触前的到位证据。不会拼接无机器人未来球轨迹与接触后机器人状态构造反事实真值。', '',
        'G 对 air_high 明确给出“站位可达、身体不覆盖、拦截不可行”。独立组 angle_right 仍出现一次“可行且接触、但最终进球”的误判，说明覆盖不保证碰撞后球离开门口。G 的独立组正站位终点均因提前接触截断，不能把可观察样本数为零解释为站位估计准确率 100%。', '',
        '## 7. 采用决定与能力边界', '',
        '保留 F 默认算法，G 作为显式实验选项。已知物理的滚动/触地预测和站位/覆盖/拦截三层判定可独立沿用；0.08 m 残差修正存在更多短促运动的代价，不默认替换原定位器。',
        '预测空窗造成的部分弹跳球机会损失，可以通过预测和高层决策改善，但必须由配对实物理结果支持。wide 与 air_wide 的现有运动预算不足，单靠更准预测不扩大行走速度或手脚可及范围；air_high 超过现有身体高度，需要新的高球动作或相应能力，而非仅移动中心。',
        '现有 20 个有效碰撞体仍是局部覆盖，肩 pitch、hip pitch、ankle pitch 的视觉几何没有直接碰撞体。未来若要评估伸手、下蹲或扑救，需要先对所需动作的真实碰撞范围和控制策略另行建模与验证；本次未扩展这些能力，也未修改阶段 E 结论。', '',
        '## 8. 复现与交付', '',
        '```bash', '.venv/bin/python scripts/audit_goalkeeper_failures.py',
        '.venv/bin/python scripts/audit_ball_prediction.py',
        '.venv/bin/python scripts/validate_prediction.py',
        '.venv/bin/python scripts/compare_prediction_models.py',
        '.venv/bin/python scripts/analyze_prediction_results.py',
        '.venv/bin/python scripts/ablate_prediction.py',
        '.venv/bin/python scripts/report_prediction.py',
        '.venv/bin/python -m pytest -q', '```', '',
        '`validate_prediction.py --phase position` 只做无球门槛；完整入口先通过该门槛再发球，两个独立进程执行配对工况，单工况重复串行。当前代码/协议匹配的 G 缓存可复用，并重新核对原始轨迹哈希；F 历史基线只读。',
        '机器结果：`results.json`、`protocol.json`、`prediction_comparison.json`、`decision_audit.json`、两项前置 audit、每工况 `summary.json` / `trajectory.npz`。NPZ 保留每 1 ms 全状态、指令、姿态、力矩、球轨迹、接触与高层 JSON 决策字节；重复核验包含实际控制命令和决策记录。',
        '交付时 `final_verification.json` 独立核对 640 个 A–F 保护文件、94 份原始轨迹/决策、时序、发射配对、严格三值分类和源文件哈希，全部通过。',
        '正式执行源码已封存在 `execution_sources/`，与 results 的 code_sha256 对应。实验结束后验证入口仅加固了历史文件缺失时只读失败、包含入口/场景哈希的缓存指纹、截断工况不计增减和重复/数值异常退出检查；控制器与物理执行参数未改，全部已有配对重新按严格三值判定复核。',
        '测试输出见 `tests.txt`。QVEL 巨值警告仍来自既有异常输入测试，E0 已核验来源；本阶段物理实验的数值异常另逐工况记录，不能用测试警告掩盖实验异常。', '']
    text = '\n'.join(lines)
    (DATA/'report.md').write_text(text)
    (ROOT/'docs/prediction_validation.md').write_text(text)


if __name__ == '__main__':
    render()
