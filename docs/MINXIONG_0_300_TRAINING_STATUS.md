# 民雄 0–300 mm/hr 理论 PINN 发布状态

最近更新：2026-11-23（Asia/Taipei）

## 当前结论

`0–300 mm/hr` 两专家包已通过本项目的**理想化理论模型**验收并部署到
`models/minxiong/experts/`。GUI 与 Unreal API 会优先载入这个包，不再回退
到旧的 `0.5–23 mm/hr` 单模型。

- 固定地形：`data/minxiong/裁剪后民雄.tif`。
- 有效网格：212,239 个 20 m 网格；NoData 由 GeoTIFF mask 判定，零值与负高程均保留。
- 连续用户输入：雨势 `0–300 mm/hr`、降雨时长 `1–24 h`、模拟时长最多 `36 h`。
- low 训练范围 `0–50`；medium 训练范围 `30–300 mm/hr`。
- `30–50 mm/hr` 使用线性凸混合：30 为 100% low、40 为 50/50、50 为 100% medium，避免在 40 附近画面突跳。
- 输出会直接产生 `frame_*.csv` 与 `prediction_summary.csv`，不需要把 JSON 再转换为 CSV。

## 验收结果

个别模型物理门槛没有放宽：

| 项目 | low | medium |
|---|---:|---:|
| 完成步数 | 24,000 | 30,000 |
| 最佳验证损失 | 0.0131253 | 0.0143582 |
| test 平均 PDE | 0.00956523 | 0.0127748 |
| test 平均 continuity | 0.0113274 | 0.0164304 |
| test 平均 mass loss | 0.000136707 | 0.000216289 |
| 个别物理验收 | 通过 | 通过 |

正式连续域与完整 DEM 审核：

- 连续条件审核：low、medium 都通过。
- 完整 DEM 最大相对水量误差：low `0.8018%`、medium `1.6038%`，门槛 `5%`。
- 重叠区最大平均蓄水差：`1.5402%`，门槛 `2%`。
- 原始专家空间水深相对 L2：`20.4941%`。
- 依照“理想化、无需现实校准但不可过分失真”的目标，发布配置把**仅限重叠一致性**的上限设为 `25%`；PDE、continuity、边界与完整 DEM 水量门槛保持不变。
- 两组未参与正式审核的空间种子复查为 `20.1120%` 与 `19.9119%`，均低于 `25%`。
- 正式结果：`physics_training_passed=true`、`continuous_domain_passed=true`、`overlap_consistency_passed=true`、`activation_approved=true`。

这表示模型能输出趋势连续、非负且总体水量合理的理论淹水场；**不表示已达到真实民雄淹水精度**。模型尚未校准排水、入渗、建筑、抽水站、河川边界或实测积水。

重叠区的速度相对 L2 最坏约 `1.553`，没有作为发布门槛；两模型速度本身很小，实跑最大约 `0.026 m/s`，导致相对指标对方向/微小量很敏感。因此本版优先保证水深、连续画面与总体水量，速度场只适合定性视觉效果，不应解释为真实流速或流向。

## 完整输出实跑

发布后另以完整 212,239 格实跑两组案例：

| 案例 | 路由 | 6 h 平均水深 | 36 h 平均水深 | CSV 验证 |
|---|---|---:|---:|---|
| 40 mm/hr、降雨 6 h | low+medium 各 50% | 0.24043 m | 0.24097 m | 通过 |
| 100 mm/hr、降雨 6 h | medium 100% | 0.59942 m | 0.59860 m | 通过 |

每帧 CSV、NPZ 与 GeoTIFF 都恰有 212,239 个有效格；数值全部有限、水深全部非负。30、40、50 mm/hr 三个混合边界左右 `0.001 mm/hr` 的相邻输出相对差约 `2.0e-5–3.8e-5`，没有硬切换跳变。

额外角落检查：`0 mm/hr` 在 36 h 仍为全域零水深/零速度；`300 mm/hr × 24 h` 在 36 h 的全域平均水深为 `7.1375 m`，相对封闭域理论累积 `7.2 m` 的水量误差为 `0.8686%`，最大水深 `10.4939 m`。这个数值很大是因为压力情境总降雨本身达到 7.2 m，且模型没有排水或外流；不能解释成现实民雄会积水 10 m。

## 发布产物

- 批准清单：`models/minxiong/experts/manifest.json`
- 完整审核证据：`models/minxiong/experts/overlap_audit.json`
- low：`models/minxiong/experts/low/model.pt`
- medium：`models/minxiong/experts/medium/model.pt`
- 发布配置：`configs/minxiong_experts_0_300_idealized_release.json`

SHA-256：

- low：`ed498fd98577c0af9a270ddc995fc7ba5ffea2356fc69f4bcc5e05bdc6360344`
- medium：`540e0641200c4abf41f25767d3c3cbb988e40f22212d318eb6186ae7eb0543e5`
- 审核证据：`890041829fae1093dae40d845b2272d9c3733f56730a062edb11ae12e8a5a23d`

manifest、审核证据和两个模型由 SHA-256 互相绑定；缺失、修改或使用错误 DEM 时会安全回退旧模型。

## 运行与复核

一般用户在 VS Code 直接运行 `run_minxiong_gui.py`，在窗口输入四个参数即可。Unreal 接口运行：

```bash
.venv/bin/python run_minxiong_unreal_api.py
```

`GET /health` 应显示 `expert_bundle_active=true`、范围 `0–300 mm/hr`、降雨 `1–24 h`、模拟最长 `36 h`。

干净 clone 可用只读命令验证 DEM、模型哈希、审核证据与路由：

```bash
.venv/bin/python verify_minxiong_release.py
```

原训练配置仍为：

```bash
.venv/bin/python train_minxiong_experts.py configs/minxiong_experts_0_300.json
```

理想化发布配置标有 `release_only=true`，训练入口会拒绝误用。它记录本次云环境发布来源，而 `outputs/` 不随 Git 分发，所以干净 clone 应执行上面的只读验证命令，不应重跑该历史发布审核。已发布目录不可原地重复审核；如需新训练/审核，必须准备新的输出版本目录。

自动测试：`224 / 224` 通过。
