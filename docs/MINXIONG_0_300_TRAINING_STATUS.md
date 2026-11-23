# 民雄 0–300 mm/hr PINN 训练状态

最近更新：2026-11-23 11:56:47（Asia/Taipei）

## 当前结论

- 固定地形：`data/minxiong/裁剪后民雄.tif`。有效格由 GeoTIFF mask/NoData 判定，零值与负高程均保留。
- 目标输入域：降雨强度连续 `0–300 mm/hr`、降雨时长 `1–24 h`、模拟时长最多 `36 h`。
- 专家路由：low 训练 `0–50`、负责 `0–40 mm/hr`；medium 训练 `30–300`、负责 `>40–300 mm/hr`；`30–50` 是独立一致性审核重叠区。
- low 已完成 24,000 步并通过个别模型物理门槛。
- medium 首次 24,000 步评估因 test 平均 continuity `0.0208804` 略高于 `0.02` 而被拒绝；没有放宽门槛。沿同一优化器轨迹安全延长至 30,000 步后，medium 已通过全部个别模型物理门槛。
- low 与 medium 个别模型现在都已通过；连续域与重叠区独立审核仍未完成，因此 `0–300` 仍是候选能力，GUI 与 Unreal API 继续安全回退已部署的 v4 模型。

## low 已完成证据

- 请求/完成步数：`24,000 / 24,000`
- 最佳验证损失：`0.0131253160`
- test 平均 PDE：`0.0095652330`
- test 平均 continuity：`0.0113274125`
- test 平均 mass loss：`0.0001367068`
- test 边界法向速度 RMS：`2.8430e-9 m/s`
- `physics_training_passed = true`

正式产物位于 `outputs/minxiong/experts_strong_0_300_v11/low/`。

## medium 已完成证据

- 请求/完成步数：`30,000 / 30,000`
- 最佳验证损失：`0.0143582078`
- test 平均 loss：`0.0149377012`
- test 平均 PDE：`0.0127748098`
- test 平均 continuity：`0.0164304007`
- test 平均 momentum x/y：`0.0104576309 / 0.0114363978`
- test 平均 mass loss：`0.0002162891`
- test 边界法向速度 RMS：`3.7653e-9 m/s`
- 最坏全事件 PDE / continuity / mass：`0.0222859083 / 0.0487959259 / 0.0004574188`
- `physics_training_passed = true`
- 通过快照：`outputs/minxiong/experts_strong_0_300_v11/medium/training_state_snapshot_step_030000_passed.pt`
- 快照 SHA-256：`d1bdeedb8ebbb5dec1a9d0591b0ac876703bec24df22f5c735cc0b9f36b8ac55`
- 模型 SHA-256：`540e0641200c4abf41f25767d3c3cbb988e40f22212d318eb6186ae7eb0543e5`
- 训练报告 SHA-256：`4e71bee67786313ac74cc6084863808f4c9104527a44987a83aa89149a68556a`

`outputs/` 被 Git 忽略，模型、报告和快照保存在当前云环境而不是 GitHub。正式发布前仍须通过独立审核，审核通过后才会把最小批准模型包部署并提交。

## 恢复与验收

若训练进程中断，从仓库根目录重新执行：

```bash
.venv/bin/python train_minxiong_experts.py configs/minxiong_experts_0_300.json
```

程序会校验地形、事件、配置与锚点身份后恢复；身份不一致会拒绝混用断点。训练结束后必须执行：

```bash
.venv/bin/python audit_minxiong_experts.py configs/minxiong_experts_0_300.json
```

只有 individual physics、连续输入域、完整 DEM 水量与 low/medium 重叠区全部通过，才允许生成并部署 manifest。模型、审核文件与 manifest 已通过 SHA-256 互相绑定。

## 已同步的代码状态

- `031a4a3`：严格验证并安全复用已完成专家训练。
- `3529993`：把发布 manifest、独立审核证据与模型 checkpoint 做哈希绑定。
- `f85dd3b`：只对 medium 安全延长至 30,000 步，low 保持 24,000 步。
- 全套自动测试：`215 / 215` 通过。

