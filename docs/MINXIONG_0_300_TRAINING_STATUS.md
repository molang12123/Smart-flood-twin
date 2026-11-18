# 民雄 0–300 mm/hr PINN 训练状态

记录时间：2026-11-18 23:42:26（Asia/Taipei）

## 当前结论

- 固定地形：`data/minxiong/裁剪后民雄.tif`。有效格由 GeoTIFF mask/NoData 判定，零值与负高程均保留。
- 目标输入域：降雨强度连续 `0–300 mm/hr`、降雨时长 `1–24 h`、模拟时长最多 `36 h`。
- 专家路由：low 训练 `0–50`、负责 `0–40 mm/hr`；medium 训练 `30–300`、负责 `>40–300 mm/hr`；`30–50` 是独立一致性审核重叠区。
- low 已完成 24,000 步并通过个别模型物理门槛。
- medium 仍在训练。本记录对应可恢复快照第 7,150 步；训练进程在记录后继续运行。
- 在 medium、连续域审核与重叠区审核全部通过前，`0–300` 仍是候选能力，GUI 与 Unreal API 继续安全回退已部署的 v4 模型。

## low 已完成证据

- 请求/完成步数：`24,000 / 24,000`
- 最佳验证损失：`0.0131253160`
- test 平均 PDE：`0.0095652330`
- test 平均 continuity：`0.0113274125`
- test 平均 mass loss：`0.0001367068`
- test 边界法向速度 RMS：`2.8430e-9 m/s`
- `physics_training_passed = true`

正式产物位于 `outputs/minxiong/experts_strong_0_300_v11/low/`。

## medium 快照

- 训练状态步数：`7,150 / 24,000`
- 当时最佳验证损失：`0.0325999683`（第 7,000 步）
- 第 7,000 步 PDE：`0.0391284703`
- 第 7,000 步 continuity：`0.0605051487`
- 第 7,000 步 momentum x/y：`0.0293305105 / 0.0275497517`
- 第 7,000 步 mass loss：`0.0002453554`
- 第 7,000 步 boundary loss：`6.39285e-21`
- 累计 medium 训练时间：`8,541.18 s`
- 快照：`outputs/minxiong/experts_strong_0_300_v11/medium/training_state_snapshot_step_007150.pt`
- 快照 SHA-256：`204b038e391a4e0fc0384c8fd1037b4679f0ba7b2a4062803229095c5a2b40ee`

`outputs/` 被 Git 忽略，快照保存在当前云环境而不是 GitHub。正式 `training_state.pt` 每 50 步原子写入，正常情况下比上述人工快照更新。

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
- 全套自动测试：`214 / 214` 通过。

