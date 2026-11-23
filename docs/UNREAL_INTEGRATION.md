# Unreal Engine 5 对接民雄 PINN

仓库提供一个只依赖 Python 标准库的本机 HTTP 桥接器。Unreal 负责 UMG 输入画面与结果呈现；Python 服务负责读取当前**实际获准**的模型范围、检查输入、执行推理，并直接生成 CSV、GeoTIFF 和 NPZ。接口不会因为请求写了 300 mm/hr 就绕过模型范围检查。

> 当前部署状态：`0–300 mm/hr` 理想化两专家包已通过审核并部署。支持降雨 1–24 小时、模拟最长 36 小时；30–50 mm/hr 自动平滑混合 low/medium。manifest、审核证据或模型哈希校验失败时，服务才会安全回退 v4。

## 启动服务

在仓库根目录使用已安装依赖的 `.venv`：

```bash
.venv/bin/python run_minxiong_unreal_api.py
```

默认地址为 `http://127.0.0.1:8765`，输出写到 `outputs/minxiong/unreal_predictions/`。可用参数包括：

```bash
.venv/bin/python run_minxiong_unreal_api.py \
  --host 127.0.0.1 --port 8765 \
  --output-root outputs/minxiong/unreal_predictions
```

默认只允许同一台电脑访问。若 Unreal 与 Python 不在同一台电脑，可显式改成 `--host 0.0.0.0`，但当前接口没有身份验证，只能放在受信任的隔离网络，并需自行配置防火墙。远端 Unreal 也无法直接读取服务器返回的本机绝对路径，需要共享目录或另加文件下载服务。

## 端点

### `GET /health`

返回服务状态、固定 DEM 有效格数、是否启用专家包、回退原因以及当前模型的真实输入范围。Unreal 开启输入按钮前应先检查这个端点。

### `GET /schema`

返回 `/predict` 的字段、单位、最大帧数及当前模型范围。UMG 不应把 300 写成永远有效的常数；应始终以 schema 当次返回范围为准。

### `POST /predict`

请求头必须是 `Content-Type: application/json`。示例：

```json
{
  "rainfall_mm_hr": 123.456,
  "duration_hr": 7.25,
  "simulation_hours": 12.5,
  "interval_minutes": 30,
  "output_name": "unreal_case_01"
}
```

四个数值字段必须有限且为正（雨势允许 0），`output_name` 可省略；若填写，只能使用 1–64 个 ASCII 字母、数字、点、横线或底线，且不能重复已有输出目录。输出时刻会自动包含 `t=0`、固定间隔、雨停时刻（若在模拟期内）和模拟终点。

成功响应的主要字段：

```json
{
  "status": "complete",
  "request": {
    "rainfall_mm_hr": 123.456,
    "duration_hr": 7.25,
    "simulation_hours": 12.5,
    "interval_minutes": 30,
    "frame_count": 27
  },
  "active_model": {
    "expert_bundle": true,
    "bundle_id": "minxiong-experts_strong_0_300_v12_idealized",
    "model_id": "medium",
    "routing_label": "medium",
    "routing_policy": "linear_blend",
    "expert_weights": {"low": 0.0, "medium": 1.0}
  },
  "output_directory": "/absolute/path/to/unreal_case_01",
  "files": {
    "csv": ["/absolute/path/to/frame_000.csv"],
    "geotiff": ["/absolute/path/to/depth_000.tif"],
    "npz": ["/absolute/path/to/frame_000.npz"]
  }
}
```

调用是同步完成后才响应，但 Unreal 必须使用异步 HTTP 任务，不能阻塞游戏线程。服务同一时间只执行一个预测；并发请求返回 HTTP 409。常见错误还包括 400 输入/范围错误、404 未知端点、413 请求过大、415 Content-Type 错误和 500 推理失败。错误响应固定包含 `status=error`、`error` 与 `message`。

## UMG / Blueprint 接线

1. 在 Widget 中建立四个数值输入框与“开始模拟”按钮。
2. Widget 初始化时异步调用 `GET /health` 和 `GET /schema`，显示当前模型范围；服务不是 ready 或请求超域时禁用开始按钮。
3. 点击按钮后把输入转换为上面的 JSON，异步发送 `POST /predict`。
4. 显示进度中的忙碌画面；收到 200 后解析 `files.csv` 或 `files.geotiff`，收到非 200 时显示响应中的 `message`。
5. 逐帧读取 `prediction_summary.csv` 的时间与对应 `frame_XXX.csv`，按模拟时间更新水面网格或材质。

UE5 内建 `HTTP`/`Json` 模块可在 C++ 调用；纯 Blueprint 项目需启用可信的 HTTP Blueprint 插件，或由 Unreal 成员写一个很薄的 Blueprint Function Library 包装 `FHttpModule`。此仓库提供的是已测试的 HTTP/文件契约，不包含 Unreal 材质、Niagara、水面网格或打包后的插件，尚不能称为完成真实 UE 场景联调。

## 直接生成的 CSV

每个时刻会生成 `frame_NNN.csv`，固定列为：

| 列 | 单位/含义 |
|---|---|
| `time_s` | 模拟秒 |
| `row`, `col` | DEM 栅格行列 |
| `x_m`, `y_m` | EPSG:3826 平面坐标，米 |
| `depth_m` | 水深，米 |
| `u_m_s`, `v_m_s` | x/y 速度，米/秒 |
| `speed_m_s` | 合速度，米/秒 |

同时生成 `prediction_summary.csv`，每帧一行，记录有效模型标签、主专家、路由策略/权重、雨情、时间、CSV/NPZ/TIF 文件名、最大/平均水深、最大流速与质量投影方式。混合时 `model_id` 会明确写成 `low+medium`，并保留 `primary_model_id` 与 `expert_weights`。JSON 仍保留作兼容，但用户不需要再从 JSON 转换 CSV。

源数据保持 GIS 米制坐标；Unreal 通常使用厘米。场景端必须先确定局部原点、东西/南北轴映射与高程基准，再进行坐标转换，不能直接把 EPSG:3826 数字当作 Unreal 世界坐标。

## `0–300 mm/hr` 已批准模型与审核

两专家为 low `0–50` 及 medium `30–300`；重叠 `30–50` 使用线性凸混合。原训练与发布审核命令为：

```bash
.venv/bin/python train_minxiong_experts.py configs/minxiong_experts_0_300.json
.venv/bin/python audit_minxiong_experts.py configs/minxiong_experts_0_300_idealized_release.json
```

low、medium 的个别物理门槛、连续输入域与完整 DEM 水量均已通过。原始重叠空间差异约 20.5%，依用户确认的理想化目标采用 25% 发布上限，并用平滑混合消除切换跳变。完整证据见 `models/minxiong/experts/overlap_audit.json` 与 `docs/MINXIONG_0_300_TRAINING_STATUS.md`。

干净 clone 不需要重跑训练审核；执行 `.venv/bin/python verify_minxiong_release.py` 可只读校验 DEM 身份、审核哈希、两个模型哈希和当前路由。
