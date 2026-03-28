# 代码审查报告（2026-03-28）

## 范围
- `tools/train.py`
- `models/upr_mvs_transformer.py`
- `models/upr_mvs.py`
- `models/point/feature_lifting.py`

## 主要问题

### 1) 阻塞级：点云子模块引用与仓库结构不一致
`models/upr_mvs_transformer.py` 依赖以下模块：
- `upr_mvs.models.point.densify`
- `upr_mvs.models.point.knn`
- `upr_mvs.models.point.point_refiner`
- `upr_mvs.models.point.unproject`

但当前仓库 `models/point/` 下仅存在 `feature_lifting.py`，缺失其余模块，且仓库根目录没有 `upr_mvs/` 包结构（仅 `models/__init__.py`）。这会导致训练/推理在导入阶段直接失败。

**建议**
- 统一包结构：要么补齐 `upr_mvs/...` 目录及 `__init__.py`，要么把导入改为与当前目录结构一致的相对/绝对导入。
- 补齐缺失的 point 子模块实现，或在配置层禁用对应路径并提供 fallback。

### 2) 高风险：`build_model` 的后备分支会命中已废弃入口
`tools/train.py` 中：
- `backbone == "dinov3"` 时构建 `UPRMVSTransformerModel`
- 否则回退到 `UPRMVSModel`

而 `UPRMVSModel` 继承的旧路径中包含 `CoarseDepthStageModel` 兼容壳，且粗阶段入口已显式废弃。当前后备逻辑可能让配置错误在更深层才暴露。

**建议**
- 在 `build_model` 中对 `backbone` 做白名单校验并尽早报错（fail fast）。
- 在配置解析阶段验证 `model.backbone` 与 `train.stage` 的合法组合。

### 3) 中风险：`resolve_resume_mode` 的路径判定过于严格
`tools/train.py` 中 `resolve_resume_mode` 仅比较 `resume_path.parent == work_dir`，在以下情形可能误判：
- checkpoint 位于 `work_dir/checkpoints/` 等子目录
- 路径存在符号链接/软链接

误判会导致应该 `full` 恢复时走 `model_only`，训练状态（优化器、调度器、scaler）丢失。

**建议**
- 使用“是否在 work_dir 树内”的判定（例如 `Path.is_relative_to`）。
- 或增加显式配置优先级：`resume_mode` 非 `auto` 时强制覆盖。

## 优先修复顺序
1. 修复导入与包结构不一致问题（阻塞运行）。
2. 增加 `build_model` 的 fail-fast 验证，减少错误定位成本。
3. 放宽恢复模式路径判定逻辑，避免隐性训练状态丢失。
