# ALFWorld / Qwen2.5-3B：训练期外置 API 路由

[CPU 检查记录](https://github.com/leiye302/rethink-alfworld-qwen25-3b-api-router/actions)

这是 RETHINK 的 **external-api-router** 消融仓库。**不训练 Actor 内化的 RETHINK 触发能力**，直接从固定 Qwen2.5-3B-Instruct 基模做150次 GRPO+OPSD；训练时由外部 API 回看完整 rollout，逐步判断 reasoning 是否有明确错误，Bad 步骤才额外给 Teacher 局部 hint。**评测完全不调用 API。**

```text
Actor rollout (unaltered reasoning/actions)
  → one complete episode → one joint API request
      all step labels + short hints only for Bad
      rejected hints → optional one grouped hint-only repair
      Good / uncertain → static-skill Teacher
      Bad              → static skill + local-hint Teacher
  → same current-Actor Teacher scoring + full-trajectory GRPO
```

## 路由与提示的边界

- 外置审核者是**训练期完整轨迹 hindsight critic**：可读任务、逐步 reasoning/action 与后续环境反馈。这不是部署时仅看当前 H/R 的 Actor 路由器，也不能把两者准确率混为一谈。
- Bad：本段 reasoning 中尚未自行修正、会误导下一步的实质错误。只错 action、探索暂未进展、最终失败本身都不自动算 Bad；合理猜测不能仅因后续没猜中就被判 Bad。缺少有效 reasoning 或证据不足返回 uncertain。
- **同一条轨迹通常一次请求**，同时返回所有 step 标签和 Bad 的短 hint；原 R 引文、step 覆盖及动作 ID 严格校验。被拒绝的 hint 在该轨迹内集中补一次，**补 hint 不能更改标签**；仍不合格则回退普通 Teacher，不逐 decision 重问。
- 本地状态记录只累计该步之前的真实反馈。hint 必须落在该步**行动前**，从当前允许动作中精确选择；另检查部分清洗/加热/冷却前提及目标容器约束。这些工程检查不等于已证明 hint 能改善成功率。
- 原 Actor token、动作、reward、GRPO mask 不改；不强制插入 marker，不替换动作，不单独训练路由器。Student 从始至终不看 hint。
- Teacher 是当前 Actor 的无梯度视角，对已采样 token 评分，不是固定 Teacher。reference KL 仍固定在本组 RL 起点。

## 固定设置

Qwen/Qwen2.5-3B-Instruct 的固定 revision，公开 SDAR 固定 commit；150 updates，每步16任务×8 rollout，SDAR门控 beta=5、sdar_coef=0.01、reference KL=0.01、entropy coefficient=0.001。其余 GRPO+OPSD 数学实现不改，见 `configs/opsd.yaml` 和配方审核。

路由与 hint 共用 `glm-5.3-flash`，temperature=0、reasoning_effort=low、最多4条轨迹并行；输出预算为 `min(14000, 1024 + 224 × 轨迹步数)`。精确缓存绑定整轨迹、模型、端点、提示词及实现版本，变更后的 rollout 不会套旧标签。完整逻辑和提示词在 [router_protocol.py](src/opsd/router_protocol.py)，缓存/补请求在 [router_worker.py](src/opsd/router_worker.py)。

整轨迹合并显著减少请求数，但不保证等比例减少 tokens、费用或时间。做过小规模真实 API 接线试测；**没有据此宣称全面语义校准通过或 RL 性能提升**。

## 准备与运行

Linux/CUDA，当前入口配置为单张空闲 **80/96GB GPU**；CUDA/vLLM 实机显存与吞吐尚未验证。建议至少64GiB主机内存、120GiB空闲磁盘。安装、模型/ALFWorld资源、自己的API配置见 [复现说明](docs/idea.md)。不需要先跑 SFT；`train.sh sft` 会明确拒绝。

```bash
# 完成 docs/idea.md 中的准备后：
bash scripts/reproduce.sh --check
bash scripts/reproduce.sh --run --allow-api-cost
# 或仅二阶段：
bash scripts/train.sh opsd --allow-api-cost
```

二阶段每10次更新在固定 **ALFWorld seen140** 验证、每20次保存、保留最近两份完整断点和最终update150。在线验证与最终评测都不调用路由 API、hint API、Analyzer 或 Teacher；任务/种子、50步上限、history2、temperature0.4、512-token预算与完整方法一致。

API 网络/额度失败保留当前 batch 并等待重试，**不会伪装为 Good 继续更新**；合法 uncertain 或不合格 hint 使用普通静态-skill Teacher。没有自动付费额度上限；`--allow-api-cost` 不是预算控制。缓存包含任务内容，请按训练数据管理，不要直接发布 runs/。

OpenAI-compatible HTTPS 服务配置见 `.env.example`。OpenCode Go 可选配置自己的 `RETHINK_API_ORG_ID`；会话标识在私有 auth 文件中持久化，重启复用。程序使用如实标识的 User-Agent，不伪装其他客户端，不自动切换付费提供商。该协议与旧逐步审核版不同，请新建运行目录，不能静默续用旧协议的训练状态。

## 这组消融能说明什么

它比较“Actor 内化触发＋纠错初始化”与“训练期外置判断替代”的**整体方案**。因为本组不做 Route-SFT，RL初始化及 reference anchor 也变了，不能把全部成绩差异单独归因于路由准确度或局部 hint。若要纯路由器对照，需另外设计相同初始化实验；本仓库不会暗中添加第三组。

本仓库提供源码/配置、CPU测试和经过小规模真实 API 检查的批处理协议，**未执行本消融的新GPU训练，没有预填成功率**。来源和逐项差异见 [审核记录](docs/experiment_log.md)。不依赖作者电脑或私有服务器，不发布凭据、私人试测数据、日志或训练状态。许可见 [LICENSE](LICENSE)；上游资源保留各自许可。
