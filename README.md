# ALFWorld / Qwen2.5-3B：训练期外置 API 路由

[CPU 检查记录](https://github.com/leiye302/rethink-alfworld-qwen25-3b-api-router/actions)

这是 RETHINK 的 **external-api-router** 消融仓库。**不训练 Actor 内化的 RETHINK 触发能力**，直接从固定 Qwen2.5-3B-Instruct 基模做150次 GRPO+OPSD；在训练时由外部 API 判断 Actor 当前 reasoning 是否有明确错误，Bad 步骤才额外给 Teacher 局部 hint。**评测完全不调用 API。**

```text
Actor rollout (unaltered reasoning/actions)
  → judge each visible H/R independently
      Good / uncertain → static-skill Teacher
      Bad              → static skill + local-hint Teacher
  → same current-Actor Teacher scoring + full-trajectory GRPO
```

## 路由与提示的边界

- 路由 API **只看 Actor 实际分词/截断后看到的 H 和当前 R**。不看当前 action、执行后反馈、后续轨迹或最终结果。每个 H/R 独立请求，不能把整条轨迹合并交给路由裁判，否则会泄露未来信息。
- Bad：有当前可见证据支持、尚未自行修正、会误导下一步的实质 reasoning 错误。探索、暂未进展、最终失败或 action 不好本身不能自动算 Bad；证据不足返回 uncertain。
- 局部 hint 的生成沿用完整方法的**整轨迹合并**、模型、状态重建、命令校验和 Teacher-only 接法。仅把提示中“Actor 已触发”的一句改为“外置裁判选中”，避免描述不存在的 Actor marker。
- 原 Actor token、动作、reward、GRPO mask 不改；不强制插入 marker，不替换动作，不单独训练路由器。Student 从始至终不看 hint。
- Teacher 是当前 Actor 的无梯度视角，对已采样 token 评分，不是固定 Teacher。reference KL 仍固定在本组 RL 起点。

## 固定设置

Qwen/Qwen2.5-3B-Instruct 的固定 revision，公开 SDAR 固定 commit；150 updates，每步16任务×8 rollout，SDAR门控 beta=5、sdar_coef=0.01、reference KL=0.01、entropy coefficient=0.001。其余 GRPO+OPSD 数学实现不改，见 `configs/opsd.yaml` 和配方审核。

路由模型暂定 `glm-5.3-flash`，temperature=0，max_tokens=384，并行4、精确 H/R 缓存。**代码接线测试不是 API 判断质量验证**；正式训练前仍应在真实样本上小测质量及费用。路由覆盖所有有效 reasoning，调用量可能明显高于仅为 Actor 触发位置生成提示的完整方法。

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

## 这组消融能说明什么

它比较“Actor 内化触发＋纠错初始化”与“训练期外置判断替代”的**整体方案**。因为本组不做 Route-SFT，RL初始化及 reference anchor 也变了，不能把全部成绩差异单独归因于路由准确度或局部 hint。若要纯路由器对照，需另外设计相同初始化实验；本仓库不会暗中添加第三组。

本仓库只准备源码/配置和CPU测试，**未执行新GPU训练、未发起付费API、没有预填成功率**。来源和逐项差异见 [审核记录](docs/experiment_log.md)。不依赖作者电脑或私有服务器，不发布凭据、私人日志或训练状态。许可见 [LICENSE](LICENSE)；上游资源保留各自许可。
