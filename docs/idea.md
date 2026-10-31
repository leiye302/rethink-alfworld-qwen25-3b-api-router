# ALFWorld 两个消融的独立复现说明

本文件同时解释两种配置，当前 clone **只运行 `configs/ablation.json` 指定的一组**。不会顺带启动基线、完整方法或另一个消融。benchmark=ALFWorld，base=Qwen2.5-3B-Instruct；没有Qwen3的enable_thinking开关。

## 1. 科学合同

| 项目 | 完整方法参考 | trigger_only | external_api_router |
|---|---|---|---|
| RL 起点 | 完整 Route-SFT step2000 | 触发专用 SFT step2000 | 固定3B基模，无SFT |
| Good SFT目标 | 自然原动作＋EOS | 不变 | 不做SFT |
| Bad SFT目标 | marker＋纠正动作＋EOS | 只保留marker；无纠正动作/EOS | 不做SFT |
| 局部hint选位 | Actor生成marker | Actor生成marker | 外置API判断H/R为Bad |
| 局部hint接收者 | 仅Teacher | 不变 | 不变 |
| 部署/评测 | Actor独立行动 | 不变 | 不变，无API |

完整方法的 suffix CE 是每样本先除以后缀长度，再对batch取平均。trigger_only 的 Bad 删除纠正/EOS项，但保留原长度分母，保证 marker 自身的 CE 系数不变；独立 BCE 权重0.1也不变。这里的长度不含被mask的 H/R。Good动作与 Bad纠正目标在冻结源库中原样保留，不重新标注、不增加样本。

marker 二分类用完整词表概率而不是裸logit的sigmoid：

```math
q_\theta=\frac{\exp z_{\tau}}{\sum_v\exp z_v},\qquad
r_\theta=z_\tau-\operatorname{logsumexp}_{v\ne\tau}z_v,\qquad
L_{\rm route}=\operatorname{BCEWithLogits}(r_\theta,y).
```

位置是 `</think>` 及约定换行后，**预测后缀第一个token的最后一个前缀logit**。不在吃入marker之后取logit；不监督 H/R token，也不额外教 Bad marker 后结束。因果mask保证删除后续目标不改变同一次前向中marker位置的条件上下文；长期参数更新仍会不同，这正是本消融。

Stage2保留 SDAR 原生 gate_beta=5、GRPO+OPSD loss/aggregation/clip、当前Actor Teacher评分、16×8 rollout、150updates。GRPO覆盖整轨迹，不是仅Good做GRPO。静态skill来自固定SDAR的skills/alfworld；没有episode Analyzer。

## 2. 从干净目录准备

只需本公开源码、公开冻结数据包、可下载基模和 ALFWorld 官方游戏资源。权重不包含在Git中；无需任何作者的服务器/硬盘/登录凭据。使用自己的Linux环境与账号。

```bash
python3.10 -m venv .venv-sft
.venv-sft/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-sft/bin/pip install -r configs/requirements-sft.txt
.venv-sft/bin/python src/tools/setup.py --framework --bundled-data
.venv-sft/bin/python src/tools/setup.py --download-base --accept-model-license

python3.12 -m venv .venv-opsd
.venv-opsd/bin/pip install --upgrade pip setuptools wheel packaging ninja
.venv-opsd/bin/pip install -r configs/requirements-opsd.txt
.venv-opsd/bin/pip install flash-attn==2.7.4.post1 --no-build-isolation
.venv-opsd/bin/pip install -e .deps/sdar -c configs/requirements-opsd.txt
.venv-opsd/bin/pip check
cp .env.example .env
```

两环境隔离是为了保留原SFT与SDAR/vLLM依赖版本，不要随意升级后宣称完全同配方。外置API组不使用SFT环境训练；它仍可用于CPU检查/下载工具。

在自己的 `.env` 中配置 Python、获配GPU、ALFWORLD_DATA和API；不要提交该文件。`RETHINK_API_ENDPOINT` 是完整HTTPS chat/completions地址，不应含用户名、密码、查询token。密钥放环境或本地私有配置，不传在命令参数里；程序运行时写入权限0600的私有auth文件，Hydra/Ray配置不保存key。

```bash
set -a
source .env
set +a
.venv-opsd/bin/alfworld-download
```

ALFWORLD_DATA下须包含 `json_2.1.1` 和 `logic`，训练环境应枚举到3553个训练游戏。程序核验seen140每个game文件SHA；版本不符会报错，不会悄悄换任务。

API的局部hint模型/格式固定在原方法patch中；当前为glm-5.3-flash JSON协议。外置路由模型见opsd.yaml。端点需支持相应模型、JSON与reasoning_effort=low。使用自己的服务额度；准备/CPU检查不付费。训练请求含任务/轨迹内容，运行者应确认可以发送到该提供商。没有自动费用上限。

## 3. 先检查再占卡

```bash
bash scripts/reproduce.sh --check
.venv-sft/bin/python src/tools/audit_recipe.py --framework
.venv-sft/bin/python -m unittest discover -s src/tools -p 'test_*.py' -v
PYTHONPATH=src/sft .venv-sft/bin/python -m pytest src/sft/tests -q
.venv-sft/bin/python src/tools/check.py --assets
.venv-opsd/bin/python src/tools/check.py --runtime opsd
```

trigger_only 另外检查SFT运行依赖，并先做真实GPU预检；一次性模型完成forward/backward/Adam状态分配后销毁，不作为训练起点。

```bash
.venv-sft/bin/python src/tools/check.py --runtime sft
bash scripts/train.sh sft --gpu-preflight-only
```

当前入口只允许1张显式获配的80/96GB卡；不是对整台机器的占卡指令。SFT要求至少70GiB总显存、65GiB空闲；总显存、CUDA/vLLM峰值与吞吐仍须在实际租用卡上确认。单卡80GB并非所有长度/系统组合都已实测。不要为绕过OOM暗改microbatch、loss或上下文长度；保存证据再做双方一致的资源适配。

## 4. 启动与恢复

```bash
bash scripts/reproduce.sh --run --allow-api-cost
```

流程根据配置自动选择：trigger_only先预检→SFT2000→OPSD150；external_api_router直接OPSD150。成功后导出update150并用同一seen140最终评测，写入 `results/tables/<variant>_seen140.json`。程序不预填成绩，不能将CPU测试视为训练完成。

SFT全局batch8、5Good/3Bad、micro1、LR2e-6、AdamW(.9,.95)、weight_decay.01、grad_clip1、seed0、max_length8192、truncation=error。训练1epoch/2000步，但scheduler_total_steps=4000，warmup400。每100步dev256；每500步自由生成32/保存权重；step2000完整保存optimizer/scheduler/单rank RNG/采样位置。后续一律用step2000，不按验证成绩挑另一个checkpoint替代。

OPSD每10次更新验证、每20次保存，最近两份完整断点和最终update150保留；可选durable-every-update不是默认，不要只对一组打开后比较耗时。SFT默认不自动删除旧权重；空间不足先报错保留断点，不能删未备份文件。

在tmux/调度器持久会话中运行。确认旧进程退出、代码配置相同、断点完整后：

```bash
bash scripts/reproduce.sh --run --allow-api-cost --resume
```

锁和阶段状态保存在runs/ablation；不会盲目重启或覆盖半写目录。SFT第一个500步之前若无完整断点不能假装续训。API transport/额度错误与不合格提示区分：75/76保留同一rollout等待重试，完整输出缓存可复用；合法uncertain或校验不通过的hint只回退普通静态skill Teacher。损坏缓存/身份不符等完整性错误停止并保留证据。

## 5. 无API的同任务评测

```bash
EVAL_GPU=0 bash scripts/eval.sh --model checkpoints/opsd_hf150 \
  --out runs/independent_seen140 --step 150
```

代码明确关闭 `algorithm.local_hints.enabled` 和 `algorithm.external_router.enabled`。固定清单/seed见seen140_tasks.json，50步、history2、temperature0.4、一次rollout、最大512token、保留特殊token；不强制think/marker，不替Actor改动作。在线验证走相同任务清单，但不经过训练中的Teacher评分路径。

RETHINK率是含marker的**决策步占比**，不是token占比，也不等于触发质量。另报reasoning完整率、触发后有效纠错及任务成功。API组报告external-Bad选择率与真实Actor-marker率两个不同指标，不能把外置选择称为Actor自主触发。

## 6. 可复现与结果边界

固定数据、tokenizer、source、task清单和配置均有SHA绑定；`implementation_lock.json`用于检测代码漂移，`reference_3b.json`用于审核共享数学实现不变，不是GPU性能证明。随机在线训练/浮点运算不保证跨硬件逐位相同。

API组是整体替代对照：没有SFT、reference anchor也改成base。只教触发组保持同样SFT样本和局部marker权重，但减少纠正监督。总loss/总训练成本差异应如实报告；不能用这两组单独证明router、hint或初始化的纯因果贡献。

发布时只用 `python src/tools/package.py --out new-source.zip` 白名单打包，不压缩整个工作目录。API缓存、rollout、auth、训练状态和本地模型不上传Git。许可证与公开依赖见docs/related_work.md。
