# 数据

本目录提交本说明及 `frozen_data.zip`，展开后的数据不重复提交。`src/data/` 中的数据处理代码正常提交。压缩包复用完整3B方法的原始文件字节，没有重采样、重标注或修改源动作目标。

本地数据布局：

```text
data/
├── README.md
├── frozen_data.zip
├── reviewed_pool.jsonl.gz
├── selection_ids.json
└── route_sft/
    ├── train.jsonl
    ├── val.jsonl
    ├── generation_val32.jsonl
    ├── excluded.jsonl
    └── manifest.json
```

直接使用仓库内冻结数据包，压缩包 SHA-256 固定在 `configs/data_bundle.json`，每个文件 SHA-256 固定在 `configs/data.json`：

```bash
python src/tools/setup.py --bundled-data
```

该命令校验全部输入后复制，不覆盖内容不同的已有文件，不依赖任何作者本机目录。也可用 `--data-source /path/to/frozen-data` 导入同一 SHA 的外部副本。`reviewed_pool.jsonl.gz` 和 `selection_ids.json` 用于复核筛选与样本顺序；`excluded.jsonl` 记录被排除的样本，不参与训练。

训练集为 10,000 Good + 6,000 Bad，开发集为 128 Good + 128 Bad，按任务互斥拆分；自由生成检查集从开发集固定选择 16 Good + 16 Bad。数据来自固定3B采集策略。

标注提示词与 H/R 盲审输入程序位于 `src/data/`，不确定样本排除。源库的Good保留原采样动作，Bad保存marker和经过验证的纠正动作，允许该动作与原动作相同。**trigger_only编码器只将Bad marker送入训练，纠正动作/EOS不送入；API路由消融不使用该库做SFT。** 源库仍保存完整目标是为了审核同一cohort和归一化长度，不表示消融偷偷训练了纠正。冻结样本的筛选复核不需要 API 或 GPU：

```bash
python src/data/build_dataset.py --out data/rebuilt
```

复核输出必须使用新目录。程序按冻结规则重建，并检查样本 ID、顺序和任务隔离；不会重新标注或补造样本。

ALFWorld 游戏资源独立准备，`ALFWORLD_DATA` 指向同时包含 `json_2.1.1/` 和 `logic/` 的目录。seen140 的任务身份、种子和资源 SHA 固定在 `configs/seen140_tasks.json`，清单不包含游戏资源本体。

