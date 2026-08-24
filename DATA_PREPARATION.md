# 生成 VPT 训练数据

在 Linux 训练机的仓库根目录依次执行以下命令。本流程从 OpenAI VPT 7.x contractor 数据中选择一个可复现、大小受控的子集，并生成 `data/canonical`。默认下载 1,200 段录像（最多 100 小时），可为 720GB 硬盘保留足够余量。

## 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[train,test]'

# Ubuntu/Debian；aria2 用于并行下载和断点续传。
sudo apt-get update
sudo apt-get install -y aria2 jq
```

## 2. 选择 1,200 段官方录像

选中路径列表就是本次数据集的选择记录。续传时必须保持该文件不变。

```bash
mkdir -p data/vpt data/canonical

curl -fL --retry 5 \
  https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json \
  -o data/vpt/vpt-7x-index.json

jq -r '.relpaths[]' data/vpt/vpt-7x-index.json \
  | shuf -n 1200 \
  > data/vpt/selected-relpaths.txt

wc -l data/vpt/selected-relpaths.txt
```

最后一条命令必须输出 `1200`。如需更换固定子集，只重新生成一次该文件；续传期间不要重新生成。

## 3. 下载 MP4 和 JSONL

生成 aria2 输入文件，同时保留 recorder version 目录和原始文件名：

```bash
set -euo pipefail

raw_root="$(mkdir -p data/vpt/raw && realpath data/vpt/raw)"
base_url="$(jq -r '.basedir' data/vpt/vpt-7x-index.json)"

while IFS= read -r relpath; do
  version="${relpath%%/*}"
  stem="${relpath##*/}"
  mkdir -p "$raw_root/$version"
  for extension in mp4 jsonl; do
    printf '%s%s.%s\n  dir=%s/%s\n  out=%s.%s\n' \
      "$base_url" "$relpath" "$extension" \
      "$raw_root" "$version" "$stem" "$extension"
  done
done < data/vpt/selected-relpaths.txt > data/vpt/aria2-input.txt

aria2c \
  --continue=true \
  --auto-file-renaming=false \
  --max-concurrent-downloads=8 \
  --max-connection-per-server=2 \
  --input-file=data/vpt/aria2-input.txt
```

下载中断后重复执行同一条 `aria2c` 命令即可续传。定期检查空间：

```bash
du -sh data/vpt
df -h data
```

剩余空间低于 150GB 时不要再增加数据。转换后不能删除 MP4：canonical manifest 只引用视频，不会复制视频。

## 4. 转换为 canonical episodes

VPT action 的 `milli` 是墙上时钟时间，而 MP4 PTS 从接近零的位置开始。下面的循环会先归一化每个动作文件，再执行导入。官方索引没有提供可靠的 world ID，因此这里保守地用 contractor alias 作为 split group：同一 contractor 的全部 session（以及其中可能重复使用的 world）只会进入一个 split。hash bucket 的目标比例为 90% train、5% validation、5% test；由于按 contractor 分组，最终 episode 数量比例可能偏离该目标。

```bash
set -euo pipefail

raw_root="$(realpath data/vpt/raw)"
canonical_root="$(realpath data/canonical)"

while IFS= read -r relpath; do
  version="${relpath%%/*}"
  stem="${relpath##*/}"
  episode_id="${relpath//\//__}"
  video="$raw_root/$version/$stem.mp4"
  actions="$raw_root/$version/$stem.jsonl"

  episode_dir="$canonical_root/episodes/$episode_id"
  if [ -f "$episode_dir/manifest.json" ] \
    && [ -f "$episode_dir/actions.jsonl" ] \
    && [ -f "$episode_dir/frame_timestamps.json" ]; then
    echo "already ingested: $episode_id"
    continue
  fi
  if [ -e "$episode_dir" ]; then
    echo "incomplete episode directory; inspect and remove it before retrying: $episode_dir" >&2
    exit 1
  fi

  without_time="${stem%-*}"
  without_date="${without_time%-*}"
  session_id="${without_date##*-}"
  contractor_alias="${without_date%-$session_id}"
  if [ -z "$contractor_alias" ] || [ "$contractor_alias" = "$without_date" ]; then
    echo "cannot parse contractor alias and session ID from: $stem" >&2
    exit 1
  fi
  split_group="contractor:$contractor_alias"
  # VPT 7.x 的公开索引没有 world ID。使用保守分组键，防止同一
  # contractor 跨 session 重用的 world 泄漏到不同 split。
  world_id="$split_group"

  digest="$(printf '%s' "$split_group" | sha256sum | cut -c1-8)"
  bucket=$((16#$digest % 100))
  if [ "$bucket" -lt 90 ]; then
    split=train
  elif [ "$bucket" -lt 95 ]; then
    split=validation
  else
    split=test
  fi

  normalized_actions="$(mktemp)"
  first_milli="$(sed -n '1p' "$actions" | jq -r '.milli')"
  jq -c --argjson origin "$first_milli" \
    'if .milli != null then .milli = (.milli - $origin) else . end' \
    "$actions" > "$normalized_actions"

  PYTHONPATH=src python3 scripts/prepare_vpt.py \
    --output "$canonical_root" \
    --video "$video" \
    --actions "$normalized_actions" \
    --episode-id "$episode_id" \
    --session-id "$session_id" \
    --world-id "$world_id" \
    --recorder-version "$version" \
    --split "$split"

  rm -f "$normalized_actions"
done < data/vpt/selected-relpaths.txt
```

转换过程会完整解码一次每个 MP4，以提取精确帧时间戳，因此可能持续数小时。重新执行循环会跳过已经完成的 episodes。

## 5. 审计与验证

```bash
PYTHONPATH=src python3 scripts/audit_data.py data/canonical \
  --output data/canonical/audit-report.json

jq '{episode_count, duration_hours, frame_count, split_leakage, issue_count: (.issues | length)}' \
  data/canonical/audit-report.json

jq '{episodes: (.episodes | length), splits: ([.episodes[].split] | group_by(.) | map({(.[0]): length}) | add)}' \
  data/canonical/dataset_manifest.json
```

`split_leakage` 必须为空。训练前检查所有非零 `issue_count`：孤立的末帧 action 警告可以单独确认；分辨率、时间戳或较大 frame gap 问题应从数据集中排除。

## 6. 启动单张 H200 训练

```bash
PYTHONPATH=src python3 -m mcwm.training.pretrain_visual \
  --config configs/pretrain_visual_1xh200.yaml \
  --data-root data/canonical \
  --output-dir artifacts/m1-visual-1xh200
```

配置中的 `7,812` 个 optimizer steps 共处理 `499,968` 个训练 clips。下载的数据池会用于随机采样；validation clips 不计入该预算。
