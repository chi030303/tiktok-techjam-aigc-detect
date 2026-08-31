# 2026-08-31, tianqi, how teammates open bad-case HTML
# 误差画册怎么看

画册是 **自包含 HTML**（缩略图已经 base64 嵌进文件）。**不要在 Vast Jupyter / 文件浏览器里点开。** 那边不会当网页渲染，看起来是空白或源码。

## 组员怎么看

1. **下到自己电脑**，用本机 Chrome / Safari 打开 `index.html`。
2. 不要指望 `http://vast-ip/...` 或 Jupyter `view` 能显示画册。

从自己电脑拉（端口以当前 Vast 控制台为准）：

```bash
# 画册 + 两两对照（约 30–40 MB，不含 jsonl）
rsync -avz -e "ssh -p <PORT> -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes" \
  root@<IP>:/workspace/kiki/tiktok-techjam-aigc-detect/outputs/tables/badcase_galleries/ \
  ./badcase_galleries/

rsync -avz -e "ssh -p <PORT> -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes" \
  root@<IP>:/workspace/kiki/tiktok-techjam-aigc-detect/outputs/tables/badcase_compare/ \
  ./badcase_compare/
```

然后打开 `badcase_galleries/index.html`。对照页链接是相对路径 `../badcase_compare/...`，两个文件夹要放在同一层。

本机已经有一份的话，直接打开仓库里的：

`outputs/tables/badcase_galleries/index.html`

不要把 `jsonl/` 整包发给组员（大，且不是给人看的）。GitHub 也 **不提交** 大 HTML；仓库只留 `index.html` 和对照 JSON。

## 录视频要不要 Streamlit

**不要。** 截止视频 ≤3 分钟。交付物是 `predict.py`，不是 demo 站。录屏用 slides + 必要时本机打开一个 HTML 画册即可。

# end
