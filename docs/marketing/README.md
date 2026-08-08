# Loreley 发布材料工作台

> 内部审阅索引，不作为对外页面。

## 对外材料

- [中文长文](2026-08-loreley-launch-article-zh.md)；
- [英文长文](2026-08-loreley-launch-article-en.md)；
- [Design-partner brief](loreley-design-partner-brief.md)：场景筛选、预算、数据边界和合作交付；
- [公开 intake 模板](https://github.com/NeapolitanIcecream/loreley/blob/main/.github/ISSUE_TEMPLATE/design-partner.yml)：公开 issue 只收非机密摘要。
- [候选 diff 索引](candidates/README.md)：三个案例的 source patch、root/candidate hash 和 Zstandard binary identity。

## 内部控制文件

- [发布口径表](2026-08-loreley-launch-claim-sheet.md)：数字、selection status、成本语义和禁用表述；
- [发布文案包](2026-08-loreley-launch-copy-kit.md)：摘要、短帖、技术社区帖、英文帖和发布顺序。

## 图片

每张图都有可编辑 SVG 和 1600×900 PNG：

1. [搜索循环 PNG](assets/loreley-search-loop.png) / [SVG](assets/loreley-search-loop.svg)
2. [三案例证据 PNG](assets/loreley-three-case-evidence.png) / [SVG](assets/loreley-three-case-evidence.svg)
3. [案例谱系 PNG](assets/loreley-case-lineages.png) / [SVG](assets/loreley-case-lineages.svg)
4. [Zstandard identity 与结果 PNG](assets/loreley-zstd-identity-results.png) / [SVG](assets/loreley-zstd-identity-results.svg)

图片由 [`render_launch_assets.py`](https://github.com/NeapolitanIcecream/loreley/blob/main/tools/marketing/render_launch_assets.py) 生成。修改数据或文案后运行：

```bash
python3 tools/marketing/render_launch_assets.py
```

## 建议审阅顺序

1. 先确认发布口径表中的定位、数字和禁用表述；
2. 审中文长文的叙事、篇幅和技术脉络；
3. 审四张图中的信息层级与措辞；
4. 审 design-partner 合作门槛、数据边界和 intake 字段；
5. 审英文稿与各平台短文案；
6. 批准后提交和发布对外材料，内部控制文件不进入导航。

本轮没有执行 git commit、push 或外部平台发布。
