# Loreley 发布材料工作台

> 内部审阅索引，不作为对外页面。

## 论文

- [arXiv:2608.19703](https://arxiv.org/abs/2608.19703)：*Loreley: Repository-Scale Program Evolution with Quality-Diversity Search*；
- [论文源码与公开证据](../../paper/README.md)；
- [GitHub 项目](https://github.com/NeapolitanIcecream/loreley)。

论文证据分为两组：1,008-job Zstandard matched policy experiment 比较 Loreley QD、Sequential Champion 和 Independent Root；三个较早的 capability campaigns 共 348 jobs。发布材料必须保留两者的目的与统计口径，不合并成一项实验。

## 对外材料

- [中文长文](2026-08-loreley-launch-article-zh.md)；
- [英文长文](2026-08-loreley-launch-article-en.md)；
- [arXiv 论文](https://arxiv.org/abs/2608.19703)：受控策略实验、三个 capability cases 和公开证据入口；
- [Design-partner brief](loreley-design-partner-brief.md)：场景筛选、预算、数据边界和合作交付；
- [公开 intake 模板](https://github.com/NeapolitanIcecream/loreley/blob/main/.github/ISSUE_TEMPLATE/design-partner.yml)：公开 issue 只收非机密摘要。
- [候选 diff 索引](candidates/README.md)：三个案例的 source patch、root/candidate hash 和 Zstandard binary identity。

## 内部控制文件

- [发布口径表](2026-08-loreley-launch-claim-sheet.md)：受控实验、capability cases、selection status、成本语义和禁用表述；
- [发布文案包](2026-08-loreley-launch-copy-kit.md)：可直接发布的中英文摘要、短帖、技术社区帖和发布顺序。

## 图片

每张图都有可编辑 SVG 和 1600×900 PNG：

1. [论文首发主图 PNG](assets/loreley-paper-overview.png) / [SVG](assets/loreley-paper-overview.svg)：arXiv 和社区帖默认使用；包含方法循环、1,008-job endpoint、mechanism activity 和 capability cases
2. [搜索循环 PNG](assets/loreley-search-loop.png) / [SVG](assets/loreley-search-loop.svg)
3. [三案例证据 PNG](assets/loreley-three-case-evidence.png) / [SVG](assets/loreley-three-case-evidence.svg)
4. [案例谱系 PNG](assets/loreley-case-lineages.png) / [SVG](assets/loreley-case-lineages.svg)
5. [Zstandard identity 与 Top 10 holdout 结果 PNG](assets/loreley-zstd-identity-results.png) / [SVG](assets/loreley-zstd-identity-results.svg)

论文首发主图是唯一同时覆盖 matched policy experiment 与 capability campaigns 的宣传图。其余四张图只用于相应的系统或 capability 细节，不作为受控策略结果总览。

图片由 [`render_launch_assets.py`](https://github.com/NeapolitanIcecream/loreley/blob/main/tools/marketing/render_launch_assets.py) 生成。修改数据或文案后运行：

```bash
uv run python tools/marketing/render_launch_assets.py
```

## 建议审阅顺序

1. 先确认发布口径表中的 1,008-job 方法实验与 348-job capability campaigns 没有混写；
2. 审中英文短帖和技术社区帖，确认 matched result 同时包含两个 baseline；
3. 审论文概览图和四张 capability 图片的使用场景；
4. 审 design-partner 合作门槛、数据边界和 intake 字段；
5. 发布后将读者问题整理为 FAQ，不在单个平台临时改变实验结论。
