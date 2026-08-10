# Case-study candidate diffs

日期：2026-08-07

这些 patch 固化案例报告中的 source diff，便于直接查看代码变化。性能结论仍以 evaluator report、artifact identity 和 validation result 为准。

| 案例 | Root | Candidate | Source patch | Patch SHA-256 |
| --- | --- | --- | --- | --- |
| `markdown-it-py` | `97aff4f564e02e24f8526d9e2cd7899c47f714a6` | `b10adb6fad0da2a9825c3d1525048fd7b177d773` | [5 files, +54/-14](markdown-it-py-winner.patch) | `3767855bd3b1e2009ef9f8af477c5fbef390031cfe225b3f4dfb8e40843f9fcb` |
| `python-pathspec` | `6568072c2703c72796cd02467feb924540157c92` | `9d977f0a73d58aec73fa36516c07cbb0ec879347` | [5 files, +127/-51](python-pathspec-winner.patch) | `779f382cbaecd65f495e301a3a21833c9c87f00ac94f26fbd5ea4fe78e7ba742` |
| Zstandard V19 registered | `5b3fe474e4df572a7588be7abf3d8b6bd4b6010e` | `7b9aef38ecd44ba1efe0ff7282c234ae1f1ef14c` | [1 file, +8/-1](zstandard-v19-registered-winner.patch) | `8931ac0a028fb0abe2fd119a983fcfed9103204c7ab825e9d57a671ec794b221` |
| Zstandard V19 Top-10 validation winner | `5b3fe474e4df572a7588be7abf3d8b6bd4b6010e` | `fe39bee8f4659b8e8da153b9a997614f4d2d4713` | [3 files, +33/-16](zstandard-v19-evolved-followup.patch) | `9953a2654247b3651e1c96cc5ca7b75201ffe399a490d14a99a4e5a660f0b0d5` |

`python-pathspec` 的 patch 只包含 case-study 报告计数的 `pathspec/**/*.py` 修改，排除了实验控制文件 `.loreleyignore` 和 `loreley.program.md`。

Zstandard registered winner 的 release-binary SHA-256 是 `e7e9ef6b060fd060303812e8374a9cb73cba86a965798098b397db3539c302c0`。Top-10 validation winner 的 release-binary SHA-256 是 `65c54a8b39d88eafe1667445b4c10e88c669b75a32d7a999192b1924d56f98c9`。Source patch、binary identity 和评测报告需要一起解释。

固定 Top 10 后在原 holdout 上的补测将第 3 代 evolved candidate `5ee53426` 排在描述性首位，`fe39bee8` 排在第二位。该 holdout 此前已为预登记 winner 揭示，因此这一排名不替代预登记结果。

完整选择状态和结果见：

- [`markdown-it-py` 正式报告](../../research/2026-08-02-markdown-it-py-deepseek-case-study.md)
- [`python-pathspec` 正式报告](../../research/2026-08-03-pathspec-deepseek-case-study.md)
- [Zstandard V19 正式报告](../../research/2026-08-07-zstandard-gpt-v19-case-study-report.md)
- [Zstandard V19 Top-10 补充](../../research/2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md)
