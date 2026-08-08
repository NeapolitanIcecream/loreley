# Zstandard V19 Top-10 Validation Supplement

Date: 2026-08-07

This supplement asks whether the V19 training shortlist was too narrow. The
registered experiment validated the training Top 3, selected `7b9aef38`, and
measured only that candidate on its sealed holdout. Those artifacts and that
conclusion remain unchanged.

## Result

Yes: Top 3 was too narrow for this run. Expanding validation to the
deterministic training Top 10 changed the validation winner from training rank
3 to training rank 10.

The expanded winner, `fe39bee8`, measured 1.01234× compression throughput on
the existing validation split, with a 1.01156× lower 95% bound. It then produced
a 1.00891× compression result on a newly generated confirmation corpus, with a
95% interval of 1.00522–1.01261×. The confirmation result is modest-positive
under the V19 thresholds.

`fe39bee8` is a generation-4 evolved candidate. This supplements the original
case-study conclusion: V19 not only preserved a useful manual seed; its
evolution also produced a positive candidate that generalized to a fresh
corpus.

## Method

The Top-10 list was the fixed prefix of the registered training ordering over
167 unique successful release binaries. Before new measurements, the analysis
froze all ten commits, their executable identities, the original result hashes,
and the following protocol:

- reuse the three existing eight-round validation reports;
- measure training ranks 4–10 with the same release build, upstream checks,
  bidirectional cross-decode, paired benchmark, compressed-size gate, and RSS
  gate;
- use one measurement lane and eight rounds per candidate;
- apply the registered validation ordering; and
- make no model calls and no new measurement on the already revealed holdout.

All seven new candidates passed. The seven measurements took 2,348 seconds in
total, or 335.5 seconds per candidate. The Top-10 expansion therefore cost
about 39 minutes of local compute and no API tokens.

| Training rank | Candidate | Training compression lower 95% | Validation compression | Validation lower 95% | Validation decompression |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `44a22df0` | 1.00984× | 1.00987× | 1.00905× | 0.99754× |
| 2 | `f80c8619` | 1.00944× | 1.01069× | 1.00923× | 1.00115× |
| 3 | `7b9aef38` | 1.00903× | 1.01066× | 1.00994× | 1.00025× |
| 4 | `be548e5c` | 1.00827× | 1.01204× | 1.00899× | 0.99982× |
| 5 | `afc8175a` | 1.00799× | 1.01084× | 1.01020× | 1.00141× |
| 6 | `b52cbe2b` | 1.00782× | 1.00992× | 1.00693× | 0.99848× |
| 7 | `5ee53426` | 1.00741× | 1.01219× | 1.01042× | 1.00210× |
| 8 | `8ec126ea` | 1.00739× | 1.00928× | 1.00762× | 1.00108× |
| 9 | `8075c7ff` | 1.00721× | 1.01123× | 1.01011× | 0.99992× |
| 10 | `fe39bee8` | 1.00708× | **1.01234×** | **1.01156×** | 1.00151× |

The expanded winner was not a favorable training point estimate. Its training
compression lower bound ranked tenth, 0.276 percentage points below the
training leader. Independent validation reversed that ordering. Conversely,
the training combined-throughput leader `8ec126ea` fell to a 1.00762×
compression lower bound on validation.

## Fresh confirmation

After the Top-10 winner was fixed, a new 16 MiB corpus was generated from the
same `zstandard-mixed-v2` family. It contained eight 2 MiB files covering code,
records, natural text, and structured binary data. Its file hashes were
disjoint from the V19 training, validation, and holdout corpora. The corpus was
sealed before reveal, and only `fe39bee8` was measured, for 12 rounds.

| Confirmation metric | Root ratio | 95% interval |
| --- | ---: | ---: |
| Compression throughput | 1.00891× | 1.00522–1.01261× |
| Decompression throughput | 0.99830× | 0.99471–1.00191× |
| Combined throughput | 1.00359× | 1.00004–1.00716× |
| Worst measured cell | 0.99818× | — |
| Maximum compressed-size ratio | 1.00000× | — |
| Peak RSS delta | +0.031 MiB | — |

The candidate passed upstream tests, release compilation, cross-decoding, size,
and RSS checks. The confirmation audit passed. The compression gain remained
positive, but did not meet the strong threshold because its point estimate was
below 1.02×. Decompression was within the registered modest-positive gate.

## Winner lineage

`fe39bee8` first appeared at logical completion 57. Its four-generation lineage
is:

1. manual seed 8, which skips a speculative compression literal copy when the
   literal length is zero;
2. a compression hot-path evolution;
3. an eight-byte `HIST_add()` unroll evolution; and
4. the final zero-literal hot-path evolution.

Relative to root, it changes `lib/compress/hist.c`,
`lib/compress/zstd_compress.c`, and
`lib/compress/zstd_compress_internal.h`: 33 insertions and 16 deletions.
The [source diff is available as a static
patch](../marketing/candidates/zstandard-v19-evolved-followup.patch).

## Interpretation and next rule

The original preregistered winner remains the only candidate measured on the
original holdout, so this supplement does not relabel the registered result.
The fresh confirmation establishes a separate claim: the generation-4 Top-10
winner retained a positive compression gain on a previously unseen corpus. It
does not establish that `fe39bee8` beats `7b9aef38` head-to-head because they
were confirmed on different fresh corpora.

Future campaigns should validate at least the training Top 10. A useful
adaptive rule is to include every candidate within 0.003 of the leading
compression lower bound, with a minimum of 10 and a fixed upper cap. In V19,
that rule would have retained `fe39bee8` while adding about 39 minutes to a
5.31-hour search.

## Evidence

- Top-10 plan SHA-256: `d469e7cc700988117869da315f07c9bd0f1c43655af283b7bdffe1ae7a4e508a`
- Top-10 result SHA-256: `0ea400c8065db2766d7521fd3da564c5646baf538dfb10c6faeb7b27790f0f1c`
- Top-10 audit SHA-256: `d3a690b9ecc830afbbc89fc7ea8f21f69af566cddca3be507a19b14ba9f5bd84`
- Fresh-confirmation plan SHA-256: `b5469d6110efaf72f0804d78dd20d9c9f23bfa66c647fbe5019af004edbe4e8d`
- Fresh-confirmation result SHA-256: `a79d017b20fea14f3c0c404421a13347eae68e223f662cee4c0e88e313b3fb05`
- Fresh-confirmation audit SHA-256: `769ccdb15b1171fff7a329f7908a54f8187e3e77fa13fa9e4dc912445539fbab`
