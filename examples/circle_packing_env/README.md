## Circle Packing Environment

This directory provides the **evaluation environment** and integration glue for the
circle packing benchmark used by Loreley.

It is intentionally separated from the candidate repository in `examples/circle-packing`
so that agents modifying `solution.py` cannot directly change the evaluation logic.

---

## Layout

- `circle-packing/` – the candidate git repository containing the problem statement,
  `solution.py`, and any other files that agents are allowed to modify.
- `evaluate.py` – the evaluation plugin used by Loreley's worker to score candidate
  `solution.py` implementations.

The evaluator treats the candidate worktree as a generic git checkout whose root
contains a `solution.py` file exposing:

```python
from typing import Iterable, Tuple

def pack_circles(n: int = 26) -> Iterable[Tuple[float, float, float]]:
    \"\"\"Return an iterable of (x, y, r) triples for n circles in the unit square.\"\"\"
```

Each triple `(x, y, r)` represents a circle with centre `(x, y)` and radius `r > 0`.
The evaluator will request exactly `n = 26` circles by calling `pack_circles(26)`.

---

## Evaluation plugin (`evaluate.py`)

The file `evaluate.py` implements an **evaluation plugin** compatible with
`loreley.core.worker.evaluator.Evaluator`.

At a high level the plugin will:

1. Locate `solution.py` at the root of the evaluated git worktree.
2. Load the `pack_circles()` function directly from that file using `importlib`.
3. Call `pack_circles(26)` to obtain the list of circles.
4. Perform geometric validity checks:
   - All elements are length-3 iterables of real numbers.
   - `r > 0` for every circle.
   - Circles stay within the unit square: `r <= x <= 1 - r` and `r <= y <= 1 - r`.
   - No two circles overlap: the distance between any pair of centres is at least
     the sum of their radii (within a small numerical tolerance).
5. Compute:
   - `sum_radii` – main objective (higher is better).
   - `packing_density` – total area of all circles inside the unit square (secondary metric).
   - `num_circles` – number of circles (for sanity checking, higher is better).
6. Return an `EvaluationResult`-compatible mapping with:
   - A human-readable `summary`.
   - The two metrics above.
   - A list of `tests_executed`.
   - Textual `logs` describing what happened.

Logging uses **Loguru** for structured logs and **Rich** for coloured console output,
consistent with the rest of the Loreley codebase.

---

## Integrating with Loreley

To use this environment with Loreley's worker evaluator, point the evaluator to this
directory and plugin:

1. Ensure your Loreley project root (the directory that contains `loreley/` and
   `examples/`) is importable (for example by running under `uv` from the project root).
2. Configure the following environment variables:

   - `WORKER_EVALUATOR_PYTHON_PATHS=["/absolute/path/to/your/loreley/examples/circle_packing_env"]`
   - `WORKER_EVALUATOR_PLUGIN=evaluate:plugin`

3. Start the worker as usual, for example:

   ```bash
   uv run python script/run_worker.py
   ```

Whenever Loreley schedules a job whose worktree contains a `solution.py` at its
repository root, the evaluator will load it and compute circle-packing metrics that
MAP-Elites can optimise.

---

## Local quick test

You can also run the evaluator directly from this directory (within the Loreley
project) without starting the full worker:

```bash
uv run python examples/circle_packing_env/evaluate.py
```

This will treat `examples/circle-packing` (or whichever worktree you configure) as the
candidate repository, load its current `solution.py`, and print a summary of the
computed metrics to the console using Rich formatting.


