# Generate managed-pipelines.json and compile managed pipelines

Generates `managed-pipelines.json` at the repository root and compiles each managed pipeline's
`pipeline.py` to `pipeline.yaml` using the KFP compiler. Only pipelines whose `metadata.yaml`
has `managed: true` are included.

## Usage

From the project root:

```bash
# Generate managed-pipelines.json at repo root
uv run python -m scripts.generate_managed_pipelines.generate_managed_pipelines

# Write to a custom path
uv run python -m scripts.generate_managed_pipelines.generate_managed_pipelines -o path/to/managed-pipelines.json
```

## Output format

The JSON is an array of objects with:

| Field         | Source                | Description |
|---------------|-----------------------|-------------|
| `name`        | `metadata.yaml` name  | Pipeline name |
| `description` | See below             | Short description for the catalog |
| `path`        | Derived               | Relative path to `pipeline.py` (e.g. `pipelines/training/automl/my_pipeline/pipeline.py`) |
| `stability`   | `metadata.yaml` stability | `alpha` → Development Preview; `beta` → Technology Preview; `stable` → General Availability. `experimental` is not allowed for managed pipelines. |

**`description` resolution (in order):**

1. If `metadata.yaml` has a non-empty `description` string, that value is used.
2. Otherwise `pipeline_description.py` parses `pipeline.py` and reads the static `description=`
   argument from `@dsl.pipeline(...)` (including implicit string concatenation).
3. If there is no decorator description, the first line of the pipeline function’s docstring is used.

The pipeline function is chosen by matching `metadata.yaml` `name` to the Python function name when
possible; otherwise the first `@dsl.pipeline` in the file is used.

## Including a pipeline

In the pipeline’s `metadata.yaml`:

1. Set `managed: true`.
2. Optionally set `description` to override the decorator/docstring for the catalog.

Only directories that contain both `metadata.yaml` and `pipeline.py` are considered.

If any pipeline has `managed: true` but invalid metadata (missing/invalid `name` or `stability`, etc.),
the command **exits with code 1** and prints an error; no `managed-pipelines.json` is written.

## Compilation

After writing `managed-pipelines.json`, the script compiles each managed pipeline's `pipeline.py`
to `pipeline.yaml` in the same directory using the KFP SDK compiler. This eliminates the need to
commit compiled YAML files to the repository.

If compilation fails for any pipeline (missing `@dsl.pipeline` decorator, import error, etc.),
the command **exits with code 1**.
