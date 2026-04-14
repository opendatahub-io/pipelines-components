"""Dataset Download Component.

This component downloads datasets from multiple sources (HuggingFace, S3, HTTP, local/PVC),
validates the format for chat templates, splits into train/eval, and saves to PVC as JSONL.

Supported URI schemes:
- hf://dataset_name or dataset_name - HuggingFace datasets
- s3://bucket/path/to/dataset.jsonl - AWS S3 datasets (JSONL format)
- http://... or https://... - HTTP/HTTPS URLs (e.g., MinIO shared links)
- pvc://path/to/dataset.jsonl or /absolute/path - Local/PVC file paths (JSONL format)
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/opendatahub/odh-training-th04-cpu-torch29-py312-rhel9:cpu-3.3",
    packages_to_install=["datasets>=2.14.0", "huggingface-hub>=0.20.0", "s3fs>=2023.1.0"],
)
def dataset_download(
    train_dataset: dsl.Output[dsl.Dataset],
    eval_dataset: dsl.Output[dsl.Dataset],
    dataset_uri: str,
    pvc_mount_path: str,
    train_split_ratio: float = 0.9,  # 1.0 = no eval split (all data for training)
    subset_count: int = 0,
    shared_log_file: str = "pipeline_log.txt",
):
    """Download and prepare datasets from multiple sources.

    Validates that datasets follow chat template format (messages/conversations with role/content).

    Args:
        train_dataset: Output artifact for training dataset (JSONL format)
        eval_dataset: Output artifact for evaluation dataset (JSONL format)
        dataset_uri: Dataset URI (hf://, s3://, https://, pvc:// or absolute path)
        pvc_mount_path: Path where the shared PVC is mounted
        train_split_ratio: Train/eval split (0.9 = 90%/10%, 1.0 = no split, all for training)
        subset_count: Number of examples to use (0 = use all)
        shared_log_file: Name of the shared log file
    """
    import os

    from datasets import Dataset, load_dataset

    # Prefer HF_TOKEN from the environment (typically injected via the `hf-token` Kubernetes secret).
    # This keeps authentication configuration in Kubernetes secrets instead of pipeline parameters.
    hf_token = (os.environ.get("HF_TOKEN") or "").strip()

    def log_message(msg: str):
        """Log message to console and shared log file."""
        print(msg)
        log_path = os.path.join(pvc_mount_path, shared_log_file)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    def parse_uri(uri: str) -> tuple[str, str]:
        """Parse dataset URI to determine source type and path.

        Args:
            uri: Dataset URI with scheme. Supported formats:
                - HuggingFace: hf://dataset-name or dataset-name
                - S3/MinIO: s3://bucket/path/to/file.jsonl
                - HTTP/HTTPS: http://... or https://... (e.g., MinIO shared links)
                - Local/PVC: pvc://path/to/file.jsonl or /absolute/path/to/file.jsonl

        Returns:
            Tuple of (source_type, path) where source_type is 'hf', 's3', 'http', or 'local'
        """
        uri = uri.strip()

        # Check for explicit schemes
        if uri.startswith("http://") or uri.startswith("https://"):
            return ("http", uri)
        elif uri.startswith("hf://"):
            # HuggingFace URIs come in two forms:
            #   hf://datasets/org/repo  (standard HF URI with type prefix)
            #   hf://org/repo           (short form)
            # Strip the scheme, then strip the optional "datasets/" prefix
            # to get the bare dataset identifier (e.g. "org/repo").
            hf_path = uri[5:]
            if hf_path.startswith("datasets/"):
                hf_path = hf_path[len("datasets/"):]
            return ("hf", hf_path)
        elif uri.startswith("s3://"):
            return ("s3", uri[5:])
        elif uri.startswith("pvc://"):
            return ("local", uri[6:])
        elif uri.startswith("/"):
            return ("local", uri)
        else:
            # Default to HuggingFace if no scheme
            return ("hf", uri)

    def validate_dataset_format(dataset: Dataset) -> None:
        """Check dataset format and log the detected structure.

        Recognises two layout families:
        - **Chat template**: each row has a ``messages`` or ``conversations``
          list of ``{role, content}`` dicts.
        - **Columnar / instruction-style**: arbitrary columns such as
          ``question``, ``context``, ``answer``, ``instruction``, ``input``,
          ``output``, etc.  These are handled downstream by the training
          component's ``dataset_type`` / field-mapping parameters.

        The function never raises on format mismatch — it only logs
        warnings so the training component can decide how to interpret the
        data.
        """
        import logging as _logging

        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        keys = list(dataset[0].keys())

        # Detect chat template format
        if "messages" in keys or "conversations" in keys:
            field = "messages" if "messages" in keys else "conversations"
            valid_roles = {"system", "user", "assistant", "function", "tool"}
            num_to_check = min(100, len(dataset))
            issues: list[str] = []

            for i in range(num_to_check):
                messages = dataset[i].get(field, [])
                if not isinstance(messages, list):
                    issues.append(f"Item {i}: '{field}' is not a list")
                    continue
                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        issues.append(f"Item {i}, msg {j}: not a dict")
                    elif "role" not in msg or "content" not in msg:
                        issues.append(f"Item {i}, msg {j}: missing role/content")
                    elif msg["role"] not in valid_roles:
                        issues.append(f"Item {i}, msg {j}: unknown role '{msg['role']}'")

            if issues:
                _logging.warning(
                    "Chat-format validation found %d issue(s) in first %d rows "
                    "(first: %s). The training component may still handle this "
                    "dataset if configured with the right field mappings.",
                    len(issues),
                    num_to_check,
                    issues[0],
                )
            else:
                log_message(f"Dataset validated: {len(dataset)} examples in chat format")
        else:
            log_message(
                f"Dataset is not in chat template format (fields: {keys}). "
                f"Ensure the training component is configured with the correct "
                f"dataset_type and field-mapping parameters."
            )

    def split_hf_id_and_config(dataset_path: str) -> tuple[str, str | None]:
        """Split an HF dataset identifier into (id, config) if a config suffix is provided.

        Examples:
            "LipengCS/Table-GPT:All" -> ("LipengCS/Table-GPT", "All")
            "bigcode/the-stack-dedup-python" -> ("bigcode/the-stack-dedup-python", None)
        """
        if ":" in dataset_path:
            base, cfg = dataset_path.split(":", 1)
            return base, (cfg or None)
        return dataset_path, None

    def download_from_huggingface(dataset_path: str) -> Dataset:
        """Download dataset from HuggingFace."""
        ds_id, ds_config = split_hf_id_and_config(dataset_path)
        if ds_config:
            log_message(f"Downloading from HuggingFace: {ds_id} (config: {ds_config})")
        else:
            log_message(f"Downloading from HuggingFace: {ds_id}")

        # Force online mode and set a writable cache directory.
        # Base images (e.g. odh-training) often set HF_HOME to a path like
        # /opt/app-root/src which contains a pre-existing "datasets/" directory.
        # When a Hub download fails the library silently falls back to that
        # local path, producing a misleading FileNotFoundError. Using /tmp
        # avoids the conflict and ensures a clean download.
        hf_cache = "/tmp/hf_home"
        os.makedirs(hf_cache, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache
        os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache, "datasets")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        log_message(f"HF_HOME set to {hf_cache} (online mode forced)")

        # Set up authentication if token provided via environment
        if hf_token:
            log_message("Using HF_TOKEN from environment for Hugging Face authentication")
        else:
            import logging as _logging

            pretty_id = f"{ds_id}:{ds_config}" if ds_config else ds_id
            _logging.warning(
                "HF_TOKEN is not set; attempting to download Hugging Face dataset "
                f"'{pretty_id}' without authentication. "
                "Only public, non-gated datasets can be downloaded. "
                "If you need access to gated datasets, configure the 'hf-token' Kubernetes "
                "secret so HF_TOKEN is available to the component."
            )

        # Try to load with "train" split first
        load_kwargs: dict = {
            "path": ds_id,
            "split": "train",
        }
        if ds_config:
            load_kwargs["name"] = ds_config

        if hf_token:
            load_kwargs["token"] = hf_token

        try:
            dataset = load_dataset(**load_kwargs)
            log_message(f"Downloaded {len(dataset)} examples from HuggingFace (split: train)")
            return dataset

        except ValueError as e:
            # If "train" split doesn't exist, try to find an alternative
            if "Unknown split" in str(e):
                log_message("'train' split not found, attempting to detect available splits...")

                # Load dataset info without specifying split
                try:
                    load_kwargs_no_split: dict = {
                        "path": ds_id,
                    }
                    if ds_config:
                        load_kwargs_no_split["name"] = ds_config
                    if hf_token:
                        load_kwargs_no_split["token"] = hf_token

                    # Load all splits
                    dataset_dict = load_dataset(**load_kwargs_no_split)

                    # Try common training split names in order of preference
                    preferred_splits = ["train_sft", "train_gen", "train", "training"]

                    for split_name in preferred_splits:
                        if split_name in dataset_dict:
                            log_message(f"Using split: {split_name}")
                            dataset = dataset_dict[split_name]
                            log_message(f"Downloaded {len(dataset)} examples from HuggingFace (split: {split_name})")
                            return dataset

                    # If none of the preferred splits found, use the first available split
                    available_splits = list(dataset_dict.keys())
                    if available_splits:
                        first_split = available_splits[0]
                        log_message(f"Using first available split: {first_split}")
                        dataset = dataset_dict[first_split]
                        log_message(f"Downloaded {len(dataset)} examples from HuggingFace (split: {first_split})")
                        return dataset
                    else:
                        raise ValueError("No splits found in dataset")

                except Exception as inner_e:
                    log_message(f"Error detecting splits: {str(inner_e)}")
                    raise
            else:
                log_message(f"Error loading dataset: {str(e)}")
                raise
        except Exception as e:
            log_message(f"Error loading dataset: {str(e)}")
            raise

    def download_from_s3(s3_path: str) -> Dataset:
        """Download dataset from AWS S3 using datasets library native S3 support."""
        log_message(f"Loading from AWS S3: s3://{s3_path}")

        # Get credentials from Kubernetes secret (environment variables)
        access_key = (os.environ.get("AWS_ACCESS_KEY_ID") or "").strip()
        secret_key = (os.environ.get("AWS_SECRET_ACCESS_KEY") or "").strip()

        # Validate that credentials are present and consistent
        if (access_key and not secret_key) or (secret_key and not access_key):
            raise ValueError(
                "S3 credentials misconfigured: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must either "
                "both be set and non-empty, or both be unset. Check the 's3-secret' Kubernetes secret."
            )
        if not access_key and not secret_key:
            raise ValueError(
                "S3 credentials missing: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be provided via "
                "the 's3-secret' Kubernetes secret when using s3:// dataset URIs."
            )

        # Build storage_options for datasets library
        storage_options = {}

        # Add credentials if available (otherwise uses default AWS credential chain)
        if access_key and secret_key:
            storage_options["key"] = access_key
            storage_options["secret"] = secret_key
            log_message("Using S3 credentials from Kubernetes secret")
        else:
            log_message("No credentials found, using default AWS credential chain (IAM role, etc.)")

        # Load dataset directly from S3 (no temp file needed)
        dataset = load_dataset("json", data_files=f"s3://{s3_path}", storage_options=storage_options, split="train")

        log_message(f"Loaded {len(dataset)} examples from AWS S3")
        return dataset

    def download_from_http(http_url: str) -> Dataset:
        """Download dataset from HTTP/HTTPS URL (e.g., MinIO shared links)."""
        log_message(f"Loading from HTTP: {http_url}")

        # Load dataset directly from HTTP URL using datasets library
        dataset = load_dataset("json", data_files=http_url, split="train")

        log_message(f"Loaded {len(dataset)} examples from HTTP")
        return dataset

    def load_from_local(file_path: str) -> Dataset:
        """Load dataset from local/PVC file path."""
        log_message(f"Loading from local path: {file_path}")

        # If relative path, make it relative to pvc_mount_path
        if not file_path.startswith("/"):
            file_path = os.path.join(pvc_mount_path, file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load using datasets library (supports .json and .jsonl)
        if file_path.endswith(".jsonl") or file_path.endswith(".json"):
            dataset = load_dataset("json", data_files=file_path, split="train")
            log_message(f"Loaded {len(dataset)} examples from local file")
            return dataset
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Expected .json or .jsonl")

    # =========================================================================
    # Main execution
    # =========================================================================

    log_message("=" * 60)
    log_message("Dataset Download Component Started")
    log_message("=" * 60)
    log_message(f"Dataset URI: {dataset_uri}")
    log_message(f"Train/Eval split: {train_split_ratio:.0%}/{1 - train_split_ratio:.0%}")
    log_message(f"Subset count: {subset_count if subset_count > 0 else 'all (no limit)'}")

    try:
        # Parse URI and determine source
        source_type, source_path = parse_uri(dataset_uri)
        log_message(f"Source type: {source_type}")
        log_message(f"Source path: {source_path}")

        # Download/load dataset based on source
        if source_type == "hf":
            dataset = download_from_huggingface(source_path)
        elif source_type == "s3":
            dataset = download_from_s3(source_path)
        elif source_type == "http":
            dataset = download_from_http(source_path)
        elif source_type == "local":
            dataset = load_from_local(source_path)
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # Apply subset if specified
        if subset_count and subset_count > 0:
            import random

            original_size = len(dataset)
            if subset_count < original_size:
                log_message(f"Applying subset: {subset_count} of {original_size} examples")
                random.seed(42)  # For reproducibility
                subset_indices = random.sample(range(original_size), subset_count)
                dataset = dataset.select(subset_indices)
                log_message(f"Subset applied: {len(dataset)} examples selected")
            else:
                log_message(f"Subset count ({subset_count}) >= dataset size ({original_size}), using all examples")

        # Check dataset format (warns on non-chat layouts, never blocks)
        log_message("Checking dataset format...")
        validate_dataset_format(dataset)

        # Split dataset (or use all for training if ratio is 1.0)
        log_message(f"Splitting dataset with {len(dataset)} examples...")

        if train_split_ratio >= 1.0:
            # No split - use all data for training, create empty eval
            log_message("train_split_ratio=1.0: Using all data for training (no eval split)")
            train_ds = dataset
            eval_ds = Dataset.from_dict({k: [] for k in dataset.features.keys()})
            log_message(f"No split: {len(train_ds)} train, 0 eval (eval dataset will be empty)")
        else:
            split_dataset = dataset.train_test_split(test_size=1 - train_split_ratio, seed=42)
            train_ds = split_dataset["train"]
            eval_ds = split_dataset["test"]
            log_message(f"Split complete: {len(train_ds)} train, {len(eval_ds)} eval")

        # Save datasets as JSONL files to KFP artifacts
        log_message(f"Saving train dataset to {train_dataset.path}")
        train_ds.to_json(train_dataset.path, orient="records", lines=True)

        log_message(f"Saving eval dataset to {eval_dataset.path}")
        eval_ds.to_json(eval_dataset.path, orient="records", lines=True)

        # Also save to shared PVC for next pipeline step
        pvc_dataset_dir = os.path.join(pvc_mount_path, "datasets")
        os.makedirs(pvc_dataset_dir, exist_ok=True)

        pvc_train_path = os.path.join(pvc_dataset_dir, "train.jsonl")
        pvc_eval_path = os.path.join(pvc_dataset_dir, "eval.jsonl")

        log_message(f"Saving train dataset to PVC: {pvc_train_path}")
        train_ds.to_json(pvc_train_path, orient="records", lines=True)

        log_message(f"Saving eval dataset to PVC: {pvc_eval_path}")
        eval_ds.to_json(pvc_eval_path, orient="records", lines=True)

        # Save metadata
        train_dataset.metadata = {
            "dataset_uri": dataset_uri,
            "num_examples": len(train_ds),
            "split": "train",
            "train_split_ratio": train_split_ratio,
            "artifact_path": train_dataset.path,
            "pvc_path": pvc_train_path,
        }

        eval_dataset.metadata = {
            "dataset_uri": dataset_uri,
            "num_examples": len(eval_ds),
            "split": "eval",
            "train_split_ratio": train_split_ratio,
            "artifact_path": eval_dataset.path,
            "pvc_path": pvc_eval_path,
        }

        log_message("=" * 60)
        log_message("Dataset Download Component Completed Successfully")
        log_message(f"  Train: {len(train_ds)} examples")
        log_message(f"    - KFP Artifact: {train_dataset.path}")
        log_message(f"    - PVC: {pvc_train_path}")
        log_message(f"  Eval: {len(eval_ds)} examples")
        log_message(f"    - KFP Artifact: {eval_dataset.path}")
        log_message(f"    - PVC: {pvc_eval_path}")
        log_message("=" * 60)

    except Exception as e:
        error_msg = f"ERROR in dataset download: {str(e)}"
        log_message(error_msg)
        raise


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        dataset_download,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print("Compiled: dataset_download_component.yaml")
