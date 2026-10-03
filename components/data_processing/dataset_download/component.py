"""Dataset Download Component.

This component downloads datasets from multiple sources (HuggingFace, S3, HTTP, local/PVC),
validates the format, splits into train/eval, and saves to PVC as JSONL.

Supported URI schemes:
- hf://dataset_name or dataset_name - HuggingFace datasets
- s3://bucket/path/to/dataset.jsonl - AWS S3 datasets (JSONL format)
- http://... or https://... - HTTP/HTTPS URLs (e.g., MinIO shared links)
- pvc://path/to/dataset.jsonl or /absolute/path - Local/PVC file paths (JSONL format)

Supported dataset formats (controlled by dataset_format parameter):
- "chat" (default): Chat template format with messages/conversations containing role/content
- "tool_call": Tool-call format for GRPO training (single-turn or multi-turn traces)
"""

from kfp import dsl


@dsl.component(
    base_image="quay.io/opendatahub/odh-th06-cpu-torch291-py312:odh-3.4",
    packages_to_install=["datasets>=2.14.0", "huggingface-hub>=0.20.0", "s3fs>=2023.1.0"],
)
def dataset_download(
    train_dataset: dsl.Output[dsl.Dataset],
    eval_dataset: dsl.Output[dsl.Dataset],
    dataset_uri: str,
    pvc_mount_path: str,
    train_split_ratio: float = 0.9,  # 1.0 = no eval split (all data for training)
    subset_count: int = 0,
    dataset_format: str = "chat",
    shared_log_file: str = "pipeline_log.txt",
):
    """Download and prepare datasets from multiple sources.

    Validates dataset format based on the dataset_format parameter:
    - "chat": Chat template format (messages/conversations with role/content)
    - "tool_call": Tool-call format for GRPO training (single-turn or multi-turn traces)

    Args:
        train_dataset: Output artifact for training dataset (JSONL format)
        eval_dataset: Output artifact for evaluation dataset (JSONL format)
        dataset_uri: Dataset URI (hf://, s3://, https://, pvc:// or absolute path)
        pvc_mount_path: Path where the shared PVC is mounted
        train_split_ratio: Train/eval split (0.9 = 90%/10%, 1.0 = no split, all for training)
        subset_count: Number of examples to use (0 = use all)
        dataset_format: Validation format - "chat" (default) or "tool_call"
        shared_log_file: Name of the shared log file
    """
    import os

    from datasets import Dataset, load_dataset

    valid_formats = ("chat", "tool_call")
    if dataset_format not in valid_formats:
        raise ValueError(f"Unsupported dataset_format: '{dataset_format}'. Must be one of {valid_formats}.")

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
            return ("hf", uri[5:])
        elif uri.startswith("s3://"):
            return ("s3", uri[5:])
        elif uri.startswith("pvc://"):
            return ("local", uri[6:])
        elif uri.startswith("/"):
            return ("local", uri)
        else:
            # Default to HuggingFace if no scheme
            return ("hf", uri)

    def validate_chat_format_dataset(dataset: Dataset) -> bool:
        """Validate that dataset follows chat template format.

        Expected format:
        - Each entry should have 'messages' or 'conversations' field
        - Messages should be a list of dicts with 'role' and 'content'
        - Roles should be from: 'system', 'user', 'assistant', 'function', 'tool'
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        valid_roles = {"system", "user", "assistant", "function", "tool"}

        # Check first 100 examples (or fewer if dataset is smaller)
        num_to_check = min(100, len(dataset))

        for i in range(num_to_check):
            item = dataset[i]

            # Check for common chat format fields
            if "messages" in item:
                messages = item["messages"]
            elif "conversations" in item:
                messages = item["conversations"]
            else:
                raise ValueError(
                    f"Item {i} missing 'messages' or 'conversations' field. Found keys: {list(item.keys())}"
                )

            if not isinstance(messages, list):
                raise ValueError(f"Item {i}: messages must be a list")

            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    raise ValueError(f"Item {i}, message {j}: must be a dict")

                if "role" not in msg or "content" not in msg:
                    raise ValueError(f"Item {i}, message {j}: must have 'role' and 'content' fields")

                if msg["role"] not in valid_roles:
                    raise ValueError(
                        f"Item {i}, message {j}: invalid role '{msg['role']}'. Must be one of {valid_roles}"
                    )

        log_message(f"Dataset validated: {len(dataset)} examples in chat format")
        return True

    def _has_tool_calls_in_messages(messages: list) -> bool:
        """Check if any assistant message in the list contains tool_calls."""
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                return True
        return False

    # Field name aliases accepted for the single-turn target-tool indicator.
    # 'target_tool_name' is the canonical singular field; 'target_tools' is a
    # common plural/list variant used by some tool-calling datasets (e.g.,
    # Agent-Ark/Toucan-1.5M) to record one or more seed tools per example.
    _TARGET_TOOL_FIELDS = ("target_tool_name", "target_tools")

    def _get_target_tool_value(item: dict):
        """Return the first non-empty target-tool field value, checking known aliases."""
        for field in _TARGET_TOOL_FIELDS:
            value = item.get(field)
            if value:
                return value
        return None

    def validate_tool_call_format_dataset(dataset: Dataset) -> bool:
        """Validate that dataset follows tool-call format for GRPO training.

        Detects format from the first sample and validates all checked samples match:
        - Single-turn: each sample has a target-tool field (`target_tool_name` or
          `target_tools`) and a 'question' field
        - Multi-turn: each sample has 'messages' with at least one assistant tool_calls entry
        """
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        num_to_check = min(100, len(dataset))
        first = dataset[0]

        # Detect format from first sample (use .get() because Arrow-backed
        # datasets always contain all columns; missing values are None)
        is_single_turn = bool(_get_target_tool_value(first)) and bool(first.get("question"))
        first_messages = first.get("messages", None)
        is_multi_turn = (
            isinstance(first_messages, list) and len(first_messages) > 0 and _has_tool_calls_in_messages(first_messages)
        )

        if not is_single_turn and not is_multi_turn:
            raise ValueError(
                f"Item 0 does not match any supported tool-call format. "
                f"Expected either (1) single-turn with a target-tool field "
                f"({' or '.join(_TARGET_TOOL_FIELDS)}) and 'question' fields, "
                f"or (2) multi-turn with 'messages' containing assistant tool_calls. "
                f"Found keys: {list(first.keys())}"
            )

        if is_single_turn and is_multi_turn:
            log_message(
                "Item 0 matches both single-turn and multi-turn tool-call formats; using single-turn validation"
            )

        if is_single_turn:
            for i in range(num_to_check):
                item = dataset[i]
                if not _get_target_tool_value(item):
                    raise ValueError(
                        f"Item {i}: none of {_TARGET_TOOL_FIELDS} is present or non-empty. "
                        f"All samples must be single-turn format (detected from first sample)."
                    )
                if not item.get("question"):
                    raise ValueError(
                        f"Item {i}: 'question' is missing or empty. "
                        f"All samples must be single-turn format (detected from first sample)."
                    )
            log_message(
                f"Dataset validated: checked {num_to_check} of {len(dataset)} examples "
                f"in tool-call format (single-turn)"
            )
        else:
            for i in range(num_to_check):
                item = dataset[i]
                messages = item.get("messages", None)
                if not isinstance(messages, list) or len(messages) == 0:
                    raise ValueError(
                        f"Item {i}: 'messages' must be a non-empty list. "
                        f"All samples must be multi-turn format (detected from first sample)."
                    )
                if not _has_tool_calls_in_messages(messages):
                    raise ValueError(
                        f"Item {i}: no assistant message with 'tool_calls' found in 'messages'. "
                        f"All samples must be multi-turn format (detected from first sample)."
                    )
            log_message(
                f"Dataset validated: checked {num_to_check} of {len(dataset)} examples in tool-call format (multi-turn)"
            )

        return True

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
        load_kwargs = {
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
                    load_kwargs_no_split = {"path": ds_id}
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

        # Validate dataset format (dataset_format already validated at top of component)
        if dataset_format == "chat":
            log_message("Validating chat template format...")
            validate_chat_format_dataset(dataset)
        elif dataset_format == "tool_call":
            log_message("Validating tool-call format...")
            validate_tool_call_format_dataset(dataset)

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

        # Save datasets as JSONL files
        import json as _json

        def _write_jsonl(ds: Dataset, path: str):
            """Write dataset to JSONL, normalizing tool-call message fields.

            The 'messages' column can arrive in different shapes depending on
            the source dataset's Arrow schema:
            - A list of message dicts (already structured) - the common case.
            - A list where individual messages are JSON-encoded strings
              (datasets>=4.8 may type non-uniform message schemas this way).
            - The entire messages column typed as a single JSON-encoded string
              representing the whole array (e.g., some HF datasets store
              complex nested fields as raw JSON strings).
            This helper normalizes all of these to a list of dicts and strips
            None-valued fields added by Arrow schema unification.
            """
            with open(path, "w") as f:
                for row in ds:
                    row = dict(row)
                    if dataset_format == "tool_call" and "messages" in row and row["messages"]:
                        messages = row["messages"]
                        if isinstance(messages, str):
                            messages = _json.loads(messages)
                        row["messages"] = [
                            {k: v for k, v in (_json.loads(m) if isinstance(m, str) else m).items() if v is not None}
                            for m in messages
                        ]
                    f.write(_json.dumps(row) + "\n")

        log_message(f"Saving train dataset to {train_dataset.path}")
        _write_jsonl(train_ds, train_dataset.path)

        log_message(f"Saving eval dataset to {eval_dataset.path}")
        _write_jsonl(eval_ds, eval_dataset.path)

        # Also save to shared PVC for next pipeline step
        pvc_dataset_dir = os.path.join(pvc_mount_path, "datasets")
        os.makedirs(pvc_dataset_dir, exist_ok=True)

        pvc_train_path = os.path.join(pvc_dataset_dir, "train.jsonl")
        pvc_eval_path = os.path.join(pvc_dataset_dir, "eval.jsonl")

        log_message(f"Saving train dataset to PVC: {pvc_train_path}")
        _write_jsonl(train_ds, pvc_train_path)

        log_message(f"Saving eval dataset to PVC: {pvc_eval_path}")
        _write_jsonl(eval_ds, pvc_eval_path)

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
