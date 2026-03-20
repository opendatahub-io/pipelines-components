import argparse
import logging
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import yaml
from semver import Version

# The following ordered fields are required in a metadata.yaml file.
REQUIRED_FIELDS = ["name", "stability", "dependencies", "lastVerified"]
# The following fields are optional in a metadata.yaml file.
OPTIONAL_FIELDS = ["tags", "ci", "links", "managed", "description"]
STABILITY_OPTIONS = ["experimental", "alpha", "beta", "stable"]
# 'Dependencies' must contain 'kubeflow' and can contain 'external_services'.
DEPENDENCIES_FIELDS = ["kubeflow", "external_services"]
# A given dependency must contain 'name' and 'version' fields.
DEPENDENCY_REQUIRED_FIELDS = ["name", "version"]
# Comparison operators for dependency versions.
COMPARISON = {">=", "<=", "=="}

OWNERS = "OWNERS"
METADATA = "metadata.yaml"


class ValidationError(Exception):
    """Custom exception for validation errors that should be displayed without traceback.

    This exception can take a custom message.
    """

    def __init__(self, message: str = "A validation error occurred."):
        """Initialize the ValidationError with a custom message.

        Args:
            message: The error message to display.
        """
        # Call the base class constructor with the message
        super().__init__(message)

        # Store the message in an attribute (optional, but good practice)
        self.message = message


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        argParse.Namespace: Parsed and validated arguments.
    """
    parser = argparse.ArgumentParser(
        description="Validate metadata schema for Kubeflow Pipelines pipelines/components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # For example, from project root:
  python -m scripts.validate_metadata --dir components/data_processing/sample_component
        """,
    )

    parser.add_argument(
        "--dir",
        type=validate_dir,
        required=True,
        help="Path to a component/pipeline directory or a subcategory containing multiple components/pipelines",
    )

    return parser.parse_args()


def validate_dir(path: str) -> Path:
    """Validate that the input path is a valid directory.

    Args:
        path: String representation of the path to the component, pipeline, or subcategory directory.

    Returns:
        Path: Validated Path object to the directory.

    Raises:
        argparse.ArgumentTypeError: If validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory '{path}' does not exist")

    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory")

    return path


def find_dirs_to_validate(input_dir: Path) -> list[Path]:
    """Find all directories that need validation (handles both components and subcategories).

    Args:
        input_dir: Path to a component/pipeline directory or a subcategory directory.

    Returns:
        List of Path objects to directories containing metadata.yaml files.

    Raises:
        argparse.ArgumentTypeError: If no valid directories are found.
    """
    # Check if this directory has metadata.yaml
    if (input_dir / METADATA).exists():
        return [input_dir]

    # This might be a subcategory - find subdirectories with metadata.yaml
    dirs_to_validate = []
    for subdir in input_dir.iterdir():
        if subdir.is_dir() and (subdir / METADATA).exists():
            dirs_to_validate.append(subdir)

    if not dirs_to_validate:
        raise argparse.ArgumentTypeError(
            f"'{input_dir}' does not contain a {METADATA} file and has no subdirectories with one. "
            f"If this is a subcategory, ensure it contains component directories."
        )

    return dirs_to_validate


def validate_owners_file(filepath: Path):
    """Validate that the OWNERS file contains at least one approver under the 'approvers' heading.

    Args:
        filepath: Path object representing the filepath to the OWNERS file.

    Raises:
        ValidationError: If filepath input is not a file, heading 'approvers:' is missing, or no approvers are listed.
    """
    if not filepath.is_file():
        raise ValidationError(f"{filepath} is not a valid filepath.")

    with open(filepath) as f:
        for line, next_line in pairwise(f):
            next_line = next_line.strip()
            if line.startswith("approvers:") and next_line.startswith("-") and len(next_line) > 2:
                logging.info(f"OWNERS file at {filepath} contains at least one approver under heading 'approvers:'.")
                return

    # If this line is reached, no approvers were found.
    raise ValidationError(f"OWNERS file at {filepath} requires 1+ approver under heading 'approvers:'.")


def validate_metadata_yaml(filepath: Path):
    """Validate that the input filepath represents a metadata.yaml file with a valid schema.

    Args:
        filepath: Path object representing the filepath to the metadata.yaml file.

    Raise:
        ValidationError: If 'lastVerified' empty, or validate_date_verified() or validate_required_fields() fails.
    """
    if not filepath.is_file():
        raise ValidationError(f"{filepath} is not a valid filepath.")
    with open(filepath) as f:
        metadata = yaml.safe_load(f)

        # Validate metadata.yaml has been verified within one year of the current date.
        if "lastVerified" not in metadata:
            raise ValidationError(
                f"Metadata at {filepath} has corresponding metadata.yaml with no 'lastVerified' value."
            )

        last_verified = metadata.get("lastVerified")
        if not validate_date_verified(last_verified):
            raise ValidationError(
                f"Metadata at {filepath} has corresponding metadata.yaml with invalid "
                f"'lastVerified' value: {last_verified}."
            )

        # Validate required fields and their corresponding values.
        validate_required_fields(metadata)


def validate_date_verified(last_verified: datetime) -> bool:
    """Validate that the input date is RFC-3339-formatted and within 1 year of the current date.

    Args:
        last_verified: Input datetime date to be validated.

    Returns:
        bool: True if the input date is valid, False otherwise.

    Examples:
        '2025-03-15T00:00:00Z'  -> True [As of November 2025]
        '2025-03-15'            -> False
        '2024-03-15T00:00:00Z'  -> False
    """
    # Validate input date formatting.
    if not isinstance(last_verified, datetime):
        logging.error(f"'lastVerified' should be format YYYY-MM-DDT00:00:00Z, but instead is: {last_verified}.")
        return False
    # Validate input date to be within 1 year of the current date.
    today = datetime.now(timezone.utc)
    delta = abs((today - last_verified).days)
    if delta >= 365:
        logging.error(f"'lastVerified' should be within 1 year of current date, but is {delta} days over.")
        return False
    return True


def validate_required_fields(metadata: dict):
    """Validates that all required fields are present in the input dictionary and have valid values.

    Also validates that no invalid fields are present.

    Args:
        metadata: dictionary object containing nested metadata fields.

    Raises:
        ValidationError: If validation fails.
    """
    # Convert metadata keys to a list for comparison purposes.
    input_metadata_fields = list(metadata.keys())
    # Optional fields should not be validated against required fields. Remove optional fields for this check.
    for field in OPTIONAL_FIELDS:
        if field in input_metadata_fields:
            input_metadata_fields.remove(field)

    # Retrieve name from metadata.
    name = metadata.get("name")
    if name is None:
        raise ValidationError(f"Missing required field 'name' in {METADATA}.")
    if not isinstance(name, str):
        type_name = type(name).__name__
        raise ValidationError(
            f"{type_name} value identified in field 'name' in {METADATA}: '{name}'. Value for 'name' must be string."
        )

    # Convert metadata keys to a set and compare against REQUIRED_FIELDS set.
    input_fields_set = set(input_metadata_fields)
    required_fields_set = set(REQUIRED_FIELDS)
    if required_fields_set != input_fields_set:
        missing_fields = required_fields_set - input_fields_set
        if len(missing_fields) > 0:
            raise ValidationError(f"Missing required field(s) in {METADATA} for '{name}': {missing_fields}.")
        extra_fields = input_fields_set - required_fields_set
        if len(extra_fields) > 0:
            raise ValidationError(f"Unexpected field(s) in {METADATA} for '{name}': {extra_fields}.")
    # Compare input fields against REQUIRED FIELDS as lists to verify elements are ordered correctly.
    if list(input_metadata_fields) != REQUIRED_FIELDS:
        raise ValidationError(
            f"Field(s) located incorrectly in {METADATA} for '{name}'. Expected order is {REQUIRED_FIELDS}."
        )

    # Validate field values.
    for field in metadata:
        value_type = type(metadata.get(field)).__name__

        if field == "stability":
            stability_val = metadata.get("stability")
            if stability_val not in STABILITY_OPTIONS:
                raise ValidationError(
                    f"Invalid 'stability' value in {METADATA} for '{name}': '{stability_val}'. "
                    f"Expected one of: {STABILITY_OPTIONS}."
                )

        elif field == "dependencies":
            # Dependencies should be a dictionary.
            dependency_val = metadata.get("dependencies")
            if not isinstance(dependency_val, dict):
                raise ValidationError(
                    f"{value_type} value identified for field 'dependencies' in {METADATA} "
                    f"for '{name}'. Value must be array."
                )
            dependency_types = set(dependency_val.keys())

            # Dependencies should contain 'kubeflow' and can contain 'external_services'.
            if not (dependency_types == {"kubeflow"} or dependency_types == {"kubeflow", "external_services"}):
                raise ValidationError(
                    f"The following field(s) were found in dependencies: {list(dependency_val.keys())}. "
                    f"Expected {DEPENDENCIES_FIELDS}."
                )

            # Kubeflow Pipelines is a required dependency.
            kf_dependencies = dependency_val.get("kubeflow")
            ext_dependencies = dependency_val.get("external_services")
            if not isinstance(kf_dependencies, list) or (
                ext_dependencies is not None and not isinstance(ext_dependencies, list)
            ):
                raise ValidationError(
                    f"Dependency sub-types for '{name}' should contain lists but instead are "
                    f"{type(kf_dependencies)} and {type(ext_dependencies)}."
                )
            kfp_present = any(d.get("name") == "Pipelines" for d in kf_dependencies)
            if not kfp_present:
                raise ValidationError(f"{METADATA} for '{name}' is missing Kubeflow Pipelines dependency.")

            for dependency_type in [kf_dependencies, ext_dependencies]:
                if dependency_type is None:
                    continue
                for dependency in dependency_type:
                    for field in DEPENDENCY_REQUIRED_FIELDS:
                        if field not in dependency:
                            raise ValidationError(f"Missing required field '{field}' in dependency: {dependency}.")

            # Dependency versions must be correctly formatted by semantic versioning.
            invalid_dependencies = get_invalid_versions(kf_dependencies) + get_invalid_versions(ext_dependencies)
            if len(invalid_dependencies) > 0:
                raise ValidationError(
                    f"{METADATA} for '{name}' contains one or more dependencies with invalid "
                    f"semantic versioning: {invalid_dependencies}."
                )

        elif field == "tags":
            tags_val = metadata.get("tags")
            if not (isinstance(tags_val, list)):
                raise ValidationError(
                    f"{value_type} value identified in field 'tags' in {METADATA} for '{name}'. "
                    f"Value must be string array."
                )
            if not all(isinstance(item, str) for item in tags_val):
                raise ValidationError(
                    f"The following tags in {METADATA} for '{name}': {tags_val}. Expected an array of scalar strings."
                )
        elif field == "ci":
            ci_val = metadata.get("ci")
            if not isinstance(ci_val, dict):
                raise ValidationError(
                    f"{value_type} value identified for field 'ci' in {METADATA} for '{name}'. "
                    f"Value must be dictionary."
                )
            keys = set(ci_val.keys())
            if not (keys == {"skip_dependency_probe"}):
                raise ValidationError(
                    f"The following field(s) were found in field 'ci' in {METADATA} for '{name}': "
                    f"{list(ci_val.keys())}. Only field 'skip_dependency_probe' is valid."
                )
            probe = ci_val.get("skip_dependency_probe")
            if probe is not None and not isinstance(probe, bool):
                raise ValidationError(
                    f"{METADATA} expects a boolean value for skip_dependency_probe but "
                    f"{type(probe).__name__} value provided: '{probe}'."
                )

        elif field == "links":
            links_value = metadata.get("links")
            if not isinstance(links_value, dict):
                raise ValidationError(
                    f"{value_type} value identified in field 'links' in {METADATA} for '{name}'. "
                    f"Value must be dictionary."
                )

        elif field == "managed":
            managed_val = metadata.get("managed")
            if not isinstance(managed_val, bool):
                raise ValidationError(
                    f"{value_type} value identified in field 'managed' in {METADATA} for '{name}'. "
                    f"Value must be boolean (true/false)."
                )

        elif field == "description":
            desc_val = metadata.get("description")
            if not isinstance(desc_val, str):
                raise ValidationError(
                    f"{value_type} value identified in field 'description' in {METADATA} for '{name}'. "
                    f"Value must be string."
                )


def get_invalid_versions(dependencies: list[dict]) -> list[dict]:
    """Return a list of the input dependencies that contain invalid semantic versioning.

    Args:
        dependencies: list[dict] of dependencies to be validated

    Return:
        dependencies: list[dict] of invalid dependencies
    """
    if dependencies is None:
        return []
    invalid: list[dict] = []
    for dependency in dependencies:
        version = dependency.get("version")
        # If the dependency version is null or non-string, it is invalid.
        if version is None or not isinstance(version, str):
            invalid.append(dependency)
            continue
        # Strip leading '==', '>=' or '<=' from dependency version, if applicable.
        if len(version) > 1 and version[:2] in COMPARISON:
            version = version[2:]
        if not Version.is_valid(version):
            invalid.append(dependency)
    return invalid


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    input_dir = args.dir

    # Find all directories to validate (handles subcategories)
    try:
        dirs_to_validate = find_dirs_to_validate(input_dir)
    except argparse.ArgumentTypeError as e:
        logging.error("Error: %s", e)
        sys.exit(1)

    has_errors = False

    # If input_dir is a subcategory (no metadata.yaml), validate its OWNERS file
    if not (input_dir / METADATA).exists():
        subcategory_owners = input_dir / OWNERS
        if subcategory_owners.is_file():
            try:
                validate_owners_file(subcategory_owners)
            except ValidationError as e:
                logging.error("Validation Error: %s", e)
                has_errors = True
        else:
            logging.error(
                "Subcategory directory '%s' is missing a required %s file.",
                input_dir,
                OWNERS,
            )
            has_errors = True

    for dir_path in dirs_to_validate:
        print(f"Validating {dir_path}...")
        dir_has_errors = False

        # Validate OWNERS
        try:
            owners_file_path = dir_path / OWNERS
            validate_owners_file(owners_file_path)
        except ValidationError as e:
            logging.error("Validation Error: %s", e)
            dir_has_errors = True

        # Validate metadata.yaml
        try:
            metadata_file_path = dir_path / METADATA
            validate_metadata_yaml(metadata_file_path)
        except ValidationError as e:
            logging.error("Validation Error: %s", e)
            dir_has_errors = True

        if dir_has_errors:
            has_errors = True
        else:
            logging.info(f"Validation successful for {dir_path}.")
            print(f"Validation successful for {dir_path}.")

    if has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
