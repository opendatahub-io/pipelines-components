#!/usr/bin/env bash
# Regenerate requirements.txt and requirements-build.txt from pyproject.toml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# RHEL UBI9 (glibc 2.34)
PYTHON_PLATFORM="${PYTHON_PLATFORM:-x86_64-manylinux_2_34}"
AIPCC_INDEX_URL="${AIPCC_INDEX_URL:-https://packages.redhat.com/api/pypi/public-rhai/cpu-torch2.13-el9.8-test/simple}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

while [ $# -gt 0 ]; do
  case "$1" in
    --index-url) AIPCC_INDEX_URL="$2"; shift 2 ;;
    --python-platform) PYTHON_PLATFORM="$2"; shift 2 ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    *) echo "Usage: $0 [--index-url URL] [--python-platform PLATFORM] [--python-version VERSION]" >&2; exit 1 ;;
  esac
done

echo "--index-url ${AIPCC_INDEX_URL}" > requirements.txt
echo "" >> requirements.txt
uv pip compile pyproject.toml --generate-hashes --no-header --no-annotate \
  --no-emit-package kfp-components \
  --python-version "${PYTHON_VERSION}" \
  --python-platform "${PYTHON_PLATFORM}" \
  --index-url "${AIPCC_INDEX_URL}" >> requirements.txt

echo "--index-url ${AIPCC_INDEX_URL}" > requirements-build.txt
echo "" >> requirements-build.txt
printf 'pip>=26.1.2\nsetuptools\nwheel\n' | uv pip compile --generate-hashes --no-header --no-annotate \
  --python-version "${PYTHON_VERSION}" \
  --python-platform "${PYTHON_PLATFORM}" \
  --index-url "${AIPCC_INDEX_URL}" - >> requirements-build.txt
