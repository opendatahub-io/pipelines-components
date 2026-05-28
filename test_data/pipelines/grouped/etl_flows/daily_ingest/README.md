# Daily Ingest ✨

## Overview 🧾

Runs a daily data ingestion workflow.

This pipeline orchestrates the daily ingestion of data from an external source into the data lake.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `source_path` | `str` | `None` | Path to the data source. |
| `batch_size` | `int` | `100` | Number of records per batch. |

## Metadata 🗂️

- **Name**: Daily Ingest
- **Description**: A daily data ingestion pipeline
- **Documentation**: https://example.com/daily-ingest
- **Tags**:
  - testing
  - subcategory
- **Owners**:
  - Approvers:
    - nsingla
    - hbelmiro
  - Reviewers:
    - nsingla
