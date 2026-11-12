"""Download and validate the ERP CORE N170 dataset with MOABB."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import mne
import moabb
from moabb.datasets import ErpCore2021_N170


DEFAULT_DESTINATION = Path("data/ERP_CORE_N170")
DATASET_DIRECTORY = "MNE-erpcore2021-data"
MANIFEST_FILENAME = "erpcore_manifest.csv"
TASK_NAME = "N170"
ALL_SUBJECTS = tuple(range(1, 41))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def subject_directory(destination: Path, subject: int) -> Path:
    return destination / DATASET_DIRECTORY / f"sub-{subject:03d}"


def required_n170_paths(destination: Path, subject: int) -> list[Path]:
    dataset_root = destination / DATASET_DIRECTORY
    manifest_path = dataset_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return []

    subject_label = f"sub-{subject:03d}"
    required_paths = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            if row["component"].upper() != TASK_NAME:
                continue
            participant_id = row["participant_id"].strip()
            if participant_id not in ("", subject_label):
                continue
            relative_path = row["local_path"].replace(
                f"erpcore/{TASK_NAME}/", "", 1
            )
            required_paths.append(dataset_root / Path(relative_path))
    return required_paths


def subject_has_n170_data(destination: Path, subject: int) -> bool:
    required_paths = required_n170_paths(destination, subject)
    return bool(required_paths) and all(
        path.is_file() and path.stat().st_size > 0 for path in required_paths
    )


def summarize_files(destination: Path) -> tuple[int, int]:
    files = [path for path in destination.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERP CORE N170 data for selected participants."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Directory that will contain the MOABB dataset cache.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(ALL_SUBJECTS),
        help="Participant numbers to download; defaults to all 40 participants.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum download attempts per participant; defaults to 5.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=15.0,
        help="Initial retry delay, doubled after each failure; defaults to 15 seconds.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> tuple[int, Path]:
    destination = args.destination.resolve()
    subjects = tuple(dict.fromkeys(args.subjects))
    invalid_subjects = sorted(set(subjects) - set(ALL_SUBJECTS))
    if invalid_subjects:
        raise ValueError(f"Invalid participant numbers: {invalid_subjects}")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative.")

    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "download_status.json"
    log_path = destination / "download.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    started_at = utc_now()
    completed_subjects = [
        subject for subject in subjects if subject_has_n170_data(destination, subject)
    ]
    status = {
        "state": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "destination": str(destination),
        "requested_subjects": list(subjects),
        "completed_subjects": completed_subjects,
        "current_subject": None,
        "moabb_version": moabb.__version__,
        "mne_version": mne.__version__,
        "python_version": sys.version.split()[0],
        "download_provider": "upstream",
    }
    write_status(status_path, status)

    try:
        moabb.set_download_provider("upstream")
        dataset = ErpCore2021_N170()

        for subject in subjects:
            if subject in completed_subjects:
                logging.info("Participant %03d is already complete; skipping.", subject)
                continue

            status["current_subject"] = subject
            status["updated_at"] = utc_now()
            write_status(status_path, status)
            logging.info("Downloading participant %03d of %d.", subject, len(subjects))

            for attempt in range(1, args.max_attempts + 1):
                status["attempt"] = attempt
                status["max_attempts"] = args.max_attempts
                status["updated_at"] = utc_now()
                write_status(status_path, status)
                try:
                    dataset.download(
                        subject_list=[subject],
                        path=str(destination),
                        force_update=False,
                        update_path=False,
                        accept=True,
                    )
                    break
                except Exception as exc:
                    if attempt == args.max_attempts:
                        raise
                    delay = args.retry_delay_seconds * (2 ** (attempt - 1))
                    status["last_retry_error"] = repr(exc)
                    status["next_retry_delay_seconds"] = delay
                    status["updated_at"] = utc_now()
                    write_status(status_path, status)
                    logging.warning(
                        "Participant %03d attempt %d/%d failed; retrying in %.1f seconds: %s",
                        subject,
                        attempt,
                        args.max_attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)

            if not subject_has_n170_data(destination, subject):
                raise RuntimeError(
                    f"Participant {subject:03d} finished without detectable N170 files."
                )

            completed_subjects.append(subject)
            status["completed_subjects"] = completed_subjects.copy()
            status["updated_at"] = utc_now()
            write_status(status_path, status)
            logging.info("Participant %03d validated.", subject)

        missing_subjects = [
            subject
            for subject in subjects
            if not subject_has_n170_data(destination, subject)
        ]
        if missing_subjects:
            raise RuntimeError(f"Missing N170 data for participants: {missing_subjects}")

        file_count, total_bytes = summarize_files(destination)
        status.update(
            {
                "state": "completed",
                "completed_at": utc_now(),
                "updated_at": utc_now(),
                "current_subject": None,
                "completed_subjects": list(subjects),
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
        )
        write_status(status_path, status)
        logging.info(
            "Download complete: %d files, %.3f GiB.",
            file_count,
            total_bytes / (1024**3),
        )
        return 0, status_path
    except Exception as exc:
        status.update(
            {
                "state": "failed",
                "failed_at": utc_now(),
                "updated_at": utc_now(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_status(status_path, status)
        logging.exception("ERP CORE N170 download failed.")
        return 1, status_path


if __name__ == "__main__":
    parsed_args = parse_args()
    exit_code, _ = main(parsed_args)
    raise SystemExit(exit_code)
