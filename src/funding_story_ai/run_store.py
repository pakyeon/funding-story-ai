from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any


class IdempotencyConflict(RuntimeError):
    pass


class RunNotFound(KeyError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class LocalRunStore:
    """Thread-safe local run store with caller-scoped idempotency."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index_path = root / "idempotency-index.json"
        self._lock = threading.Lock()

    def _read_index(self) -> dict[str, dict[str, str]]:
        if not self.index_path.exists():
            return {}
        value = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Run store index must be a JSON object")
        return value

    def _write_json_atomic(self, path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def begin(
        self,
        *,
        caller_id: str,
        idempotency_key: str,
        request_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        request_hash = _canonical_hash(request_payload)
        index_key = sha256(f"{caller_id}\0{idempotency_key}".encode()).hexdigest()
        with self._lock:
            index = self._read_index()
            existing = index.get(index_key)
            if existing:
                if existing["request_hash"] != request_hash:
                    raise IdempotencyConflict(
                        "The idempotency key was already used with a different request"
                    )
                record = self.get(existing["run_id"])
                return record, False

            run_id = f"run-{uuid.uuid4()}"
            now = _now()
            record = {
                "run_id": run_id,
                "status": "running",
                "caller_id": caller_id,
                "idempotency_key_sha256": sha256(idempotency_key.encode()).hexdigest(),
                "request_hash": request_hash,
                "result_uri": f"story://runs/{run_id}",
                "created_at": now,
                "updated_at": now,
                "result": None,
                "error_type": None,
            }
            self._write_json_atomic(self.root / f"{run_id}.json", record)
            index[index_key] = {"run_id": run_id, "request_hash": request_hash}
            self._write_json_atomic(self.index_path, index)
            return record, True

    def complete(self, run_id: str, result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = self.get(run_id)
            record.update(
                {
                    "status": "completed",
                    "updated_at": _now(),
                    "result": result,
                    "error_type": None,
                }
            )
            self._write_json_atomic(self.root / f"{run_id}.json", record)
            return record

    def fail(self, run_id: str, error: Exception) -> dict[str, Any]:
        with self._lock:
            record = self.get(run_id)
            record.update(
                {
                    "status": "failed",
                    "updated_at": _now(),
                    "result": None,
                    "error_type": type(error).__name__,
                }
            )
            self._write_json_atomic(self.root / f"{run_id}.json", record)
            return record

    def get(self, run_id: str) -> dict[str, Any]:
        if not run_id.startswith("run-"):
            raise RunNotFound(run_id)
        path = self.root / f"{run_id}.json"
        if not path.is_file():
            raise RunNotFound(run_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Invalid run record: {run_id}")
        return value
