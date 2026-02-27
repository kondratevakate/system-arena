"""Dataset loading from various formats."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from arena.core.event import Event, EventLog, EventType
from arena.core.types import Actor, ActorType, CaseId, Channel
from arena.datasets.manifest import WorkflowManifest


def load_event_log(
    path: str | Path,
    manifest: WorkflowManifest | None = None,
) -> EventLog:
    """
    Load an event log from a file.

    Supports:
    - Parquet (.parquet)
    - JSONL (.jsonl, .ndjson)
    - JSON (.json)

    Args:
        path: Path to the data file
        manifest: Optional manifest for field mapping

    Returns:
        Loaded EventLog
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return _load_parquet(path, manifest)
    elif suffix in (".jsonl", ".ndjson"):
        return _load_jsonl(path, manifest)
    elif suffix == ".json":
        return _load_json(path, manifest)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_parquet(
    path: Path,
    manifest: WorkflowManifest | None,
) -> EventLog:
    """Load event log from Parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    df = table.to_pandas()

    return _dataframe_to_event_log(df, manifest)


def _load_jsonl(
    path: Path,
    manifest: WorkflowManifest | None,
) -> EventLog:
    """Load event log from JSONL file."""
    events = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            event = _dict_to_event(data, manifest)
            events.append(event)

    return EventLog(events)


def _load_json(
    path: Path,
    manifest: WorkflowManifest | None,
) -> EventLog:
    """Load event log from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Support both array of events and {"events": [...]}
    if isinstance(data, list):
        events_data = data
    elif isinstance(data, dict) and "events" in data:
        events_data = data["events"]
    else:
        raise ValueError("JSON file must contain an array or {events: [...]}")

    events = [_dict_to_event(d, manifest) for d in events_data]
    return EventLog(events)


def _dataframe_to_event_log(
    df: Any,  # pandas DataFrame
    manifest: WorkflowManifest | None,
) -> EventLog:
    """Convert a DataFrame to an EventLog."""
    events = []

    # Get field mappings from manifest
    id_field = "case_id"
    timestamp_field = "timestamp"

    if manifest:
        id_field = manifest.case.id_field
        timestamp_field = manifest.case.timestamp_field

    for _, row in df.iterrows():
        event = _row_to_event(row.to_dict(), id_field, timestamp_field, manifest)
        events.append(event)

    return EventLog(events)


def _row_to_event(
    row: dict[str, Any],
    id_field: str,
    timestamp_field: str,
    manifest: WorkflowManifest | None,
) -> Event:
    """Convert a data row to an Event."""
    # Parse case ID
    case_id_raw = row.get(id_field)
    if case_id_raw is None:
        raise ValueError(f"Missing case ID field: {id_field}")

    case_id = _parse_case_id(case_id_raw)

    # Parse timestamp
    timestamp_raw = row.get(timestamp_field)
    if timestamp_raw is None:
        raise ValueError(f"Missing timestamp field: {timestamp_field}")

    timestamp = _parse_timestamp(timestamp_raw)

    # Parse event type
    event_type = _parse_event_type(row.get("event_type", "interaction.occurred"))

    # Parse actor
    actor = _parse_actor(row)

    # Parse channel
    channel = _parse_channel(row.get("channel"))

    # Build payload from remaining fields
    exclude_fields = {
        id_field,
        timestamp_field,
        "event_type",
        "event_id",
        "actor_type",
        "actor_id",
        "actor_name",
        "channel",
        "workflow_type",
        "sequence_number",
        "state_before",
        "state_after",
        "outcome_labels",
        "source_system",
        "correlation_id",
    }

    payload = {k: v for k, v in row.items() if k not in exclude_fields and v is not None}

    # Parse outcome labels
    outcome_labels = {}
    if "outcome_labels" in row and row["outcome_labels"]:
        if isinstance(row["outcome_labels"], dict):
            outcome_labels = row["outcome_labels"]
        elif isinstance(row["outcome_labels"], str):
            try:
                outcome_labels = json.loads(row["outcome_labels"])
            except json.JSONDecodeError:
                outcome_labels = {"label": row["outcome_labels"]}

    # Get optional fields
    event_id = row.get("event_id")
    if event_id:
        event_id = UUID(str(event_id))

    workflow_type = row.get("workflow_type", "")
    if manifest and not workflow_type:
        workflow_type = manifest.name

    return Event(
        case_id=case_id,
        event_id=event_id or None,
        timestamp=timestamp,
        sequence_number=row.get("sequence_number", 0),
        event_type=event_type,
        workflow_type=workflow_type,
        actor=actor,
        channel=channel,
        payload=payload,
        state_before=row.get("state_before"),
        state_after=row.get("state_after"),
        outcome_labels=outcome_labels,
        source_system=row.get("source_system"),
        correlation_id=UUID(str(row["correlation_id"])) if row.get("correlation_id") else None,
    )


def _dict_to_event(
    data: dict[str, Any],
    manifest: WorkflowManifest | None,
) -> Event:
    """Convert a dictionary to an Event."""
    id_field = "case_id"
    timestamp_field = "timestamp"

    if manifest:
        id_field = manifest.case.id_field
        timestamp_field = manifest.case.timestamp_field

    return _row_to_event(data, id_field, timestamp_field, manifest)


def _parse_case_id(value: Any) -> CaseId:
    """Parse a case ID from various formats."""
    if isinstance(value, UUID):
        return CaseId(value)
    if isinstance(value, str):
        try:
            return CaseId(UUID(value))
        except ValueError:
            # Use string hash as UUID for non-UUID case IDs
            import hashlib

            hash_bytes = hashlib.md5(value.encode()).digest()
            return CaseId(UUID(bytes=hash_bytes))
    if isinstance(value, int):
        # Convert int to UUID-like format
        return CaseId(UUID(int=value))

    raise ValueError(f"Cannot parse case ID: {value}")


def _parse_timestamp(value: Any) -> datetime:
    """Parse a timestamp from various formats."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Try ISO format first
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
        # Try common formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse timestamp: {value}")
    if isinstance(value, (int, float)):
        # Assume Unix timestamp
        return datetime.fromtimestamp(value)

    raise ValueError(f"Cannot parse timestamp: {value}")


def _parse_event_type(value: Any) -> EventType:
    """Parse an event type."""
    if isinstance(value, EventType):
        return value
    if isinstance(value, str):
        try:
            return EventType(value)
        except ValueError:
            # Map common variations
            mappings = {
                "created": EventType.ENTITY_CREATED,
                "state_changed": EventType.STATE_CHANGED,
                "interaction": EventType.INTERACTION_OCCURRED,
                "decision_requested": EventType.DECISION_REQUESTED,
                "decision_made": EventType.DECISION_EXECUTED,
                "outcome": EventType.OUTCOME_RECORDED,
                "timer": EventType.TIMER_ELAPSED,
                "escalation": EventType.ESCALATION_TRIGGERED,
            }
            if value.lower() in mappings:
                return mappings[value.lower()]
            # Default to interaction
            return EventType.INTERACTION_OCCURRED

    return EventType.INTERACTION_OCCURRED


def _parse_actor(data: dict[str, Any]) -> Actor:
    """Parse actor information from data."""
    actor_type_raw = data.get("actor_type", "system")
    actor_id = data.get("actor_id", "unknown")
    actor_name = data.get("actor_name")

    try:
        actor_type = ActorType(actor_type_raw)
    except ValueError:
        actor_type = ActorType.SYSTEM

    return Actor(
        actor_type=actor_type,
        actor_id=str(actor_id),
        name=actor_name,
    )


def _parse_channel(value: Any) -> Channel | None:
    """Parse a channel."""
    if value is None:
        return None
    if isinstance(value, Channel):
        return value
    if isinstance(value, str):
        try:
            return Channel(value.lower())
        except ValueError:
            return None
    return None


def load_from_directory(
    directory: str | Path,
    manifest_name: str = "manifest.yaml",
    data_pattern: str = "*.parquet",
) -> tuple[WorkflowManifest, EventLog]:
    """
    Load manifest and event log from a directory.

    Args:
        directory: Path to the workflow directory
        manifest_name: Name of the manifest file
        data_pattern: Glob pattern for data files

    Returns:
        Tuple of (manifest, event_log)
    """
    from arena.datasets.manifest import load_manifest

    directory = Path(directory)

    # Load manifest
    manifest_path = directory / manifest_name
    manifest = load_manifest(manifest_path)

    # Find and load data files
    data_files = list(directory.glob(data_pattern))

    if not data_files:
        raise FileNotFoundError(f"No data files matching {data_pattern} in {directory}")

    # Load and merge all data files
    all_events: list[Event] = []

    for data_file in data_files:
        event_log = load_event_log(data_file, manifest)
        all_events.extend(event_log.events)

    # Sort by timestamp
    all_events.sort(key=lambda e: e.timestamp)

    return manifest, EventLog(all_events)
