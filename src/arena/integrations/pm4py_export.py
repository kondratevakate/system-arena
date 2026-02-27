"""PM4Py integration for process discovery and conformance checking.

This module provides helpers to export arena EventLogs to PM4Py-compatible
formats for process discovery, conformance checking, and visualization.

Usage:
    from arena.integrations.pm4py_export import export_to_pm4py, discover_process

    # Export to PM4Py event log
    pm4py_log = export_to_pm4py(event_log)

    # Discover process model
    net, im, fm = discover_process(event_log, algorithm="inductive")

    # Check conformance
    results = check_conformance(event_log, net, im, fm)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING

from arena.core.event import Event, EventLog, EventType
from arena.core.types import CaseId

if TYPE_CHECKING:
    import pandas as pd


# PM4Py column name constants
CASE_ID_KEY = "case:concept:name"
ACTIVITY_KEY = "concept:name"
TIMESTAMP_KEY = "time:timestamp"
RESOURCE_KEY = "org:resource"


def to_pm4py_dataframe(
    event_log: EventLog,
    activity_mapper: dict[EventType, str] | None = None,
    include_payload: bool = True,
) -> "pd.DataFrame":
    """
    Convert an arena EventLog to a PM4Py-compatible DataFrame.

    PM4Py expects specific column names:
    - case:concept:name - case identifier
    - concept:name - activity name
    - time:timestamp - event timestamp
    - org:resource - resource/actor (optional)

    Args:
        event_log: The arena EventLog to convert
        activity_mapper: Optional mapping from EventType to activity names.
                        If not provided, uses event_type + payload hints.
        include_payload: Whether to include payload fields as additional columns

    Returns:
        pandas DataFrame in PM4Py format
    """
    import pandas as pd

    records = []

    for event in event_log.sorted_by_time():
        record = {
            CASE_ID_KEY: str(event.case_id),
            ACTIVITY_KEY: _get_activity_name(event, activity_mapper),
            TIMESTAMP_KEY: event.timestamp,
            RESOURCE_KEY: event.actor.actor_id,
            # Additional useful fields
            "arena:event_type": event.event_type.value,
            "arena:channel": event.channel.value if event.channel else None,
            "arena:workflow_type": event.workflow_type,
            "arena:sequence_number": event.sequence_number,
        }

        # Add object references
        for i, ref in enumerate(event.object_refs):
            record[f"arena:object_{i}_type"] = ref.object_type
            record[f"arena:object_{i}_id"] = ref.object_id

        # Add payload fields
        if include_payload:
            for key, value in event.payload.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    record[f"payload:{key}"] = value

        records.append(record)

    df = pd.DataFrame(records)

    # Ensure timestamp is datetime
    df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY])

    return df


def export_to_pm4py(
    event_log: EventLog,
    activity_mapper: dict[EventType, str] | None = None,
) -> Any:
    """
    Export an arena EventLog to a PM4Py EventLog object.

    This enables direct use with PM4Py algorithms for:
    - Process discovery (Alpha, Heuristics, Inductive miner)
    - Conformance checking
    - Process enhancement
    - Visualization

    Args:
        event_log: The arena EventLog to convert
        activity_mapper: Optional mapping from EventType to activity names

    Returns:
        pm4py EventLog object

    Raises:
        ImportError: If pm4py is not installed
    """
    try:
        import pm4py
        from pm4py.objects.conversion.log import converter as log_converter
    except ImportError:
        raise ImportError(
            "pm4py is required for this function. "
            "Install it with: pip install pm4py"
        )

    df = to_pm4py_dataframe(event_log, activity_mapper)

    # Convert DataFrame to PM4Py EventLog
    pm4py_log = log_converter.apply(
        df,
        parameters={
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: CASE_ID_KEY,
        },
    )

    return pm4py_log


def discover_process(
    event_log: EventLog,
    algorithm: str = "inductive",
    activity_mapper: dict[EventType, str] | None = None,
) -> tuple[Any, Any, Any]:
    """
    Discover a process model from an event log.

    Uses PM4Py process discovery algorithms to create a Petri net
    model from the historical event data.

    Args:
        event_log: The arena EventLog to analyze
        algorithm: Discovery algorithm to use:
                  - "inductive" (recommended, most robust)
                  - "alpha" (classic, simple processes)
                  - "heuristics" (handles noise well)
        activity_mapper: Optional mapping from EventType to activity names

    Returns:
        Tuple of (net, initial_marking, final_marking) - Petri net model

    Raises:
        ImportError: If pm4py is not installed
        ValueError: If unknown algorithm specified
    """
    try:
        import pm4py
    except ImportError:
        raise ImportError(
            "pm4py is required for process discovery. "
            "Install it with: pip install pm4py"
        )

    pm4py_log = export_to_pm4py(event_log, activity_mapper)

    if algorithm == "inductive":
        net, im, fm = pm4py.discover_petri_net_inductive(pm4py_log)
    elif algorithm == "alpha":
        net, im, fm = pm4py.discover_petri_net_alpha(pm4py_log)
    elif algorithm == "heuristics":
        net, im, fm = pm4py.discover_petri_net_heuristics(pm4py_log)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Use 'inductive', 'alpha', or 'heuristics'."
        )

    return net, im, fm


def check_conformance(
    event_log: EventLog,
    net: Any,
    initial_marking: Any,
    final_marking: Any,
    activity_mapper: dict[EventType, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Check conformance of event log against a process model.

    Compares actual process executions against an expected model
    to identify deviations.

    Args:
        event_log: The arena EventLog to check
        net: Petri net model
        initial_marking: Initial marking of the Petri net
        final_marking: Final marking of the Petri net
        activity_mapper: Optional mapping from EventType to activity names

    Returns:
        List of conformance results per trace, including:
        - fitness: How well the trace fits the model (0-1)
        - deviations: List of specific deviations

    Raises:
        ImportError: If pm4py is not installed
    """
    try:
        import pm4py
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    except ImportError:
        raise ImportError(
            "pm4py is required for conformance checking. "
            "Install it with: pip install pm4py"
        )

    pm4py_log = export_to_pm4py(event_log, activity_mapper)

    # Run token-based replay
    replayed_traces = token_replay.apply(
        pm4py_log,
        net,
        initial_marking,
        final_marking,
    )

    return replayed_traces


def visualize_process(
    event_log: EventLog,
    algorithm: str = "inductive",
    activity_mapper: dict[EventType, str] | None = None,
    output_path: str | None = None,
) -> Any:
    """
    Discover and visualize a process model.

    Args:
        event_log: The arena EventLog to analyze
        algorithm: Discovery algorithm to use
        activity_mapper: Optional mapping from EventType to activity names
        output_path: Optional path to save the visualization (PNG/SVG)

    Returns:
        Graphviz visualization object
    """
    try:
        import pm4py
    except ImportError:
        raise ImportError(
            "pm4py is required for visualization. "
            "Install it with: pip install pm4py"
        )

    net, im, fm = discover_process(event_log, algorithm, activity_mapper)

    # Create visualization
    gviz = pm4py.visualization.petri_net.visualizer.apply(net, im, fm)

    if output_path:
        pm4py.visualization.petri_net.visualizer.save(gviz, output_path)

    return gviz


def _get_activity_name(
    event: Event,
    mapper: dict[EventType, str] | None,
) -> str:
    """
    Get a human-readable activity name for an event.

    Uses mapper if provided, otherwise derives from event type and payload.
    """
    if mapper and event.event_type in mapper:
        return mapper[event.event_type]

    # Default: combine event type with key payload fields
    base = event.event_type.value

    # Add context from payload
    if event.event_type == EventType.INTERACTION_OCCURRED:
        direction = event.payload.get("direction", "")
        channel = event.channel.value if event.channel else ""
        if direction and channel:
            return f"{direction}_{channel}"
        elif direction:
            return f"interaction_{direction}"

    elif event.event_type == EventType.STATE_CHANGED:
        new_state = event.payload.get("new_state", "")
        if new_state:
            return f"state_to_{new_state}"

    elif event.event_type == EventType.OUTCOME_RECORDED:
        outcome = event.payload.get("outcome", "")
        if outcome:
            return f"outcome_{outcome}"

    return base


# Default activity mappings for common workflows
SALES_ACTIVITY_MAPPER: dict[EventType, str] = {
    EventType.ENTITY_CREATED: "Lead Created",
    EventType.INTERACTION_OCCURRED: "Contact Attempt",
    EventType.STATE_CHANGED: "Status Update",
    EventType.DECISION_REQUESTED: "Decision Required",
    EventType.DECISION_EXECUTED: "Action Taken",
    EventType.OUTCOME_RECORDED: "Outcome Recorded",
    EventType.TIMER_ELAPSED: "Follow-up Due",
    EventType.ESCALATION_TRIGGERED: "Escalated",
}

CLINICAL_ACTIVITY_MAPPER: dict[EventType, str] = {
    EventType.ENTITY_CREATED: "Episode Started",
    EventType.INTERACTION_OCCURRED: "Patient Contact",
    EventType.STATE_CHANGED: "Status Change",
    EventType.DECISION_REQUESTED: "Clinical Decision Needed",
    EventType.DECISION_EXECUTED: "Intervention Made",
    EventType.OUTCOME_RECORDED: "Outcome Documented",
    EventType.TIMER_ELAPSED: "Follow-up Due",
    EventType.ESCALATION_TRIGGERED: "Escalated to Physician",
}
