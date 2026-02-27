# System Arena

A benchmark and development framework for case-based agents.

Replay historical decision points, compare agent policies against baselines, and evaluate operational outcomes before production deployment.

## Installation

```bash
pip install system-arena

# With process mining integrations
pip install system-arena[integrations]
```

## Core Concepts

- **Event Log** — Canonical schema for process events (entity.created, state.changed, interaction.occurred, etc.)
- **Decision Point** — A moment where a policy could intervene
- **Policy** — Interface for agent decision-making (threshold, rule-based, or custom)
- **Replay** — Offline evaluation with no-future-leakage guarantee
- **Metrics** — Precision@K, conversion, policy violations

## Usage

Each workflow/use-case should be a separate repository that depends on `system-arena`.

### Workflow Repository Structure

```
my-workflow/
├── manifest.yaml          # Workflow definition
├── data/
│   └── events.parquet     # Historical event log
├── policies/
│   └── my_policy.py       # Custom policies
└── analysis/
    └── notebooks/         # Analysis notebooks
```

### Example: manifest.yaml

```yaml
name: my_workflow
version: "1.0"

case:
  id_field: lead_id
  timestamp_field: event_timestamp

decision_points:
  - name: followup_due
    trigger: "timer.elapsed"
    condition: "days_since_last_contact > 3"

allowed_actions:
  - action: wait
  - action: send_reminder
  - action: escalate

outcomes:
  positive: [converted, meeting_booked]
  negative: [went_cold, lost]
```

### Example: Running a Benchmark

```python
from arena.datasets import load_event_log, load_manifest
from arena.replay import BenchmarkRunner, BenchmarkConfig
from arena.policy import ThresholdPolicy, ThresholdConfig

# Load data
manifest = load_manifest("manifest.yaml")
event_log = load_event_log("data/events.parquet", manifest)

# Configure policy
policy = ThresholdPolicy(
    thresholds=[
        ThresholdConfig(
            field="days_since_last_contact",
            threshold=3.0,
            action_above="send_reminder",
            action_below="wait",
        )
    ]
)

# Run benchmark
config = BenchmarkConfig(
    decision_rules=manifest.get_decision_rules(),
    constraints=manifest.get_constraints(),
)
runner = BenchmarkRunner(config)
result = runner.run(event_log, policy)

# View results
print(f"Decisions: {result.total_decisions}")
print(f"Violations: {result.total_violations}")
```

## Object-Centric Process Mining

Events can reference multiple objects (not just a single case_id):

```python
from arena.core.types import ObjectRef

event = Event(
    case_id=lead_id,
    event_type=EventType.INTERACTION_OCCURRED,
    object_refs=(
        ObjectRef("lead", "lead_123"),
        ObjectRef("clinic", "clinic_456", role="target"),
        ObjectRef("assistant", "asst_789", role="owner"),
    ),
)

# Filter by object
clinic_events = event_log.filter_by_object("clinic", "clinic_456")
```

## Integrations

### PM4Py (Process Discovery)

```python
from arena.integrations import export_to_pm4py, discover_process

# Export for PM4Py analysis
pm4py_log = export_to_pm4py(event_log)

# Discover process model
net, im, fm = discover_process(event_log, algorithm="inductive")
```

### Variant Analysis

```python
from arena.integrations import compare_variants

# Compare successful vs unsuccessful trajectories
comparison = compare_variants(event_log)
print(comparison.summary())
```

### Performance Analysis

```python
from arena.integrations import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
report = analyzer.analyze(event_log)
print(report.summary())
```

## Modules

| Module | Purpose |
|--------|---------|
| `arena.core` | Event, EventLog, CaseSnapshot, ObjectRef |
| `arena.decision` | DecisionPoint extraction, labeling |
| `arena.policy` | Policy interface, baselines (Random, Rule, Threshold) |
| `arena.replay` | BenchmarkRunner, no-future-leakage timeline |
| `arena.eval` | Metrics, comparison, reports |
| `arena.datasets` | Parquet/JSONL loading, manifest parsing |
| `arena.integrations` | PM4Py, variant analysis, performance overlays |
| `arena.cli` | Command-line interface |

## License

MIT
