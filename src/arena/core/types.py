"""Shared type definitions for the arena framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, NewType
from uuid import UUID

# Type aliases
CaseId = NewType("CaseId", UUID)


# =============================================================================
# Object-Centric Process Mining Support
# =============================================================================
# Based on van der Aalst's object-centric process mining research:
# Events can relate to multiple objects, not just a single case_id.
# This prevents information loss in multi-entity processes like:
# - Medical tourism: lead + clinic + patient + assistant
# - Patient care: patient + screening episode + appointment
# - Sales: lead + contact + opportunity + account


@dataclass(frozen=True)
class ObjectRef:
    """
    Reference to a related object in the process.

    In object-centric process mining, an event can relate to multiple
    objects of different types. This captures those relationships.

    Examples:
    - ObjectRef(object_type="lead", object_id="abc123")
    - ObjectRef(object_type="clinic", object_id="clinic_456")
    - ObjectRef(object_type="patient", object_id="patient_789")
    """

    object_type: str  # e.g., "lead", "clinic", "patient", "opportunity"
    object_id: str  # ID within that object type
    role: str = ""  # Optional role: "primary", "target", "owner", etc.

    def __str__(self) -> str:
        if self.role:
            return f"{self.object_type}:{self.object_id}[{self.role}]"
        return f"{self.object_type}:{self.object_id}"


@dataclass
class ObjectRelation:
    """
    A relation between two objects in the process.

    Captures structural relationships like:
    - lead -> clinic (assigned_to)
    - patient -> appointment (scheduled)
    - opportunity -> contact (primary_contact)
    """

    source: ObjectRef
    target: ObjectRef
    relation_type: str  # e.g., "assigned_to", "scheduled", "primary_contact"
    metadata: dict[str, Any] = field(default_factory=dict)


class Channel(str, Enum):
    """Communication/interaction channels."""

    PHONE = "phone"
    SMS = "sms"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    WEB_CHAT = "web_chat"
    IN_PERSON = "in_person"
    SYSTEM = "system"
    API = "api"


class ActorType(str, Enum):
    """Types of actors that can create events."""

    HUMAN_AGENT = "human_agent"
    AI_AGENT = "ai_agent"
    CUSTOMER = "customer"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"


@dataclass(frozen=True)
class Actor:
    """Represents who/what created an event."""

    actor_type: ActorType
    actor_id: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """
    An action that can be taken by a policy.

    Actions are domain-agnostic at the type level;
    domain-specific semantics are in the action_type and params.
    """

    action_type: str
    params: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.params:
            param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"{self.action_type}({param_str})"
        return self.action_type


# Common actions
WAIT = Action(action_type="wait")
ESCALATE = Action(action_type="escalate")
CLOSE = Action(action_type="close")


@dataclass
class OutcomeLabel:
    """Label for a case outcome, used for evaluation."""

    label: str
    category: str  # positive, negative, neutral
    recorded_at: datetime
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
