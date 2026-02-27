"""Policy interface definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from arena.core.snapshot import CaseSnapshot
from arena.core.types import Action
from arena.policy.result import PolicyResult


@runtime_checkable
class Policy(Protocol):
    """
    Protocol for policies that make decisions.

    This is the core interface that all agent policies must implement.
    A policy takes a case snapshot and returns a decision.
    """

    @property
    def name(self) -> str:
        """The name of this policy."""
        ...

    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        """
        Choose an action given the case state and allowed actions.

        Args:
            snapshot: Point-in-time view of the case (no future information)
            allowed_actions: The set of actions the policy can choose from

        Returns:
            PolicyResult containing the chosen action and metadata
        """
        ...


class BasePolicy(ABC):
    """
    Abstract base class for policies.

    Provides common functionality and enforces the Policy interface.
    """

    def __init__(self, name: str = "", version: str = "1.0") -> None:
        self._name = name or self.__class__.__name__
        self._version = version

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @abstractmethod
    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        """Choose an action. Must be implemented by subclasses."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class CallablePolicy:
    """
    A policy that wraps a callable function.

    Useful for simple policies or quick prototyping.
    """

    def __init__(
        self,
        fn: callable,
        name: str = "callable_policy",
    ) -> None:
        self._fn = fn
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        result = self._fn(snapshot, allowed_actions)

        if isinstance(result, PolicyResult):
            return result
        elif isinstance(result, Action):
            return PolicyResult(action=result)
        elif isinstance(result, str):
            # Assume it's an action type
            matching = [a for a in allowed_actions if a.action_type == result]
            if matching:
                return PolicyResult(action=matching[0])
            raise ValueError(f"Action type '{result}' not in allowed actions")
        else:
            raise TypeError(
                f"Policy function must return PolicyResult, Action, or str, got {type(result)}"
            )


def policy_from_callable(
    fn: callable,
    name: str = "callable_policy",
) -> CallablePolicy:
    """Create a policy from a callable function."""
    return CallablePolicy(fn=fn, name=name)


class CompositePolicy(BasePolicy):
    """
    A policy that combines multiple policies.

    Uses a selector function to choose which sub-policy to use.
    """

    def __init__(
        self,
        policies: dict[str, Policy],
        selector: callable,
        name: str = "composite_policy",
    ) -> None:
        super().__init__(name=name)
        self.policies = policies
        self.selector = selector

    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        # Select which policy to use
        policy_key = self.selector(snapshot, allowed_actions)
        policy = self.policies.get(policy_key)

        if policy is None:
            raise ValueError(f"No policy found for key '{policy_key}'")

        result = policy.decide(snapshot, allowed_actions)

        # Add metadata about which sub-policy was used
        result.metadata["sub_policy"] = policy_key

        return result


class FallbackPolicy(BasePolicy):
    """
    A policy that tries multiple policies in order until one succeeds.
    """

    def __init__(
        self,
        policies: list[Policy],
        name: str = "fallback_policy",
    ) -> None:
        super().__init__(name=name)
        self.policies = policies

    def decide(
        self,
        snapshot: CaseSnapshot,
        allowed_actions: list[Action],
    ) -> PolicyResult:
        errors = []

        for i, policy in enumerate(self.policies):
            try:
                result = policy.decide(snapshot, allowed_actions)
                result.metadata["fallback_index"] = i
                result.metadata["fallback_policy"] = policy.name
                return result
            except Exception as e:
                errors.append((policy.name, str(e)))
                continue

        # All policies failed
        raise RuntimeError(
            f"All fallback policies failed: {errors}"
        )
