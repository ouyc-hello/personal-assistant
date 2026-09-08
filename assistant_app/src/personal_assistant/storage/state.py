from __future__ import annotations

from enum import StrEnum


class MemoryStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    PUBLISHED = "PUBLISHED"
    CONFLICT = "CONFLICT"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"
    DELETED = "DELETED"
    REJECTED = "REJECTED"


class TaskStatus(StrEnum):
    CREATED = "CREATED"
    SCHEDULED = "SCHEDULED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ToolRunStatus(StrEnum):
    CREATED = "CREATED"
    EXECUTING = "EXECUTING"
    EXECUTED_FAILED = "EXECUTED_FAILED"
    EXECUTED_SUCCESS_UNACK = "EXECUTED_SUCCESS_UNACK"
    COMPLETED = "COMPLETED"
    SUPERSEDED = "SUPERSEDED"


class ApprovalStatus(StrEnum):
    DRAFT = "DRAFT"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


TASK_TRANSITIONS: dict[str, set[str]] = {
    TaskStatus.CREATED: {TaskStatus.SCHEDULED, TaskStatus.CANCELLED},
    TaskStatus.SCHEDULED: {TaskStatus.EXECUTING, TaskStatus.CANCELLED},
    TaskStatus.EXECUTING: {TaskStatus.COMPLETED, TaskStatus.FAILED},
    TaskStatus.FAILED: {TaskStatus.SCHEDULED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.CANCELLED: set(),
}

TOOL_TRANSITIONS: dict[str, set[str]] = {
    ToolRunStatus.CREATED: {ToolRunStatus.EXECUTING, ToolRunStatus.SUPERSEDED},
    ToolRunStatus.EXECUTING: {
        ToolRunStatus.EXECUTED_FAILED,
        ToolRunStatus.EXECUTED_SUCCESS_UNACK,
        ToolRunStatus.SUPERSEDED,
    },
    ToolRunStatus.EXECUTED_FAILED: {ToolRunStatus.EXECUTING, ToolRunStatus.SUPERSEDED},
    ToolRunStatus.EXECUTED_SUCCESS_UNACK: {ToolRunStatus.COMPLETED},
    ToolRunStatus.COMPLETED: set(),
    ToolRunStatus.SUPERSEDED: set(),
}

APPROVAL_TRANSITIONS: dict[str, set[str]] = {
    ApprovalStatus.DRAFT: {ApprovalStatus.PENDING_CONFIRMATION},
    ApprovalStatus.PENDING_CONFIRMATION: {ApprovalStatus.SUBMITTING, ApprovalStatus.REJECTED},
    ApprovalStatus.SUBMITTING: {ApprovalStatus.SUBMITTED, ApprovalStatus.UNKNOWN},
    ApprovalStatus.SUBMITTED: {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.UNKNOWN},
    ApprovalStatus.APPROVED: set(),
    ApprovalStatus.REJECTED: set(),
    ApprovalStatus.UNKNOWN: {ApprovalStatus.SUBMITTING, ApprovalStatus.REJECTED},
}
