"""The adapter's error type, kept apart from the orchestration that raises it.

Both the pipeline itself (:mod:`nekron.data_adapter.adapter`) and the fold
schedules it cuts (:mod:`nekron.data_adapter.splitting`) report failures with the
same exception, and the pipeline consumes the schedules. Defining the type here —
as every other package in the project does — is what keeps that dependency going
in one direction.
"""

from __future__ import annotations


class AdapterError(Exception):
    """Base class for pipeline-orchestration errors.

    Every other package raises its own type, so a caller that wants to distinguish
    "the pipeline was wired wrong" from "pandas disagreed with me" needs one here
    too.
    """
