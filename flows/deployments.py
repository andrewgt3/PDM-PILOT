"""
Prefect deployment definitions with schedules.

- weekly_drift_check: every Monday 06:00
- weekly_retraining_all: every Sunday 02:00 (off-peak)
- new-machine-onboarding: event-triggered (no schedule)
- model-shadow-evaluation: event or manual (no schedule in first iteration)
"""

from prefect import serve

from flows.maintenance_flows import (
    weekly_drift_check_flow,
    weekly_retraining_all_flow,
    onboarding_flow,
    shadow_evaluation_flow,
)


def get_scheduled_deployments():
    """Build deployment objects with schedules for serve()."""
    return [
        weekly_drift_check_flow.to_deployment(
            name="weekly-drift-check",
            cron="0 6 * * 1",  # Monday 06:00
            tags=["maintenance", "drift"],
        ),
        weekly_retraining_all_flow.to_deployment(
            name="scheduled-retraining-all",
            cron="0 2 * * 0",  # Sunday 02:00
            tags=["maintenance", "retraining"],
        ),
        onboarding_flow.to_deployment(
            name="new-machine-onboarding",
            tags=["onboarding", "event-triggered"],
        ),
        shadow_evaluation_flow.to_deployment(
            name="model-shadow-evaluation",
            tags=["shadow", "evaluation"],
        ),
    ]


if __name__ == "__main__":
    deployments = get_scheduled_deployments()
    serve(*deployments)
