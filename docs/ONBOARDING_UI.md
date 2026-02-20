# Onboarding Progress Widget (UI Reference)

The frontend should provide an **onboarding progress widget** that allows operators to see the status of new-machine onboarding and restart failed or stalled runs.

## API Endpoints

- **GET /api/onboarding**  
  Returns a list of all machines and their current onboarding status (latest row per machine).  
  Response: array of `{ machine_id, started_at, completed_at, current_step, status, error_message, model_id }`.

- **GET /api/onboarding/{machine_id}**  
  Returns the current onboarding status for a single machine.  
  Response: `{ machine_id, started_at, completed_at, current_step, status, error_message, model_id }`.  
  Returns 404 if there is no onboarding record for that machine.

- **POST /api/onboarding/{machine_id}/restart**  
  Restarts onboarding for a machine (e.g. after STALLED or FAILED). Sets status to PENDING, clears error/step, and triggers the Prefect flow `new-machine-onboarding` for that `machine_id`.  
  Returns 202 Accepted with `{ status, message }`.

## Widget Behavior

The widget should:

1. Call **GET /api/onboarding** (or **GET /api/onboarding/{machine_id}** for a single-machine view) to load status.
2. Display per machine: **status**, **current_step**, **started_at**, **completed_at**, **error_message**, and a **Restart** action.
3. For the Restart action, call **POST /api/onboarding/{machine_id}/restart**, then refresh the status.

Actual widget implementation is deferred to a separate frontend task (e.g. prompt 22).
