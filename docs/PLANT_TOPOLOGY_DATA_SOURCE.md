# Plant Topology & Fleet Data Source

## Where Fleet Topology Gets Its Data

The **Fleet Topology** on Plant Overview uses the **same machine list** as the rest of the app:

1. **Source**: `GET /api/machines` (and live updates via WebSocket).
2. **Backend**: `MachineService.get_all_machines()` in `services/machine_service.py`:
   - Reads from `cwru_features` (only machines that have at least one feature row appear).
   - Returns up to **100** machines (`limit=100`).
   - For each `machine_id`, **shop** and **line** come from `EQUIPMENT_METADATA` in `machine_service.py`.

3. **Layout**: `FleetTopology.jsx` groups machines by `machine.shop`:
   - **Body Shop**, **Stamping**, **Paint Shop**, **Final Assembly** (columns).
   - Machines without metadata get `shop: "Unassigned Shop"` and appear in the topology only if the backend returns them; they are not assigned to a specific column in the current layout (they go into "Unassigned").

So the topology is driven by:
- Which machine IDs have rows in `cwru_features`.
- `EQUIPMENT_METADATA` (name, shop, line, type) for those IDs. Currently **10** machines are defined there (WB-001, WB-002, WB-003, HP-200, TD-450, PR-101, CO-050, TS-001, LA-003, CV-100).

## Showing More Assets (e.g. 100)

- The API already supports **up to 100** machines; the overview subtitle shows **"X active assets"** where X = `machines.length`.
- To display many more assets in the topology with correct shop/line:
  - Add entries to `EQUIPMENT_METADATA` in `services/machine_service.py` for each machine (or derive shop/line from DB or config).
  - Ensure those machines have feature data in `cwru_features` so they are returned by `get_all_machines()`.

No separate "100 assets" UI mode exists; the count is whatever the API returns (max 100).
