"""Explainable checks with explicit missing/failed/pass outcomes, never loss-only approval."""
import math


def assess(metrics, summary, standards=None):
    """Evaluate predeclared limits; missing standards or evidence remain unassessed.

    Returns a diagnostic, not a production-domain promotion. Scenario references
    and held-out evidence must still be reviewed against their provenance.
    """
    standards = standards or {}
    gates = []
    for scenario in metrics.get("scenarios", []):
        sid = scenario["scenario_id"]
        for metric, threshold_name in [
            ("depth_rmse_m", "depth_rmse_m_max"),
            ("depth_mae_m", "depth_mae_m_max"),
            ("velocity_rmse_m_s", "velocity_rmse_m_s_max"),
            ("ic_h_rmse_m", "ic_h_rmse_m_max"),
            ("bc_discharge_rmse_m2_s", "bc_discharge_rmse_m2_s_max")]:
            limit, value = standards.get(threshold_name), scenario.get(metric)
            gates.append(_gate(sid+":"+metric, value, limit))
        residuals = scenario.get("pde_rms_normalized")
        gates.append(_gate(sid+":pde_rms_normalized_max",
                           max(residuals) if residuals else None,
                           standards.get("pde_rms_normalized_max")))
    gates.append(_gate("max_global_balance_error_m3", summary.get("max_global_balance_error_m3"),
                       standards.get("max_global_balance_error_m3")))
    gates.append(_gate("max_cell_balance_error_m3", summary.get("max_cell_balance_error_m3"),
                       standards.get("max_cell_balance_error_m3")))
    status = "failed" if any(g["status"] == "failed" for g in gates) else (
        "passed" if gates and all(g["status"] == "passed" for g in gates) else "not_evaluated")
    return {"status": status, "gates": gates,
            "scope": "configured diagnostic limits only; not external accuracy certification",
            "reference_validation_status": metrics.get("reference_validation_status", "not_completed")}


def _gate(name, value, limit):
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit < 0):
        raise ValueError("Acceptance limits must be finite nonnegative numbers")
    if value is None or limit is None:
        return {"name":name,"value":value,"limit":limit,"status":"not_evaluated"}
    if not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError("Nonfinite acceptance metric")
    return {"name":name,"value":value,"limit":limit,"status":"passed" if value <= limit else "failed"}
