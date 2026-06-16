"""
Twin Validation
===============
Compares the PyBaMM digital twin's degradation predictions against
real NASA battery aging data, quantifying accuracy with RMSE / MAE / R2.
"""

import numpy as np


def calculate_validation_metrics(predicted_soh, actual_soh):
    predicted = np.array(predicted_soh)
    actual = np.array(actual_soh)

    n = min(len(predicted), len(actual))
    predicted = predicted[:n]
    actual = actual[:n]

    errors = predicted - actual
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "rmse_soh": round(float(rmse), 3),
        "mae_soh": round(float(mae), 3),
        "r2": round(float(r2), 4),
        "n_points": int(n),
        "max_error": round(float(np.max(np.abs(errors))), 3),
    }


def interpret_validation(metrics):
    r2 = metrics["r2"]
    rmse = metrics["rmse_soh"]

    if r2 > 0.95:
        quality = "Excellent"
        note = "The twin closely tracks real battery degradation."
    elif r2 > 0.85:
        quality = "Good"
        note = "The twin captures the degradation trend well."
    elif r2 > 0.7:
        quality = "Fair"
        note = "The twin captures general behavior; parameters could be tuned."
    else:
        quality = "Needs tuning"
        note = "Consider recalibrating chemistry parameters to this cell."

    return {
        "quality": quality,
        "note": note,
        "summary": f"R2={r2}, RMSE={rmse}% SoH — {quality}. {note}"
    }
