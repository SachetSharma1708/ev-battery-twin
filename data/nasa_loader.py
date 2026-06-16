"""
NASA Battery Aging Dataset Loader
=================================
Loads and parses the NASA Prognostics Center of Excellence (PCoE)
battery aging dataset — real lithium-ion cells cycled to failure.

Download:
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Requires: scipy, pandas, numpy
"""

import os
import numpy as np
import pandas as pd

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class NASABatteryData:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_cell(self, cell_id="B0005"):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required. Run: pip install scipy")

        filepath = os.path.join(self.data_dir, f"{cell_id}.mat")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Could not find {filepath}.\n"
                f"Download the NASA battery dataset and place {cell_id}.mat "
                f"in the '{self.data_dir}/' folder. See data/README.md."
            )

        mat = loadmat(filepath)
        cell_data = mat[cell_id][0, 0]
        cycles = cell_data["cycle"][0]

        records = []
        cycle_num = 0
        nominal_capacity = None

        for c in cycles:
            cycle_type = c["type"][0]
            if cycle_type != "discharge":
                continue

            cycle_num += 1
            data = c["data"][0, 0]

            try:
                capacity = float(data["Capacity"][0, 0])
            except (KeyError, IndexError, ValueError):
                continue

            if nominal_capacity is None:
                nominal_capacity = capacity

            soh = (capacity / nominal_capacity) * 100
            records.append({
                "cycle": cycle_num,
                "capacity_ah": round(capacity, 4),
                "soh_percent": round(soh, 2),
            })

        return pd.DataFrame(records)

    def list_available_cells(self):
        if not os.path.exists(self.data_dir):
            return []
        return [f.replace(".mat", "") for f in os.listdir(self.data_dir) if f.endswith(".mat")]
