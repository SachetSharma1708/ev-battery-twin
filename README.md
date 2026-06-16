# 🔋 EV Battery Digital Twin

> A physics-based digital twin of an EV battery pack — built on real electrochemical models (PyBaMM) and validated against real NASA battery aging data.

## What makes this a real digital twin

Runs the Doyle-Fuller-Newman (DFN) electrochemical model via PyBaMM, modeling real degradation mechanisms: SEI growth, lithium plating, particle cracking, and active material loss. Validated against real NASA cells cycled to failure.

## Features
- Live Twin: configure your charging/driving profile and watch the battery age cycle-by-cycle, with a Remaining Useful Life prediction
- Scenario Comparison: gentle vs aggressive usage side by side
- NASA Validation: compare twin predictions against real measured data (RMSE / R2)

## Setup
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
PyBaMM is a scientific library; install takes a few minutes.
Optional: download NASA data for validation (see data/README.md).

## Tech stack
PyBaMM (DFN model) · NumPy · SciPy · Pandas · Plotly · Streamlit
Chemistry: Chen2020 parameter set (LG M50 21700 — a real EV cell)

## License
MIT
