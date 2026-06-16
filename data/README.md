# NASA Battery Aging Dataset — Download Guide

This digital twin validates its physics predictions against real lithium ion
battery aging data from NASA's Prognostics Center of Excellence (PCoE).

## How to download

1. Go to: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
2. Find the "Battery Data Set" entry.
3. Download the ZIP and extract these into this data/ folder:
   - B0005.mat
   - B0006.mat
   - B0007.mat
   - B0018.mat

The loader (nasa_loader.py) will parse them automatically.

Note: *.mat files are excluded from git by default (.gitignore).
