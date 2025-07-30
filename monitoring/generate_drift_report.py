import pandas as pd
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
import os

# Create report folder
os.makedirs("reports", exist_ok=True)

# Load reference and current data
ref = pd.read_csv("data/iris.csv")
cur = pd.read_csv("data/iris.csv")  # Use same for now

# Generate dashboard
dashboard = Dashboard(tabs=[DataDriftTab()])
dashboard.calculate(ref, cur)
dashboard.save("reports/data_drift_dashboard.html")

print("✅ Drift report saved at: reports/data_drift_dashboard.html")

