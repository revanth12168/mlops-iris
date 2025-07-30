import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftTable

ref = pd.DataFrame({"col": [1, 2, 3, 4]})
cur = pd.DataFrame({"col": [1, 2, 3, 10]})

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("drift_report.html")

