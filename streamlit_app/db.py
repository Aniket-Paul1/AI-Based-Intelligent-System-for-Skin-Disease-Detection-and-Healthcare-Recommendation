import csv
import os
from datetime import datetime

DB_FILE = "prediction_history.csv"


def save_prediction(disease, confidence):
    """
    Saves prediction history to CSV file.
    """

    file_exists = os.path.isfile(DB_FILE)

    with open(DB_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow(["Timestamp", "Disease", "Confidence (%)"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            disease,
            f"{confidence:.2f}"
        ])
