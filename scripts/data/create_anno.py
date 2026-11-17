# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:43:02 2026

@author: chris
"""
from pathlib import Path
import os
import csv

data_path = Path(r"")
patient_ids = os.listdir(data_path)
basepath = ""

csv_filename = 'ncct_anno.csv'

# Open the file and write the data
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(["sample_id", "filepath"])

    # Loop through patient IDs and construct rows
    for patient_id in patient_ids:
        # Using os.path.join handles slashes cleanly
        filename = f"ncct_{patient_id}.nii.gz"
        filepath = os.path.join(basepath, filename)

        writer.writerow([patient_id, filepath])

print(f"File saved successfully as {csv_filename}")