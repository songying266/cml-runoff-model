# Import required libraries
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import random
import os
from collections import defaultdict
from scipy.signal import savgol_filter
import seaborn as sns
from PIL import Image
import math
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Load different datasets used as model inputs
# dataset1: Individual CML link data
# dataset2: Individual rain gauge data
# dataset3: Mean of grouped CML data
# dataset4: Mean of grouped rain gauge data
dataset1 = read_csv(r'..\model_CML.csv',header=None)
dataset2 = read_csv(r'..\model_rain.csv',header=None)
dataset3 = read_csv(r'../model_CML_mean.csv', header=None)
dataset4 = read_csv(r'../model_rain_mean.csv', header=None)
# Prepare various input combinations
datasets = {
    "S1": dataset1,# CML only
    "S2": dataset2,# RG only
    "S3": pd.concat([dataset1.iloc[:, :-2], dataset2], axis=1, ignore_index=True),# CML+ RG
    "S4": pd.concat([dataset3.iloc[:,:31],dataset4],axis=1,ignore_index=True),# MEAN CML+ MEAN RG
    "S5":dataset3,# Mean CML only
    "S6":dataset4 # Mean RG only
}
#################################cross-validation
#################events selection
all_events = range(0,26)
# Output folder to save validation results
output_folder = r'..\results\cross-validation'
os.makedirs(output_folder, exist_ok=True)
# Settings for cross-validation
events_per_group = [1, 1, 1, 1, 1]
assert sum(events_per_group) == 5, "at least 5 events in validation"
validation_schemes = []
#every event in the validation for at least 5 times
num_vali_events = 5
num_iterations = 35
event_selection_count = defaultdict(int)
validation_schemes = []
for _ in range(num_iterations):
    vali_events = []
    for group in event_groups:
        available_events = [e for e in group if event_selection_count[e] < 5]
        if not available_events:
            available_events = group
        selected_event = random.choice(available_events)
        vali_events.append(selected_event)
        event_selection_count[selected_event] += 1
    validation_schemes.append(vali_events)
# Train and evaluate Random Forest models under different input datasets
event_results = {event: [] for event in all_events}
for dataset_name, values in datasets.items():
    grouped_data = values.groupby('event')
    results = []
    for iteration, vali_events in enumerate(validation_schemes):
        train_events = [event for event in all_events if event not in vali_events]
        train = pd.concat([grouped_data.get_group(event) for event in train_events])
        vali = pd.concat([grouped_data.get_group(event) for event in vali_events])
        train_X = train.iloc[:, :-2]
        train_y = train.iloc[:, -2]
        vali_X = vali.iloc[:, :-2]
        vali_y = vali.iloc[:, -2]

        rf_model = RandomForestRegressor(n_estimators=60, n_jobs=-1, max_depth=25, oob_score=True,
                                         criterion='squared_error', random_state=42)
        rf_model.fit(train_X, train_y)
        Ypred = rf_model.predict(vali_X)

        YPred = pd.DataFrame({
            'Predicted': Ypred,
            'Observed': vali_y.values,
            'event': vali.iloc[:, -1]
        })
        vali_output_path = os.path.join(output_folder, f"{dataset_name}_vali_{iteration}.csv")
        YPred.to_csv(vali_output_path, index=False)


