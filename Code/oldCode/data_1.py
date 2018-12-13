import os
import pandas as pd

# data_dir = 'stage1'
# patients = os.listdir(data_dir)
labels_data = pd.read_csv('LIDC-IDRI_MetaData.csv', index_col = 0)
print(labels_data.dtypes)
print(labels_data.columns)
# label = labels_df.at[patient, 'cancer']\n",
patient_id = []
