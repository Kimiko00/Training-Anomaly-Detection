import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from datetime import datetime
from keras import regularizers
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dropout, LSTM, TimeDistributed, RepeatVector

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
sns.set(color_codes=True)
tf.random.set_seed(10)
np.random.seed(10)

#DATA LOADING / PRE-PROCESSING
data_dir = 'Bearing_Sensor_Data'
merged_data = pd.DataFrame()

for filename in os.listdir(data_dir):
    dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t', error_bad_lines=False)
    dataset_mean_abs = np.array(dataset.mean().abs())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,-4))
    dataset_mean_abs.index = [filename]
    merged_data = merged_data.append(dataset_mean_abs)
    
merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

merged_data.index = pd.to_datetime(merged_data.index, format='%Y%m%d%H%M%S', errors='ignore')
merged_data = merged_data.sort_index()
merged_data.to_csv('Average_BearingTest_Dataset.csv')
print("Dataset shape:", merged_data.shape)
merged_data.head()

#DEFINE TEST/TEST DATA
train = merged_data['2004.02.12 10:52:39': '2004.02.15 12:52:39']
test = merged_data['2004.02.15 12:52:39':]
print("Training dataset shape:", train.shape)
print("Testing dataset shape:", test.shape)

fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train['Bearing 1'], label='Bearing 1', color='blue', animated = True, linewidth=1)
ax.plot(train['Bearing 2'], label='Bearing 2', color='red', animated = True, linewidth=1)
ax.plot(train['Bearing 3'], label='Bearing 3', color='green', animated = True, linewidth=1)
ax.plot(train['Bearing 4'], label='Bearing 4', color='black', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Bearing Sensor Training Data', fontsize=16)
plt.show()