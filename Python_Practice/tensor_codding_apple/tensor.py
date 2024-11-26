import pandas as pd


data = pd.read_csv("gpascore.csv")
data = data.dropna()

x = []
y = data['admit'].values

for i, rows in data.iterrows():
    x.append([rows['gre'], rows['gpa'], rows['rank']])



import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss = 'binary_crossentropy' , metrics=['accuracy'])

model.fit(np.array(x),np.array(y),epochs = 1000)



predict = model.predict([[750, 3.7,3], [400,2.2,1]])
print(predict)