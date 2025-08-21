import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv(r'archive (3)\Dataset.csv')
print(df.shape)

# Selected features
# Define the features as per the training data
features = [
    'HR', 'Temp', 'Resp', 'O2Sat', 'SBP', 'MAP', 'DBP',
    'WBC', 'Lactate', 'Creatinine', 'BUN', 'Platelets', 'Glucose',
    'pH', 'HCO3', 'PaCO2', 'FiO2', 'PTT', 'Fibrinogen', 'Age',
    'Gender', 'ICULOS'
]
target = 'SepsisLabel'

# Handle missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Separate features and target
X = df[features]
y = df[target]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the deep learning model_1
model = Sequential()

# Input layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Hidden layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(1, activation='sigmoid'))  # Sigmoid because it's binary classification

# Compile the model_1
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model_1 with reduced epochs (e.g., 10)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model_1 on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)  # Threshold of 0.5 for classification

# Print classification report
print(classification_report(y_test, y_pred))

# Save the Keras model_1 using the Keras save method
model.save('sepsis_model.h5')  # Save the model_1 as a .h5 file

# Save the scaler using joblib
joblib.dump(scaler, 'model/scaler.pkl')
