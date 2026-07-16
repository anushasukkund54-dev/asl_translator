import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load data
# Assumes final_dataset.csv has 63 columns of features and 1 column of labels
df = pd.read_csv('final_dataset.csv', header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# 3. Split data (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.3, random_state=42
)

# 4. Build Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Define Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

# 6. Train the model
print("Starting training...")
model.fit(
    X_train, y_train, 
    epochs=25, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# 7. Evaluate the model
print("\n--- Final Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 8. Detailed Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# --- REPLACEMENT STARTS HERE ---
# Get only the classes that are present in the current test set
unique_labels = np.unique(y_true_classes)
target_names = [le.classes_[i] for i in unique_labels]

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))
# --- REPLACEMENT ENDS HERE ---
# 9. Save the assets
model.save('asl_model.h5')
joblib.dump(le, 'label_encoder.pkl')
print("\nModel and Label Encoder saved successfully.")