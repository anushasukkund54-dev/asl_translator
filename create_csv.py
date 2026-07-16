import cv2
import mediapipe as mp
import pandas as pd
import os

# 1. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
data = []

# 2. Update this path to your ASL dataset folder
# Use the exact path where your 'A', 'B', 'C'... folders are located
root_path = r"C:/Users/anush/Downloads/dataset/asl_alphabet_train/asl_alphabet_train"

# 3. Process every folder
for label in os.listdir(root_path):
    label_dir = os.path.join(root_path, label)
    if os.path.isdir(label_dir):
        print(f"Converting folder: {label}")
        
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            
            # Convert to RGB (MediaPipe requirement)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(image_rgb)
            
            # Extract landmarks if hand is detected
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                # Create a list of 63 coordinates (x, y, z for each of 21 points)
                row = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]
                row.append(label) # Save the letter/label at the end
                data.append(row)

# 4. Save to CSV
df = pd.DataFrame(data)
df.to_csv('final_dataset.csv', index=False, header=False)
print("Conversion complete! 'final_dataset.csv' is ready in your folder.")