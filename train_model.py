import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd # Pandas ka istemal data ko manage karne ke liye

# --- Configuration ---
BASE_DATA_DIRS = ['train', 'valid', 'test']
CLASSES_TO_USE = [
    'brown_rust', 'fusarium_head_blight', 'healthy', 'leaf_blight', 
    'mildew', 'septoria', 'smut', 'yellow_rust'
]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50 # Ek high limit set karein, EarlyStopping sabse achha epoch dhoondh lega
MODEL_SAVE_PATH = 'wheat_disease_model.h5'
LABELS_SAVE_PATH = 'labels.json'

# --- 1. Automatic Data Re-split ---
print("Step 1: Consolidating and splitting all available data...")

all_filepaths = []
all_labels = []

# Saare folders (train, valid, test) se images ko ikattha karna
for data_dir in BASE_DATA_DIRS:
    for class_name in CLASSES_TO_USE:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            for image_name in os.listdir(class_path):
                all_filepaths.append(os.path.join(class_path, image_name))
                all_labels.append(class_name)

# Pandas DataFrame ka istemal karke data ko manage karna
df = pd.DataFrame({'filepath': all_filepaths, 'label': all_labels})
print(f"Found a total of {len(df)} images across {len(df['label'].unique())} classes.")

# Pehla split: 80% training, 20% temporary (validation + test)
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
# Doosra split: 20% temporary ko 10% validation aur 10% test mein baantna
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")


# --- 2. Data Augmentation & Generators ---
print("\nStep 2: Setting up data augmentation from dataframes...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES_TO_USE,
    shuffle=True
)
validation_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col='filepath',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES_TO_USE,
    shuffle=False
)
test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col='filepath',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASSES_TO_USE,
    shuffle=False
)

# --- 3. Class Imbalance Handle Karna ---
print("\nStep 3: Calculating class weights...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Calculated Class Weights:", class_weights_dict)

# App ke liye labels map save karna
labels_map = {v: k for k, v in train_generator.class_indices.items()}
with open(LABELS_SAVE_PATH, 'w') as f:
    json.dump(labels_map, f)
print(f"Labels map saved to '{LABELS_SAVE_PATH}'")

# --- 4. Model Banana (Stronger Regularization ke saath) ---
print("\nStep 4: Building the AI Model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model.trainable = False # Base model ko freeze karna

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Stronger regularization
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x) # Ek aur dropout layer
predictions = Dense(len(CLASSES_TO_USE), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- 5. Smart Training ke liye Callbacks ---
# NEW: EarlyStopping aur ReduceLROnPlateau
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, # 10 epochs tak improvement na hone par training rok dein
    restore_best_weights=True # Sabse achhe model weights ko restore karein
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, # Learning rate ko 5 se divide karein
    patience=5, # 5 epochs tak improvement na hone par
    min_lr=1e-6
)

# --- 6. Model ko Compile aur Train Karna ---
print("\nStep 5: Compiling and training the model...")
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    class_weight=class_weights_dict,
    callbacks=[early_stopping, reduce_lr] # Naye callbacks ka istemal
)

# --- 7. Model ko Evaluate aur Save Karna ---
print("\nStep 6: Evaluating the final model on the test set...")
results = model.evaluate(test_generator)
print("-" * 30)
print(f"Final Test Loss: {results[0]:.4f}")
print(f"Final Test Accuracy: {results[1]:.4f}")
print("-" * 30)

model.save(MODEL_SAVE_PATH)
print(f"Training complete! Final model saved to '{MODEL_SAVE_PATH}'.")

# Temporary dataframes ko saaf karna (optional, memory bachane ke liye)
del train_df, val_df, test_df, df
print("Cleanup complete.")
