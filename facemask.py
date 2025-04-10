import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.mixed_precision import Policy, set_global_policy
import cv2
import os
import time
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import ImageTk
import threading
import glob
from sklearn.model_selection import train_test_split

# Enable mixed precision training for faster performance
policy = Policy('mixed_float16')
set_global_policy(policy)

# Define the optimized mask removal model (U-Net architecture)
def build_mask_removal_model(input_shape=(256, 256, 3)):
    """Build an optimized U-Net model for mask removal."""
    inputs = Input(shape=input_shape)
    
    # Contracting path (encoder) with BatchNorm and reduced channels
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = BatchNormalization()(p1)
    p1 = Dropout(0.1)(p1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = BatchNormalization()(p2)
    p2 = Dropout(0.2)(p2)
    
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = BatchNormalization()(p3)
    p3 = Dropout(0.3)(p3)
    
    # Bottleneck
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = BatchNormalization()(c4)
    p4 = Dropout(0.4)(p4)
    
    # Expansive path (decoder)
    u5 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(p4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    
    # Output layer (use float32 for final layer)
    outputs = Conv2D(3, (1, 1), activation='sigmoid', dtype='float32')(c7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['accuracy'])
    
    return model

class DataGenerator:
    """Optimized class to prepare and load training data from single images containing both masked and unmasked faces"""
    
    def __init__(self, data_dir="masked_unmasked", batch_size=8, img_size=(256, 256), validation_split=0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.validation_split = validation_split
        
    def validate_data(self):
        """Verify that the data directory contains valid images"""
        image_files = glob.glob(os.path.join(self.data_dir, '*.[jJpP][pPnN][gG]'))  # Look directly in masked_unmasked
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in directory: {self.data_dir}")
            
        print(f"Found {len(image_files)} images")
        return True
        
    def detect_and_extract_faces(self, image_path):
        """Detect and extract both masked and unmasked faces from a single image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return None, None
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
            
            if len(faces) < 2:
                print(f"Could not find at least 2 faces in image: {image_path}")
                return None, None
                
            # Sort faces by x-coordinate (assuming left is masked, right is unmasked)
            faces = sorted(faces, key=lambda x: x[0])
            
            # Extract first face (assumed to be masked)
            x1, y1, w1, h1 = faces[0]
            masked_face = img[y1:y1+h1, x1:x1+w1]
            
            # Extract second face (assumed to be unmasked)
            x2, y2, w2, h2 = faces[1]
            unmasked_face = img[y2:y2+h2, x2:x2+w2]
            
            # Resize both faces to target size
            masked_face = cv2.resize(masked_face, self.img_size)
            unmasked_face = cv2.resize(unmasked_face, self.img_size)
            
            return masked_face, unmasked_face
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None
    
    def create_generators(self):
        """Create optimized data generators for training with validation split"""
        # Get all image files
        image_files = glob.glob(os.path.join(self.data_dir, '*.[jJpP][pPnN][gG]'))  # Look directly in masked_unmasked
        
        # Split into train and validation sets
        train_files, val_files = train_test_split(image_files, test_size=self.validation_split, random_state=42)
        
        # Data augmentation config
        data_gen_args = dict(
            rescale=1./255,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Custom generator that yields pairs of masked/unmasked faces
        def pair_generator(image_files, batch_size, augment=False):
            while True:
                # Shuffle files at start of each epoch
                np.random.shuffle(image_files)
                
                for i in range(0, len(image_files), batch_size):
                    batch_files = image_files[i:i+batch_size]
                    masked_batch = []
                    unmasked_batch = []
                    
                    for file_path in batch_files:
                        masked, unmasked = self.detect_and_extract_faces(file_path)
                        if masked is not None and unmasked is not None:
                            masked_batch.append(masked)
                            unmasked_batch.append(unmasked)
                    
                    if len(masked_batch) == 0:
                        continue
                        
                    masked_batch = np.array(masked_batch, dtype=np.float32) / 255.0
                    unmasked_batch = np.array(unmasked_batch, dtype=np.float32) / 255.0
                    
                    if augment:
                        # Apply the same augmentation to both masked and unmasked faces
                        seed = np.random.randint(99999)
                        
                        # Create temporary generator for augmentation
                        temp_gen = ImageDataGenerator(**data_gen_args)
                        
                        # Apply same transformation to both
                        masked_aug = temp_gen.flow(masked_batch, batch_size=len(masked_batch), 
                                                  shuffle=False, seed=seed)
                        unmasked_aug = temp_gen.flow(unmasked_batch, batch_size=len(unmasked_batch), 
                                                    shuffle=False, seed=seed)
                        
                        yield next(masked_aug), next(unmasked_aug)
                    else:
                        yield masked_batch, unmasked_batch
        
        # Create generators
        train_gen = pair_generator(train_files, self.batch_size, augment=True)
        val_gen = pair_generator(val_files, self.batch_size, augment=False)
        
        # Calculate steps
        train_steps = max(1, len(train_files) // self.batch_size)
        val_steps = max(1, len(val_files) // self.batch_size)
        
        return train_gen, val_gen, train_steps, val_steps

class MaskRemover:
    def __init__(self, model_path=None):
        """Initialize the mask remover with optimized model loading."""
        self.model = None
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print(f"Model loaded from {model_path}")
                # Test the model with dummy data
                dummy_input = np.random.rand(1, 256, 256, 3).astype('float32')
                _ = self.model.predict(dummy_input)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Building new optimized model")
                self.model = build_mask_removal_model()
        else:
            print("No model found - initializing optimized model")
            self.model = build_mask_removal_model()
    
    def preprocess_face(self, face_img):
        """Preprocess face image for the model."""
        face_img = cv2.resize(face_img, (256, 256))
        face_img = face_img / 255.0
        return np.expand_dims(face_img, axis=0)
    
    def remove_mask(self, image_path, output_path=None):
        """Remove mask from faces in the image with error handling."""
        try:
            # Read image with explicit UTF-8 handling
            if isinstance(image_path, str):
                image_path = image_path.encode('utf-8').decode('utf-8')
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")
        
        result_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))
        
        if len(faces) == 0:
            print("No faces detected in the image.")
            return img, None
        
        unmasked_faces = []
        for (x, y, w, h) in faces:
            try:
                face_img = img[y:y+h, x:x+w]
                processed_face = self.preprocess_face(face_img)
                unmasked_face = self.model.predict(processed_face, verbose=0)[0]
                unmasked_face = (unmasked_face * 255).astype('uint8')
                unmasked_face = cv2.resize(unmasked_face, (w, h))
                result_img[y:y+h, x:x+w] = unmasked_face
                unmasked_faces.append((unmasked_face, (x, y, w, h)))
            except Exception as e:
                print(f"Error processing face at ({x}, {y}): {e}")
                continue
        
        if output_path:
            try:
                cv2.imwrite(output_path, result_img)
            except Exception as e:
                print(f"Error saving result: {e}")
        
        return result_img, unmasked_faces
        
    def train(self, data_dir="masked_unmasked", epochs=50, batch_size=8, callbacks=None):
        """Train the mask removal model with validation."""
        data_generator = DataGenerator(data_dir, batch_size=batch_size)
        data_generator.validate_data()
        
        train_gen, val_gen, train_steps, val_steps = data_generator.create_generators()
        
        if callbacks is None:
            callbacks = [
                ModelCheckpoint("mask_removal_model.h5", save_best_only=True, 
                               monitor='val_loss', mode='min'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, 
                                 min_lr=1e-6, verbose=1),
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                TensorBoard(log_dir='./logs', histogram_freq=1)
            ]
        
        print(f"\nStarting training with {train_steps} train steps and {val_steps} validation steps per epoch")
        
        history = self.model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model.save('mask_removal_model.h5')
        print("Model saved as 'mask_removal_model.h5'")
        
        return history

class MaskRemovalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Removal - Optimized")
        self.root.geometry("850x650")
        
        self.input_image_path = tk.StringVar()
        self.output_image_path = tk.StringVar()
        self.mask_remover = None
        self.training_active = False
        
        # Initialize mask remover with error handling
        try:
            self.mask_remover = MaskRemover('mask_removal_model.h5')
        except Exception as e:
            print(f"Error initializing mask remover: {e}")
            messagebox.showwarning("Warning", 
                                 "Could not load pre-trained model. A new model will be created.")
            self.mask_remover = MaskRemover()
        
        self.create_ui()
    
    def create_ui(self):
        """Create and arrange all UI components"""
        # Main frames
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = tk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = tk.Frame(image_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Progress bar
        self.progress = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Image labels
        tk.Label(left_frame, text="Original Image").pack()
        self.input_image_label = tk.Label(left_frame, bg="lightgray", width=40, height=20)
        self.input_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(right_frame, text="Processed Image").pack()
        self.output_image_label = tk.Label(right_frame, bg="lightgray", width=40, height=20)
        self.output_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        button_frame = tk.Frame(top_frame)
        button_frame.pack(fill=tk.X)
        
        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.process_button = tk.Button(button_frame, text="Remove Mask", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.train_button = tk.Button(button_frame, text="Train Model", command=self.show_training_dialog)
        self.train_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.save_button = tk.Button(button_frame, text="Save Result", command=self.save_result, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
    
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.input_image_path.set(file_path)
            try:
                # Display input image
                img = Image.open(file_path)
                img.thumbnail((400, 400))
                img_tk = ImageTk.PhotoImage(img)
                self.input_image_label.config(image=img_tk)
                self.input_image_label.image = img_tk
                
                # Clear output image
                self.output_image_label.config(image='')
                
                self.process_button.config(state=tk.NORMAL)
                self.status_var.set(f"Image loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {e}")
                self.status_var.set(f"Error loading image: {e}")
    
    def process_image(self):
        """Process the loaded image to remove masks"""
        if not self.input_image_path.get():
            messagebox.showerror("Error", "Please select an image first")
            return
            
        try:
            # Create output directory if it doesn't exist
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Process image
            input_path = self.input_image_path.get()
            output_path = os.path.join(output_dir, f"unmasked_{os.path.basename(input_path)}")
            self.output_image_path.set(output_path)
            
            self.status_var.set("Processing image... This may take a moment.")
            self.root.update()
            
            # Start processing in a separate thread
            self.progress.start()
            
            def process_task():
                try:
                    result_img, _ = self.mask_remover.remove_mask(input_path, output_path)
                    self.root.after(0, lambda: self.update_result(result_img, output_path))
                except Exception as e:
                    self.root.after(0, lambda: self.handle_process_error(e))
            
            thread = threading.Thread(target=process_task)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error", f"Error processing image: {e}")
            self.status_var.set(f"Error processing image: {e}")
    
    def update_result(self, result_img, output_path):
        """Update the UI with the processed image"""
        self.progress.stop()
        
        # Display output image
        img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.output_image_label.config(image=img_tk)
        self.output_image_label.image = img_tk
        
        self.save_button.config(state=tk.NORMAL)
        self.status_var.set(f"Mask removed and saved to {output_path}")
    
    def handle_process_error(self, error):
        """Handle errors during image processing"""
        self.progress.stop()
        messagebox.showerror("Error", f"Error processing image: {error}")
        self.status_var.set(f"Error processing image: {error}")
    
    def save_result(self):
        """Save the processed image"""
        if not self.output_image_path.get():
            messagebox.showerror("Error", "No processed image to save")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        
        if save_path:
            try:
                import shutil
                shutil.copy(self.output_image_path.get(), save_path)
                self.status_var.set(f"Image saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {e}")
                self.status_var.set(f"Error saving image: {e}")
    
    def show_training_dialog(self):
        """Show the training configuration dialog"""
        train_dialog = tk.Toplevel(self.root)
        train_dialog.title("Train Model")
        train_dialog.geometry("450x250")
        train_dialog.resizable(False, False)
        train_dialog.transient(self.root)
        train_dialog.grab_set()
        
        tk.Label(train_dialog, text="Training Parameters", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Data directory
        dir_frame = tk.Frame(train_dialog)
        dir_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(dir_frame, text="Dataset Directory:").pack(side=tk.LEFT)
        
        self.dataset_path = tk.StringVar(value="masked_unmasked")
        dataset_entry = tk.Entry(dir_frame, textvariable=self.dataset_path, width=25)
        dataset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        browse_btn = tk.Button(dir_frame, text="Browse", command=lambda: self.select_data_directory(self.dataset_path))
        browse_btn.pack(side=tk.RIGHT)
        
        # Parameters frame
        params_frame = tk.Frame(train_dialog)
        params_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Number of epochs
        tk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=20)
        tk.Spinbox(params_frame, from_=1, to=100, textvariable=self.epochs_var, width=5).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Batch size
        tk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        self.batch_size_var = tk.IntVar(value=8)
        tk.Spinbox(params_frame, from_=1, to=32, textvariable=self.batch_size_var, width=5).grid(row=0, column=3, sticky='w', padx=5, pady=5)
        
        # Note about data organization
        note_text = "Note: Dataset should contain images with both masked\nand unmasked faces of the same person."
        tk.Label(train_dialog, text=note_text, fg="blue").pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(train_dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=train_dialog.destroy)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        train_btn = tk.Button(button_frame, text="Start Training", command=lambda: self.start_training(train_dialog))
        train_btn.pack(side=tk.RIGHT, padx=5)
    
    def select_data_directory(self, path_var):
        """Select the data directory for training"""
        directory = filedialog.askdirectory(title="Select Dataset Directory")
        if directory:
            path_var.set(directory)
    
    def start_training(self, dialog):
        """Start the optimized model training process."""
        data_dir = self.dataset_path.get()
        epochs = self.epochs_var.get()
        batch_size = self.batch_size_var.get()
        
        if not data_dir:
            messagebox.showerror("Error", "Please select a dataset directory")
            return
        
        dialog.destroy()
        
        # Validate dataset structure
        try:
            if not os.path.exists(data_dir):
                messagebox.showerror("Error", f"Directory does not exist: {data_dir}")
                return
                
            image_files = glob.glob(os.path.join(data_dir, '*.[jJpP][pPnN][gG]'))
            if len(image_files) == 0:
                messagebox.showerror("Error", f"No images found in directory: {data_dir}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Error validating dataset: {e}")
            return
        
        if not messagebox.askyesno("Confirm Training", 
            f"Training Parameters:\n\nDataset: {data_dir}\n"
            f"Epochs: {epochs}\nBatch Size: {batch_size}\n\n"
            "Training may take a while. Continue?"):
            return
        
        # Create training progress window
        train_progress = tk.Toplevel(self.root)
        train_progress.title("Training Progress")
        train_progress.geometry("400x250")
        train_progress.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close button
        
        tk.Label(train_progress, text="Training in Progress", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.progress_var = tk.StringVar()
        self.progress_var.set("Initializing training...")
        tk.Label(train_progress, textvariable=self.progress_var).pack(pady=5)
        
        self.epoch_var = tk.StringVar()
        self.epoch_var.set("Epoch: 0/0")
        tk.Label(train_progress, textvariable=self.epoch_var).pack(pady=5)
        
        self.metrics_var = tk.StringVar()
        self.metrics_var.set("Loss: - | Accuracy: -")
        tk.Label(train_progress, textvariable=self.metrics_var).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(train_progress, orient=tk.HORIZONTAL, 
                                          length=300, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        cancel_btn = tk.Button(train_progress, text="Cancel Training", 
                             command=lambda: self.cancel_training(train_progress))
        cancel_btn.pack(pady=5)
        
        # Start training in a separate thread
        self.training_active = True
        thread = threading.Thread(
            target=self.run_training,
            args=(data_dir, epochs, batch_size, train_progress),
            daemon=True
        )
        thread.start()
    
    def run_training(self, data_dir, epochs, batch_size, progress_window):
        """Run the training process with progress updates."""
        try:
            class TrainingCallback(tf.keras.callbacks.Callback):
                def __init__(self, app):
                    self.app = app
                
                def on_epoch_begin(self, epoch, logs=None):
                    if not self.app.training_active:
                        self.model.stop_training = True
                    self.app.root.after(0, lambda: self.app.update_training_progress(
                        epoch+1, epochs, "Starting epoch..."))
                
                def on_epoch_end(self, epoch, logs=None):
                    loss = logs.get('loss', 0)
                    acc = logs.get('accuracy', 0)
                    val_loss = logs.get('val_loss', 0)
                    val_acc = logs.get('val_accuracy', 0)
                    status = (f"Epoch {epoch+1}/{epochs}\n"
                             f"Loss: {loss:.4f} | Accuracy: {acc:.4f}\n"
                             f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                    self.app.root.after(0, lambda: self.app.update_training_progress(
                        epoch+1, epochs, status))
            
            callbacks = [
                TrainingCallback(self),
                ModelCheckpoint("mask_removal_model.h5", save_best_only=True),
                EarlyStopping(patience=5, restore_best_weights=True)
            ]
            
            self.root.after(0, lambda: self.progress_var.set("Preparing data..."))
            
            start_time = time.time()
            history = self.mask_remover.train(
                data_dir=data_dir,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )
            
            training_time = time.time() - start_time
            self.root.after(0, lambda: self.training_complete(
                progress_window, training_time))
            
        except Exception as e:
            self.root.after(0, lambda: self.training_error(progress_window, str(e)))
    
    def update_training_progress(self, current_epoch, total_epochs, status):
        """Update the training progress UI."""
        self.epoch_var.set(f"Epoch: {current_epoch}/{total_epochs}")
        self.metrics_var.set(status)
        self.progress_bar['value'] = (current_epoch / total_epochs) * 100
        self.progress_bar.update()
    
    def cancel_training(self, progress_window):
        """Cancel the ongoing training."""
        self.training_active = False
        progress_window.destroy()
        self.status_var.set("Training cancelled by user")
    
    def training_complete(self, progress_window, training_time):
        """Handle training completion."""
        self.training_active = False
        progress_window.destroy()
        messagebox.showinfo("Training Complete", 
                          f"Training completed in {training_time:.2f} seconds.\n"
                          "Model saved as 'mask_removal_model.h5'")
        self.status_var.set("Training complete")
    
    def training_error(self, progress_window, error_msg):
        """Handle training errors."""
        self.training_active = False
        progress_window.destroy()
        messagebox.showerror("Training Error", 
                           f"An error occurred during training:\n\n{error_msg}")
        self.status_var.set("Training error occurred")

# Run application
if __name__ == "__main__":
    root = tk.Tk()
    app = MaskRemovalApp(root)
    root.mainloop()