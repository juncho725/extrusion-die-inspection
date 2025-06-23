# ----------------------Version----------------------
# v01.05.04 (Faster R-CNN version - Multi-directory support + Model save/load)

# ----------------------Updates----------------------
# 1. Support for loading data from multiple directories
# 2. Changed to Faster R-CNN (removed mask part from Mask R-CNN)
# 3. Simplified image preprocessing
# 4. Added polygon coordinate validation and clipping
# 5. Added model save and load functionality

# -----------------------Notes-----------------------
# Model visualized_outputs_faster_rcnn

import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Model save function
def save_model(model, save_path, optimizer=None, epoch=None, loss=None):
    """
    Save the model.
    
    Args:
        model: Model to save
        save_path: Save path
        optimizer: Optimizer (optional)
        epoch: Current epoch (optional)
        loss: Current loss value (optional)
    """
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data to save
    if optimizer is not None and epoch is not None:
        # Save full checkpoint (for resuming training)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if loss is not None:
            checkpoint['loss'] = loss
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    else:
        # Save model only (for inference)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}")

# Model load function
def load_model(model, load_path, optimizer=None, device=None):
    """
    Load saved model.
    
    Args:
        model: Model structure (model to apply loaded weights)
        load_path: Path to load file
        optimizer: Optimizer (optional, for resuming training)
        device: Device to load model (GPU/CPU)
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer (if provided)
        epoch: Saved epoch (if in checkpoint)
        loss: Saved loss value (if in checkpoint)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load file
    checkpoint = torch.load(load_path, map_location=device)
    
    # Check if checkpoint or simple model
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Load model weights from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)
        
        # If optimizer is provided and optimizer state exists in checkpoint
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded (epoch: {epoch}): {load_path}")
            return model, optimizer, epoch, loss
        else:
            print(f"Model weights only loaded (epoch: {epoch}): {load_path}")
            return model, None, epoch, loss
    else:
        # Load simple model weights only
        model.load_state_dict(checkpoint)
        print(f"Model weights loaded: {load_path}")
        return model, None, 0, None

# 1. Dataset class definition (mask removed)
class LabelMeDataset(Dataset):
    def __init__(self, json_dir, transform=None):
        self.json_dir = json_dir
        self.files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.transform = transform
        print(f"Loaded {len(self.files)} JSON files from {json_dir}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        json_path = os.path.join(self.json_dir, self.files[idx])
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Load image
            img_path = os.path.join(self.json_dir, data['imagePath'])
            if not os.path.exists(img_path):
                print(f"Error: Image not found - {img_path}")
                # Return dummy image
                dummy_img = Image.new('RGB', (100, 100), color='grey')
                dummy_target = {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros(0, dtype=torch.int64)}
                return transforms.ToTensor()(dummy_img), dummy_target
                
            img = Image.open(img_path).convert('RGB')
            
            # Parse annotations (boxes only, no masks)
            boxes, labels = [], []
            for shape in data['shapes']:
                points = shape['points']
                label = shape['label']

                # Convert string label to integer
                try:
                    label = int(label)
                except ValueError:
                    print(f"Warning: Invalid label: {label}, using default value 1.")
                    label = 1  # Use default value

                # Process coordinates
                if len(points) == 2:  # Rectangle
                    x_min, y_min = points[0]
                    x_max, y_max = points[1]
                elif len(points) > 2:  # Polygon
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                else:
                    continue  # Skip if not enough points
                
                # Check image boundaries and clip
                width, height = img.size
                x_min = max(0, min(float(x_min), width-1))
                y_min = max(0, min(float(y_min), height-1))
                x_max = max(0, min(float(x_max), width-1))
                y_max = max(0, min(float(y_max), height-1))
                
                # Validate box (width and height must be positive)
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)
                else:
                    print(f"Warning: Skipping invalid box coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

            # Create target dictionary (no masks)
            target = {
                'boxes': boxes,
                'labels': labels
            }

            # Apply image transforms
            if self.transform:
                img = self.transform(img)

            return img, target
            
        except Exception as e:
            print(f"Error occurred while loading data ({self.files[idx]}): {e}")
            # Return dummy data on error
            dummy_img = Image.new('RGB', (100, 100), color='grey')
            dummy_target = {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros(0, dtype=torch.int64)}
            return transforms.ToTensor()(dummy_img), dummy_target

# 2. Initialize and combine multiple datasets
data_dirs = [
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\S01_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\S02_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\S03_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\S04_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T13_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T14_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T15_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T16_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T17_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T18_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T19_dual_rotated",    
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T20_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T21_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T22_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T23_dual_rotated",
    r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\T24_dual_rotated"
]

# Create dataset for each directory
datasets = []
for data_dir in data_dirs:
    if os.path.exists(data_dir):
        dataset = LabelMeDataset(
            json_dir=data_dir,
            transform=transforms.ToTensor()
        )
        datasets.append(dataset)
    else:
        print(f"Warning: Directory does not exist - {data_dir}")

# Combine all datasets
if datasets:
    train_dataset = ConcatDataset(datasets)
    print(f"Total {len(train_dataset)} data loaded.")
else:
    print("Error: No valid datasets!")
    # Create dummy dataset
    train_dataset = LabelMeDataset(
        json_dir=r"C:\Users\Jun\Desktop\241226_RtoB\204.2x\S01_dual_rotated",
        transform=transforms.ToTensor()
    )

def collate_fn(batch):
    images, targets = [], []
    for b in batch:
        images.append(b[0])
        targets.append(b[1])
    return images, targets

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# 3. Model initialization (Faster R-CNN)
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 4  # Including background class
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 4. Define training function
def train_model(model, train_loader, num_epochs=40, lr=0.0001):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print("Starting training...")
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                
                # Check if all targets have boxes
                valid_batch = all(len(t['boxes']) > 0 for t in targets)
                if not valid_batch:
                    print(f"Warning: Batch {batch_idx} has targets without valid boxes. Skipping.")
                    continue
                
                try:
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

            # Calculate average loss (valid batches only)
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            scheduler.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Processed Batches: {batch_count}/{len(train_loader)}")
            
            # Save intermediate model (optional)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                checkpoint_path = f"model_checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Model checkpoint saved: {checkpoint_path}")
                
    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        print(f"Error during training: {e}")
    
    print("Training completed!")
    return model, optimizer, avg_loss

# 5. Visualization and save predictions for all images function
def visualize_and_save_predictions(model, dataset, output_dir, num_samples=None, threshold=0.5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Result save directory created: {output_dir}")
    
    # Determine number of samples
    total_samples = len(dataset)
    if num_samples is None or num_samples > total_samples:
        num_samples = total_samples
    
    # Select random indices (comment out if entire dataset needed)
    # indices = np.random.choice(total_samples, num_samples, replace=False)
    
    for idx in range(num_samples):
        try:
            # Get image and target from dataset
            # Direct access from ConcatDataset
            img, target = dataset[idx]
            
            img = img.to(device)

            with torch.no_grad():
                prediction = model([img])

            img = transforms.ToPILImage()(img.cpu())
            draw = ImageDraw.Draw(img)

            # Draw ground truth boxes (green)
            for box in target['boxes']:
                draw.rectangle(box.tolist(), outline="green", width=3)

            # Draw prediction boxes (red)
            for i, box in enumerate(prediction[0]['boxes']):
                if prediction[0]['scores'][i] > threshold:
                    draw.rectangle(box.tolist(), outline="red", width=3)
                    
                    # Display label and score
                    label = prediction[0]['labels'][i].item()
                    score = prediction[0]['scores'][i].item()
                    draw.text((box[0], box[1]), f"Class: {label}, {score:.2f}", fill="red")

            output_path = os.path.join(output_dir, f"visualized_{idx}.png")
            img.save(output_path)
            
            if (idx + 1) % 10 == 0:
                print(f"Image {idx + 1}/{num_samples} saved")
        except Exception as e:
            print(f"Error visualizing image {idx}: {e}")

# Execute training
trained_model, optimizer, avg_loss = train_model(model, train_loader, num_epochs=20, lr=0.0001)

# Save final model after training completion
model_save_dir = "C:\\Users\\Jun\\Desktop\\241226_RtoB\\models"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Save final checkpoint (for resuming training)
final_model_path = os.path.join(model_save_dir, "final_model.pth")
save_model(trained_model, final_model_path, optimizer, 20, avg_loss)

# Save model only (for inference)
model_only_path = os.path.join(model_save_dir, "model_only.pth")
save_model(trained_model, model_only_path)

# Save results
output_directory = "C:\\Users\\Jun\\Desktop\\241226_RtoB\\visualized_outputs_faster_rcnn_multi"
visualize_and_save_predictions(trained_model, train_dataset, output_directory, threshold=0.8)

print("All tasks completed.")

# Example code for loading saved model (commented out)
"""
# Load model for inference
loaded_model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 4
in_features = loaded_model.roi_heads.box_predictor.cls_score.in_features
loaded_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
loaded_model, _, _, _ = load_model(loaded_model, model_only_path)
loaded_model.eval()  # Set to inference mode

# Load model for resuming training
new_model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = new_model.roi_heads.box_predictor.cls_score.in_features
new_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
optimizer = torch.optim.Adam(new_model.parameters(), lr=0.0001)
new_model, optimizer, start_epoch, loss = load_model(new_model, final_model_path, optimizer)
# Resume training from start_epoch
"""