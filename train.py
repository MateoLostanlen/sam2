# Train/Fine-Tune SAM 2 on smokeseg Dataset
# Inspired by the tutorial: 
# "Train/Fine-Tune Segment Anything 2 (SAM 2) in 60 Lines of Code"
# Tutorial Link: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Original Repository: https://github.com/facebookresearch/segment-anything-2

import wandb
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import glob
import random
import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
import random

# Function to Load Data
def load_smokeseg_data(ds_path):
    imgs = glob.glob(ds_path)
    random.seed(13)
    random.shuffle(imgs)

    data = []
    for file in imgs:
        if os.path.isfile(file.replace("images", "masks")):
            data.append({"image": file, "annotation": file.replace("images", "masks")})

    print(f"Loaded {len(data)} samples.")
    return data


class SmokeSegDataset(Dataset):
    def __init__(self, ds_path, train=True, img_size=1024, transform=None, threshold=127):
        """
        Custom PyTorch Dataset for loading smokeseg images and binary annotations.
        
        Args:
            ds_path (str): Path to the dataset.
            train (bool): If True, use a random point; if False, use the point closest to the centroid.
            img_size (int): Size to which images and masks will be resized.
            transform (callable, optional): Optional transform to apply to images and masks.
            threshold (int): Threshold value for binarizing the annotation mask.
        """
        self.data = load_smokeseg_data(ds_path)
        self.img_size = img_size
        self.transform = transform
        self.threshold = threshold
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]

        # Load image and annotation
        Img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
        ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Load annotation as grayscale

        # Resize image and annotation
        r = np.min([self.img_size / Img.shape[1], self.img_size / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

        # Threshold annotation to binary (0 and 255)
        _, binary_mask = cv2.threshold(ann_map, self.threshold, 255, cv2.THRESH_BINARY)
        binary_mask[binary_mask > 0] = 1

        # Ensure binary_mask has the correct shape (H, W)
        binary_mask = binary_mask.astype(np.uint8)

        # Extract all positive points
        points = np.argwhere(binary_mask > 0)  # Get all positive points

        if len(points) > 0:
            if self.train:
                # Random point for training
                random_point = points[np.random.randint(len(points))]
                points = np.array([[[random_point[1], random_point[0]]]])  # Single random point (x, y)
            else:
                # Closest point to the centroid for evaluation
                centroid = np.mean(points, axis=0)  # Calculate the centroid
                distances = np.linalg.norm(points - centroid, axis=1)  # Compute Euclidean distances
                closest_point_index = np.argmin(distances)  # Index of the closest point
                closest_point = points[closest_point_index]  # The closest point itself
                points = np.array([[[closest_point[1], closest_point[0]]]])  # Format as (1, 1, 2)
        else:
            points = np.zeros((1, 1, 2), dtype=np.float32)  # Default point if no positive points

        # Prepare output
        labels = np.ones((1, 1), dtype=np.float32)  # Binary label (always 1 for binary masks)

        return Img, np.expand_dims(binary_mask, axis=0), points, labels


# Initialize wandb
wandb.init(project="sam2-smokeseg", name="fine-tune-sam2", config={"learning_rate": 1e-5, "batch_size": 1})

# Load dataset
train_dataset = SmokeSegDataset("../ds_ft_sam2_v2/train/images/*")
val_dataset = SmokeSegDataset("../ds_ft_sam2_v2/val/images/*")


# Load model
sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters
predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder

optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scaler = torch.amp.GradScaler('cuda')  # mixed precision

# Training loop
epochs = 10
# Initialize variables to track the best validation loss
best_val_loss = float("inf")  # Set to infinity initially
for e in range(epochs):
    metrics = {"seg_loss": [], "score_loss": [], "mean_iou": [], "iou_std": [], "step_time": []}
    for itr, (image, mask, input_point, input_label) in enumerate(train_dataset):
        start_time = time.time()
        with torch.amp.autocast('cuda'):
            predictor.set_image(image) # apply SAM image encoder to the image
    
            # prompt encoding
    
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)
    
            # mask decoder
    
            batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution
    
            # Segmentaion Loss caclulation
    
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss
    
            # Score loss calculation (intersection over union) IOU
    
            inter = (gt_mask * (prd_mask > 0.5)).sum()
            iou = inter / (gt_mask.sum() + (prd_mask > 0.5).sum() - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05  # mix losses
    
            # apply back propogation
    
            predictor.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision
    
            # Metrics
            mean_iou = np.mean(iou.cpu().detach().numpy())
            iou_std = np.std(iou.cpu().detach().numpy())
            step_time = time.time() - start_time
            metrics["seg_loss"].append(seg_loss.item())
            metrics["score_loss"].append(score_loss.item())
            metrics["mean_iou"].append(mean_iou)
            metrics["iou_std"].append(iou_std)
            metrics["step_time"].append(step_time)
    
        if (itr + 1) % 10 == 0:  # Log metrics every 10 steps
            wandb.log({
                "step": itr,
                "seg_loss": np.mean(metrics["seg_loss"][-10:]),
                "score_loss": np.mean(metrics["score_loss"][-10:]),
                "mean_iou": np.mean(metrics["mean_iou"][-10:]),
                "iou_std": np.mean(metrics["iou_std"][-10:]),
                "step_time": np.mean(metrics["step_time"][-10:]),
            })
    
        if itr % 1000 == 0:
            torch.save(predictor.model.state_dict(), "model.torch")
            print(f"Model saved at step {itr}.")
    
        print(f"Step {itr} | Mean IoU: {mean_iou:.4f} | Loss: {loss.item():.4f}")
    
    # Validation step
    val_metrics = {"mean_iou": [], "seg_loss": []}
    with torch.no_grad():
        for val_image, val_mask, val_input_point, val_input_label in val_dataset:
            predictor.set_image(val_image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(val_input_point, val_input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels), boxes=None, masks=None)
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            gt_mask = torch.tensor(val_mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()
            inter = (gt_mask * (prd_mask > 0.5)).sum()
            iou = inter / (gt_mask.sum() + (prd_mask > 0.5).sum() - inter)
            val_metrics["mean_iou"].append(iou.item())
            val_metrics["seg_loss"].append(seg_loss.item())
    # Calculate average validation metrics
    val_mean_iou = np.mean(val_metrics["mean_iou"])
    val_avg_seg_loss = np.mean(val_metrics["seg_loss"])

    wandb.log({
        "epoch": e,
        "val_mean_iou": val_mean_iou,
        "val_seg_loss": val_avg_seg_loss,
    })

    print(f"Epoch {e} | Validation Mean IoU: {val_mean_iou:.4f} | Validation Loss: {val_avg_seg_loss:.4f}")

    # Save the model checkpoint if this is the best validation loss
    if val_avg_seg_loss < best_val_loss:
        best_val_loss = val_avg_seg_loss  # Update best loss
        model_dict = {"model": predictor.model.state_dict()}
        checkpoint_path = "best_model.pt"
        torch.save(model_dict, checkpoint_path)  # Save model checkpoint
        
        # Upload checkpoint to W&B as an artifact
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        
        print(f"Saved and uploaded new best model to W&B at epoch {e} with validation loss {best_val_loss:.4f}.")
    print(f"Epoch {e} | Validation Mean IoU: {np.mean(val_metrics['mean_iou']):.4f} | Validation Loss: {np.mean(val_metrics['seg_loss']):.4f}")
