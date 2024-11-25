import argparse
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import UNet

def load_model(checkpoint_path, device):
    """
    Load the trained model from the checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model on.
        
    Returns:
        model (nn.Module): Loaded model.
    """
    model = UNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=2,
        decoder_attention=True,
        dropout_rate=0.5,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, device, image_size=(256, 256)):
    """
    Preprocess the input image for the model.
    
    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device to load the image on.
        image_size (tuple): Desired image size.
        
    Returns:
        image_tensor (torch.Tensor): Preprocessed image tensor.
        original_size (tuple): Original image size.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    return image_tensor, original_size

def postprocess_mask(mask, original_size):
    """
    Postprocess the predicted mask to match the original image size.
    
    Args:
        mask (np.ndarray): Predicted mask.
        original_size (tuple): Original image size.
        
    Returns:
        mask_resized (np.ndarray): Resized mask.
    """
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0.5).astype(np.uint8)
    return mask_resized

def save_segmented_image(original_image_path, mask, output_path):
    """
    Overlay the mask on the original image and save the result.
    
    Args:
        original_image_path (str): Path to the original image.
        mask (np.ndarray): Binary mask.
        output_path (str): Path to save the segmented image.
    """
    image = cv2.imread(original_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = mask[:, :, 0] * 255  # Red channel for neoplastic
    colored_mask[:, :, 1] = mask[:, :, 1] * 255  # Green channel for non-neoplastic
    
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    plt.imsave(output_path, overlay)
    
def main():
    parser = argparse.ArgumentParser(description="Inference Script for NeoPolyp Segmentation")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image (e.g., image.jpeg)")
    args = parser.parse_args()
    
    # Configuration
    CHECKPOINT_PATH = "checkpoints/best_model.pth.tar"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = load_model(CHECKPOINT_PATH, DEVICE)
    
    # Preprocess the image
    image_tensor, original_size = preprocess_image(args.image_path, DEVICE)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = torch.sigmoid(outputs).cpu().numpy()[0]  # [2, H, W]
    
    # Process predictions for each class
    mask_neoplastic = predictions[0]
    mask_non_neoplastic = predictions[1]
    
    # Postprocess masks
    mask_neoplastic = postprocess_mask(mask_neoplastic, original_size)
    mask_non_neoplastic = postprocess_mask(mask_non_neoplastic, original_size)
    
    # Combine masks (for visualization)
    combined_mask = np.stack([mask_neoplastic, mask_non_neoplastic], axis=-1)
    
    # Save the segmented image
    output_image_path = "segmented_" + os.path.basename(args.image_path)
    save_segmented_image(args.image_path, combined_mask, output_image_path)
    
    print(f"Segmented image saved to {output_image_path}")

if __name__ == "__main__":
    main()
