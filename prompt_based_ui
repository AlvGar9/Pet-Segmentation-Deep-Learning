import gradio as gr
import torch
import numpy as np
import cv2
from torchvision.transforms import Normalize
from collections import OrderedDict
# Assuming the model class PointBasedSegmentationModel is defined in another file
# from model_definition import PointBasedSegmentationModel 

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointBasedSegmentationModel(num_classes=2).to(device)

# Load the model file
state_dict = torch.load("path/to/your/model.pt", map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.eval()

# Normalization
normalize_transform = Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))

# Pre-process the images
def preprocess(image, point):
    x_orig, y_orig = point
    
    # Resize and create heatmap
    image_resized = cv2.resize(image, (224, 224))
    x_scaled = int(x_orig * 224 / image.shape[1])
    y_scaled = int(y_orig * 224 / image.shape[0])

    # Normalize image
    image_norm = normalize_transform(torch.tensor(image_resized / 255.).permute(2, 0, 1)).unsqueeze(0)
    
    # Create heatmap
    heatmap = np.zeros((224, 224), dtype=np.float32)
    cv2.circle(heatmap, (x_scaled, y_scaled), 5, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (13, 13), 3)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    
    return image_norm.float().to(device), heatmap_tensor.float().to(device), (x_orig, y_orig)

def predict(image, point):
    if image is None:
        return None
    if point is None:
        return image # Return original image if no click
    
    try:
        image_tensor, heatmap_tensor, click_coords = preprocess(image, point)
        
        with torch.no_grad():
            logits = model(image_tensor, heatmap_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        # Resize mask to original image size
        pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # Create overlay
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[pred_mask_resized == 1] = [255, 0, 0] # Red for segmented object
        overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
        
        # Draw the click point
        cv2.circle(overlay, click_coords, 8, (255, 255, 255), -1) # White border
        cv2.circle(overlay, click_coords, 6, (0, 255, 0), -1) # Green center
        
        return overlay
    
    except Exception as e:
        print(f"Error: {e}")
        return image

with gr.Blocks() as demo:
    gr.Markdown("## Click anywhere to segment")
    with gr.Row():
        input_img = gr.Image(label="Input Image", type="numpy")
        output_img = gr.Image(label="Segmentation Result")

    click_point = gr.State()

    def capture_click(evt: gr.SelectData):
        print(f"Clicked at: {evt.index}")
        return evt.index

    input_img.select(
        fn=capture_click,
        inputs=None,
        outputs=click_point
    ).then(
        fn=predict,
        inputs=[input_img, click_point],
        outputs=output_img
    )

    def reset_click_point():
        return None
        
    input_img.change(
        fn=reset_click_point,
        inputs=None,
        outputs=click_point
    )

demo.launch()
