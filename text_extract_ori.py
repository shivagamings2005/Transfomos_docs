import sys
import torch
import numpy as np
import cv2

# Add CRAFT-pytorch directory to Python path
sys.path.append(r"D:\sih_check\CRAFT-pytorch")

from craft import CRAFT
from imgproc import resize_aspect_ratio, normalizeMeanVariance
from craft_utils import getDetBoxes, adjustResultCoordinates
from collections import OrderedDict

def copyStateDict(state_dict):
    """Copy state dictionary to remove module prefix if present."""
    if list(state_dict.keys())[0].startswith("module"):
        start_prefix = "module."
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[len(start_prefix):]
            new_state_dict[name] = v
        return new_state_dict
    return state_dict

def detect_text(image_path, trained_model_path='craft_mlt_25k.pth'):
    """
    Detect text regions in an image using CRAFT.
    
    Args:
        image_path (str): Path to input image
        trained_model_path (str): Path to pre-trained CRAFT model
    
    Returns:
        Annotated image with text detection boxes
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CRAFT model
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location=device)))
    net.eval()
    net = net.to(device)
    
    # Read image
    image = cv2.imread(image_path)
    image_copy = image.copy()
    
    # Resize and normalize image
    # Use cv2.INTER_AREA as interpolation, square_size as 1280, mag_ratio as 1 (default)
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, 
        square_size=1280, 
        interpolation=cv2.INTER_AREA
    )
    ratio_h = ratio_w = 1 / target_ratio
    
    # Prepare image tensor
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)   # [c, h, w] to [b, c, h, w]
    x = x.to(device)
    
    # Inference
    with torch.no_grad():
        y, feature = net(x)
    
    # Post-processing
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    
    # Get bounding boxes
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
    # Scale boxes back to original image size
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    # Draw rectangular boxes on image
    for box in boxes:
        box = np.int32(box)
        cv2.polylines(image_copy, [box], True, (0, 255, 0), 2)
    
    return image_copy

def main():
    # Example usage
    image_path = r"C:\Users\shiva\Downloads\test.jpg"  # Replace with your image path
    #image_path="img2.jpg"
    result_image = detect_text(image_path)
    
    # Display result
    result_image=cv2.resize(result_image,(1080,1920))
    cv2.imshow('Text Detection Result', result_image)
    #cv2.imwrite("crop2.png",result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()