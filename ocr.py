"""
CONFIGURATION STEPS
- git clone https://github.com/clovaai/CRAFT-pytorch.git
- cd CRAFT-pytorch
- Download : https://drive.usercontent.google.com/download?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ&export=download&authuser=0
- set path in line 26 and 156
- open CRAFT-pytorch\basenet\vgg16_bn.py 
    - remove line 7 and 26
"""
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import warnings
import sys
import torch
import numpy as np
from collections import OrderedDict
import numpy as np
from scipy import fftpack
warnings.filterwarnings("ignore")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-stage1')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-stage1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sys.path.append(r"D:\sih_check\CRAFT-pytorch")
from craft import CRAFT
from imgproc import resize_aspect_ratio, normalizeMeanVariance
from craft_utils import getDetBoxes, adjustResultCoordinates

def remove_background(image):
    f_transform = fftpack.fft2(image)
    f_shifted = fftpack.fftshift(f_transform)
    
    # Create a mask to suppress high-frequency components
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-1:crow+1, ccol-1:ccol+1] = 0
    
    # Apply frequency domain filtering
    f_shifted_filtered = f_shifted * mask
    f_ishift = fftpack.ifftshift(f_shifted_filtered)
    img_back = fftpack.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize the image
    img_back = cv2.normalize(img_back,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoising(img_back, None, 8, 5, 21)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(10,10))
    final_image = clahe.apply(denoised)
    
    return final_image

def remove_bg(image):
    preprocessed = cv2.normalize(image, None, 0, 25, cv2.NORM_MINMAX)
    background_removed = remove_background(preprocessed)
    background_removed=255-background_removed
    background_removed = np.where(background_removed < 120, 0, 255).astype(np.uint8)
    kernel = np.ones((1,1), np.uint8)
    background_removed = cv2.morphologyEx(background_removed, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("imgfinal",background_removed)
    return background_removed
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_prefix = "module."
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[len(start_prefix):]
            new_state_dict[name] = v
        return new_state_dict
    return state_dict

def merge_boxes_into_lines(boxes, overlap_threshold=0.5):
    if len(boxes) == 0:
        return []

    # Convert boxes to numpy arrays if they aren't already
    boxes = [np.array(box) for box in boxes]
    
    # Calculate vertical bounds for each box
    box_bounds = []
    for box in boxes:
        min_y = np.min(box[:, 1])
        max_y = np.max(box[:, 1])
        height = max_y - min_y
        box_bounds.append((min_y, max_y, height))
    
    # Initialize clusters
    clusters = []
    used_boxes = set()
    
    # Sort boxes by vertical position
    sorted_indices = sorted(range(len(boxes)), key=lambda i: box_bounds[i][0])
    
    for i in sorted_indices:
        if i in used_boxes:
            continue
            
        current_cluster = [i]
        used_boxes.add(i)
        
        # Compare with remaining boxes
        for j in sorted_indices:
            if j in used_boxes:
                continue
                
            # Get box properties
            min_y1, max_y1, h1 = box_bounds[current_cluster[0]]
            min_y2, max_y2, h2 = box_bounds[j]
            
            # Calculate overlap
            overlap_height = min(max_y1, max_y2) - max(min_y1, min_y2)
            smaller_height = min(h1, h2)
            
            if overlap_height > 0:
                overlap_ratio = overlap_height / smaller_height
                if overlap_ratio >= overlap_threshold:
                    current_cluster.append(j)
                    used_boxes.add(j)
        
        clusters.append([boxes[idx] for idx in current_cluster])
    
    # Merge boxes in each cluster
    merged_boxes = []
    for cluster in clusters:
        if cluster:
            merged_box = merge_line_boxes(cluster)
            merged_boxes.append(merged_box)
    
    return merged_boxes

def merge_line_boxes(line_boxes):
    """Create a single bounding box that encompasses all boxes in a line."""
    all_points = np.vstack([box.reshape(-1, 2) for box in line_boxes])
    min_x = np.min(all_points[:, 0])
    max_x = np.max(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_y = np.max(all_points[:, 1])
    
    return np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
def crop_text_line(image, box, padding=5):
    min_x = max(0, int(np.min(box[:, 0])) - padding)
    max_x = min(image.shape[1], int(np.max(box[:, 0])) + padding)
    min_y = max(0, int(np.min(box[:, 1])) - padding)
    max_y = min(image.shape[0], int(np.max(box[:, 1])) + padding)
    
    # Crop the image section
    return image[min_y:max_y, min_x:max_x]
def detect_text_lines(image,original_image,trained_model_path='craft_mlt_25k.pth', overlap_threshold=0.5):
    # Load CRAFT model
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location=device)))
    net.eval()
    net = net.to(device)
    
    # Read image
    image_copy = original_image.copy()
    
    # Resize and normalize image
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image, 
        square_size=1280, 
        interpolation=cv2.INTER_AREA
    )
    ratio_h = ratio_w = 1 / target_ratio
    
    # Prepare image tensor
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = x.to(device)
    
    # Inference
    with torch.no_grad():
        y, _ = net(x)
    
    # Post-processing
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    
    # Get word-level boxes
    boxes, _ = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    # Merge boxes into lines
    line_boxes = merge_boxes_into_lines(boxes, overlap_threshold)
    
    # Draw line boxes on image
    cropped_lines = []
    for _, box in enumerate(line_boxes):
        box = np.int32(box)
        # Draw box on the annotated image
        cv2.polylines(image_copy, [box], True, (0, 255, 0), 2)
        # Crop text line
        cropped_line = crop_text_line(original_image, box)
        cropped_lines.append(cropped_line)
    
    return image_copy, cropped_lines
def make_box(image,original_image):
    # Adjust overlap_threshold based on your needs
    image=remove_bg(image)
    result_image,cropped_lines = detect_text_lines(image,original_image,overlap_threshold=0.5)
    
    # Display result
    cv2.imshow('Text Line Detection Result', result_image)
    return cropped_lines
def extract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images=make_box(gray,image)
    result=""
    for image in images:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result+=('\n'+generated_text)
        print(result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result
img=cv2.imread(r"C:\Users\shiva\Downloads\written_image.png")
print("\n\n")
print(extract(img))