import sys
import torch
import cv2
import numpy as np

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

def detect_text(image, device, trained_model_path='craft_mlt_25k.pth', y_threshold=5):
    # Set up device
    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location=device)))
    net.eval()
    net = net.to(device)
    
    # Read image
    image = cv2.imread(image)
    image_copy = image.copy()
    
    # Resize and normalize image
    img_resized, target_ratio, _ = resize_aspect_ratio(
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
        y, _ = net(x)
    
    # Post-processing
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    
    # Get bounding boxes
    boxes, _ = getDetBoxes(score_text, score_link, text_threshold=0.62, link_threshold=0.4, low_text=0.2)
    
    # Scale boxes back to original image size
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    # Precompute box centers
    centers = [(np.mean(box[:, 0]), np.mean(box[:, 1])) for box in boxes]
    
    # Combine boxes with their centers for easy processing
    boxes_with_centers = list(zip(boxes, centers))
    
    # Sort boxes by Y-axis center for initial clustering
    boxes_with_centers.sort(key=lambda b: b[1][1])
    
    # Efficient row clustering using a sweep line approach
    rows = []
    current_row = [boxes_with_centers[0]]
    
    for i in range(1, len(boxes_with_centers)):
        _, center = boxes_with_centers[i]
        _, prev_center = current_row[-1]
        
        if abs(center[1] - prev_center[1]) <= y_threshold:
            current_row.append(boxes_with_centers[i])
        else:
            rows.append(current_row)
            current_row = [boxes_with_centers[i]]
    
    # Add the last row
    if current_row:
        rows.append(current_row)
    
    # Sort each row by X-axis center
    for row in rows:
        row.sort(key=lambda b: b[1][0])
    
    # Extract and draw bounding boxes
    ordered_words = []
    word_border=[]
    h,w,_=image.shape
    for row in rows:
        for box, _ in row:
            box = np.int32(box)
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            xmin, xmax = int(np.min(x_coords)), int(np.max(x_coords))
            ymin, ymax = int(np.min(y_coords)), int(np.max(y_coords))
            xmin=max(0,xmin)
            xmax=min(w,xmax)
            ymin=max(0,ymin)
            ymax=min(h,ymax)
            # Crop and store the word
            word_image = image_copy[ymin:ymax, xmin:xmax]
            ordered_words.append(word_image)
            word_border.append(box)
            #cv2.imshow("img",word_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            # Draw rectangle using cv2.rectangle()
            cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    
    return image_copy, ordered_words,word_border



def main():
    # Example usage
    image_path = r"C:\Users\shiva\Downloads\hindi_hand.jpg"  # Replace with your image path
    result_image, ordered_words,word_border = detect_text(image_path, torch.device('cpu'))
    print(word_border)
    # Display result
    cv2.imwrite("img2.png",result_image)

if __name__ == "__main__":
    main()