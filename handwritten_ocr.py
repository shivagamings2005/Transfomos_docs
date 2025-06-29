import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import warnings
warnings.filterwarnings("ignore")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def extract(image):
    result=""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray=b_w.preprocess_image(r"C:\Users\shiva\Downloads\written_image.png")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),key=lambda b: b[1][1], reverse=False))
    heights = [b[3] for b in bounding_boxes]
    avg_height = sum(heights) / len(heights)
    lines = []
    current_line = [bounding_boxes[0]]
    for box in bounding_boxes[1:]:
        if abs(box[1] - current_line[-1][1]) < avg_height * 0.5:  
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]
    lines.append(current_line)
    for line in lines:
        x = min([box[0] for box in line])
        y = min([box[1] for box in line])
        w = max([box[0] + box[2] for box in line]) - x
        h = max([box[1] + box[3] for box in line]) - y
        if w>5:
            img=image[y:y+h,x:x+w]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            result+=(' '+generated_text)
        print(result)
    #cv2.imwrite("{c}.jpg",image)
    return result
#img=cv2.imread(r"C:\Users\shiva\Downloads\written_image.png")
#img=cv2.imread("img.jpg")
#print(extract(img))