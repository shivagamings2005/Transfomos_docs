import cv2
import numpy as np
from scipy import fftpack
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessed = cv2.normalize(gray, None, 0, 25, cv2.NORM_MINMAX)
    background_removed = remove_background(preprocessed)
    background_removed=255-background_removed
    background_removed = np.where(background_removed < 120, 0, 255).astype(np.uint8)
    kernel = np.ones((1,1), np.uint8)
    background_removed = cv2.morphologyEx(background_removed, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow("imgfinal",background_removed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
image = cv2.imread(r"C:\Users\shiva\Downloads\test.jpg")  # Replace with your image path
remove_bg(image)
