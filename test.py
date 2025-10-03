import cv2
import numpy as np

def classify_and_mark_colors(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gaussian_kernel= np.array([
        [0.075, 0.124, 0.075],
        [0.124, 0.204, 0.124],
        [0.075, 0.124, 0.075]
    ], dtype=np.float32)
    hsv_smoothed = cv2.filter2D(hsv, -1, gaussian_kernel) # Apply Gaussian smoothing -1 means same depth as source
    color_ranges = {
        'black_n_grey': {
            'lower': np.array([0, 0, 0]),
            'upper': np.array([180, 10, 255]),
            'bgr_color': (0, 0, 0)
        },
        'white': {

            'lower': np.array([0, 0, 220]),
            'upper': np.array([180, 30, 255]),  
            'bgr_color': (255, 255, 0)
        },
        'red': {
            'lower1': np.array([0, 5,0]),
            'upper1': np.array([50, 255, 255]),
            'lower2': np.array([150, 5, 0]),
            'upper2': np.array([180, 255, 255]),
            'bgr_color': (0, 0, 255)
        },
    }
    output_img = img.copy()

    for color_name, params in color_ranges.items():
        if color_name == 'red':
            mask1 = cv2.inRange(hsv_smoothed, params['lower1'], params['upper1'])
            mask2 = cv2.inRange(hsv_smoothed, params['lower2'], params['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_smoothed, params['lower'], params['upper'])

        output_img[mask > 0] = params['bgr_color']

    # cv2.imshow('Original Image', img)
    # cv2.imshow('Color Classified Image', output_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    file_name = image_path.split('/')
    cv2.imwrite(f'output/{file_name[len(file_name) - 1]}.jpg', output_img)
    print(f'Output saved to output/{file_name[len(file_name) - 1]}')
import os
if __name__ == "__main__":
    # open /input folder to input image path
    input_folder = 'input'
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            classify_and_mark_colors(image_path)
            print(f'Processed {image_path}')