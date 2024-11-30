from ultralytics import YOLO
import cv2
import numpy as np
import os


model_path = 'D:/Users/Eryn/Git/Road Damage Segmentation/run/train5/weights/best.pt'
input_dir = 'D:/Users/Eryn/Git/Road Damage Segmentation/test-images'
output_dir = 'D:/Users/Eryn/Git/Road Damage Segmentation/test-result' 
 

# Create test-result directory 
os.makedirs(output_dir, exist_ok=True)


model = YOLO(model_path)

class_colors = {
    0: (100, 100, 255),  # Crack: Red
    1: (255, 100, 100),  # Pothole: Blue    
}


for filename in os.listdir(input_dir):
    
        test_path = os.path.join(input_dir, filename)
        pred_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}-pred.jpg")

        # Load test image
        img = cv2.imread(test_path)
        H, W, _ = img.shape

        # Runs the model on the image
        results = model(img)

        
        for result in results:
            if len(result) < 1:
                continue

            # Iterate through each mask detected
            for mask, class_id in zip(result.masks.data, result.boxes.cls):

                # Convert mask tensor to NumPy array
                mask = mask.numpy() * 255

                # Resize mask to input image dimensions
                mask = cv2.resize(mask, (W, H)).astype(np.uint8)

                # Get the mask color from class_colors based on class_id
                color = class_colors.get(int(class_id), (255, 255, 255))  

                # Create a black mask
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                
                # Blue channel
                colored_mask[:, :, 0] = (mask / 255) * color[0]  

                # Green channel
                colored_mask[:, :, 1] = (mask / 255) * color[1]  

                # Red channel
                colored_mask[:, :, 2] = (mask / 255) * color[2]  

                # Add the colored mask on top of the original image
                img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

       
        cv2.imwrite(pred_path, img)

        print(f"Prediction is saved to: {pred_path}")

print("Model test completed.")
