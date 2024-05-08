import cv2
import numpy as np


# Create Edge Mask
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edge = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

    return edge


# Color Quantization
def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result


# Cartoon Filter
def cartoon_filter(frame, line_size, blur_value, k):
    # Create Edge Mask
    edge = edge_mask(frame, line_size, blur_value)

    # # Color Quantization
    #img_quantized = color_quantization(frame, k)
    
    # Reduce Noise
    #blurred = cv2.bilateralFilter(img_quantized, d=7, sigmaColor=200, sigmaSpace=200)
    blurred = cv2.bilateralFilter(frame, d=9, sigmaColor=9, sigmaSpace=7)

    # Apply Edge Mask
    cartoon_image = cv2.bitwise_and(blurred, blurred, mask=edge)

    return cartoon_image


# Open Camera
cap = cv2.VideoCapture(0)

# Get camera properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
output = cv2.VideoWriter("output_video4.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

count_frame = 0
set_line_size = 3
set_blur_value = 3

while True:
    print(count_frame)

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Apply Cartoon Filter to the frame
    # cartoon_image = cartoon_filter(frame, line_size=9, blur_value=5, k=9) if count_frame > 100 else frame
    # cartoon_image = cartoon_filter(frame, line_size=3, blur_value=3, k=9) if count_frame > 100 else frame
    set_line_size = 3 + 2 * (count_frame // 80) if set_line_size < 17 else 17
    set_blur_value = 3 + 2 * (count_frame // 100) if set_blur_value < 5 else 5
    cartoon_image = cartoon_filter(frame, line_size=set_line_size, blur_value=set_blur_value, k=9) if count_frame > 50 else frame

    count_frame += 1

    # Write the cartoon image to the output video
    output.write(cartoon_image)

    # Display the cartoon image
    cv2.imshow("Cartoon Filter", cartoon_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
output.release()

# Close all windows
cv2.destroyAllWindows()
