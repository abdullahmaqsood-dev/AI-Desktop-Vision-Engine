import cv2
import numpy as np
img = cv2.imread('pic.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', gray)

blur = cv2.bilateralFilter(gray, 5, 20, 20)
edges = cv2.Canny(blur, 10, 70)
cv2.imwrite('edges.png', edges)

kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
edges = cv2.dilate(edges, kernel_h, iterations=1)
cv2.imwrite('edges_dilated.png', edges)

countours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
result = img.copy()
detected_buttons = []

for cnt in countours:
    perimeter = cv2.arcLength(cnt,True)
    if perimeter == 0:
        continue
    vertices = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

    area = cv2.contourArea(cnt)
    if area < 100:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    is_rectangle = False
    is_circle = False
    if circularity > 0.8:
        is_circle = True
    elif len(vertices) >=4 and len(vertices) <=8:
        if w>10 or h>10:
            is_rectangle = True
    if not is_rectangle and not is_circle:
        continue
    
    is_duplicate = False
    for btn in detected_buttons:
        bcx = btn['x'] + btn['w']//2
        bcy = btn['y'] + btn['h']//2
        if abs(bcx - (x + w//2)) < 10 and abs(bcy - (y + h//2)) < 10:
            is_duplicate = True
            break
    if (is_rectangle or is_circle) and not is_duplicate:
        detected_buttons.append(
            {'x': x, 'y': y, 'w': w, 'h': h, 'shape': 'rectangle' if is_rectangle else 'circle'}
        )

print(f"Detected {len(detected_buttons)} buttons:")

print(f"Total buttons to draw: {len(detected_buttons)}")

for btn in detected_buttons:
    x, y, w, h = btn['x'], btn['y'], btn['w'], btn['h']
    
    # Check the shape we saved earlier (default to rectangle just in case)
    shape = btn.get('shape', 'rectangle') 
    
    if shape == 'circle':
        # Calculate the center point of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Calculate the radius (using max ensures the whole button fits inside)
        radius = max(w, h) // 2
        
        # Draw the circle: image, center, radius, color (BGR), thickness
        cv2.circle(result, (center_x, center_y), radius, (255, 0, 0), 2)
        
    else:
        # Draw the standard rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite('detected_buttons.png', result)


