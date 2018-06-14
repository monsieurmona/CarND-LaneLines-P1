# **Writeup: Finding Lane Lines on the Road** 

## Overview
When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

The **goal** of this project is to find lane lines on the road

## Reflection

### 1. Pipeline Description 

The pipeline to process an image consists of the following steps. 

1. Convert image to HSV color space. 
	
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
This allows to mask colors in an image, way better than using the RGB color space. Trying to mask colors in RGB space is difficult, as mixing the Red, Green and Blue channel gives in different colors. We are interested only in yellow and white, in a certain range of saturation and brightness. 

2. Masking colors of interest

        # mask image for white pixels
        mask_white_pixels = cv2.inRange(hsv_image, np.array([0, 0, 178]),np.array([180, 25, 255]) )
             
        # mask image for yellow pixels
        mask_yellow_pixels = cv2.inRange(hsv_image, np.array([15, 80, 191]),np.array([25, 255, 255]) 
       
        # combine masked images
        masked_image = cv2.bitwise_or(mask_white_pixels,mask_yellow_pixels)
Two different masks are generated to seperate the colors yellow and white from the original image with two range sets. The three values of each arrays above are Hue, Saturation and Value (brightness), while in OpenCV Hue counts from 0 to 180, Saturation and Value from 0 to 255. The left array and right array specify the lower and upper bounds. Both masks are combined afterwards to a single one. This resulting mask is directly used for the next step. 

3. Masks the region of interest. 

        # Mask the region of interest
        imshape = original_image.shape
        vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
        masked_image = region_of_interest(masked_image, vertices)
The mask is shaped like a trapeze. All pixels outside of the mask are blacked out to retain the lane in front of the car.

4. Blur and Edge detection

        # edge detection
        masked_image = gaussian_blur(masked_image,7)
        edge_image = canny(masked_image, 50, 150)
The image is blured first to reduce noise in the image. Edges are dected with the algorithm from Canny. 

5. Hough Lines
 
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2               # distance resolution in pixels of the Hough grid
        theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 7        # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 19      # maximumgap in pixels between connectable line segments
    
        # Get the hough lines
        hough_lines_image = hough_lines(edge_image, rho, theta, threshold, min_line_length, max_line_gap)
All the edges found by the Canny edge detector are used as input for the hough_lines algorithm to find lines along the lane markings. 

6. Draw lines on top of the image

        # Draw hough lines on top of the orignal image
        hough_lines_image_color = cv2.cvtColor(hough_lines_image,cv2.COLOR_RGB2BGR)
        hough_lines_mask = grayscale(hough_lines_image)
        not_needed, hough_lines_mask = cv2.threshold(hough_lines_mask, 10, 255, cv2.THRESH_BINARY)
        hough_lines_mask = cv2.bitwise_not(hough_lines_mask)
        original_image_masked = cv2.bitwise_and(original_image,original_image,mask = hough_lines_mask)
        original_image_with_hough_lines = cv2.bitwise_or(original_image_masked, hough_lines_image_color)
The found lines are used first to generate a mask. Those are taken to black out the the original images at the line locations. Finally the red hough lines are drawn on top. I have chosen this way of merging lines with the original images it improves a lot the appearance, as the contrast is larger. 

#### Combine Hough Lines to Two Single Lines
In order to draw a single line on the left and right lanes, I modified the draw_lines() function. I assume one line should be drawn on the left side of the image and another one on the right side. 

    line_length_left = np.array([])
    line_length_right = np.array([])
    xMid = img.shape[1] / 2
    max_extend = 330

At the next step, I try to find a line length threshold for each side. As the longes lines are probably the ones that define lane markings best. All line length are calculated and inserted into an array. 
    
    # calculate the treshold for a minimum length
    for line in lines:
        for x1,y1,x2,y2 in line:
            dy = y2-y1
            dx = x2-x1

            # length calculated without square root
            # as the real length is not needed
            length = dx**2 + dy**2
            if (x1 < xMid and x2 < xMid):
                line_length_left = np.append(line_length_left, [length])
            elif (x1 > xMid and x2 > xMid):
                line_length_right = np.append(line_length_right, [length])
                     
The line lengths will be sorted in descending order. The seventh length is the one with minimum threshold. An initial threshold is also give to get rid of noise.
         
    line_length_left[::-1].sort()
    line_length_right[::-1].sort()
    min_line_length_left = 300
    min_line_length_right = 300
    max_element_idx = 6
    
    # get the length of the 7th longest line
    # as minmum length threshold
    if (len(line_length_left) > max_element_idx):
        length = line_length_left[max_element_idx]
        if (length > min_line_length_left):
            min_line_length_left = length 

    if (len(line_length_right) > max_element_idx):
        length = line_length_right[max_element_idx]
        if (length > min_line_length_right):
            min_line_length_right = length

The code belows how the lines are filtered by minimum threshold and slope. 

    linesLeftX = np.array([])
    linesLeftY = np.array([])
    linesRightX = np.array([])
    linesRightY = np.array([])
        
    # group lines to left or right lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            dy = y2-y1
            dx = x2-x1
            
            # length calculated witout square root
            # as the real length is not needed
            length = dx**2 + dy**2

            # filter by minimum line length
            if (dy < 0 and x1 < xMid and x2 < xMid and length >= min_line_length_left):
                # left side
                angle = dx / np.sqrt(length)
                
                if (angle > 0.7 and angle < 0.9):
                    linesLeftX = np.append(linesLeftX, [x1,x2])
                    linesLeftY = np.append(linesLeftY, [y1,y2])
                    
            elif (x1 > xMid and x2 > xMid and length >= min_line_length_right):
                # right side
                angle = dx / np.sqrt(length)

                if (angle > 0.7 and angle < 0.9):
                    linesRightX = np.append(linesRightX, [x1,x2])
                    linesRightY = np.append(linesRightY, [y1,y2])

The remaining points of the lines are used for curve fitting. 
                        
    if (len(linesLeftX) > 1):
        z = np.polyfit(linesLeftY, linesLeftX, 1)
        p = np.poly1d(z)
        cv2.line(img, (int(p(img.shape[0])), img.shape[0]), (int(p(max_extend)), max_extend), color, thickness)
            
    if (len(linesRightX) > 1):
        z = np.polyfit(linesRightY, linesRightX, 1)
        p = np.poly1d(z)
        cv2.line(img, (int(p(img.shape[0])), img.shape[0]), (int(p(max_extend)), max_extend), color, thickness)

The two liniar functions are drawn then into the image.

### 2. Identify potential shortcomings with your current pipeline

Left turns, right turns, changing lanes, different kinds of lane markings are not handled with the pipeline. Only a straight lines without to much of curvature are assumed to be lane divider. 

The pipeline is not able to interpolate or correct lane lines based on anothers. 

### 3. Suggest possible improvements to your pipeline

Change the perspective, so that straight lanes a parallel to each other. Use vertical line (Haar) features for line detection. Use an integral image to speed up the processes.

Measure confidence and update lane dividers accordingly. A line doesn't go criss cross from one image to another. 

Keep track of lanes. That may help to estimate lane dividers.

Use a lane marking with high confidence to interpolate the other with low confidence.

Use a polynomial line along lane dividers

