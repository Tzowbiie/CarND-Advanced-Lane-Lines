class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
        # yvalues for ploting
        self.ploty = None
        
def imgSobel(img, s_thresh, sx_thresh, v_thresh):
    print('we are somewhere esle')
    img = np.copy(img)
    """Convert to HLS and HSV color space. Using S and V channel for finding lane pixels"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
    # Stack each channel
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, v_binary)) * 255
    binary = (s_binary + v_binary + sxbinary)*255
    ret,thresh1 = cv2.threshold(binary,0,255,cv2.THRESH_BINARY)
    
    return thresh1

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def setROI(img):
    """Sets a Region of interest"""
    x_fac=int(img.shape[0]/20)
    y_fac=int(img.shape[1]/20)
    print('Image width ',img.shape[1])
    #setting ROI - independent of image size
    vertices = np.array([[[0,18*x_fac],[8.5*y_fac,450],[11.5*y_fac,450],[20*y_fac,18*x_fac]]],dtype=np.int32) 
    roi=region_of_interest(img, vertices)
    return roi

def transformImage(img):
        '''Transforms image of road to birds eye view'''
        x_fac=int(img.shape[0]/20)
        y_fac=int(img.shape[1]/20)
        '''Src are the source points in the original image, dst are the same points but on the road viewed from above'''
        src = np.float32([[0,18*x_fac],[8.6*y_fac,450],[11.4*y_fac,450],[20*y_fac,18*x_fac]])
        dst = np.float32([[0,20*x_fac],[0,0],[20*y_fac,0],[20*y_fac,20*x_fac]])
        
        M=cv2.getPerspectiveTransform(src,dst)
        warped = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
        return warped
    
def transformImageBack(img):
        '''Transforms image of bird eye view to road'''
        x_fac=int(img.shape[0]/20)
        y_fac=int(img.shape[1]/20)
        '''Src are the source points in the original image, dst are the same points but on the road viewed from above'''
        dst = np.float32([[0,18*x_fac],[8.6*y_fac,450],[11.4*y_fac,450],[20*y_fac,18*x_fac]])
        src = np.float32([[0,20*x_fac],[0,0],[20*y_fac,0],[20*y_fac,20*x_fac]])
        
        M=cv2.getPerspectiveTransform(src,dst)
        warped = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
        return warped
    
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 30
    # Set the width of the windows +/- margin
    margin = 150
    # Set minimum number of pixels found to recenter window
    minpix = 15

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = int(leftx_current-margin)  # Update this
        win_xleft_high = int(leftx_current+margin)  # Update this
        win_xright_low = int(rightx_current-margin)  # Update this
        win_xright_high = int(rightx_current+margin) # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current=int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current=int(np.mean(nonzerox[good_right_inds]))
       # else: 
        #if not enough pixels, change margin (in case of tight curves)
           # margin = 300
            # Set minimum number of pixels found to recenter window
            #minpix = 20
            #window=window-1
            #pass # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    #set parameters of Line
    left_lane.allx=left_fitx
    left_lane.ally=ploty
    right_lane.allx=right_fitx
    right_lane.ally=ploty
    print('Right Lane Current fit ',right_fit)
    right_lane.current_fit=right_fit
    left_lane.current_fit=left_fit
    ## Visualization ##
    # Colors in the left and right lane regions+plt.figure(5)
    plt.figure(4)
       
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imshow(out_img)
    return out_img, left_fit, right_fit, ploty

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    left_lane.current_fit=left_fit
    right_lane.current_fit=right_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_lane.allx=left_fitx
    left_lane.ally=ploty
    right_lane.allx=right_fitx
    right_lane.ally=ploty
    
    return left_fitx, right_fitx

def search_around_poly(binary_warped,left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
   
    # Fit new polynomials
    left_fitx, right_fitx = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
   
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, left_lane.ally]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              left_lane.ally])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, left_lane.ally]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              left_lane.ally])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    plt.figure(5)
    plt.imshow(binary_warped)
    plt.figure(4)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    
    #draw_points = np.vstack((left_fitx,ploty)).astype(np.int32).T
    #cv2.polylines(window_img, [draw_points], False, (255,255,0),15)  
    #draw_points_right = np.vstack((right_fitx,ploty)).astype(np.int32).T
    #cv2.polylines(window_img, [draw_points_right], False, (255,255,0),15) 
    
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    #Plot the polynomial lines onto the image
    #plt.plot(left_fitx, left_lane.ally, color='yellow')
    #plt.plot(right_fitx, left_lane.ally, color='yellow')
    #plt.imshow(result)
    # End visualization steps ##
    return result

def measure_curvature_pixels(ploty,left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    #definition of scale: Pixel to real world mm
    # 700 pixels = width of US lane = 3.7 m
    # 1 pixel = 0.00503 m
    ym_per_pix = 25/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty).astype(int)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature in m) #####
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    left_lane_pos=np.mean(left_lane.allx[y_eval-10:y_eval])
    right_lane_pos=np.mean(right_lane.allx[y_eval-10:y_eval])
    image_mid=1280/2
    lane_mid=735/2
    left_lane.line_base_pos=(lane_mid-(image_mid-left_lane_pos))*xm_per_pix
    right_lane.line_base_pos=(lane_mid-(right_lane_pos-image_mid))*xm_per_pix
    print('dist from car center to left line: ',left_lane.line_base_pos,'dist from lane car center to left line: ',right_lane.line_base_pos)
    #print('all left lane pixels',left_lane.allx)
    #print('all left lane pixels',left_lane.ally)
    return left_curverad, right_curverad

def undistort_image(image):
    img_undist = cv2.undistort(image, dist_pickle["mtx"], dist_pickle["dist"], None, mtx)
    return img_undist

def sanity_check_lines():
    """sanity check for lane candidates
    1. check wether polynomial coefficients of the last lanes are similar to new candidate of the actual frame.
       IF yes -> left_lane.best_fit is updated with new lane
    2. Check wether poly coefficients of both lanes are similar compared to each other
       IF yes -> both lanes are found = TRUE"""
    abs_val_left=np.absolute(left_lane.current_fit - left_lane.best_fit)
    abs_val_right=np.absolute(right_lane.current_fit - right_lane.best_fit)
    
    if abs_val_left[0]<np.float64(1.1):
        if abs_val_left[1]<np.float64(1.1):
            if abs_val_left[2]<np.float64(500):
                best_fit=np.add(left_lane.best_fit,left_lane.best_fit)
                left_lane.best_fit =np.divide(best_fit+left_lane.current_fit,3)
                left_lane.detected=True
                print('found left', left_lane.best_fit)
    else:
        left_lane.detected=False
    if abs_val_right[0]<np.float64(1.1):
        if abs_val_right[1]<np.float64(1.1):
            if abs_val_right[2]<np.float64(1000):
                best_fit_r=np.add(right_lane.best_fit,right_lane.best_fit)
                right_lane.best_fit =np.divide(best_fit_r+right_lane.current_fit,3)
                right_lane.detected=True
                print('found right', right_lane.best_fit)
    else:
        right_lane.detected=False
    abs_val_best=np.absolute(right_lane.best_fit - left_lane.best_fit)
    if (left_lane.detected==True) & (right_lane.detected==True):
        if abs_val_best[0]<np.float64(1.1):
            if abs_val_best[1]<np.float64(1.1):
                if abs_val_best[2]<np.float64(800):
                    left_lane.detected=True
                    right_lane.detected=True
                    print('Both Lines detected!')
    else:
        left_lane.detected=False
        right_lane.detected=False
    
    return
def draw_lanes_mean(img):
    """Drawing the lanes with the averaged poly coefficients: .best_fit"""
    left_fitx = left_lane.best_fit[0]*left_lane.ally**2 + left_lane.best_fit[1]*left_lane.ally + left_lane.best_fit[2]
    right_fitx = right_lane.best_fit[0]*left_lane.ally**2 + right_lane.best_fit[1]*left_lane.ally + right_lane.best_fit[2]
    draw_points = np.vstack((left_fitx,left_lane.ally)).astype(np.int32).T
    cv2.polylines(img, [draw_points], False, (255,255,0),15)
    
   # draw_points_right = (np.asarray([right_fitx, ploty]).T).astype(np.int32)
    draw_points_right = np.vstack((right_fitx,right_lane.ally)).astype(np.int32).T
    cv2.polylines(img, [draw_points_right], False, (255,255,0),15) 

    return img
def visualize_radius(image,lanes_warped):
    #visualisation
    road_with_lanes = cv2.addWeighted(image, 0.5, lanes_warped, 1, 0)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 0.5
    fontColor              = (255,0,0)
    lineType               = 1
    cv2.putText(road_with_lanes,left_lane.radius_of_curvature.astype(str), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    bottomLeftCornerOfText = (10,480)
    cv2.putText(road_with_lanes,'Left Lane Radius in m:', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        (255,0,0),
        lineType)
    bottomLeftCornerOfText = (1100,500)
    cv2.putText(road_with_lanes,right_lane.radius_of_curvature.astype(str), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        (0,255,0),
        lineType)
    bottomLeftCornerOfText = (1100,480)
    cv2.putText(road_with_lanes,'Right Lane Radius in m:', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        (0,255,0),
        lineType)
    bottomLeftCornerOfText = (600,200)
    cv2.putText(road_with_lanes,left_lane.line_base_pos.astype(str), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        (0,255,255),
        lineType)
    bottomLeftCornerOfText = (600,180)
    cv2.putText(road_with_lanes,'Deviation from lane center in m:', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        (0,255,255),
        lineType)
    
    plt.imshow(road_with_lanes)
    
    return road_with_lanes