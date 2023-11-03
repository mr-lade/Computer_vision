#%matplotlib inline
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from part1 import imgpoints,objp,objpoints


def calibrate():
    print ("********* CALIBRATION STARTING ...")
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("camera_cal/calibration_results/calibration_dist.p", "wb" ))
    
    print ("Testing One Sample Image ensuring it is working fine... ")
    udt = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img, cmap="gray")
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(udt)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    print ("********* CALIBRATION COMPLETED AND COEFFICIENTS STORED **********")


def undistortAll(source_location, target_location, mtx, dist):
    print ("UNDISTORT IMAGES STARTING")   
    test_images = glob.glob(source_location)
    count = 0
    
    for idx, fname in enumerate(test_images):
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        fn = target_location + fname[12:]
        x = cv2.imwrite(fn,dst)
        count = count + 1 if x else None

    print ("UNDISTORT IMAGES COMPLETED WITH COUNT - ", count, " ")
def undistort(image):
        cam_meta = pickle.load(open("camera_cal/calibration_results/calibration_dist.p", "rb"))
        mtx, dist = cam_meta['mtx'], cam_meta['dist']
        undst = cv2.undistort(image, mtx, dist, None, mtx)
        return undst
def abs_sobel_thresh(image_gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel=cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel=cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel= np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    sobelx=cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely=cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag= np.sqrt(sobelx*2 + sobely*2)
    scaled_factor = np.max(gradmag)/255
    gradmag = (gradmag/scaled_factor).astype(np.uint8)
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1  
    return mag_binary
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate directional gradient
    sobelx=cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely=cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir= np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    scaled_factor = np.max(absgraddir)/255
    scaled_absgraddir = (absgraddir/scaled_factor).astype(np.uint8)
    dir_binary = np.zeros_like(scaled_absgraddir)
    dir_binary[(scaled_absgraddir >= thresh[0]) & (scaled_absgraddir <= thresh[1])] = 1

    return dir_binary

# APPLY PERSPECTIVE ADJUSTMENT: INPUT - IMAGE, SOURCE COORDINATES, DESTINATION COORDINATES 
# OUTPUT - INVERSE COEFFICENTS & OPTIMIZED IMAGE
def warp (image, src, dst):
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped_img = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return Minv, warped_img

# APPLY COLOR GRADIENT TO S CHANNEL AND THERSHOLD: INPUT - IMAGE, THRESHOLD 
# OUTPUT - OPTIMIZED IMAGE
def color_threshold(img, thresh):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls_img[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

# FIND LANE LEFT(X.Y) AND RIGHT(X,Y) COORDINATES - INPUT: WARPED & GRADIENT OPTIMIZED IMAGE
# OUTPUT: LEFT(X.Y) AND RIGHT(X,Y)
def lane_finding(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int32(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 15
    window_height = np.int32(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 50
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
        
    return leftx, lefty, rightx, righty

def poly_fit(binary_warped, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return ploty, left_fitx, right_fitx

def curvature(ploty, leftx, lefty, rightx, righty):
    y_eval = np.max(ploty)
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700 

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def offset(result, pts_left, pts_right):
    xm_per_pix = 3.7 / 700
    lane_difference = pts_right[0, pts_right[0,:,1].argmax(), 0] - pts_left[0, pts_left[0,:,1].argmax(), 0]
    offset_center = (lane_difference - result.shape[1]/2) * xm_per_pix
    return offset_center 
def visualization(warped, ploty, left_fitx, right_fitx, Minv, undist, left_curverad, right_curverad,drawLane=True,drawZone=True,drawText=True):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    result = warped
    
    if drawLane:        
        #Draw Lane Boundaries
        margin=40
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        cv2.fillPoly(color_warp, np.int32([left_line_pts]), (0,0,255))
        cv2.fillPoly(color_warp, np.int32([right_line_pts]), (0,0,255))
        newwarp1 = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        result1 = cv2.addWeighted(undist, 1, newwarp1, 1, 0)

    if drawZone:
        # Draw the lane area onto the warped blank image as green
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        if drawLane:
            result = cv2.addWeighted(result1, 1, newwarp, 0.4, 0)
        else:
            result = cv2.addWeighted(color_warp, 1, newwarp, 0.4, 0)

    if drawText:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.5
        color = (255,255,255)
        text = "Left Curvature: {} m".format(round(left_curverad, 3))
        cv2.putText(result, text, (400, 50), font, font_size, color, 2)
        text = "Right Curvature: {} m".format(round(right_curverad, 3))
        cv2.putText(result, text, (400, 100), font, font_size, color, 2)
        center_offset = offset(result, pts_left, pts_right)
        text = "Vehicle is {}m left of center".format(round(center_offset, 3))
        cv2.putText(result, text,(400,150), font, font_size, color, 2)

    return result

def pipeline(image):
        
    image = undistort(image)
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    gradx = abs_sobel_thresh(image_gray, orient='x', sobel_kernel=11, thresh=(30, 150))
    
    grady = abs_sobel_thresh(image_gray, orient='y', sobel_kernel=11, thresh=(30, 200))
    
    gradmag = mag_thresh(image_gray, sobel_kernel=11, thresh=(30, 150))
    
    graddir = dir_threshold(image_gray, sobel_kernel=21, thresh=(100, 180))
    
    combined_gradient = np.zeros_like(graddir)
    combined_gradient[((gradx == 1) & (grady == 1)) | ((gradmag == 1) & (graddir == 1))] = 1
    
    grad_s = color_threshold(image, thresh=(80, 255))
        
    combined = np.zeros_like(grad_s)
    combined[(grad_s == 1) | (combined_gradient == 1)] = 1
    
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

    dst = np.float32([[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
    Minv, warp_img = warp(combined, src, dst)
        
    leftx, lefty, rightx, righty = lane_finding(warp_img)
    
    ploty, left_fitx, right_fitx = poly_fit(warp_img, leftx, lefty, rightx, righty)
    
    left_curverad, right_curverad = curvature(ploty, leftx, lefty, rightx, righty)
    
    image_with_overlay = visualization(warp_img, ploty, left_fitx, right_fitx, Minv, image, left_curverad, right_curverad)
        
    return image_with_overlay
