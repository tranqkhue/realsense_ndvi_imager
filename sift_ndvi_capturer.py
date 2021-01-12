import pyrealsense2 as rs
import numpy as np
import cv2

cv2.ocl.setUseOpenCL(True)

#=================================================================================

def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
       
    # Detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        #descriptor = cv2.xfeatures2d.SURF_create()
        descriptor = cv2.cuda.SIFT_SURF_create(300,_nOctaveLayers=2)
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # Get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

#=================================================================================

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

#=================================================================================
  
def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

#=================================================================================

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            							 reprojThresh)

        return (matches, H, status)
    else:
        return None

#=================================================================================

# Configure stream
pipeline  = rs.pipeline()
config    = rs.config()
config.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8,   30)
config.enable_stream(rs.stream.color,    1280, 720, rs.format.rgb8, 30)

#---------------------------------------------------------------------------------

# Start input streaming
pipeline.start(config)
# Disable laser emitter
ir_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
ir_sensor.set_option(rs.option.emitter_enabled, 0)
ir_sensor.set_option(rs.option.enable_auto_exposure, 1)
#ir_sensor.set_option(rs.option.exposure, 40000)
rgb_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
rgb_sensor.set_option(rs.option.enable_auto_white_balance, 0)
rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)
#rgb_sensor.set_option(rs.option.exposure, 400)

#---------------------------------------------------------------------------------
FEATURE_EXTRACTOR = 'sift'
CONNECTIVITY = 8
DRAW_CIRCLE_RADIUS = 4

# Ignore first 1sec for camera warm-up
for i in range(30):
	frames = pipeline.wait_for_frames()

try:
	while True:
		# Read frames
		frames     = pipeline.wait_for_frames()
		ir_frames  = frames.get_infrared_frame()
		rgb_frames = frames.get_color_frame()

		ir_intrin  = ir_frames.profile.as_video_stream_profile().intrinsics
		#print(ir_intrin.fx, ir_intrin.fy) 
		rgb_intrin = rgb_frames.profile.as_video_stream_profile().intrinsics
		#print(rgb_intrin.fx, rgb_intrin.fy) 

		# Convert to numpy image
		ir_frame   = np.asanyarray(ir_frames.get_data())
		rgb_frame  = cv2.resize(np.asanyarray(rgb_frames.get_data()), (854,480))
		gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

        # Resize to get the same scale as FOV of 2 sensors are different
		center  = np.array(ir_frame.shape) / 2
		w       = 1280*ir_intrin.fx/rgb_intrin.fx
		h       = 720*ir_intrin.fy/rgb_intrin.fy      
		crop_ir = ir_frame[int(center[0]-h/2):int(center[0]+h/2), \
						   int(center[1]-w/2):int(center[1]+w/2)]
		ir_frame 	= cv2.resize(crop_ir, 	 (854,480))
		#gray_frame 	= cv2.resize(gray_frame, (640,480)) 

		# Extract ir, red green and blue channel  
		ir_channel    = (ir_frame/256.0).astype('float32')
		blue_channel  = (rgb_frame[:,:,0]/256.0).astype('float32')
		green_channel = (rgb_frame[:,:,1]/256.0).astype('float32')  
		red_channel   = (rgb_frame[:,:,2]/256.0).astype('float32') 
		cv2.imshow('red', red_channel) 

		kpsA, featuresA = detectAndDescribe(ir_frame,   method=FEATURE_EXTRACTOR)
		kpsB, featuresB = detectAndDescribe(gray_frame,	method=FEATURE_EXTRACTOR)
		matches   = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, \
	    							  method=FEATURE_EXTRACTOR)
		M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
		if M is None:
		    print("Error!")
		else:
			(matches, H, status) = M
			print(H)
			aligned_ir = cv2.warpPerspective(ir_channel, H, (854,480))

			cv2.imshow('red',  red_channel)
			cv2.imshow('aligned_ir', aligned_ir)
			# Calculate ndvi  
			ndvi_image = cv2.subtract(aligned_ir.astype('float32'), red_channel.astype('float32'))/\
						 cv2.add(aligned_ir.astype('float32'), red_channel.astype('float32'))  
			ndvi_image = (ndvi_image+1)/2  
			ndvi_image = cv2.convertScaleAbs(ndvi_image*255)  
			ndvi_image = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_JET)  

			cv2.imshow('ndvi', ndvi_image) 

		# Exit on ESC key
		c = cv2.waitKey(1) % 0x100
		if c == 27:
			break

finally:
	pipeline.stop() 
	cv2.destroyAllWindows()