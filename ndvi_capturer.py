import pyrealsense2 as rs
import numpy as np
import cv2

# Configure NDVI
# Define the motion model  
#warp_mode = cv2.MOTION_TRANSLATION  
warp_mode = cv2.MOTION_AFFINE
#warp_mode = cv2.MOTION_HOMOGRAPHY  
# Define 2x3 or 3x3 matrices and initialize the matrix to identity  
if (warp_mode == cv2.MOTION_HOMOGRAPHY):   
  warp_matrix = np.eye(3, 3, dtype=np.float32)  
else :  
  warp_matrix = np.eye(2, 3, dtype=np.float32)  
# Specify the number of iterations.  
number_of_iterations = 5000;  
# Specify the threshold of the increment  
# in the correlation coefficient between two iterations   
termination_eps = 1e-10;  
# Define termination criteria  
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)  


# Configure stream
pipeline  = rs.pipeline()
config    = rs.config()
config.enable_stream(rs.stream.infrared, 1280, 720, rs.format.y8,   30)
config.enable_stream(rs.stream.color,    1280, 720, rs.format.rgb8, 30)

# Start input streaming
pipeline.start(config)
# Disable laser emitter
ir_sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
ir_sensor.set_option(rs.option.emitter_enabled, 0)
ir_sensor.set_option(rs.option.enable_auto_exposure, 1)
#ir_sensor.set_option(rs.option.exposure, 100)
rgb_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
#rgb_sensor.set_option(rs.option.enable_auto_white_balance, 0)
rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)
#rgb_sensor.set_option(rs.option.exposure, 10)

# Ignore first 1sec for camera warm-up
for i in range(30):
	frames = pipeline.wait_for_frames()

try:
	frames     = pipeline.wait_for_frames()
	ir_frames  = frames.get_infrared_frame()
	rgb_frames = frames.get_color_frame()

	ir_intrin  = ir_frames.profile.as_video_stream_profile().intrinsics
	#print(ir_intrin.fx, ir_intrin.fy) 
	rgb_intrin = rgb_frames.profile.as_video_stream_profile().intrinsics
	#print(rgb_intrin.fx, rgb_intrin.fy) 

	# Convert to numpy image
	ir_image   = np.asanyarray(ir_frames.get_data())
	rgb_image  = np.asanyarray(rgb_frames.get_data()) 

	# Extract ir, red green and blue channel  
	ir_channel    = (ir_image/256.0).astype('float32')
	blue_channel  = (rgb_image[:,:,0]/256.0).astype('float32')
	green_channel = (rgb_image[:,:,1]/256.0).astype('float32')  
	red_channel   = (rgb_image[:,:,2]/256.0).astype('float32')  
  
	# Align the images  
	# Run the ECC algorithm. The results are stored in warp_matrix.  
	# Find size of image1  
	sz = rgb_image.shape  
	(cc, warp_matrix) = cv2.findTransformECC(green_channel, \
											 ir_channel,\
											 warp_matrix,\
											 warp_mode, criteria,\
											 inputMask=None, gaussFiltSize=1) 
	print('ok')
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
		ir_image   = np.asanyarray(ir_frames.get_data())
		rgb_image  = np.asanyarray(rgb_frames.get_data()) 

		# Extract ir, red green and blue channel  
		ir_channel    = (ir_image/256.0).astype('float32')
		blue_channel  = (rgb_image[:,:,0]/256.0).astype('float32')
		green_channel = (rgb_image[:,:,1]/256.0).astype('float32')  
		red_channel   = (rgb_image[:,:,2]/256.0).astype('float32')  

		if (warp_mode == cv2.MOTION_HOMOGRAPHY):  
			# Use warpPerspective for Homography   
			ir_aligned = cv2.warpPerspective(ir_channel, warp_matrix, (sz[1],sz[0]), \
											  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)  
		else :  
			# Use warpAffine for nit_channel, Euclidean and Affine  
			ir_aligned = cv2.warpAffine(ir_channel, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);  
	  
		# calculate ndvi  
		ndvi_image = (ir_aligned - red_channel)/(ir_aligned + red_channel)  
		ndvi_image = (ndvi_image+1)/2  
		ndvi_image = cv2.convertScaleAbs(ndvi_image*255)  
		ndvi_image = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_JET)  

		cv2.imshow("Image", ndvi_image) 

		# Exit on ESC key
		c = cv2.waitKey(1) % 0x100
		if c == 27:
			break

finally:
	pipeline.stop() 
	cv2.destroyAllWindows()