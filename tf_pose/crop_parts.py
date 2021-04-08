# Helper file to crop body parts identified by pose estimation
# Body Joint Id 
# 0 : Nose, 1: Neck
# 2: L Shoulder, 3: L Elbow, 4: L Wrist
# 5: R Shoulder, 6: R Elbow, 7: R Wrist
# 8: L Hip, 9: L Knee, 10: L Ankle
# 11: R Hip, 12: R Knee, 13: R Ankle
# 14: L Eye, 15: R Eye, 16: L Ear, 17: R Ear
# Body Part Id:
# 0: Head
# 1: Left arm, 2: Right Arm
# 3: Left hand, 4: Right hand
# 5: Left leg, 6:Right leg
import cv2
import math
import numpy as np

def get_distance(x1,y1,x2,y2):
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def check_part(part_id, part_list):
	for i in range(len(part_list)):
		if str(part_id) == part_list[i][0]:
			return i
	return -1

def get_left_arm(part_list):
	l_elbow = check_part(3, part_list)
	#l_elbow_x, l_elbow_y = part_list[l_elbow][1], part_list[l_elbow][2]
	if l_elbow>=0:
		l_elbow_x, l_elbow_y = part_list[l_elbow][1], part_list[l_elbow][2]
		l_shoulder = check_part(2, part_list)
		#l_shoulder_x, l_shoulder_y = part_list[l_shoulder][1], part_list[l_shoulder][2]
		if l_shoulder>=0:
			l_shoulder_x, l_shoulder_y = part_list[l_shoulder][1], part_list[l_shoulder][2]
			#print("l elbow and l shoulder", l_elbow_x,l_elbow_y,l_shoulder_x, l_shoulder_y)

			d = get_distance(l_elbow_x,l_elbow_y, l_shoulder_x, l_shoulder_y)

			tan_theta = (l_shoulder_y - l_elbow_y)/(l_shoulder_x - l_elbow_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)

			#print( l_elbow_x, l_elbow_y, int(d * max(abs(cos_theta), abs(sin_theta))) )
			#return (l_elbow_x, l_elbow_y, int(d/1.414))
			return (l_elbow_x, l_elbow_y, int(d * max(abs(cos_theta), abs(sin_theta))) )
			
			#l_elbow_x+
		else:
			print("l elbow", l_elbow_x, l_shoulder_y)
			return (-1,-1,0)
			#pass
	else:
		return (-1,-1,0)

def get_right_arm(part_list):
	r_elbow = check_part(6, part_list)
	#l_elbow_x, l_elbow_y = part_list[l_elbow][1], part_list[l_elbow][2]
	if r_elbow>=0:
		r_elbow_x, r_elbow_y = part_list[r_elbow][1], part_list[r_elbow][2]
		r_shoulder = check_part(5, part_list)
		#l_shoulder_x, l_shoulder_y = part_list[l_shoulder][1], part_list[l_shoulder][2]
		if r_shoulder>=0:
			r_shoulder_x, r_shoulder_y = part_list[r_shoulder][1], part_list[r_shoulder][2]
			#print("r elbow and r shoulder", r_elbow_x,r_elbow_y,r_shoulder_x, r_shoulder_y)

			d = get_distance(r_elbow_x,r_elbow_y, r_shoulder_x, r_shoulder_y)

			tan_theta = (r_shoulder_y - r_elbow_y)/(r_shoulder_x - r_elbow_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)

			#print(r_elbow_x, r_elbow_y, int(d * max(abs(cos_theta), abs(sin_theta))))
			#return (r_elbow_x, r_elbow_y, int(d/1.414))
			return (r_elbow_x, r_elbow_y, int(d * max(abs(cos_theta), abs(sin_theta))) )
			#l_elbow_x+
		else:
			print("l elbow", r_elbow_x, r_shoulder_y)
			return (-1,-1,0)
			#pass
	else:
		return (-1,-1,0)

def get_left_hand(part_list):
	l_wrist = check_part(4, part_list)
	if l_wrist>=0:
		l_wrist_x, l_wrist_y = part_list[l_wrist][1], part_list[l_wrist][2]
		l_elbow = check_part(3, part_list)
		if l_elbow>=0:
			l_elbow_x, l_elbow_y = part_list[l_elbow][1], part_list[l_elbow][2]

			d = get_distance(l_wrist_x, l_wrist_y, l_elbow_x,l_elbow_y)
			
			tan_theta = (l_elbow_y - l_wrist_y)/(l_elbow_x - l_wrist_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)

			#print(l_wrist_x+int(d/1.414),l_wrist_y+int(d/1.414), l_wrist_x-int(d/1.414),l_wrist_y-int(d/1.414))
			#print("left_hand ",l_wrist_x, l_wrist_y, theta, int(d * max(abs(cos_theta), abs(sin_theta))/2))
			return(l_wrist_x, l_wrist_y, int(d * max(abs(cos_theta), abs(sin_theta)) / 2 ))
			#print("l Wrist and l elbow")
		else:
			print("l wrist")
			return (-1,-1,0)
	else:
		return (-1,-1,0)

def get_right_hand(part_list):
	r_wrist = check_part(7, part_list)
	if r_wrist>=0:
		r_wrist_x, r_wrist_y = part_list[r_wrist][1], part_list[r_wrist][2]
		r_elbow = check_part(6, part_list)
		if r_elbow>=0:
			r_elbow_x, r_elbow_y = part_list[r_elbow][1], part_list[r_elbow][2]

			d = get_distance(r_wrist_x, r_wrist_y, r_elbow_x,r_elbow_y)
			
			tan_theta = (r_elbow_y - r_wrist_y)/(r_elbow_x - r_wrist_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)

			#print("right_hand ",r_wrist_x, r_wrist_y, theta, int(d * max(abs(cos_theta), abs(sin_theta))/2))
			return(r_wrist_x, r_wrist_y, int(d * max(abs(cos_theta), abs(sin_theta)) / 2 ))

		else:
			print("r wrist")
			return (-1,-1,0)
	else:
		return (-1,-1,0)

def get_left_leg(part_list):
	l_knee = check_part(9, part_list)
	if l_knee>=0:
		l_knee_x, l_knee_y = part_list[l_knee][1], part_list[l_knee][2]
		l_hip = check_part(8, part_list)
		if l_hip>=0:
			l_hip_x, l_hip_y = part_list[l_hip][1], part_list[l_hip][2]
			
			d = get_distance(l_knee_x, l_knee_y, l_hip_x, l_hip_y)
			tan_theta = (l_hip_y - l_knee_y)/(l_hip_x - l_knee_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)

			return (l_knee_x, l_knee_y, int(d * max(abs(cos_theta), abs(sin_theta))) )

		else:
			print("l knee")
			return (-1,-1,0)
	else:
		return (-1, -1,0)

def get_right_leg(part_list):
	r_knee = check_part(12, part_list)
	if r_knee>=0:
		r_knee_x, r_knee_y = part_list[r_knee][1], part_list[r_knee][2]
		r_hip = check_part(11, part_list)
		if r_hip>=0:
			r_hip_x, r_hip_y = part_list[r_hip][1], part_list[r_hip][2]
			
			d = get_distance(r_knee_x, r_knee_y, r_hip_x, r_hip_y)
			tan_theta = (r_hip_y - r_knee_y)/(r_hip_x - r_knee_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)

			return (r_knee_x, r_knee_y, int(d * max(abs(cos_theta), abs(sin_theta))) )

		else:
			print("r knee")
			return (-1,-1,0)
	else:
		return (-1, -1,0)

def get_left_foot(part_list):
	l_ankle = check_part(10, part_list)
	if l_ankle >= 0:
		l_ankle_x, l_ankle_y = part_list[l_ankle][1], part_list[l_ankle][2]
		l_knee = check_part(9, part_list)
		if l_knee >= 0:
			l_knee_x, l_knee_y = part_list[l_knee][1], part_list[l_knee][2]

			d = get_distance(l_ankle_x, l_ankle_y, l_knee_x, l_knee_y)
			tan_theta = (l_knee_y - l_ankle_y)/(l_knee_x - l_ankle_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)

			return (l_ankle_x, l_ankle_y, int(d * max(abs(cos_theta), abs(sin_theta)) /2) )


			#print("l ankle and l knee")
		else:
			print("l ankle")
			return (-1,-1,0)
	else:
		return (-1, -1,0)

def get_right_foot(part_list):
	r_ankle = check_part(13, part_list)
	if r_ankle >= 0:
		r_ankle_x, r_ankle_y = part_list[r_ankle][1], part_list[r_ankle][2]
		r_knee = check_part(12, part_list)
		if r_knee >= 0:
			r_knee_x, r_knee_y = part_list[r_knee][1], part_list[r_knee][2]

			d = get_distance(r_ankle_x, r_ankle_y, r_knee_x, r_knee_y)
			tan_theta = (r_knee_y - r_ankle_y)/(r_knee_x - r_ankle_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)

			return (r_ankle_x, r_ankle_y, int(d * max(abs(cos_theta), abs(sin_theta)) /2) )


			#print("l ankle and l knee")
		else:
			print("l ankle")
			return (-1,-1,0)
	else:
		return (-1, -1,0)

def get_head(part_list):
	nose = check_part(0,part_list)
	if nose >= 0:
		nose_x, nose_y = part_list[nose][1], part_list[nose][2]
		neck = check_part(1,part_list)
		if neck>=0:
			neck_x, neck_y = part_list[neck][1], part_list[neck][2]
			d = get_distance(nose_x, nose_y, neck_x, neck_y)
			tan_theta = (neck_y - nose_y)/(neck_x - nose_x + 0.0001)
			theta = math.atan(tan_theta)
			cos_theta = math.cos(theta)
			sin_theta = math.sin(theta)
			#print("head")
			return (nose_x, nose_y, int(d * max(abs(cos_theta), abs(sin_theta))) )
		else:
			return (-1,-1,0)
	else:
		return (-1,-1,0)

def crop_parts(part_list):
	print("This script will crop body parts")
	# Call functions to get co-ordinates of body part bounding box if detected
	part_bbox = []
	part_bbox.append([0, get_head(part_list)])
	part_bbox.append([1 ,get_left_arm(part_list)])
	part_bbox.append([2 ,get_right_arm(part_list)])
	part_bbox.append([3 ,get_left_hand(part_list)])
	part_bbox.append([4 ,get_right_hand(part_list)])
	part_bbox.append([5, get_left_leg(part_list)])
	part_bbox.append([6, get_right_leg(part_list)])
	part_bbox.append([7, get_left_foot(part_list)])
	part_bbox.append([8, get_right_foot(part_list)])
	

	#get_left_leg(part_list)
	#get_right_leg(part_list)
	#get_left_foot(part_list)
	#get_right_foot(part_list)
	return part_bbox