import os
# File to take 100 random images from HAKE training dataset
ct_total = 0
ct = 0 
images_dir = "/media/prth/My Passport/HAKE-Action/Data/hico_20160224_det/images/train2015/"
idir = "\"/media/prth/My Passport/HAKE-Action/Data/hico_20160224_det/images/train2015/"

for file in os.scandir(images_dir):
	ct+=1
	if ct == 42:
		command = "cp "+idir+file.name+'"'+" /home/prth/Desktop/pose_estimate/tf-pose-estimation/images/hake/"+str(ct_total)+".jpg"
		print(command)
		ct_total+=1
		ct = 0
		os.system(command)
	if ct_total>=500:
		break