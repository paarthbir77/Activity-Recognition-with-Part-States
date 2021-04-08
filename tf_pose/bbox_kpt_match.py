import numpy as np

def check_point_in_box(point, lower_corner, upper_corner):

	if point[0] < upper_corner[0] and point[0] > lower_corner[0]:
		if point[1] < upper_corner[1] and point[1] > lower_corner[1]:
			return True
	return False

def bbox_kpt_match(persons_part_list, persons_bbox):

	matches = []
	num_people_bbox = len(persons_bbox)
	num_people_kpts = len(persons_part_list)
	#print("b ", num_people_bbox, "k ",num_people_kpts)
	scores = np.zeros((num_people_kpts, num_people_bbox))

	for i in range(num_people_bbox):
		for j in range(num_people_kpts):
			for k in range(len(persons_part_list[j])):
				if check_point_in_box([persons_part_list[j][k][1], persons_part_list[j][k][2]], persons_bbox[i][0], persons_bbox[i][1]):
					scores[j][i]+=1
	#print(scores)

	if num_people_kpts<=num_people_bbox:
		# that is distinct keypoint skeletons < person bounding boxes detected then kpts act as limitation. so assign each kpt to a box
		maxi = -1
		for i in range(num_people_kpts):
			maxi = -1
			max_index = -1
			for j in range(num_people_bbox):
				if maxi<scores[i][j]:
					maxi = scores[i][j]
					max_index = j
			matches.append([i, max_index])
	else:
		pass

	return matches


def clean_kpts(persons_part_list,persons_bbox):

	matches = bbox_kpt_match(persons_part_list, persons_bbox)
	for i in range(len(matches)):
		p_kpt = persons_part_list[matches[i][0]]
		p_box = persons_bbox[matches[i][1]]
		#print("kpt skeleton")
		#print(p_kpt)
		#print("bounding box")
		#print(p_box)
		out_of_box_pts = []
		for kpt in p_kpt:
			point = [kpt[1], kpt[2]]
			if check_point_in_box(point, p_box[0], p_box[1]):
				pass
			else:
				print(kpt, " not in box ", p_box)
				#persons_part_list[matches[i][0]].remove(kpt)
				out_of_box_pts.append(kpt)
		for kpt in out_of_box_pts:
			persons_part_list[matches[i][0]].remove(kpt)
		#cleaned_persons_part_list.append(p_kpt)
	return persons_part_list



#bbox = [[(12, 9), (526, 569)], [(679, 74), (993, 500)], [(383, 150), (706, 553)]]
'''kpts = [[['0', 746, 200], ['1', 805, 216], ['2', 718, 219], ['3', 699, 351], ['4', 721, 435], ['5', 893, 213], ['6', 927, 341], 
['7', 915, 457], ['8', 774, 429], ['11', 865, 423], ['14', 740, 178], ['15', 768, 188], ['17', 824, 175]]

, [['0', 659, 232], ['1', 574, 310], ['2', 490, 316], ['3', 503, 485], ['5', 662, 313], ['6', 674, 448], 
['7', 603, 463], ['14', 640, 216], ['16', 574, 238]], 

[['0', 322, 169], ['1', 219, 307], ['2', 62, 329], ['5', 334, 272], 
['14', 281, 135], ['15', 334, 138], ['16', 194, 157]]]'''

#bbox = [[(22, 371), (239, 1016)], [(500, 289), (811, 929)], [(209, 229), (589, 979)], [(646, 416), (1003, 972)]]
#kpts = [[['0', 378, 328], ['1', 381, 456], ['2', 278, 451], ['3', 234, 573], ['4', 244, 740], ['5', 487, 456], ['6', 496, 601], ['7', 475, 757], ['8', 284, 829], ['11', 418, 840], ['14', 356, 312], ['15', 393, 301], ['16', 328, 334], ['17', 428, 323]], [['0', 659, 367], ['1', 674, 506], ['2', 593, 506], ['3', 528, 668], ['4', 546, 562], ['5', 759, 501], ['8', 603, 935], ['11', 727, 941], ['14', 640, 345], ['15', 677, 345], ['16', 618, 378], ['17', 718, 367]], [['0', 905, 534], ['1', 855, 668], ['2', 759, 657], ['3', 765, 840], ['4', 715, 751], ['5', 943, 685], ['6', 930, 863], ['14', 877, 506], ['15', 927, 512], ['16', 821, 529], ['17', 955, 545]], [['0', 119, 501], ['1', 78, 573], ['2', 9, 562], ['5', 178, 584], ['6', 197, 757], ['7', 150, 685], ['11', 122, 974], ['14', 100, 462], ['15', 134, 473], ['16', 47, 445]]]
# out = [[0, 2], [1, 1], [2, 3], [3, 0]]  ---- kpt, bbox

#bbox = [[(214, 43), (352, 201)]]
#kpts = [[['0', 334, 86], ['1', 267, 162], ['2', 219, 159], ['5', 320, 164], ['6', 418, 235], ['7', 474, 295], ['14', 326, 73], ['16', 295, 83]]]
#print(bbox_kpt_match(kpts, bbox))
#print(clean_kpts(kpts, bbox))
