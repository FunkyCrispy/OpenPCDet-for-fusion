import os 
import copy
import glob
import cv2 as cv
import numpy as np


def cart_to_hom(pts):
	"""
	:param pts: (N, 3 or 2)
	:return pts_hom: (N, 4 or 3)
	"""
	pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
	return pts_hom


def lidar_to_rect(pts_lidar, V2C, R0):
	"""
	:param pts_lidar: (N, 3)
	:return pts_rect: (N, 3)
	"""
	pts_lidar_hom = cart_to_hom(pts_lidar)
	pts_rect = np.dot(pts_lidar_hom, np.dot(V2C.T, R0.T))
	return pts_rect


def rect_to_img(pts_rect, P2):
	"""
	:param pts_rect: (N, 3)
	:return pts_img: (N, 2)
	"""
	pts_rect_hom = cart_to_hom(pts_rect)
	pts_2d_hom = np.dot(pts_rect_hom, P2.T)
	pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
	pts_rect_depth = pts_2d_hom[:, 2] - P2.T[3, 2]  # depth in rect camera coord
	return pts_img, pts_rect_depth


def lidar_to_img(pts_lidar, V2C, R0, P2):
	"""
	:param pts_lidar: (N, 3)
	:return pts_img: (N, 2)
	"""
	pts_rect = lidar_to_rect(pts_lidar, V2C, R0)
	pts_img, pts_depth = rect_to_img(pts_rect, P2)
	return pts_img, pts_depth


def get_lidar(velodyne_name):
	pts_lidar = np.fromfile(velodyne_name, dtype=np.float32).reshape(-1, 4)
	return pts_lidar


def get_image(image_name):
	img_arr = cv.imread(image_name)
	# print(img_arr.shape) # (375, 1242, 3)
	return img_arr


def cls_type_to_id(cls_type):
	type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
	if cls_type not in type_to_id.keys():
		return -1
	return type_to_id[cls_type]


def get_label(label_name):
	lines = []
	with open(label_name, 'r') as f:
		lines = f.readlines()
		objs_2d = []
	for line in lines:
		label = line.strip().split(' ')
		cls_type = label[0]
		cls_id = cls_type_to_id(cls_type)
		obj_2d = np.array((int(float(label[4])), int(float(label[5])), int(float(label[6])), 
			int(float(label[7])), cls_id))
		objs_2d.append(obj_2d)
	return objs_2d


def get_calib(calib_name):
	lines = []
	with open(calib_name) as f:
		lines = f.readlines()

	obj = lines[2].strip().split(' ')[1:]
	P2 = np.array(obj, dtype=np.float32).reshape(3, 4)
	obj = lines[3].strip().split(' ')[1:]
	P3 = np.array(obj, dtype=np.float32).reshape(3, 4)
	obj = lines[4].strip().split(' ')[1:]
	R0 = np.array(obj, dtype=np.float32).reshape(3, 3)
	obj = lines[5].strip().split(' ')[1:]
	Tr_velo_to_cam = np.array(obj, dtype=np.float32).reshape(3, 4)

	return P2, P3, R0, Tr_velo_to_cam


def lidar_add_sem(velodyne_name):
	# lidar points
	pts_lidar = get_lidar(velodyne_name)
	pts_xyz = pts_lidar[:, :3]

	# create arr as image shape
	img_shape = (375, 1242)
	sem_arr = np.zeros(img_shape)

	# labels of objects
	label_name = velodyne_name.replace('velodyne', 'label_2').replace('bin', 'txt')
	objs_2d = get_label(label_name)
	for bbox in objs_2d:
		x1, y1, x2, y2, cls_id = bbox
		x1 = max(x1, 0)
		x2 = min(x2, img_shape[1])
		y1 = max(y1, 0)
		y2 = min(y2, img_shape[0])
		sem_arr[y1:y2+1, x1:x2+1] = cls_id

	# calib arrays
	calib_name = velodyne_name.replace('velodyne', 'calib').replace('bin', 'txt')
	P2, _, R0, V2C = get_calib(calib_name)
	
	# lidar points to image
	pts_img, _ = lidar_to_img(pts_xyz, V2C, R0, P2)

	# add semantic information to points
	pad = np.zeros((pts_lidar.shape[0], 4))
	for i in range(pts_img.shape[0]):
		x, y = pts_img[i].astype(np.int)
		if x >= img_shape[1] or x < 0 or y >= img_shape[0] or y < 0:
			pad[i][0] = 1
			continue
		if sem_arr[y, x] == 1:
			pad[i][1] = 1
		elif sem_arr[y, x] == 2:
			pad[i][2] = 1
		elif sem_arr[y, x] == 3:
			pad[i][3] = 1
		else:
			pad[i][0] = 1
	pts_lidar = np.concatenate((pts_lidar, pad), axis=1)

	# notify specific data type, because np.tofile cannot define data type
	# so you have to save and load with the same data type when using np.fromfile
	pts_lidar = pts_lidar.astype(np.float32)

	return pts_lidar


if __name__ == '__main__':
	
	path = '/home/fengchen/data/kitti/data_object_velodyne'
	velodyne_names = glob.glob(path + '/training/*/*.bin')

	save_path = path.replace('data_object_velodyne', 'data_object_velodyne_sem')
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	for v_name in velodyne_names:
		pts_lidar = lidar_add_sem(v_name)
		save_name = v_name.replace('data_object_velodyne', 'data_object_velodyne_sem')
		pts_lidar.tofile(save_name)
		print(v_name.split('/')[-1])