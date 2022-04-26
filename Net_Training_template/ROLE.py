from __future__ import absolute_import
import random
from random import randint
import math
import pyblur
import os
from PIL import Image
from PIL import ImageEnhance
from skimage.measure import label as skimage_label
import numpy as np
import cv2


def CheckCollision(DropList):
	"""
	This function handle the collision of the drops
	"""
	
	listFinalDrops = []
	Checked_list = []
	list_len = len(DropList)
	# because latter raindrops in raindrop list should has more colision information
	# so reverse list	
	DropList.reverse()
	drop_key = 1
	for drop in DropList:
		# if the drop has not been handle	
		if drop.getKey() not in Checked_list:			
			# if drop has collision with other drops
			if drop.getIfColli():
				# get collision list
				collision_list = drop.getCollisionList()
				# first get radius and center to decide how  will the collision do
				final_x = drop.getCenters()[0] * drop.getRadius()
				final_y = drop.getCenters()[1]  * drop.getRadius()
				tmp_devide = drop.getRadius()
				final_R = drop.getRadius()  * drop.getRadius()
				for col_id in collision_list:				
					Checked_list.append(col_id)
					# list start from 0				
					final_x += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[0]
					final_y += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[1]
					tmp_devide += DropList[list_len - col_id].getRadius()
					final_R += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getRadius() 
				final_x = int(round(final_x/tmp_devide))
				final_y = int(round(final_y/tmp_devide))
				final_R = int(round(math.sqrt(final_R)))
				# rebuild drop after handled the collisions
				newDrop = raindrop(drop_key, (final_x, final_y), final_R)
				drop_key = drop_key+1
				listFinalDrops.append(newDrop)
			# no collision
			else:
				drop.setKey(drop_key)
				drop_key = drop_key+1
				listFinalDrops.append(drop)			
	

	return listFinalDrops


def generateDrops(imagePath, cfg, inputLabel = None):
	"""
	This function generate the drop with random position
	"""
	
	maxDrop = cfg["maxDrops"]
	minDrop = cfg["minDrops"]
	drop_num = randint(minDrop, maxDrop)
	maxR = cfg["maxR"]
	minR = cfg["minR"]
	ifReturnLabel = cfg["return_label"]
	edge_ratio = cfg["edge_darkratio"]

	PIL_bg_img = Image.open(imagePath)
	bg_img = np.asarray(PIL_bg_img)
	# to check if collision or not
	label_map = np.zeros_like(bg_img)[:,:,0]
	imgh, imgw, _ = bg_img.shape
	
	
	# random drops position
	ran_pos = [(int(random.random() * imgw), int(random.random() * imgh)) for _ in range(drop_num)]

	listRainDrops = []
	#########################
	# Create Raindrop
	#########################
	# create raindrop by default
	if inputLabel is None:
		for key, pos in enumerate(ran_pos):
			# label should start from 1
			key = key+1
			radius = random.randint(minR, maxR)
			drop = raindrop(key, pos, radius)
			listRainDrops.append(drop)
	#using input label			
	else:
		arrayLabel = np.asarray(inputLabel)
		# get alpha						
		condition = (arrayLabel[:,:,0]>cfg["label_thres"])		
		label = np.where(condition, 1, 0)

		label_part, label_nums = skimage_label(label, connectivity=2, return_num = True)		
		for idx in range(label_nums):
			# 0 is bg
			i = idx+1
			label_index = np.argwhere(label_part==i)
			U = np.min(label_index[:,0])
			D = np.max(label_index[:,0]) + 1
			L = np.min(label_index[:,1])
			R = np.max(label_index[:,1]) + 1
			cur_alpha = arrayLabel[U:D, L:R, 0].copy()
			#cur_alpha[(cur_alpha<=cfg["label_thres"])] = 0
			
			cur_label = (cur_alpha>cfg["label_thres"]) * 1	
			
			# store left top
			centerxy = (L, U)			
			drop = raindrop(idx, centerxy = centerxy, input_alpha = cur_alpha, input_label = cur_label)
			listRainDrops.append(drop)
			
		
			
	
	
		
			
	#########################
	# Handle Collision
	#########################
	
	collisionNum = len(listRainDrops)
	listFinalDrops = list(listRainDrops)
	loop = 0
	
	# only check when using default raindrop
	if inputLabel is None:
		while collisionNum > 0:		
			loop = loop+1
			listFinalDrops = list(listFinalDrops)
			collisionNum = len(listFinalDrops)
			label_map = np.zeros_like(label_map)
			# Check Collision
			for drop in listFinalDrops:		
				# check the bounding 
				(ix, iy) = drop.getCenters()
				radius = drop.getRadius()
				ROI_WL = 2*radius
				ROI_WR = 2*radius
				ROI_HU = 3*radius
				ROI_HD = 2*radius
				if (iy-3*radius) <0 :
					ROI_HU = iy	
				if (iy+2*radius)>imgh:
					ROI_HD = imgh - iy
				if (ix-2*radius)<0:
					ROI_WL = ix			
				if  (ix+2*radius) > imgw:
					ROI_WR = imgw - ix

				# apply raindrop label map to Image's label map
				drop_label = drop.getLabelMap()


				# check if center has already has drops
				if (label_map[iy, ix] > 0):
					col_ids = np.unique(label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR])
					col_ids = col_ids[col_ids!=0]
					drop.setCollision(True, col_ids)
					label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR] = drop_label[3*radius - ROI_HU:3*radius + ROI_HD, 2*radius - ROI_WL: 2*radius+ROI_WR] * drop.getKey()
				else:			
					label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR] = drop_label[3*radius - ROI_HU:3*radius + ROI_HD, 2*radius - ROI_WL: 2*radius+ROI_WR] * drop.getKey()
					# no collision 
					collisionNum = collisionNum-1


			if collisionNum > 0:
				listFinalDrops = CheckCollision(listFinalDrops)	
			

	
	# add alpha for the edge of the drops
	alpha_map = np.zeros_like(label_map).astype(np.float64)
	
	if inputLabel is None:
		for drop in listFinalDrops:
			(ix, iy) = drop.getCenters()
			radius = drop.getRadius()
			ROI_WL = 2*radius
			ROI_WR = 2*radius
			ROI_HU = 3*radius
			ROI_HD = 2*radius
			if (iy-3*radius) <0 :
				ROI_HU = iy	
			if (iy+2*radius)>imgh:
				ROI_HD = imgh - iy
			if (ix-2*radius)<0:
				ROI_WL = ix			
			if  (ix+2*radius) > imgw:
				ROI_WR = imgw - ix

			drop_alpha = drop.getAlphaMap()	

			alpha_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix+ROI_WR] += drop_alpha[3*radius - ROI_HU:3*radius + ROI_HD, 2*radius - ROI_WL: 2*radius+ROI_WR]
			
	else:
		for drop in listFinalDrops:
			(ix, iy) = drop.getCenters()
			drop_alpha = drop.getAlphaMap()

			h, w = drop_alpha.shape

			alpha_map[iy:iy + h, ix: ix+w] += drop_alpha[:h, : w]
		# alpha_map = arrayLabel[:,:,0].copy()			
		'''
		alpha_map = pyblur.GaussianBlur(Image.fromarray(np.uint8(alpha_map)), 10)
		alpha_map = np.asarray(alpha_map).astype(np.float)
		
		alpha_map = alpha_map/np.max(alpha_map)*255.0
		'''
	alpha_map = alpha_map/np.max(alpha_map)*255.0
	#cv2.imwrite("test.bmp", alpha_map)
	#sys.exit()
	# alpha_map[label<1] = 0									

	

	

	PIL_bg_img = Image.open(imagePath)
	for drop in listFinalDrops:
		# check bounding
		if inputLabel is None:
			(ix, iy) = drop.getCenters()
			radius = drop.getRadius()		
			ROIU = iy - 3*radius
			ROID = iy + 2*radius
			ROIL = ix - 2*radius
			ROIR = ix + 2*radius
			if (iy-3*radius) <0 :
				ROIU = 0
				ROID = 5*radius		
			if (iy+2*radius)>imgh:
				ROIU = imgh - 5*radius
				ROID = imgh
			if (ix-2*radius)<0:
				ROIL = 0
				ROIR = 4*radius
			if  (ix+2*radius) > imgw:		
				ROIL = imgw - 4*radius
				ROIR = imgw
		else:
			# left top 
			(ix, iy) = drop.getCenters()
			h, w = drop.getLabelMap().shape			
			ROIU = iy
			ROID = iy + h 
			ROIL = ix
			ROIR = ix + w 


		tmp_bg = bg_img[ROIU:ROID, ROIL:ROIR,:]
		drop.updateTexture(tmp_bg)
		tmp_alpha_map  = alpha_map[ROIU:ROID, ROIL:ROIR]
		
		
		
		
		output = drop.getTexture()		
		tmp_output = np.asarray(output).astype(np.float)[:,:,-1]
		tmp_alpha_map = tmp_alpha_map * (tmp_output/255)
		tmp_alpha_map  = Image.fromarray(tmp_alpha_map.astype('uint8'))		
		tmp_alpha_map.save("test.bmp")
		
		
		edge = ImageEnhance.Brightness(output)
		edge = edge.enhance(edge_ratio)
		
		
		
		if inputLabel is None:
			PIL_bg_img.paste(edge, (ix-2*radius, iy-3*radius), tmp_alpha_map)
			PIL_bg_img.paste(output, (ix-2*radius, iy-3*radius), output)
		else:
			PIL_bg_img.paste(edge, (ix, iy), tmp_alpha_map)
			PIL_bg_img.paste(output, (ix, iy), output)
		
	

	
	if ifReturnLabel:
		output_label = np.array(alpha_map)
		output_label.flags.writeable = True
		output_label[output_label>0] = 1
		output_label = Image.fromarray(output_label.astype('uint8'))	
		return PIL_bg_img, output_label

	return PIL_bg_img


class raindrop():
	def __init__(self, key, centerxy = None, radius = None, input_alpha = None, input_label = None):
		if input_label is None:
			self.key = key		
			self.ifcol = False	
			self.col_with = []
			self.center = centerxy
			self.radius = radius
			self.type = "default"
			# label map's WxH = 4*R , 5*R
			self.labelmap = np.zeros((self.radius * 5, self.radius*4))
			self.alphamap = np.zeros((self.radius * 5, self.radius*4))
			self.background = None
			self.texture = None
			self._create_label()
			self.use_label = False
		else:
			self.key = key
			assert input_alpha is not None, "Please also input the alpha map"
			self.alphamap = input_alpha
			self.labelmap =input_label
			self.ifcol = False
			self.col_with = []
			# default shape should be [h,w]			
			h, w = self.labelmap.shape
			# set the label center
			self.center = centerxy
			self.radius = min(w//4, h//4)
			self.background = None
			self.texture = None
			self.use_label = True

	def setCollision(self, col, col_with):
		self.ifcol = col
		self.col_with = col_with

	def updateTexture(self, bg):		
		
		fg = pyblur.GaussianBlur(Image.fromarray(np.uint8(bg)), 5)
		fg = np.asarray(fg)
		# add fish eye effect to simulate the background
		K = np.array([[30*self.radius, 0, 2*self.radius],
				[0., 20*self.radius, 3*self.radius],
				[0., 0., 1]])
		D = np.array([0.0, 0.0, 0.0, 0.0])
		Knew = K.copy()
		Knew[(0,1), (0,1)] = math.pow(self.radius, 1/3)*2 * Knew[(0,1), (0,1)]
		fisheye = cv2.fisheye.undistortImage(fg, K, D=D, Knew=Knew)
		
		
		tmp = np.expand_dims(self.alphamap, axis = -1)		
		tmp = np.concatenate((fisheye, tmp), axis = 2)
		
		self.texture = Image.fromarray(tmp.astype('uint8'), 'RGBA')
		# most background in drop is flipped
		self.texture = self.texture.transpose(Image.FLIP_TOP_BOTTOM)

		
	# create the raindrop label
	def _create_label(self):
		if self.type == "default":
			self._createDefaultDrop()
		elif self.type == "splash":
			self._createSplashDrop()

	def _createDefaultDrop(self):
		cv2.circle(self.labelmap, (self.radius * 2, self.radius * 3), self.radius, 128, -1)
		cv2.ellipse(self.labelmap, (self.radius * 2, self.radius * 3), (self.radius, int(1.3*math.sqrt(3) * self.radius)), 0, 180, 360, 128, -1)		
		# set alpha map for png 
		self.alphamap = pyblur.GaussianBlur(Image.fromarray(np.uint8(self.labelmap)), 10)
		self.alphamap = np.asarray(self.alphamap).astype(np.float)
		self.alphamap = self.alphamap/np.max(self.alphamap)*255.0
		# set label map
		self.labelmap[self.labelmap>0] = 1

	def _createSplashDrop(self):
		pass

	def setKey(self, key):
		self.key = key

	def getLabelMap(self):
		return self.labelmap

	def getAlphaMap(self):
		return self.alphamap

	def getTexture(self):
		return self.texture

	def getCenters(self):
		return self.center
		
	def getRadius(self):
		return self.radius

	def getKey(self):
		return self.key

	def getIfColli(self):
		return self.ifcol

	def getCollisionList(self):
		return self.col_with
	
	def getUseLabel(self):
		return self.use_label