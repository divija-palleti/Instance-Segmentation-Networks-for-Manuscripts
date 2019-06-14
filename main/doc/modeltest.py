import cv2
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model
import keras.backend as K
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from main.doc import train
from PIL import Image
#edited divija.p
from app import *
import urllib.request
from skimage import io
# 
config = train.Config()
DOCDATA = ROOT_DIR+"datasets/doc"
OUTPUTPATH=  ROOT_DIR+"/main/doc/static/images/2.jpg"
IMG1PATH= ROOT_DIR+"/main/doc/static/images/1.jpg"
#edited divija.p

#edited divija.p
# with urllib.request.urlopen(filepath) as url:

#     s = url.read()
#     # I'm guessing this would output the html source code ?
#     print(s)

# arguments 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c","--filepath",help="filepath for JSON")
args = vars(parser.parse_args())
print(args['filepath'])
#edited divija.p
filepath= args['filepath']
filename=str(filepath)

#edited divija.p
def image_url_to_numpy_array_skimage(url,format=None):
    from skimage import io
    image = io.imread(url)
    image = np.asarray(image, dtype="uint8")
    if format=='BGR' :
        ## return BGR format array
        return image[...,[2,1,0]]
    return image



image=image_url_to_numpy_array_skimage(url=filepath)

# image = Image.open(urllib.request.urlopen(filepath))
# # resp = urllib.request.urlopen(filepath)
# # image = np.asarray(bytearray(resp.read()), dtype="uint8")
# # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
img=image
print(type(img),"p")
# cv2.imshow('ImageWindow', img)
# cv2.waitKey()
print("uuu")
# 
# added  divija.p
class InferenceConfig(config.__class__):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	IMAGE_RESIZE_MODE = "square"
	DETECTION_MIN_CONFIDENCE = 0.6
	DETECTION_NMS_THRESHOLD = 0.3
	PRE_NMS_LIMIT = 12000
	RPN_ANCHOR_SCALES = (8,32,64,256,1024)
	RPN_ANCHOR_RATIOS = [1,3,10]

	POST_NMS_ROIS_INFERENCE = 12000
	
	'''
	
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	IMAGE_RESIZE_MODE = "square"
	DETECTION_MIN_CONFIDENCE = 0.3
	DETECTION_NMS_THRESHOLD = 0.3
	PRE_NMS_LIMIT = 12000
	RPN_ANCHOR_SCALES = (8,32,64,256,1024)
	RPN_ANCHOR_RATIOS = [1,3,10]

	POST_NMS_ROIS_INFERENCE = 12000
	'''
def load_model():
	config = InferenceConfig()
	DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
	TEST_MODE = "inference"
	global dataset
	dataset = train.Dataset()
	dataset.load_data(MAIN_DIR, "val")
	dataset.prepare()

	print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
	global model
	with tf.device(DEVICE):
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
							  config=config)
	weights_path= ROOT_DIR+"/pretrained_model_indiscapes.h5"
	print("Loading weights ", weights_path)
	model.load_weights(weights_path, by_name=True)
	global graph
	graph = tf.get_default_graph()



# 
if __name__ == '__main__':
	load_model()
	# app.run('0.0.0.0', debug=True)
	with graph.as_default(): 
		runtest(img,model,dataset)

def get_ax(rows=1, cols=1, size=16):
	_, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
	return ax
def runtest(img,model,dataset):

	import json
	image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	image,_,scale,padding,_=utils.resize_image(image,min_dim=256, max_dim=1024)
	results = model.detect([image], verbose=1)
	ax = get_ax(1)
	r = results[0]
	ccc,contours=visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
	                            dataset.class_names, r['scores'], ax=ax,
	                            title="Predictions",show_bbox=False,show_mask=True)
	print(len(contours))
	arr = np.array(contours)
	print(arr[0][0][0][0],"shape")

	# print(contours[0][0],"contours")
	# print(contours[0])
	cls=r['class_ids']
	classes = ['Background','Hole(Virtual)','Hole(Physical)','Character Line Segment',
	           'Physical Degradation','Page Boundary','Character Component','Picture',
	           'Decorator','Library Marker','Boundary Line']

	strt="""
	{
	  "_via_settings": {
	    "ui": {
	      "annotation_editor_height": 30,
	      "annotation_editor_fontsize": 0.6000000000000001,
	      "leftsidebar_width": 18,
	      "image_grid": {
	        "img_height": 80,
	        "rshape_fill": "none",
	        "rshape_fill_opacity": 0.3,
	        "rshape_stroke": "yellow",
	        "rshape_stroke_width": 2,
	        "show_region_shape": true,
	        "show_image_policy": "all"
	      },
	      "image": {
	        "region_label": "region_id",
	        "region_label_font": "10px Sans"
	      }
	    },
	    "core": {
	      "buffer_size": 18,
	      "filepath": {},
	      "default_filepath": ""
	    },
	    "project": {
	      "name": "corrected_3"
	    }
	  },
	  "_via_img_metadata": {
	    "": {
	      "filename": \""""+str(filename)+"""\",
	      "size": -1,
	      "regions": [
	"""

	end="""
	],
	      "file_attributes": {}
	    }
	  },
	  "_via_attributes": {
	    "region": {
	      "Spatial Annotation": {
	        "type": "dropdown",
	        "description": "",
	        "options": {
	          "Hole(Virtual)": "",
	          "Hole(Physical)": "",
	          "Character Line Segment": "",
	          "Boundary Line": "",
	          "Physical Degradation": "",
	          "Page Boundary": "",
	          "Character Component": "",
	          "Picture": "",
	          "Decorator": "",
	          "Library Marker": ""
	        },
	        "default_options": {}
	      },
	      "Comments": {
	        "type": "text",
	        "description": "",
	        "default_value": ""
	      }
	    },
	    "file": {}
	  }
	}
	"""

	rgns=""
	for i in range(len(cls)):
		if i!=(-1):
			k = np.array(contours[i][0])
			print(k.shape,"kshape")
			print(k," ith k",i)
			ln=len(contours[i][0])
			mid = int(ln/2)
			# k1=k[0:mid,:]
			# k2=k[mid:ln-1,:]
			k1=k[0:ln-1,:]
			print(k1,"ith k1",i)
			print(k1.shape,"k1shape")
			# print(k2,"ith k2",i)
			# print(k2.shape,"k2shape")
			from rdp import rdp
			# from simplification.cutil import simplify_coords, simplify_coordsvw
			# from polysimplify import VWSimplifier
			import visvalingamwyatt as vw
			# simplifier = vw.Simplifier(points)

# Simplify by percentage of points to keep
      # simplifier.simplify(ratio=0.5)
			simplifier1 = vw.Simplifier(k1)
			# simplifier.simplify(ratio=0.5)
			n1=int(0.020*k1.shape[0])
			print(n1,"n1")
			# n2=int(0.025*k2.shape[0])
			# print(n2,"n2")
			rdpk1=np.array(simplifier1.simplify(number=n1))
			# simplifier2 = vw.Simplifier(k2)
			# simplifier.simplify(ratio=0.5)
			# rdpk2=np.array(simplifier2.simplify(number=n2))
			# rdpk1= rdp(k1,epsilon=1)
			# rdpk2= rdp(k2,epsilon=1)
			print(rdpk1,"ith rdpk1",i)
			print(rdpk1.shape,"rdpk1shape")
			# print(rdpk2,"ith rdpk2",i)
			# print(rdpk2.shape,"rdpk2shape")
			# final= np.concatenate((rdpk1, rdpk2), axis=0)
			final=rdpk1
			print(final.shape[0],"finalshape")
			print(final)
			length=final.shape[0]
			str1=""
			str2=""
			for j in range(length):

				str1+=str(final[j][0]-padding[0][0])
				
				str1+=","
				str1+='\n'
			for j in range(length):

				str2+=str(final[j][1]-padding[1][0])
				# g=0
				str2+=","
				str2+='\n'
			str1=str1[:-2]
			str2=str2[:-2]
			rg="""{
	          "shape_attributes": {
	            "name": "polygon",
	            "all_points_x": [ """+ str2+"""],
	            "all_points_y": ["""+str1+ """]
	          },
	          "region_attributes": {
	            "Spatial Annotation":\""""+str(classes[cls[i]])+"""\",
	            "Comments": ""
	          },
	          "timestamp": {
	            "StartingTime": 6016533,
	            "EndTime": 6035060
	          }
	        }"""
			
			if(i!=len(cls)-1):
				rg+=","
			rgns+=rg

	    # if(i!=len(cls)-1):

	    #   rg+=","
	    # rgns+=rg
				
  
				
				


	# k=np.array(contours)
	# print(k,"k")
	# print(k.shape[0],"shape1")


	# for i in range(len(cls)):

	#     str1=""
	#     str2=""
	#     ln=len(contours[i][0])
			
	#     print(ln,"lrn")
	#     for j in range(ln):

	#         if(j%20==0):
	#             str1+=str(contours[i][0][j][0]-padding[0][0])
	#             str1+=','
	#             str1+='\n'
	#     for j in range(ln):
	#         if(j%20==0):
	#             str2+=str(contours[i][0][j][1]-padding[1][0])
	#             str2+=','
	#             str2+='\n'
	#     str1=str1[:-2]
	#     str2=str2[:-2]
	    # rg="""{
	    #       "shape_attributes": {
	    #         "name": "polygon",
	    #         "all_points_x": [ """+ str2+"""],
	    #         "all_points_y": ["""+str1+ """]
	    #       },
	    #       "region_attributes": {
	    #         "Spatial Annotation":\""""+str(classes[cls[i]])+"""\",
	    #         "Comments": ""
	    #       },
	    #       "timestamp": {
	    #         "StartingTime": 6016533,
	    #         "EndTime": 6035060
	    #       }
	    #     }"""
	    # if(i!=len(cls)-1):
	    #     rg+=","
	    # rgns+=rg

	with open ('save.json','w') as f:
	    f.write(strt)
	    f.write(rgns)
	    f.write(end)
	h, w = image.shape[:2]
	image=image[padding[0][0]:h-padding[0][1],padding[1][0]:w-padding[1][1]]
	plt.savefig(OUTPUTPATH,bbox_inches='tight')
