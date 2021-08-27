import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from scipy.signal.signaltools import wiener
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
#from keras.models import load_model
from scipy.stats import kurtosis
from scipy.stats import skew
from tensorflow import keras

import tensorflow as tf
import pandas as pd
import numpy as np

import cv2
import csv

app = Flask(__name__)

app.config["file_UPLOADS"] = "./img_uploads"
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])


@app.route('/upload-api', methods=['GET','POST'])
def upload_file_api():
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config["file_UPLOADS"], file.filename))
		print('save image locally')
		gmbr = app.config["file_UPLOADS"]+"/"+file.filename


		imageS, data_glcmS = set_csv(gmbr)
		glcm_all_aglsS, columnsS = set_glcm(imageS, data_glcmS)
		resultS = merge_csv(glcm_all_aglsS, columnsS)
		XS = set_scalling(resultS)

		predictS,score, bcc, kulit_normal, melanoma = predict(XS)


		resp = jsonify({'message' : 'file successfully uploaded', 'result' : predictS, 'bcc' : bcc, 'kulit_normal' : kulit_normal, 'melanoma' : melanoma})
		resp.status_code = 201

		os.remove(gmbr)
		print('remove image')

		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg'})
		resp.status_code = 400
		return resp

def predict(X):
	model = tf.keras.models.load_model('model_new.h5', compile=False)
	imgX = np.expand_dims(X,axis=0)
	imgsX = np.vstack([imgX])

	val = model.predict(imgsX)

	print(val)

	bcc = val[0][0][0]*100
	kulit_normal =  val[0][0][1]*100
	melanoma =  val[0][0][2]*100

	z= val.max(axis=-1)*100
	z1 = z[0]
	score = str(z1[0])+' %'

	print("{:.3f}".format(bcc))
	print("{:.3f}".format(kulit_normal))
	print("{:.3f}".format(melanoma))

	bcc = str("{:.3f}".format(bcc))+ ' %'
	kulit_normal =  str("{:.3f}".format(kulit_normal))+ ' %'
	melanoma =  str("{:.3f}".format(melanoma))+ ' %'

	print(score)

	classes = val.argmax(axis=-1)
	#print(classes)
	predict = ''
	if classes == 0:
		predict = 'Bcc'
	elif classes == 1:
		predict = 'Kulit Normal'
	else :
		predict = 'Melanoma'

	
	return predict, score, bcc, kulit_normal, melanoma

def set_scalling(result):
	X = decimal_scaling(
            result[['mean_h','mean_s', 'mean_v', 'std_h', 'std_s', 'std_v', 'var_h', 'var_s', 'var_v',
                    'skew_h','skew_s','skew_v','kurt_h','kurt_s','kurt_v',
                    'contrast_0', 'correlation_0', 'energy_0','homogeneity_0',
                    'contrast_45', 'correlation_45', 'energy_45','homogeneity_45',
                    'contrast_90', 'correlation_90', 'energy_90','homogeneity_90',
                    'contrast_135', 'correlation_135', 'energy_135','homogeneity_135']].values
                    )
	return X

# ------------------------ Data Normalization menggunakan Decimal Scaling --------------------------------
def decimal_scaling(data):
    data = np.array(data, dtype=np.float32)
    max_row = data.max(axis=0)
    c = np.array([len(str(int(number))) for number in np.abs(max_row)])
    return data/(10**c)

def merge_csv(glcm_all_agls, columns):
	# Create the pandas DataFrame for GLCM features data
	glcm_df = pd.DataFrame(glcm_all_agls, columns = columns)
	#save to csv
	glcm_df.to_csv("glcm_hsv.csv", index=False)

	#open CSV (HSV LBP and GLCM feature)
	df1 = pd.read_csv('fitur_hsv_glcm.csv')
	df2 = pd.read_csv('glcm_hsv.csv')
	# df3 = pd.read_csv('glcm_lbp_hsv.csv')

	#mergered CSV HSV LBP and GLCM feature to 1 CSV
	result = pd.concat([df1, df2], axis=1)
	result.to_csv('merged_hsv_glcm.csv', index=False)

	return result


def set_glcm(image, data_glcm):
	# call calc_glcm_all_agls() for all properties 
	properties = [ 'contrast', 'correlation', 'energy','homogeneity']

	#call function denoising
	glcm_features = get_denoising(image)
	#set data for calculate GLCM
	# gray_glcm = cv2.cvtColor(glcm_features, cv2.COLOR_BGR2GRAY)
	data_glcm.append(glcm_features)

	glcm_all_agls = []
	glcm_all_agls.append(calc_glcm_all_agls(glcm_features, props=properties))

	columns = []
	angles = ['0', '45', '90','135']
	for name in properties :
		for ang in angles:
			columns.append(name + "_" + ang)

	return glcm_all_agls, columns

# calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 
def calc_glcm_all_agls(img, props, dists=[2], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
	glcm = greycomatrix(img,
						distances=dists,
						angles=agls,
						levels=lvl,
						symmetric=sym,
						normed=norm)
	feature = []
	glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
	for item in glcm_props:
		feature.append(item)
	return feature

def set_csv(imagePath):
	image = cv2.imread(imagePath)
	resizing = cv2.resize(image, (224, 224))

	# create variabel
	data_hsv = []
	# data_lbp =[]
	data_glcm=[]

	#call function denoising
	features_hsv = get_denoising(resizing)
	img = cv2.inpaint(resizing,features_hsv,1,cv2.INPAINT_TELEA)
	#call function HSV Extraction
	feat_hsv = hsv_image(img)

	#save to csv
	data_hsv.append(feat_hsv)
	csv_columns = ['mean_h','mean_s', 'mean_v', 'std_h', 'std_s', 'std_v', 'var_h', 'var_s', 'var_v','skew_h','skew_s','skew_v','kurt_h','kurt_s','kurt_v']
	csv_file ="fitur_hsv_glcm.csv"
	try:
		with open(csv_file, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
			writer.writeheader()
			for data in data_hsv:
				writer.writerow(data)
	except IOError:
		print("I/O error")

	#call function denoising and LBP Extraction
	# lbp_features = get_denoising(image)
	# gray_lbp = cv2.cvtColor(lbp_features, cv2.COLOR_BGR2GRAY)
	# lbp_get= lbp_extraction(gray_lbp)

	# data_lbp.append(lbp_get)

	# Create the pandas DataFrame for LBP features data
	# lbp_df = pd.DataFrame(data_lbp)
	#save to csv
	# lbp_df.to_csv("lbp_glcm_hsv.csv", index=False)

	return image, data_glcm

#fungi denoising (resize,gray,wiener,morphology,threshold)
def get_denoising(image):
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gauss =gray.astype("float32")
	filtered_img = wiener(gauss, (5, 5))  #Filter the image
	filtered_img = np.uint8(filtered_img / filtered_img.max() * 255)

	kernel = np.ones((7, 7), np.uint8)
	blackhat = cv2.morphologyEx(filtered_img, cv2.MORPH_BLACKHAT, kernel)

	ret,th = cv2.threshold(blackhat,1,255,cv2.THRESH_BINARY)
	ret,thresh2 = cv2.threshold(th,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	img = cv2.inpaint(gray,thresh2,1,cv2.INPAINT_TELEA)
	return img

#fungi LBP Extraction


#fungi HSV Extraction
def hsv_image(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	features = [
		np.mean(h),
		np.mean(s),
		np.mean(v),

		np.std(h),
		np.std(s),
		np.std(v),

		np.var(h),
		np.var(s),
		np.var(v),

		skew(h, axis=None),
		skew(s, axis=None),
		skew(v, axis=None),

		kurtosis(h, None, fisher=False),
		kurtosis(s, None, fisher=False),
		kurtosis(v, None, fisher=False)]
	fitur={
		"mean_h" :features[0],
		"mean_s" :features[1],
		"mean_v" :features[2],

		"std_h":features[3],
		"std_s":features[4],
		"std_v":features[5],

		"var_h":features[6],
		"var_s":features[7],
		"var_v":features[8],

		"skew_h":features[9],
		"skew_s":features[10],
		"skew_v":features[11],

		"kurt_h":features[12],
		"kurt_s":features[13],
		"kurt_v":features[14]}
	return fitur

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
	app.run(host="0.0.0.0", port=5000, debug=False)