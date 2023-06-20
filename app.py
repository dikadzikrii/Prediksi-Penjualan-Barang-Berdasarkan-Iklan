from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# data = pd.read_csv('prediksi_sales.csv')

# x = np.array(data["TV", "Radio", "Newspaper"])
# y = np.array(data["Sales"])
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=41)

# model = LinearRegression()
# model.fit(xtrain, ytrain)

model = 'model.joblib'

loaded_model = joblib.load(model)

def predict_sales(tv, radio, newspaper):
	features = np.array([[tv, radio, newspaper]])
	hasil = loaded_model.predict(features)
	return hasil


# dic = {0 : 'Cat', 1 : 'Dog'}

# model = load_model('model.h5')

# model.make_predict_function()

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		# img = request.files['my_image']
		tv = request.form['tv']
		radio = request.form['radio']
		newspaper = request.form['newspaper']

		p = predict_sales(int(tv), int(radio), int(newspaper))

	return render_template("index.html", prediction = p)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)