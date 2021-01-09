from flask import Flask, render_template, request, url_for, redirect
import pickle
import io
import csv

app = Flask(__name__)

model = pickle.load(open('evc.pickle', 'rb'))

names = ["Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria", "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume", "White Blood Cell Count", "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia"]

min = [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0]
max = [120, 360, 1.1, 5.5, 5.5, 1, 1, 1, 1, 700, 500, 100, 200, 100, 50, 100, 30000, 20, 1, 1, 1, 1, 1, 1]

@app.route("/upload", methods=["POST", "GET"])
def file():
	if request.method == "POST":
		if request.files:
			f = request.files['data_file']

			if f:
				stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
				data = list(csv.reader(stream))
				print(data)

				for i in range(24):
					data[0][i] = float((float(data[0][i]) - min[i]) / (max[i] - min[i]))

				res = model.predict(data)[0]
				return redirect(url_for("result", res=res))
		else:
			return "no files added"
	else:
		return render_template("index.html")

@app.route("/", methods=["POST", "GET"])
def home():

	data = []
	if request.method == "POST":
		for n in names:
			data.append(float(request.form.get(n)))
		
		for i in range(24):
			data[i] = (data[i] - min[i]) / (max[i] - min[i])

		res = model.predict([data])[0]
		return redirect(url_for("result", res=res))
	else:	
		return render_template("index.html")

@app.route("/<res>")
def result(res):
	print("res : " + res)
	if res == '1':
		return render_template("result.html", text=["You should really got to a specialist for advice", "we have detected severe problem in your kidney."])
	else:
		return render_template("result.html", text=["congrates,we are happy for you.", "your kidney is sound and healthy."])

if __name__ == "__main__":
	app.run(debug=True)