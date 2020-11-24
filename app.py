from flask import Flask, render_template, request, url_for, redirect
import pickle

app = Flask(__name__)

names = ["Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", "Red Blood Cells", "Pus Cell", "Pus Cell clumps", "Bacteria", "Blood Glucose Random", "Blood Urea", "Serum Creatinine", "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume", "White Blood Cell Count", "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease", "Appetite", "Pedal Edema", "Anemia"]

@app.route("/", methods=["POST", "GET"])
def home():

	data = []
	if request.method == "POST":
		for n in names:
			data.append(float(request.form.get(n)))
		
		print(data)

		min = [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0]
		max = [120, 360, 1.1, 5.5, 5.5, 1, 1, 1, 1, 700, 500, 100, 200, 100, 50, 100, 30000, 20, 1, 1, 1, 1, 1, 1]
		
		for i in range(24):
			data[i] = (data[i] - min[i]) / (max[i] - min[i])

		model = pickle.load(open('evc.pickle', 'rb'))

		res = model.predict([data])[0]
		return redirect(url_for("result", res=res))
	else:	
		return render_template("index.html")

@app.route("/<res>")
def result(res):
	if {res} == 0:
		return render_template("result.html", text="CKD not predicted")
	else:
		return render_template("result.html", text="CKD predicted")

if __name__ == "__main__":
	app.run(debug=True)