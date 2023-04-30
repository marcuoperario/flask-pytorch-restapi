from flask import Flask, request, send_file
import torch
import os 
import shutil
import json
from dotenv import load_dotenv
import pyrebase

app = Flask(__name__)

load_dotenv()
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
FIREBASE_AUTH_DOMAIN = os.getenv("FIREBASE_AUTH_DOMAIN")
FIREBASE_DATABASE_URL = os.getenv("FIREBASE_DATABASE_URL")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
FIREBASE_MESSENGER_SENDER_ID = os.getenv("FIREBASE_MESSENGER_SENDER_ID")
FIREBASE_APP_ID = os.getenv("FIREBASE_APP_ID")
FIREBASE_MEASUREMENT_ID = os.getenv("FIREBASE_MEASUREMENT_ID")

firebase_config = {
  "apiKey": FIREBASE_API_KEY,
  "authDomain": FIREBASE_AUTH_DOMAIN,
  "databaseURL": FIREBASE_DATABASE_URL,
  "projectId": FIREBASE_PROJECT_ID,
  "storageBucket": FIREBASE_STORAGE_BUCKET,
  "messagingSenderId": FIREBASE_MESSENGER_SENDER_ID,
  "appId": FIREBASE_APP_ID,
  "measurementId": FIREBASE_MEASUREMENT_ID,
  "serviceAccount": "serviceAccount.json"
}

model = torch.hub.load("ultralytics/yolov5", "custom", path = "opt/render/project/src/ML/weights/best.pt", trust_repo=True)
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()

@app.route("/detect/<string:user_id>", methods = ["POST"])
def detect(user_id):
    image = request.files["image"]
    timestamp = request.form.get("timestamp")
    file_name = image.filename
    file_type = file_name.split(".")[1]
    image.save(file_name)
    detect = model(file_name)
    print(detect)
    results = detect.pandas().xyxy[0].to_json()
    print(results)
    os.remove(file_name)
    results_json = json.loads(results)
    results_json["timestamp"] = timestamp

    if results_json["class"] == {}:
        results_json["status"] = "null"
        results_json["image_uri"] = "null"
        results_json = { "something": results_json }
    else:
        detect.save()
        if file_type == "png":
            file_name = file_name.replace(".png", ".jpg")
        elif file_type == "jpeg":   
            file_name = file_name.replace(".jpeg", ".jpg")
        local_path = "opt/render/project/src/runs/detect/exp/" + file_name
        cloud_path = "images/" + file_name
        storage.child(cloud_path).put(local_path)
        image_uri = storage.child(cloud_path).get_url(None)
        results_json["image_uri"] = image_uri
        results_json["status"] = "OK"
        entry = db.child("detections").child(user_id).push(results_json)
        data_id = entry["name"]
        results_json = { data_id: results_json}
        shutil.rmtree("opt/render/project/src/runs")
    final_response = json.dumps(results_json)
    print(final_response)
    return final_response       

@app.route("/<string:user_id>/recent", methods = ["GET"])
def get_recent_logs(user_id):
    data = db.child("detections").child(user_id).order_by_child("timestamp").limit_to_last(5).get()
    if data.val() == None:
        response = { "message": "empty"}
        return response
    else:
        return data.val()

@app.route("/<string:user_id>/all", methods = ["GET"])
def get_all_logs(user_id):
    data = db.child("detections").child(user_id).order_by_child("timestamp").get()
    if data.val() == None:
        response = { "message": "empty"}
        return response
    else:
        return data.val()

@app.route("/<string:user_id>/<string:entry_id>", methods = ["DELETE"])
def delete_log(user_id, entry_id):
    db.child("detections").child(user_id).child(entry_id).remove()
    response = { "message": "deleted"}
    return response

@app.route("/<string:user_id>", methods = ["DELETE"])
def delete_all_logs(user_id):
    db.child("detections").child(user_id).remove()
    response = { "message": "deleted"}
    return response

@app.route("/", methods = ["GET"])
def home():
    return "DurTect API"

def run_server():
    if os.path.exists("opt/render/project/src/runs") and os.path.isdir("opt/render/project/src/runs"):
        shutil.rmtree("opt/render/project/src/runs")
    app.run(host = "0.0.0.0", port = 5000) 

if __name__ == "__main__":
    run_server()
    
