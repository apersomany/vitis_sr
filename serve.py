
import flask
import requests
import onnxruntime
import cv2
import numpy
import time
import threading
import math
import os

if not(os.path.exists("model.onnx")):
    import model

app = flask.Flask(__name__)
sem = threading.Semaphore()
exp = 20

@app.route("/upscale")
def upscale():
    url = flask.request.args.get("src")
    res = requests.get(url)
    if res.status_code == 200:
        img = numpy.frombuffer(bytearray(res.content), dtype=numpy.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        [h, w, _] = numpy.shape(img)
        img = numpy.array_split(img, math.ceil(w * h / 2 ** exp), 0)
        sem.acquire()
        t = time.time()
        img = list(map(partial, img))
        print("Upscaling completed in: " + str(round((time.time() - t) * 1000)))
        sem.release()
        img = numpy.concatenate(img, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.imencode(".png", img)[1]
        img = img.tobytes()
        return (img, 200, [["content-type", "image/png"]])
    else:
        return (res.content, res.status_code, res.headers.items())
    
mod = onnxruntime.InferenceSession("model.onnx", providers=["DmlExecutionProvider"])
bnd = mod.io_binding()

def partial(img):
    img = onnxruntime.OrtValue.ortvalue_from_numpy(img)
    bnd.bind_ortvalue_input("i", img)
    [h, w, c] = img.shape()
    out = onnxruntime.OrtValue.ortvalue_from_shape_and_type([h * 4, w * 4, c], numpy.uint8)
    bnd.bind_ortvalue_output("o", out)
    mod.run_with_iobinding(bnd)
    return out.numpy()
        
app.run("127.0.0.1", 1234)