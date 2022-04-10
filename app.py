from flask import Flask, request, render_template, send_from_directory
import os

from Denoise import doDenoise
from SR import doSR

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def main():
    target = os.path.join(APP_ROOT, 'static/images/')
    for i in os.listdir(target):
        file_data = target + "\\" + i
        os.remove(file_data)
    return render_template('index.html', sr='sr.png', dn='dn.png')


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    if not os.path.isdir(target):
        os.mkdir(target)

    upload = request.files.getlist("file")[0]

    filename = upload.filename

    ext = os.path.splitext(filename)[1]
    if (ext != ".jpg") and (ext != ".png"):
        return render_template("error.html", message="wrong format"), 400

    destination = "/".join([target, filename])

    upload.save(destination)

    return render_template("restore.html", image_name=filename)


@app.route("/denoise", methods=["POST"])
def denoise():

    if 'level25' in request.form['mode']:
        mode = 'l25'
    elif 'level15' in request.form['mode']:
        mode = 'l15'
    else:
        return render_template("error.html", message="Command Error"), 400
    filename = request.form['image']

    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    doDenoise(destination, mode, target)

    return render_template("done.html", image_name1=filename, image_name2='temp.jpg')


@app.route("/sr", methods=["POST"])
def sr():
    # retrieve parameters from html form
    if 'x4' in request.form['mode']:
        mode = 'x4'
    elif 'x2' in request.form['mode']:
        mode = 'x2'
    else:
        return render_template("error.html", message="Command Error"), 400
    filename = request.form['image']

    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    doSR(destination, mode, target)

    return render_template("done.html", image_name1=filename, image_name2='temp.png')


@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


@app.route('/static/example/<filename>')
def send_example(filename):
    return send_from_directory("static/example", filename)


if __name__ == "__main__":
    app.run()

