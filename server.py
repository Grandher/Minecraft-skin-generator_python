from flask import Flask, render_template, jsonify
from generator import imageGenerator  # Импортируйте вашу функцию для генерации изображения
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image')
def generate_image():
    image = imageGenerator()
    
    # Сохраняем изображение в формате base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Возвращаем base64-кодированное изображение
    return jsonify({'image': img_str})

if __name__ == '__main__':
    app.run(debug=True)
