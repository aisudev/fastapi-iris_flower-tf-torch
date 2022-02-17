import fastapi
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from utils import request
from constant import labels

if tf.test.gpu_device_name():
    print('GPU')
else:
    print('CPU')

# Craete Router
router = fastapi.APIRouter()
# Load Tensorflow Model
model = load_model('./models/weights/tf_iris.h5')

# Prediction Function (usecase)
async def ModelPrediction(data):
    x = np.array([[data.a, data.b, data.c, data.d]])

    pred = model.predict(x)

    res = np.argmax(pred, axis=1)[0]
    category = labels.classNameCat[res]
    confidence = float(pred[0][res])

    return category, confidence

# Prediction API
@router.post('/predict')
async def prediction(data: request.Data):
    category, _ = await ModelPrediction(data)
    return {
        'Category': category
    }
