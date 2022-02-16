import fastapi
from models import model
import torch
from utils import request
from constant import labels

# API Router
router = fastapi.APIRouter()
# Init Pytorch Model
tmodel = model.Model()
# Load weight
tmodel.load('./models/weights/model.pth')

# prediction function (usecase)
def ModelPrediction(data):
    x = torch.FloatTensor([[data.a, data.b, data.c, data.d]])
    pred = tmodel(x)
    pred = int(torch.argmax(pred[0]))
    return labels.classNameCat[pred]

# pytorch prediction api
@router.post('/predict')
async def prediction(data: request.Data):
    pred = ModelPrediction(data)
    return {
        'Category': pred
    }
