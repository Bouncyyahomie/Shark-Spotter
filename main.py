from model import load_ml_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import preprocess_input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from model import load_ml_model
from PIL import Image
from io import BytesIO
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, SourceUser, StickerMessage, ImageMessage
from dotenv import load_dotenv
import os
load_dotenv()

channel_access_token = os.getenv('LINE_CHANNEL_SECRET', None)
line_bot_api = LineBotApi(channel_access_token)

app = FastAPI(title="Shark Spotter")

model_1 = load_ml_model('model/densenet_model.h5') 
model_2 = load_ml_model('model/svc_model.pkl')
model_3 = load_ml_model('model/sgd_model.pkl') 

base_model = DenseNet201(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

class_labels = ['grey reef shark', 'great hammerhead', 'scalloped hammerhead', 'coral shark', 'bull shark', 'spottail shark', 'pigeye shark', 'whitespotted bambooshark', 'indonesian bambooshark', 'zebra shark', 'tiger shark', 'slender bambooshark', 'blacktip reef shark', 'graceful shark', 'spinner shark', 'brownbanded shark', 'whitetip reef shark', 'grey bambooshark']
encoder_labels = LabelEncoder()
encoder_labels.fit(class_labels)

def predict_ensemble(features):
    best_weights = [0.25, 0.5, 0.25]

    # Make predictions on the new data with each model
    preds_1 = model_1.predict(features)
    preds_2 = model_2.predict(features)
    preds_3 = model_3.predict(features)

    # Apply the best weight values to the predicted outputs of each model
    weighted_preds_1 = preds_1 * best_weights[0]
    weighted_preds_2 = preds_2 * best_weights[1]
    weighted_preds_3 = preds_3 * best_weights[2]

    final_preds = np.argmax(weighted_preds_1 + weighted_preds_2 + weighted_preds_3, axis=1)
    
    return final_preds

def extract_feature(img):
    data = []
    labels = []
    nb_features = 1920  
    features = np.empty((1, nb_features))

    img_resized = img.resize((299, 299))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Extract features and store in features array
    features[0,:] = np.squeeze(model.predict(x))

    return features, labels

@app.get("/")
def helloWorld():
 return {"Hello": "World"}

@app.post("/predict")
async def predict(file: UploadFile):
    # Load the image 
    img = Image.open(BytesIO(await file.read()))
    # Extract features from the image
    features, labels = extract_feature(img)
    # Predict as number
    img_pred_class = predict_ensemble(features)
    result = encoder_labels.inverse_transform(img_pred_class)[0]
    return {"predict": result}


@app.post("/webhook")
async def callback(request: Request):
    data = await request.json()
    print(f"data: {data}")
    for i in range(len(data['events'])):
        event = data['events'][i]
        event_handle(event)
        
def event_handle(event):
    print("event")
    print(event["type"])
