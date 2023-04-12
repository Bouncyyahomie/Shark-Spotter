from model import load_ml_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import preprocess_input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import numpy as np
from fastapi import FastAPI, UploadFile, Request, Header, HTTPException
from model import load_ml_model
from PIL import Image
from io import BytesIO
from linebot import LineBotApi, WebhookHandler
from linebot import LineBotApi
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
from linebot.exceptions import InvalidSignatureError
from dotenv import load_dotenv
import os
import random
import shutil
import csv

load_dotenv()

channel_access_token = os.getenv('CHANNEL_ACCESS_TOKEN', None)
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

app = FastAPI(title="Shark Spotter")

model_1 = load_ml_model('model/densenet_model.h5')
model_2 = load_ml_model('model/svc_model.pkl')
model_3 = load_ml_model('model/sgd_model.pkl')

base_model = DenseNet201(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

class_labels = ['grey reef shark', 'great hammerhead', 'scalloped hammerhead', 'coral shark', 'bull shark', 'spottail shark', 'pigeye shark', 'whitespotted bambooshark', 'indonesian bambooshark',
                'zebra shark', 'tiger shark', 'slender bambooshark', 'blacktip reef shark', 'graceful shark', 'spinner shark', 'brownbanded shark', 'whitetip reef shark', 'grey bambooshark']
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

    final_preds = np.argmax(
        weighted_preds_1 + weighted_preds_2 + weighted_preds_3, axis=1)

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
    features[0, :] = np.squeeze(model.predict(x))

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
async def callback(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError as e:
        raise HTTPException(
            status_code=400, detail="chatbot handle body error.%s" % e.message)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def message_text(event):

    text = event.message.text

    profile = line_bot_api.get_profile(event.source.user_id)

    start_word = ['สวัสดีครับคุณพี่ ', 'ยินดีต้อนรับครับ คุณ']
    response_word = random.choice(start_word) + profile.display_name + \
        " สงสัยว่าฉลามที่เจอเป็นฉลามสายพันธุ์ไหนสามารถส่งมาถามน้องหลามได้เลยครับ"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response_word)
    )


@handler.add(MessageEvent, message=(ImageMessage))
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    else:
        return

    message_content = line_bot_api.get_message_content(event.message.id)
    print(type(message_content))

    image_folder = "uploads/images/"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    file_path = image_folder+event.message.id+'.'+ext

    with open(file_path, 'wb') as f1:
        for chunk in message_content.iter_content():
            f1.write(chunk)
    img = Image.open(file_path)

    features, labels = extract_feature(img)
    # Predict as number
    img_pred_class = predict_ensemble(features)
    result = encoder_labels.inverse_transform(img_pred_class)[0]

    line_bot_api.reply_message(
        event.reply_token, [
            TextSendMessage(
                text='จากการประมวลผมฉลามพันธุ์นี้คือ '+result + ' ครับ'),
        ])

    # close the file after processing
    f1.close()

    # # for collect data
    # collect_path = 'collect/'+ result + event.message.id + '.' + ext
    # if not os.path.exists(collect_path):
    #     os.makedirs(collect_path)
    # shutil.copy2(file_path, collect_path)

    # # remove the image file after processing
    # os.remove(file_path)
    
    csv_path = 'uploads/collect.csv'
    # header
    header = ['species', 'file path']
    # collect in dict
    new_row = {'species': result, 'file path': file_path}
    # for collect data in csv 
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'a', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(new_row)
    else:
        with open(csv_path, 'w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerow(new_row)