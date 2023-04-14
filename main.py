from model import load_ml_model
from shark_payload import generate_payload
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import preprocess_input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from rembg import remove
import numpy as np
from fastapi import FastAPI, Request, Header, HTTPException
from model import load_ml_model
from PIL import Image
from linebot import LineBotApi, WebhookHandler
from linebot import LineBotApi
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage, TemplateSendMessage, ConfirmTemplate, MessageAction, ImageSendMessage
from linebot.exceptions import InvalidSignatureError
from dotenv import load_dotenv
import os
import random
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
extractor_model = Model(inputs=base_model.input, outputs=x)

class_labels = ['grey reef shark', 'great hammerhead', 'scalloped hammerhead', 'coral shark', 'bull shark', 'spottail shark', 'pigeye shark', 'whitespotted bambooshark', 'indonesian bambooshark',
                'zebra shark', 'tiger shark', 'slender bambooshark', 'blacktip reef shark', 'graceful shark', 'spinner shark', 'brownbanded shark', 'whitetip reef shark', 'grey bambooshark']
best_weights = [0.25, 0.5, 0.25]

encoder_labels = LabelEncoder()
encoder_labels.fit(class_labels)


def extract_feature(img):
    features = []
    labels = []
    nb_features = 1920
    features = np.empty((1, nb_features))

    img_resized = img.resize((299, 299))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Extract features and store in features array
    features[0, :] = np.squeeze(extractor_model.predict(x))

    return features, labels


def extract_feature_no_bg(img):
    features = []
    labels = []
    nb_features = 1920
    features = np.empty((1, nb_features))

    img_resized = img.resize((299, 299))
    x = image.img_to_array(img_resized)
    output1 = remove(img)
    output2 = output1.convert('RGB')
    x = image.img_to_array(output2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features[0, :] = np.squeeze(extractor_model.predict(x))

    return features, labels


def predict_ensemble(features):
    preds_1 = model_1.predict(features)
    sorted_preds1 = np.sort(preds_1)
    top_preds1 = sorted_preds1[0][-1:-3:-1]
    sorted_idx1 = np.argsort(preds_1)
    top_label1 = sorted_idx1[0][-1:-4:-1]

    preds_2 = model_2.predict(features)
    preds_prob_2 = model_2.predict_proba(features)
    sorted_preds2 = np.sort(preds_prob_2)
    top_preds2 = sorted_preds2[0][-1:-3:-1]
    sorted_idx2 = np.argsort(preds_prob_2)
    top_label2 = sorted_idx2[0][-1:-4:-1]

    preds_3 = model_3.predict(features)
    preds_prob_3 = model_3.predict_proba(features)
    sorted_preds3 = np.sort(preds_prob_3)
    top_preds3 = sorted_preds3[0][-1:-3:-1]
    sorted_idx3 = np.argsort(preds_prob_3)
    top_label3 = sorted_idx3[0][-1:-4:-1]

    # Apply the best weight values to the predicted outputs of each model
    weighted_preds_1 = preds_1 * best_weights[0]
    weighted_preds_2 = preds_2 * best_weights[1]
    weighted_preds_3 = preds_3 * best_weights[2]

    final_preds = np.argmax(
        weighted_preds_1 + weighted_preds_2 + weighted_preds_3, axis=1)
    final_preds_label = encoder_labels.inverse_transform(final_preds)

    print('Top 3 class for Model 1:', top_label1)
    print('Top 3 class for Model 2:', top_label2)
    print('Top 3 class for Model 3:', top_label3)

    print(top_preds1)
    print(top_preds2)
    print(top_preds3)

    if top_preds1[0] - top_preds1[1] >= 0.25 or top_preds2[0] - top_preds2[1] >= 0.25 or top_preds3[0] - top_preds3[1] >= 0.25:
        if top_preds1[0] - top_preds1[1] > 0.01 and top_preds2[0] - top_preds2[1] > 0.01 and top_preds3[0] - top_preds3[1] > 0.01:
            print('This image is', final_preds_label[0])
            return "Y", final_preds
        else:
            return "N", final_preds
    else:
        return "N", final_preds


def predict_ensemble_no_shark(features):
    preds_1 = model_1.predict(features)
    sorted_preds1 = np.sort(preds_1)
    top_preds1 = sorted_preds1[0][-1:-3:-1]
    sorted_idx1 = np.argsort(preds_1)
    top_label1 = sorted_idx1[0][-1:-4:-1]

    preds_2 = model_2.predict(features)
    preds_prob_2 = model_2.predict_proba(features)
    sorted_preds2 = np.sort(preds_prob_2)
    top_preds2 = sorted_preds2[0][-1:-3:-1]
    sorted_idx2 = np.argsort(preds_prob_2)
    top_label2 = sorted_idx2[0][-1:-4:-1]

    preds_3 = model_3.predict(features)
    preds_prob_3 = model_3.predict_proba(features)
    sorted_preds3 = np.sort(preds_prob_3)
    top_preds3 = sorted_preds3[0][-1:-3:-1]
    sorted_idx3 = np.argsort(preds_prob_3)
    top_label3 = sorted_idx3[0][-1:-4:-1]

    # Apply the best weight values to the predicted outputs of each model
    weighted_preds_1 = preds_1 * best_weights[0]
    weighted_preds_2 = preds_2 * best_weights[1]
    weighted_preds_3 = preds_3 * best_weights[2]

    final_preds = np.argmax(
        weighted_preds_1 + weighted_preds_2 + weighted_preds_3, axis=1)
    final_preds_label = encoder_labels.inverse_transform(final_preds)

    print('Top 3 class for Model 1:', top_label1)
    print('Top 3 class for Model 2:', top_label2)
    print('Top 3 class for Model 3:', top_label3)

    return final_preds


def final_answer(pred1, pred2):
    if pred1 == pred2:
        p1 = encoder_labels.inverse_transform(pred1)[0]
        return p1
    else:
        p1 = encoder_labels.inverse_transform(pred1)[0]
        p2 = encoder_labels.inverse_transform(pred2)[0]
        return p1, p2


@app.get("/")
def helloWorld():
    return {"Hello": "World"}


# old ver for test in swagger
# @app.post("/predict")
# async def predict(file: UploadFile):
#     # Load the image
#     img = Image.open(BytesIO(await file.read()))
#     # Extract features from the image
#     features, labels = extract_feature(img)
#     # Predict as number
#     img_pred_class = predict_ensemble(features)
#     result = encoder_labels.inverse_transform(img_pred_class)[0]
#     return {"predict": result}


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
def get_user_response(event):
    profile = line_bot_api.get_profile(event.source.user_id)
    start_word = ['สวัสดีครับคุณพี่ ', 'ยินดีต้อนรับครับ คุณ']

    if event.message.text == "retry":
        with open('uploads/collect.csv', "r") as f1:
            line = f1.readlines()
        last_line = line[-1]
        arr = last_line.split(',')
        file_path = arr[1].strip()

        img = Image.open(file_path)

        features, labels = extract_feature(img)
        img_pred_class1 = predict_ensemble_no_shark(features)

        features2, labels2 = extract_feature_no_bg(img)
        img_pred_class2 = predict_ensemble_no_shark(features2)

        x = final_answer(img_pred_class1, img_pred_class2)
        print(type(x))
        if type(x) is tuple:
            message = x[0].capitalize() + ' หรือ ' + x[1].capitalize() 
            f_csv = message
        else:
            print(x)
            message = x
            f_csv = message

        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(
                    text= "จากการประมาลผลรูปนี้มีแนวโน้มที่จะเป็น "+message+" และ หากท่านต้องการข้อมูลเพิ่มเติมของสายพันธุ์ฉลามสามารถพิมชื่อของสายพันธุ์นั้นมาได้เลยครับ"),
            ])

        csv_path = 'uploads/collect.csv'
        # header
        header = ['species', 'file path']
        # collect in dict
        new_row = {'species': f_csv, 'file path': file_path}
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

    elif event.message.text == "no-retry":
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(
                    text="สามารถส่งรูปมาใหม่ได้เลยครับ"),
            ])

    elif event.message.text == "classify shark":
        response_word = random.choice(start_word) + profile.display_name + \
            " สงสัยว่าฉลามที่เจอเป็นฉลามสายพันธุ์ไหนสามารถส่งมาถามน้องหลามได้เลยครับ"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response_word)
        )
    elif event.message.text == "shark info":
        response_word = random.choice(start_word) + profile.display_name + \
            " ต้องการข้อมูลของฉลามสายพันธุ์ไหน สามารถพิมชื่อมาได้เลยครับ"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response_word)
        )
    elif event.message.text == "more info about shark":
        response_word = random.choice(start_word) + profile.display_name + \
            " หากสนใจเกี่ยวกับ ฉลามและปลากระดูกอ่อนในน่านน้ำไทย สามารถสแกน QR Code ด่านล่างได้เลยเพื่อศึกษาเพิ่มเติมจาก หนังสือ ปลากระดูกอ่อนของไทยและใกล้เคียง (THE CARTILAGINOUS FISHES OF THAILAND AND ADJUSCENT WATERS) ได้เลยครับ"

        messages = [TextSendMessage(text=response_word),
                   ImageSendMessage(original_content_url="https://sv1.picz.in.th/images/2023/04/14/mVe6nI.jpg",
                                    preview_image_url="https://sv1.picz.in.th/images/2023/04/14/mVe6nI.jpg")
                   ]
        
        line_bot_api.reply_message(event.reply_token, messages)
        
    else:
        payload = generate_payload(event.message.text)
        response_word = random.choice(start_word) + profile.display_name + \
            " พันธ์ฉลามที่ท่านได้ส่งมานั้น มีข้อมูลตามนี้ครับ"
            
        messages = [TextMessage(text=response_word),
                    TextMessage(text= payload)]
        
        line_bot_api.reply_message(event.reply_token, messages)

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
    print(file_path)

    with open(file_path, 'wb') as f1:
        for chunk in message_content.iter_content():
            f1.write(chunk)
    img = Image.open(file_path)

    features1, labels1 = extract_feature(img)
    # Predict as number
    img_pred_class1 = predict_ensemble(features1)
    print(img_pred_class1[0])

    features2, labels2 = extract_feature_no_bg(img)
    img_pred_class2 = predict_ensemble(features2)
    print(img_pred_class2[0])
    if img_pred_class2[0] == "Y" and img_pred_class1[0] == "Y":
        x = final_answer(img_pred_class1[1], img_pred_class2[1])
        print(type(x))
        if type(x) is tuple:
            message = x[0].capitalize() + ' หรือ ' + x[1].capitalize()
            f_csv = message
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(
                        text= "จากการประมาลผลรูปนี้มีแนวโน้มที่จะเป็น "+message+" และ หากท่านต้องการข้อมูลเพิ่มเติมของสายพันธุ์ฉลามสามารถพิมชื่อของสายพันธุ์นั้นมาได้เลยครับ"),
                ])
        else:
            print(x)
            message = x.capitalize()
            f_csv = message
            payload = generate_payload(message)
            
            messages = [TextMessage(text= "จากการประมาลผลรูปนี้มีแนวโน้มที่จะเป็น "+message + "และนี่คือข้อมูลเพิ่มเติมครับ"),
                        TextMessage(text= payload)]
            
            line_bot_api.reply_message(event.reply_token, messages)
    else:
        message = ""
        f_csv = 'no-prediction'

        """Send message to ask user if they want to retry image prediction"""
        confirm_template = TemplateSendMessage(
            alt_text='Confirm template',
            template=ConfirmTemplate(
                text="ต้องขออภัยรูปภาพนี้อาจจะมีความชัดไม่มากพอหรืออาจจะไม่ใช่สายพันธุ์ฉลามที่สามารถพบเจอได้ในน่านน้ำไทย โปรดเลือกรูปอื่นเพื่อนำมาประมวลผลใหม่ครับ หรือหากท่านมั่นใจว่านี่คือรูปปลาฉลามและต้องการทำนายภาพนี้อีกครั้งโปรดเลือกใช่ ",
                actions=[
                    MessageAction(label="ใช่", text="retry"),
                    MessageAction(label="ไม่", text="no-retry")
                ]
            )
        )

        line_bot_api.reply_message(event.reply_token, [confirm_template])

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
    new_row = {'species': f_csv, 'file path': file_path}
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
