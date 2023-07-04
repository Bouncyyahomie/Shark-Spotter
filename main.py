from model import load_ml_model
# from shark_payload import generate_payload
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import preprocess_input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
# from rembg import remove
import numpy as np
from fastapi import FastAPI, Request, Header, HTTPException,File, UploadFile
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
from io import BytesIO
import tensorflow as tf
from mangum import Mangum



def set_seed(seed):
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)

# Set the seed to a fixed value
set_seed(42)


load_dotenv()

channel_access_token = os.getenv('CHANNEL_ACCESS_TOKEN', None)
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)

line_bot_api = LineBotApi(channel_access_token)
line_handler = WebhookHandler(channel_secret)

app = FastAPI(title="Shark Spotter")

model_1 = load_ml_model('model/cnn_model.h5')
model_2 = load_ml_model('model/svc_model.pkl')
model_3 = load_ml_model('model/knn_model.pkl')

base_model = DenseNet201(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
extractor_model = Model(inputs=base_model.input, outputs=x)

class_labels = ['blacktip reef shark',
 'brownbanded bambooshark',
 'bull shark',
 'coral catshark',
 'graceful shark',
 'great hammerhead',
 'grey bambooshark',
 'grey reef shark',
 'indonesian bambooshark',
 'pigeye shark',
 'scalloped hammerhead',
 'slender bambooshark',
 'spinner shark',
 'spottail shark',
 'tiger shark',
 'whitespotted bambooshark',
 'whitetip reef shark',
 'zebra shark']

available_shark = [
    ["Grey bambooshark", "ฉลามกบเทา || ฉลามตุ๊กแก"],
    ["Indonesian bambooshark", "ฉลามกบ || ฉลามกบครีบเหลี่ยม || ฉลามกบอินโดนีเซีย || ฉลามลายตุ๊กแก"],
    ["Slender bambooshark", "ฉลามกบเหลือง || ฉลามกบหลังสัน || ฉลามกบลาย || ฉลามหิน || ฉลามลาย"],
    ["Whitespotted bambooshark", "ฉลามกบจุดขาว || ฉลามกบลายเสือน้ำตาล || ฉลามกบลาย || ฉลามหิน"],
    ["Brownbanded bambooshark", "ฉลามกบแถบน้ำตาล || ฉลามกบครีบเว้า || ฉลามกบจุด || ฉลามกบลาย"],
    ["Zebra shark", "ฉลามเสือดาว || ฉลามลายเสือดาว || ฉลามม้าลาย || ฉลามเซบราเอียน"],
    ["Coral catshark", "ฉลามกบลายหินอ่อน || ฉลามลายหินอ่อน || ฉลามแมว || ฉลามแมวปะการัง"],
    ["Graceful shark", "ฉลามหูดำ || ฉลามหน้าหมู || ฉลามจ้าวมัน || จ้าวมัน"],
    ["Grey reef shark", "ฉลามครีบดำใหญ่ || ฉลามปะการังสีเทา || ฉลามจ้าวมัน || จ้าวมัน || ฉลามสีเทา || ฉลามหูดำ"],
    ["Pigeye shark", "ฉลามตาเล็ก || ฉลามหน้าหมู || ฉลามชวา"],
    ["Spinner shark", "ฉลามหัวแหลม || ฉลามหูดำ || ฉลามจมูกยาว || ฉลามครีบยาว || ชายกรวย"],
    ['Bull shark', 'ฉลามหัวบาตร || ฉลามวัวกระทิง || ฉลามปากแม่น้ำ || ฉลามแม่น้ำ'],
    ['Blacktip reef shark', 'ฉลามปะการังครีบดำ || ฉลามหูดำ || ฉลามครีบดำ'],
    ['Spottail shark', 'ฉลามหางจุด || ฉลามหูดำ || ฉลามครีบดำ || ฉลามหนูหัวแหลม || ฉลามจุดดำ'],
    ['Whitetip reef shark', 'ฉลามครีบขาว || ฉลามขี้เซา || ฉลามปลายครีบขาว || ฉลามปะการังครีบขาว'],
    ['Tiger shark', 'ฉลามเสือ || เสือทะเล || ตะเพียนทอง || พิมพา'],
    ["Scalloped hammerhead", "ฉลามหัวค้อน || ฉลามหัวค้อนสีน้ำเงิน || ฉลามหัวฆ้อนสีเงิน || อ้ายแบ้"],
    ["Great hammerhead", "ฉลามหัวค้อนใหญ่ || ฉลามหัวฆ้อนยักษ์"]
]

species = ['Grey bambooshark', 'Indonesian bambooshark', 'Slender bambooshark', 'Whitespotted bambooshark', 'Brownbanded bambooshark', 'Zebra shark', 'Coral catshark', 'Graceful shark', 'Grey reef shark', 'Pigeye shark', 'Spinner shark', 'Bull shark', 'Blacktip reef shark', 'Spottail shark', 'Whitetip reef shark', 'Tiger shark', 'Scalloped hammerhead', 'Great hammerhead']
thai_name = [' ฉลามกบเทา || ฉลามตุ๊กแก', ' ฉลามกบ || ฉลามกบครีบเหลี่ยม || ฉลามกบอินโดนีเซีย || ฉลามลายตุ๊กแก', ' ฉลามกบเหลือง || ฉลามกบหลังสัน || ฉลามกบลาย || ฉลามหิน || ฉลามลาย', ' ฉลามกบจุดขาว || ฉลามกบลายเสือน้ำตาล || ฉลามกบลาย || ฉลามหิน', ' ฉลามกบแถบน้ำตาล || ฉลามกบครีบเว้า || ฉลามกบจุด || ฉลามกบลาย', ' ฉลามเสือดาว || ฉลามลายเสือดาว ||  ฉลามม้าลาย || ฉลามเสือ || เสือทะเล ', ' ฉลามกบลายหินอ่อน || ฉลามลายหินอ่อน || ฉลามแมว || ฉลามแมวปะการัง', ' ฉลามหูดำ || ฉลามหน้าหมู || ฉลามจ้าวมัน || จ้าวมัน', ' ฉลามครีบดำใหญ่ || ฉลามปะการังสีเทา || ฉลามจ้าวมัน || จ้าวมัน || ฉลามสีเทา || ฉลามหูดำ', ' ฉลามตาเล็ก || ฉลามหน้าหมู || ฉลามชวา', ' ฉลามหัวแหลม || ฉลามหูดำ || ฉลามจมูกยาว || ฉลามครีบยาว || ชายกรวย', ' ฉลามหัวบาตร || ฉลามวัวกระทิง || ฉลามปากแม่น้ำ || ฉลามแม่น้ำ', ' ฉลามปะการังครีบดำ || ฉลามหูดำ || ฉลามครีบดำ', ' ฉลามหางจุด || ฉลามหูดำ || ฉลามครีบดำ || ฉลามหนูหัวแหลม || ฉลามจุดดำ', ' ฉลามครีบขาว || ฉลามขี้เซา || ฉลามปลายครีบขาว || ฉลามปะการังครีบขาว', ' ฉลามเสือ || เสือทะเล || ตะเพียนทอง || พิมพา', ' ฉลามหัวค้อน || ฉลามหัวค้อนสีน้ำเงิน || ฉลามหัวฆ้อนสีเงิน || อ้ายแบ้', ' ฉลามหัวค้อนใหญ่ || ฉลามหัวค้อนยักษ์ || ฉลามหัวฆ้อนยักษ์']
sci_name = [' Chiloscyllium griseum', ' Chiloscyllium hasselti', ' Chiloscyllium indicum', ' Chiloscyllium plagiosum', ' Chiloscyllium punctatum', ' Stegostoma tigrinum', ' Stegostoma tigrinum', ' Carcharhinus amblyrhynchoides', ' Carcharhinus amblyrhynchos', ' Carcharhinus amboinensis', ' Carcharhinus brevipinna', ' Carcharhinus leucas', ' Carcharhinus melanopterus', ' Carcharhinus sorrah', ' Triaenodon obesus', ' Galeocerdo cuvier', ' Sphyrna lewini', ' Sphyrna mokarran']
jenus = [' Hemiscylliidae', ' Hemiscylliidae', ' Hemiscylliidae', ' Hemiscylliidae', ' Hemiscylliidae', ' Stegostomatidae', ' Scyliorhinidae', ' Carcharhinidae', ' Carcharhinidae', ' Carcharhinidae', ' Carcharhinidae', ' Carcharhinidae', ' Carcharhinidae', ' Carcharhinidae', ' Carcharhinidae', ' Galeocerdidae', '  Sphyrnidaea', ' Sphyrnidaea']
num_egg = [' 2', ' 2', ' 2', ' 2', ' 2', ' 2-4', ' 2', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่', ' ไม่ได้สืบพันธุ์ด้วยวิธีการวางไข่']
size_egg = [' 7-9 cm', ' 7-9 cm', ' -', ' 8 cm', ' 10-15 cm', ' 13-17 cm', ' 6-8 cm', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -']
size = [' 77 -90 cm', ' 61 - 91 cm', ' 65 - 82 cm', ' 95 cm', ' 132-144 cm', ' 250-354 cm', ' 70 cm', ' 182-243 cm', ' 265 cm', ' 280-303 cm', ' 304 cm', ' 366 - 400 cm', ' 180 - 205 cm', ' 90-128 cm', ' 213 cm', ' 226-305 cm', ' 140 -198 cm', ' 610 cm']
live = [' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพอยู่ระหว่างทะเลและน้ำจืด', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพระหว่างทะเลและน้ำจืดตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลเปิด', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพระหว่างทะเลและน้ำจืดตามแนวปะการัง', ' อพยพระหว่างทะเลและน้ำจืดตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลตามแนวปะการัง', ' อพยพอยู่ในเฉพาะในทะเลลึก', ' อพยพอยู่ระหว่างทะเลเปิดและน้ำจืด', ' อพยพอยู่ในเฉพาะในทะเลเปิด']
where = [' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย', ' ทะเลอันดามันและอ่าวไทย']
miss = [' ฉลามกบ (Indonesian bambooshark)', ' ฉลามกบเทา (Grey bambooshark)', ' -', ' - ', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -', ' -']
status = [' ONEP/IUCN - มีแนวโน้มใกล้สูญพันธุ์', ' IUCN - ใกล้สูญพันธุ์ || ONEP -  มีแนวโน้มใกล้สูญพันธ์', ' ONEP/IUCN - มีแนวโน้มใกล้สูญพันธุ์', 'ONEP -  มีแนวโน้มใกล้สูญพันธุ์ || IUCN - ใกล้ถูกคุกคาม', 'ONEP -  มีแนวโน้มใกล้สูญพันธุ์ || IUCN - ใกล้ถูกคุกคาม', ' ONEP -  ใกล้สูญพันธุ์ || IUCN - ใกล้สูญพันธุ์', ' ONEP -  ใกล้ถูกคุกคาม || IUCN - ใกล้ถูกคุกคาม', ' ONEP -  ใกล้สูญพันธุ์ || IUCN -  มีแนวโน้มใกล้สูญพันธุ์', ' ONEP -  ใกล้สูญพันธุ์ || IUCN -  ใกล้สูญพันธุ์', ' ONEP -  มีแนวโน้มใกล้สูญพันธุ์ || IUCN -  มีแนวโน้มใกล้สูญพันธุ์', ' ONEP -  มีแนวโน้มใกล้สูญพันธุ์ || IUCN -  มีแนวโน้มใกล้สูญพันธุ์', ' ONEP -  ใกล้สูญพันธุ์ || IUCN -  มีแนวโน้มใกล้สูญพันธุ์', 'ONEP -  มีแนวโน้มใกล้สูญพันธุ์ || IUCN -  มีแนวโน้มใกล้สูญพันธุ์', ' ONEP -  มีแนวโน้มใกล้สูญพันธุ์ || IUCN -  ใกล้ถูกคุกคาม', ' ONEP -  มีแนวโน้มใกล้สูญพันธุ์ || IUCN -  มีแนวโน้มใกล้สูญพันธุ์', ' ONEP -  ใกล้สูญพันธุ์ || IUCN -  ใกล้ถูกคุกคาม', ' ONEP -  ใกล้สูญพันธุ์อย่างยิ่ง || IUCN -  ใกล้สูญพันธุ์อย่างยิ่ง', ' ONEP -  ใกล้สูญพันธุ์อย่างยิ่ง || IUCN -  ใกล้สูญพันธุ์อย่างยิ่ง']
img = ['https://sv1.picz.in.th/images/2023/04/15/maebPn.md.jpg', 'https://sv1.picz.in.th/images/2023/04/15/mamccJ.jpg', 'https://sv1.picz.in.th/images/2023/04/15/mamPtt.jpg', 'https://sv1.picz.in.th/images/2023/04/15/mam8SN.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpWbs9.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpWDjf.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpWV9I.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpWXjP.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpWavt.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpWvNe.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpWGUl.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpWmcE.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpYPnW.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpYCK1.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpY8OJ.md.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpYX4f.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpYaxa.md.jpg', 'https://sv1.picz.in.th/images/2023/04/16/mpYpnq.jpg']


def generate_payload(name):
    # Loop through the arrays
    for i in range(len(species)):
        # Check if the name matches the species or is in the Thai name array
        if name in thai_name[i] or species[i].lower() == name.lower():
            payload = f"ชื่อไทย: {thai_name[i].replace('||', ',')}\nชื่อภาษาอังกฤษ: {species[i]}\nชื่อวิทยาศาสตร์: {sci_name[i]}\nชื่อวงศ์: {jenus[i]}\nจำนวนไข่: {num_egg[i]}\nขนาดของไข่: {size_egg[i]}\nขนาดของลำตัวสูงสุด: {size[i]}\nลักษณะการอพยพ: {live[i]}\nสถานที่พบในไทย: {where[i]}\nมักเข้าใจผิดเป็น: {miss[i]}\nสถานะปัจจุบัน: {status[i].replace('||',',')}"
            img_url = img[i]
            return payload, img_url
    else:
        if name == 'ฉลามนั้นชอบงับคุณ':
            payload1 = 'ส่วนผมนั้นชอบคุณงับ'
            return payload1
        elif 'ไม่' in str(name):
            payload2 = 'ไม่เป็นไรครับ หวังว่าน้องหลามจะได้ช่วยท่านนะครับ'
            return payload2
    return "ไม่พบข้อมูลหรือคำสั่งที่ท่านต้องการ หรือท่านอาจพิมผิดครับ ลองพิมแล้วส่งข้อความมาใหม่ครับ"


os.chdir('/tmp')


best_weights = [0.6, 0.2, 0.2]

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


# def extract_feature_no_bg(img):
#     features = []
#     labels = []
#     nb_features = 1920
#     features = np.empty((1, nb_features))

#     img_resized = img.resize((299, 299))
#     x = image.img_to_array(img_resized)
#     output1 = remove(img)
#     output2 = output1.convert('RGB')
#     x = image.img_to_array(output2)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features[0, :] = np.squeeze(extractor_model.predict(x))

#     return features, labels


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

    ensemble_predictions = (preds_1 + preds_prob_2 + preds_prob_3) / 3

    confidence_threshold = 0.6

    # get the maximum confidence for each prediction
    max_confidence = np.max(ensemble_predictions, axis=1)

    low_confidence_mask = max_confidence > confidence_threshold

    print(max_confidence)
    print(low_confidence_mask)

    if top_label1[0] == top_label2[0] == top_label3[0] :
        return "Y", top_label1[0]
    elif low_confidence_mask:
        return "Y", final_preds
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
        line_handler.handle(body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError as e:
        raise HTTPException(
            status_code=400, detail="chatbot handle body error.%s" % e.message)
    return 'OK'


@line_handler.add(MessageEvent, message=TextMessage)
def get_user_response(event):
    profile = line_bot_api.get_profile(event.source.user_id)
    start_word = ['สวัสดีครับคุณพี่ ', 'ฉลามนั้นชอบงับคุณ แต่ผมมาช่วยคุณงับ คุณ']
    print(event.message.text)
    print(profile.display_name)

    if event.message.text == "retry\u200B":
        with open('collect.csv', "r") as f1:
            line = f1.readlines()
        last_line = line[-1]
        arr = last_line.split(',')
        file_path = arr[1].strip()

        img = Image.open(file_path)

        features, labels = extract_feature(img)
        img_pred_class1 = predict_ensemble_no_shark(features)

        # features2, labels2 = extract_feature_no_bg(img)
        # img_pred_class2 = predict_ensemble_no_shark(features2)

        # x = final_answer(img_pred_class1, img_pred_class2)
        x= encoder_labels.inverse_transform(img_pred_class1)
        print(x)
        if type(x) is tuple:
            message = x[0].capitalize() + ' หรือ ' + x[1].capitalize() 
            f_csv = message
        else:
            print(x)
            message = x[0]
            f_csv = message

        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(
                    text= "จากการประมาลผลรูปนี้มีแนวโน้มที่จะเป็น "+message+" และ หากท่านต้องการข้อมูลเพิ่มเติมของสายพันธุ์ฉลามสามารถพิมชื่อของสายพันธุ์นั้นมาได้เลยครับ"),
            ])

        csv_path = 'collect.csv'
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

    elif event.message.text == "no-retry\u200B":
        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(
                    text="หากท่านต้องการทำนายสายพันธุ์ปลาฉลามจากรูปใหม่ สามารถส่งรูปมาใหม่ได้เลยครับ"),
            ])

    elif event.message.text == "classify shark":
        response_word = random.choice(start_word) + profile.display_name + \
            "สงสัยว่าฉลามที่เจอเป็นฉลามสายพันธุ์ไหนสามารถส่งมาถามน้องหลามได้เลยครับ"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response_word)
        )
    elif event.message.text == "shark info":
        response_word = random.choice(start_word) + profile.display_name + \
                        f"ต้องการข้อมูลของฉลามสายพันธุ์ไหน สามารถพิมชื่อมาได้เลยครับ นี้คือรายชื่อของฉลามที่เราสามารถตอบได้:\n"
        for i in available_shark:
           
            response_word += i[0] + ' - ' + i[1].replace('||', ',') + '\n' + '\n'

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response_word)
        )
    elif event.message.text == "more info about shark":
        response_word = random.choice(start_word) + profile.display_name + \
            "หากสนใจเกี่ยวกับ ฉลามและปลากระดูกอ่อนในน่านน้ำไทย สามารถสแกน QR Code ด่านล่างได้เลยเพื่อศึกษาเพิ่มเติมจาก หนังสือ ปลากระดูกอ่อนของไทยและใกล้เคียง (THE CARTILAGINOUS FISHES OF THAILAND AND ADJUSCENT WATERS) ได้เลยครับ"

        messages = [TextSendMessage(text=response_word),
                   ImageSendMessage(original_content_url="https://sv1.picz.in.th/images/2023/04/14/mVe6nI.jpg",
                                    preview_image_url="https://sv1.picz.in.th/images/2023/04/14/mVe6nI.jpg")
                   ]
        
        line_bot_api.reply_message(event.reply_token, messages)
        
    else:
        result = generate_payload(event.message.text)
        if len(result) == 2:

            payload = result[0]
            img = result[1]
            print(img)
            response_word = random.choice(start_word) + profile.display_name + \
                "พันธุ์ฉลามที่ท่านได้ส่งมานั้น มีข้อมูลตามนี้ครับ"
                
            messages = [TextMessage(text=response_word),
                        TextMessage(text= payload),
                        ImageSendMessage(original_content_url=img,
                                        preview_image_url=img)]
            
            line_bot_api.reply_message(event.reply_token, messages)
        else: 
            payload = result
            response_word =  f"นี่คือรายชื่อของฉลามที่เราสามารถตอบได้:\n"
            for i in available_shark:
                response_word += i[0] + ' - ' + i[1].replace('||', ',') + '\n' + '\n'
                
            messages = [TextMessage(text=payload), TextMessage(text=response_word)]
            line_bot_api.reply_message(event.reply_token, messages)



@line_handler.add(MessageEvent, message=(ImageMessage))
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        ext = 'jpg'
    else:
        return

    message_content = line_bot_api.get_message_content(event.message.id)
    print(type(message_content))

    # static_path = os.path.join(os.path.dirname(__file__), "tmp")
    
    image_folder = "images/"
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

    # features2, labels2 = extract_feature_no_bg(img)
    # img_pred_class2 = predict_ensemble(features2)
    # print(img_pred_class2[0])
    if  img_pred_class1[0] == "Y":
        x= encoder_labels.inverse_transform([img_pred_class1[1]])
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
            message = x[0].capitalize()
            print(message)
            f_csv = message
            payload, img = generate_payload(message)
            
            messages = [TextMessage(text= "จากการประมาลผลรูปนี้มีแนวโน้มที่จะเป็น "+message + "และนี่คือข้อมูลเพิ่มเติมครับ"),
                        TextMessage(text= payload),
                        ImageSendMessage(original_content_url=img,
                                    preview_image_url=img)
                        ]
            
            line_bot_api.reply_message(event.reply_token, messages)
    else:
        message = ""
        f_csv = 'no-prediction'

        """Send message to ask user if they want to retry image prediction"""
        confirm_template = TemplateSendMessage(
            alt_text='Confirm template',
            template=ConfirmTemplate(
                text="ต้องขออภัยรูปภาพนี้ไม่สามารถระบุสายพันธุ์ปลาฉลามในรูปภาพนี้ได้ เนื่องจากอาจจะมีความชัดไม่มากพอหรืออาจจะไม่ใช่สายพันธุ์ฉลามที่สามารถพบเจอได้ในน่านน้ำไทย โปรดเลือกรูปอื่นเพื่อนำมาประมวลผลใหม่ครับ หรือหากท่านมั่นใจว่านี่คือรูปปลาฉลามและต้องการทำนายภาพนี้อีกครั้งโปรดเลือกใช่ ",
                actions=[
                    MessageAction(label="ใช่", text="retry\u200B"),
                    MessageAction(label="ไม่", text="no-retry\u200B"),
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

    csv_path = 'collect.csv'
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


handler = Mangum(app, lifespan="off")