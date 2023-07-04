import csv

# Define the function to generate the message payload
# def generate_payload(name):
#     with open('info/shark.csv', newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         # print(reader.fieldnames)
#         for row in reader:
#             thai_name = row['thai-name']
#             thai_name_split =[name.strip() for name in thai_name.split('||')]
#             if name in thai_name_split  or row['species'].lower() == name.lower():
#                 payload = f"ชื่อไทย: {row['thai-name'].replace('||', ',')}\nชื่อภาษาอังกฤษ: {row['species']}\nชื่อวิทยาศาสตร์: {row['sci-name']}\nชื่อวงศ์: {row['jenus']}\nจำนวนไข่: {row['num-egg']}\nขนาดของไข่: {row['size-egg']}\nขนาดของลำตัวสูงสุด: {row['size']}\nลักษณะการอพยพ: {row['live']}\nสถานที่พบในไทย: {row['where']}\nมักเข้าใจผิดเป็น: {row['miss']}\nสถานะปัจจุบัน: {row['status'].replace('||',',')}"
#                 img = row['img']
#                 return payload, img
#             else:
#                 if name == 'ฉลามนั้นชอบงับคุณ':
#                     payload1 = 'ส่วนผมนั้นชอบคุณงับ'
#                     return payload1
#                 elif 'ไม่' in str(name):
#                     payload2 = 'ไม่เป็นไรครับ หวังว่าน้องหลามจะได้ช่วยท่านนะครับ'
#                     return payload2
#         return "ไม่พบข้อมูลหรือคำสั่งที่ท่านต้องการ หรือท่านอาจพิมผิดครับ ลองพิมแล้วส่งข้อความมาใหม่ครับ"

import csv

# Read CSV file
with open('info/shark.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # Initialize arrays for each column
    species = []
    thai_name = []
    sci_name = []
    jenus = []
    num_egg = []
    size_egg = []
    size = []
    live = []
    where = []
    miss = []
    status = []
    img = []

    # Extract data from each row and append to respective arrays
    for row in reader:
        species.append(row['species'])
        thai_name.append(row['thai-name'])
        sci_name.append(row['sci-name'])
        jenus.append(row['jenus'])
        num_egg.append(row['num-egg'])
        size_egg.append(row['size-egg'])
        size.append(row['size'])
        live.append(row['live'])
        where.append(row['where'])
        miss.append(row['miss'])
        status.append(row['status'])
        img.append(row['img'])

# Print the arrays
print('species:', species)
print('thai-name:', thai_name)
print('sci-name:', sci_name)
print('jenus:', jenus)
print('num-egg:', num_egg)
print('size-egg:', size_egg)
print('size:', size)
print('live:', live)
print('where:', where)
print('miss:', miss)
print('status:', status)
print('img:', img)


# name = "pigeye shark"
# print(generate_payload(name))

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