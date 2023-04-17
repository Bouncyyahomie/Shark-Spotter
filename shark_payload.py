import csv

# Define the function to generate the message payload
def generate_payload(name):
    with open('info/shark.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        # print(reader.fieldnames)
        for row in reader:
            thai_name = row['thai-name']
            thai_name_split =[name.strip() for name in thai_name.split('||')]
            if name in thai_name_split  or row['species'].lower() == name.lower():
                payload = f"ชื่อไทย: {row['thai-name'].replace('||', ',')}\nชื่อภาษาอังกฤษ: {row['species']}\nชื่อวิทยาศาสตร์: {row['sci-name']}\nชื่อวงศ์: {row['jenus']}\nจำนวนไข่: {row['num-egg']}\nขนาดของไข่: {row['size-egg']}\nขนาดของลำตัวสูงสุด: {row['size']}\nลักษณะการอพยพ: {row['live']}\nสถานที่พบในไทย: {row['where']}\nมักเข้าใจผิดเป็น: {row['miss']}\nสถานะปัจจุบัน: {row['status'].replace('||',',')}"
                img = row['img']
                return payload, img
            else:
                if name == 'ฉลามนั้นชอบงับคุณ':
                    payload1 = 'ส่วนผมนั้นชอบคุณงับ'
                    return payload1
                elif 'ไม่' in str(name):
                    payload2 = 'ไม่เป็นไรครับ หวังว่าน้องหลามจะได้ช่วยท่านนะครับ'
                    return payload2
        return "ไม่พบข้อมูลหรือคำสั่งที่ท่านต้องการ หรือท่านอาจพิมผิดครับ ลองพิมแล้วส่งข้อความมาใหม่ครับ"


# name = "pigeye shark"
# print(generate_payload(name))