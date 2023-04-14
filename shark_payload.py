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
                payload = f"ชื่อไทย: {row['thai-name']}\nชื่อภาษาอังกฤษ: {row['species']}\nnชื่อวิทยาศาสตร์: {row['sci-name']}\nชื่อวงศ์: {row['jenus']}\nจำนวนไข่: {row['num-egg']}\nขนาดของไข่: {row['size-egg']}\nขนาดของลำตัวสูงสุด: {row['size']}\nลักษณะการอพยพ: {row['live']}\nสถานที่พบในไทย: {row['where']}\nมักเข้าใจผิดเป็น: {row['miss']}\nสถานะปัจจุบัน: {row['status']}"
                return payload
    return "ไม่พบข้อมูล หรือท่านอาจพิมผิดครับ ลองพิมมาใหม่ดูครับ"

            
# name = "pigeye shark"
# print(generate_payload(name))