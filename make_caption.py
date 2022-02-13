import json

with open(r'C:\Users\VIP444\Documents\COCO-Dataset\annotations\MSCOCO_train_val_Korean.json', 'r', encoding="UTF-8") as file:
    data = json.load(file)
    
with open('data/origin_caption.json', 'w', encoding="UTF-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

with open(r'data/origin_caption.json', 'r', encoding="UTF-8") as file:
    data = json.load(file)
    
print(len(data))

txt_data = []

for elem in data:
    for caption in elem['caption_ko']:
        path = elem['file_path'].split('/')[1]
        txt_data.append(path + ',' + caption)

print(len(txt_data))

file_path = 'data/caption.txt'

with open(file_path, 'w', encoding='UTF-8') as file:
    for elem in txt_data:
        file.write(elem + "\n")