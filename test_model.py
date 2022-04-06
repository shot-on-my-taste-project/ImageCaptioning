from glob import glob
from caption_model.models import EncoderCNN, DecoderRNN
from caption_model.vocab import Vocabulary
from PIL import Image
from torchvision import transforms
import pickle
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)

    # 이미지 정규화
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = transform(image).unsqueeze(0)

    return image

def main():
    # 캡션을 생성할 이미지가 존재하는 디렉토릭 경로
    image_dir = r"data\resize_test"
    encoder_path = "caption_model/encoder-5.ckpt" 
    decoder_path = "caption_model/decoder-5.ckpt" 
    vocab_path = "caption_model/vocab_26076.pkl" 

    # 모델 파라미터
    embed_size = 256 # 임베딩 차원
    hidden_size = 512 # 히든 스테이트 차원
    num_layers = 1 # LSTM 레이어 수

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary()
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # 모델 빌드 및 로드
    encoder = EncoderCNN(embed_size).eval() 
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    image_path = os.listdir(image_dir)

    for path in image_path:
        path = path = os.path.dirname(os.path.abspath(path))
        image = load_image(path)
        image_tensor = image.to(device)

        # 캡션 생성
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy() 

        # 문자열로 변환
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id] 
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        # 이미지 및 경로, 생성된 캡션 출력
        image = Image.open(image_path)
        plt.imshow(np.asarray(image))
        plt.show()
        print(sentence)
        print(path.split("\\")[-1])

if __name__ == "__main__":
    print(123)
    main()