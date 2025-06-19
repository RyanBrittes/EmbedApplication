import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    #Executa uma vez quando instanciando o objeto Dataset, inicializamos o diretório contendo a imagem e 
    # anotamos os arquivos, então ambos são transformados
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        #O transform serve para processar os dados e torná-los adequados ao treinamento
        self.transform = transform
        self.target_transform = target_transform

    #Retorna o número de amostras dentro do conjunto de dados
    def __len__(self):
        return len(self.img_labels)

    #Carreaga e retorna uma amostra de um conjunto dedados pelo index provido
    #A imagem será localizada e convertida em um tensor usando o decode_image
    #Recupera o rótulo correspondente do arquivo csv em self.img_labels
    #Após chama a função de transformação e então retorna o tensor da imagem 
    # e o correspondente rótulo dentro de uma tupla
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label