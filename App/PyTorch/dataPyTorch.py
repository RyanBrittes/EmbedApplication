import torch

#Utilizamos as propriedades Datasets e DataLoaders para trabalhar com conjuntos de dados
#São úteis para criar uma camada entre o código e os dados armazenados no modelos treinados

#Datasets: armazenam os dados em suas correspondentes camadas
torch.utils.data.Dataset

#DataLoader: envolve um iterável em torno do Dataset para permitir acesso fácil às amostras.
torch.utils.data.DataLoader
