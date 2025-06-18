import torch

#Formas dos tensores:
a = torch.eye(3) # -> Matriz identidade 3X3
b = torch.rand(2, 3) # -> Valores aleatórios entre 0 e 1
c = torch.ones((3, 3)) # -> Tensor de 3X3 preenchidos com 1
d = torch.zeros(2, 3) # -> Tensor 2X3 preenchido com zeros
e = torch.tensor([[3.109210, 2.091201], [6.123163, 0.87491]]) # -> Matriz 2D
f = torch.tensor([1, 2, 3]) # -> Vetor 1D
torch.manual_seed(1850) # -> Semente que irá fixar os valores aleatórios que deverão aparecer
g = torch.rand(2, 2) # -> Usará a semente acima para prover os valores aleatórios
h = torch.empty(3, 4, 2) # Cria um conjunto de matrizes 1°N de matrizes 2°Nº de linhas 3°N de colunas
i = torch.empty_like(h) # Cria um conjunto de matrizes igual ao informado
j = torch.zeros_like(h) # Cria um conjunto de matrizes com zeros igual ao informado
k = torch.ones_like(h) # Cria um conjunto de matrizes com ums igual ao informado
l = torch.rand_like(h) # Cria um conjunto de matrizes com aleatórios igual ao informado
m = torch.tensor((2, 5, 12, 1, 8)) # Conjunto de valores inteiros
n = torch.tensor(((2, 5, 7), [5, 1, 9])) # -> Matriz 2D

if torch.accelerator.is_available():
    print('We have an accelerator!')
else:
    print('Sorry, CPU only.')