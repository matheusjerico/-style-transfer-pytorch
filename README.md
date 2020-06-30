# Transferência de Estilo com Deep Learning

Este notebook é baseado no conteúdo do [github da Udacity para o Deep Learning Nanodegree](https://github.com/udacity/deep-learning-v2-pytorch).

Neste notebook, iremos replicar a técnica de transferência de estilo em PyTorch descrita neste paper: [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

Este paper utiliza a VGG com 19 layers para extrair features. Essa rede é composta por camadas convolucionais e de pooling. Na imagem abaixo, Conv_1_1 indica a primeira camada convolucional do primeiro conjunto de camadas. Conv_2_1 indica a primeira camada convolucional do segundo conjunto de camadas. A camada convolucional mais profunda nessa rede é a Conv_5_4.

<img src='notebook_ims/vgg19_convlayers.png' width=80% />

### Separando Estilo e Conteúdo

Transferência de estilo consiste em separar o conteúdo e o estilo de uma imagem. Dadas uma imagem de conteúdo e uma imagem de estilo, queremos criar uma nova imagem _target_ que deve conter os componentes de conteúdo e de estilo desejados:
* objetos e suas disposições são a **imagem de conteúdo**
* estilo, cores e texturas são a **imagem de estilo**

O exemplo abaixo exibe a imagem de conteúdo (um gato) e uma image de estilo ([Hokusai's Great Wave](https://en.wikipedia.org/wiki/The_Great_Wave_off_Kanagawa)).

A imagem _target_ gerada ainda contém o gato, mas ele está estilizado com as ondas, as cores azul e bege e blocos de texturas da imagem de estilo!

<img src='notebook_ims/style_tx_cat.png' width=80% />

Iremos utilizar a VGG19 pré-treinada para extrair conteúdo ou features de estilo de uma imagem. Iremos formalizar, mais para frente, os conceitos de _loss_ de conteúdo e de estilo, e utilizá-las de forma iterativa para atualizar nossa imagem _target_ até que consigamos o resultado que queremos.


```python
%matplotlib inline

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
```

## Carregando a VGG19 (features)

A VGG199 é dividida em duas partes:
* `vgg19.features`, que possui todas as camadas convolucionais e de pooling
* `vgg19.classifier`, que possui o classificador formado por três camadas lineares

Nós precisamos apenas da parte `features`, cujos pesos iremos carregar e "congelar".


```python
vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)
```

    Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" to /home/matheuspalhares/.cache/torch/checkpoints/vgg19-dcbb9e9d.pth



    HBox(children=(FloatProgress(value=0.0, max=574673361.0), HTML(value='')))


    



```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)
```




    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True)
      (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace=True)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace=True)
      (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): ReLU(inplace=True)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace=True)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace=True)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace=True)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace=True)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )



### Carregando As Imagens de Conteúdo e de Estilo

Você pode carregar as imagens que quiser. A função abaixo nos permite carregar uma imagem de qualquer tipo e tamanho.

A função `load_image` também converte as imagens para Tensors normalizados.

Será mais fácil termos imagens menores e termos as imagens de conteúdo e estilo reduzidas para as mesmas dimensões.


```python
def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image
```

Agora, iremos carregar os arquivos das imagens e forçaremos a imagem de estilo para o mesmo tamanho da imagem de conteúdo.


```python
# load in content and style image
content = load_image('images/foto_praia.jpg').to(device)
# Resize style to match content, makes code easier
style = load_image('images/romero_02.jpg', shape=content.shape[-2:]).to(device)
```


```python
# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image
```


```python
# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))
```




    <matplotlib.image.AxesImage at 0x7f26d151b828>




![png](imagens/output_10_1.png)


---
## Camadas da VGG19

Para obtermos as representações de conteúdo e de estilo de uma imagem, devemos passar a imagem através da rede VGG19 até chegarmos nas camadas desejadas e, então, pegaremos a saída dessa camada.


```python
# print out VGG19 structure so you can see the names of various layers
print(vgg)
```

    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True)
      (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace=True)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace=True)
      (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): ReLU(inplace=True)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace=True)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace=True)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace=True)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace=True)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )


## Features de Conteúdo e de Estilo

Na célula abaixo, iremos dar nomes às camadas de acordo com os nomes no paper para a _representação de conteúdo_ e para a _representação de estilo_.


```python
def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features
```

---
## Matriz Gram

A saída de cada camada convolucional é um Tensor cujas dimensões estão associadas ao `batch_size`, uma profundidade `d` e alguma altura e largura (`h`, `w`). A matriz Gram de uma camada convolucional é calculada conforme abaixo:
* Pegamos a profundidade, altura e largura do tensor usando `batch_size, d, h, w = tensor.size`
* Redimensionamos esse tensor de tal forma que as dimensões espaciais sejam achatadas (flattened)
* Calculamos a matriz Gram por meio da multiplicação do tensor redimensionado por sua transposta

*Nota: podemos multiplicar duas matrizes utilizando `torch.mm(matrix1, matrix2)`.*


```python
def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram 
```

## Ligando Os Pontos

Agora que já escrevemos funções para extrair features e calcular a matriz Gram de uma camada convolucional, vamos ligar os pontos. Iremos extrair nossas features das imagens e calcular as matrizes Gram para cada camada na nossa representação de estilo.


```python
# get content and style features only once before training
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)
```

---
## Loss e Pesos

#### Pesos de Estilo de Camada Individual

Na célula abaixo, você terá a oportunidade de dar pesos para cada camada relevante. É sugerido que você utilize pesos no intervalo 0-1. Aumentando o peso em layers mais próximos do começo da rede (`conv1_1` e `conv2_1`), é esperado alcançar artefatos de estilo _maiores_ na imagem final. Por outro lado, caso você opte por aumentar os pesos das camadas finais, a ênfase será dada em features menores.

Isso ocorre porque cada camada possui um tamanho diferente e, juntas, elas criam uma representação de estilo multi-escala.

#### Pesos de Conteúdo e Estilo

Assim como no paper, iremos definir um alpha (`content_weight`) e um beta (`style_weight`).

Essa taxa irá afetar o quão estilizada a imagem final será. É recomendado utilizar `content_weight = 1` e ajustar o `style_weight` para alcançar a relação desejada.


```python
# weights for each style layer 
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1  # alpha
style_weight = 1e6  # beta
```

## Atualizando o Target e Calculando as Losses

Aqui, iremos decidir a quantidade de épocas em que iremos atualizar nossa imagem. Será algo similar ao loop de treinamento que temos feito até hoje, porém estaremos alterando apenas nossa imagem de saída (nada na VGG19 nem em outras imagens).

**São recomendados pelo menos 2000 épocas para alcançar bons resultados**. Entretanto, comece com um valor bem menor apenas para testar seu código com diferentes pesos, parâmetros e imagens.

Dentro do loop de treinamento, iremos calcular as losses de conteúdo e de estilo, e atualizar a imagem de saída.

#### Loss de Conteúdo

A loss de conteúdo será a diferença média quadrática entre as features do target e do conteúdo na camada `conv4_2`. Isso será calculado da seguinte maneira:
```
content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
```

#### Loss de Estilo

A loss de estilo é calculada de forma similar, porém iremos iterar ao longo das camadas especificadas por `name` no nosso dicionário `style_weights`.
> Calcularemos a matriz Gram para a imagem target (`target_gram`) e para a imagem de estilo (`style_gram`) em cada uma das camadas e compararemos essas matrizes Gram, calculando o `layer_style_loss`.
> Depois, esse valor será normalizado pelo tamanho da camada.

#### Loss Total

Por fim, iremos calcular a loss total somando as duas losses calculadas acima e as pesando de acordo com alpha e beta.

Iremos imprimir, de tempos em tempos, essa loss. Não se assuste se ela for bem alta. Demora um bom tempo para que o estilo de uma imagem mude e você deve focar na aparência da imagem de saída, então não se preocupe com o valor da loss. Entretanto, note que esse valor decresce com o tempo.


```python
# for displaying the target image, intermittently
show_every = 50

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 5000  # decide how many iterations to update your image (5000)

for ii in range(1, steps+1):
    
    # get the features from your target image
    target_features = get_features(target, vgg)
    
    # the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)
        
    # calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # display intermediate images and print the loss
    if  ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()
```

    Total loss:  819438080.0



![png](imagens/output_22_1.png)


    Total loss:  542704896.0



![png](imagens/output_22_3.png)


    Total loss:  411975392.0



![png](imagens/output_22_5.png)


    Total loss:  333755072.0



![png](imagens/output_22_7.png)


    Total loss:  278132864.0



![png](imagens/output_22_9.png)


    Total loss:  235787424.0



![png](imagens/output_22_11.png)


    Total loss:  202574896.0



![png](imagens/output_22_13.png)


    Total loss:  176136224.0



![png](imagens/output_22_15.png)


    Total loss:  154843616.0



![png](imagens/output_22_17.png)


    Total loss:  137402704.0



![png](imagens/output_22_19.png)


    Total loss:  122772648.0



![png](imagens/output_22_21.png)


    Total loss:  110254944.0



![png](imagens/output_22_23.png)


    Total loss:  99376368.0



![png](imagens/output_22_25.png)


    Total loss:  89792640.0



![png](imagens/output_22_27.png)


    Total loss:  81268296.0



![png](imagens/output_22_29.png)


    Total loss:  73649992.0



![png](imagens/output_22_31.png)


    Total loss:  66868192.0



![png](imagens/output_22_33.png)


    Total loss:  60850960.0



![png](imagens/output_22_35.png)


    Total loss:  55526524.0



![png](imagens/output_22_37.png)


    Total loss:  50823432.0



![png](imagens/output_22_39.png)


    Total loss:  46695364.0



![png](imagens/output_22_41.png)


    Total loss:  43091420.0



![png](imagens/output_22_43.png)


    Total loss:  39951184.0



![png](imagens/output_22_45.png)


    Total loss:  37202348.0



![png](imagens/output_22_47.png)


    Total loss:  34796916.0



![png](imagens/output_22_49.png)


    Total loss:  32679054.0



![png](imagens/output_22_51.png)


    Total loss:  30798708.0



![png](imagens/output_22_53.png)


    Total loss:  29117488.0



![png](imagens/output_22_55.png)


    Total loss:  27601762.0



![png](imagens/output_22_57.png)


    Total loss:  26232062.0



![png](imagens/output_22_59.png)


    Total loss:  24985010.0



![png](imagens/output_22_61.png)


    Total loss:  23840542.0



![png](imagens/output_22_63.png)


    Total loss:  22786544.0



![png](imagens/output_22_65.png)


    Total loss:  21809866.0



![png](imagens/output_22_67.png)


    Total loss:  20901558.0



![png](imagens/output_22_69.png)


    Total loss:  20056220.0



![png](imagens/output_22_71.png)


    Total loss:  19266220.0



![png](imagens/output_22_73.png)


    Total loss:  18528052.0



![png](imagens/output_22_75.png)


    Total loss:  17834576.0



![png](imagens/output_22_77.png)


    Total loss:  17182616.0



![png](imagens/output_22_79.png)


    Total loss:  16570057.0



![png](imagens/output_22_81.png)


    Total loss:  15992542.0



![png](imagens/output_22_83.png)


    Total loss:  15445900.0



![png](imagens/output_22_85.png)


    Total loss:  14928993.0



![png](imagens/output_22_87.png)


    Total loss:  14437803.0



![png](imagens/output_22_89.png)


    Total loss:  13972543.0



![png](imagens/output_22_91.png)


    Total loss:  13530054.0



![png](imagens/output_22_93.png)


    Total loss:  13108809.0



![png](imagens/output_22_95.png)


    Total loss:  12707406.0



![png](imagens/output_22_97.png)


    Total loss:  12323727.0



![png](imagens/output_22_99.png)


    Total loss:  11956448.0



![png](imagens/output_22_101.png)


    Total loss:  11604621.0



![png](imagens/output_22_103.png)


    Total loss:  11267581.0



![png](imagens/output_22_105.png)


    Total loss:  10944738.0



![png](imagens/output_22_107.png)


    Total loss:  10635712.0



![png](imagens/output_22_109.png)


    Total loss:  10339543.0



![png](imagens/output_22_111.png)


    Total loss:  10054292.0



![png](imagens/output_22_113.png)


    Total loss:  9780658.0



![png](imagens/output_22_115.png)


    Total loss:  9516442.0



![png](imagens/output_22_117.png)


    Total loss:  9261645.0



![png](imagens/output_22_119.png)


    Total loss:  9016673.0



![png](imagens/output_22_121.png)


    Total loss:  8780886.0



![png](imagens/output_22_123.png)


    Total loss:  8553117.0



![png](imagens/output_22_125.png)


    Total loss:  8333329.0



![png](imagens/output_22_127.png)


    Total loss:  8121414.0



![png](imagens/output_22_129.png)


    Total loss:  7916936.5



![png](imagens/output_22_131.png)


    Total loss:  7719941.0



![png](imagens/output_22_133.png)


    Total loss:  7530434.5



![png](imagens/output_22_135.png)


    Total loss:  7347936.5



![png](imagens/output_22_137.png)


    Total loss:  7171939.5



![png](imagens/output_22_139.png)


    Total loss:  7002148.5



![png](imagens/output_22_141.png)


    Total loss:  6838000.5



![png](imagens/output_22_143.png)


    Total loss:  6679098.5



![png](imagens/output_22_145.png)


    Total loss:  6525334.5



![png](imagens/output_22_147.png)


    Total loss:  6376251.5



![png](imagens/output_22_149.png)


    Total loss:  6231810.5



![png](imagens/output_22_151.png)


    Total loss:  6092147.0



![png](imagens/output_22_153.png)


    Total loss:  5956545.5



![png](imagens/output_22_155.png)


    Total loss:  5825005.5



![png](imagens/output_22_157.png)


    Total loss:  5697645.0



![png](imagens/output_22_159.png)


    Total loss:  5574315.0



![png](imagens/output_22_161.png)


    Total loss:  5454559.0



![png](imagens/output_22_163.png)


    Total loss:  5338634.0



![png](imagens/output_22_165.png)


    Total loss:  5225729.5



![png](imagens/output_22_167.png)


    Total loss:  5115792.0



![png](imagens/output_22_169.png)


    Total loss:  5009159.5



![png](imagens/output_22_171.png)


    Total loss:  4905533.0



![png](imagens/output_22_173.png)


    Total loss:  4804947.0



![png](imagens/output_22_175.png)


    Total loss:  4707334.5



![png](imagens/output_22_177.png)


    Total loss:  4612703.5



![png](imagens/output_22_179.png)


    Total loss:  4521017.0



![png](imagens/output_22_181.png)


    Total loss:  4431694.5



![png](imagens/output_22_183.png)


    Total loss:  4344792.5



![png](imagens/output_22_185.png)


    Total loss:  4260256.0



![png](imagens/output_22_187.png)


    Total loss:  4178184.0



![png](imagens/output_22_189.png)


    Total loss:  4098491.75



![png](imagens/output_22_191.png)


    Total loss:  4020838.25



![png](imagens/output_22_193.png)


    Total loss:  3945225.5



![png](imagens/output_22_195.png)


    Total loss:  3871609.5



![png](imagens/output_22_197.png)


    Total loss:  3799876.75



![png](imagens/output_22_199.png)


## Visualizando a Imagem Final


```python
# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))
```




    <matplotlib.image.AxesImage at 0x7f26c4e17470>




![png](imagens/output_24_1.png)


# Mini-projeto

Não se desesperem! Este mini-projeto será apenas para diversão. Teste diferentes imagens de estilo e de conteúdo, altere os parâmetros que vimos no decorrer deste notebook e tente gerar resultados legais.

Uma sugestão: aumente as dimensões da imagem que será processada (na função `load_image`) e tente gerar alguma imagem estilizada de resolução HD ou até mesmo 4K.
