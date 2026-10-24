# 01 · Carga y exploración de datos: Fashion-MNIST

> Anterior: — · Siguiente: [02 · Arquitectura MLP y forward pass](02-arquitectura-mlp-forward-pass.md)
> Lab que lo acompaña: [labs/01-carga-fashion-mnist](../labs/01-carga-fashion-mnist/)

Antes de construir ninguna red hace falta tener datos en un formato que la red pueda consumir.
**Fashion-MNIST** es el dataset de partida: 70.000 imágenes en escala de grises de 28×28 píxeles,
repartidas en 10 categorías de ropa (camiseta, pantalón, jersey, vestido, abrigo, sandalia, camisa,
zapatilla, bolso, bota). Es el sustituto "moderno" del MNIST de dígitos: mismo tamaño y formato,
pero un problema algo menos trivial.

## 1. De imagen a tensor

Un modelo no entiende archivos de imagen, entiende tensores (arrays multidimensionales de
números). `torchvision.transforms.ToTensor()` hace dos cosas a la vez:

- Convierte la imagen a un tensor de PyTorch con forma `[canales, alto, ancho]` — para
  Fashion-MNIST, `[1, 28, 28]` (1 canal porque es escala de grises).
- **Normaliza** los valores de píxel de `[0, 255]` a `[0.0, 1.0]`.

```python
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
```

> Analogía (C#): es como pasar un `byte[]` de píxeles crudos a un `float[]` ya escalado — el
> modelo trabaja siempre en el mismo rango, independientemente de cómo venga codificada la imagen
> original.

## 2. Por qué se normaliza

Los algoritmos de optimización (descenso de gradiente) convergen mejor cuando las entradas están
en un rango pequeño y consistente (aquí, `[0, 1]`). Si se dejaran los valores en `[0, 255]`, los
gradientes iniciales serían mucho más grandes y entrenar sería más inestable. Es una preparación de
datos, igual que la limpieza de nulos o el escalado de columnas numéricas en un problema tabular.

## 3. Etiquetas y one-hot encoding

Cada imagen viene con una etiqueta: un entero de `0` a `9` que identifica la categoría. Para
algunas funciones de pérdida (y para razonar sobre "probabilidad por clase") conviene representar
esa etiqueta como un vector de 10 posiciones, con un `1` en la posición de la clase correcta y `0`
en el resto: **one-hot encoding**.

```python
def to_one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes=num_classes).float()
```

Por ejemplo, la etiqueta `7` ("Sneaker"/zapatilla) se convierte en:

```
tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
```

> Analogía (C#): es como pasar de un `enum` (`Sneaker = 7`) a un vector de `bool[]` donde solo la
> posición `7` es `true`. Pierdes el orden implícito del número (el 7 no es "más" que el 3), que es
> justo lo que se busca: las clases son categorías, no una escala.

## 4. Mi exploración real del dataset

| Dato | Valor |
|---|---|
| Imágenes de entrenamiento | 60.000 |
| Imágenes de test | 10.000 |
| Forma de una imagen | `torch.Size([1, 28, 28])` |
| Valor mínimo de píxel (tras `ToTensor()`) | 0.000 |
| Valor máximo de píxel (tras `ToTensor()`) | 1.000 |

La normalización funciona como se espera: todo el dataset queda en `[0, 1]` sin necesidad de
escalarlo a mano.
