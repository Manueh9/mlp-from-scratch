# 02 · Arquitectura MLP y forward pass

> Anterior: [01 · Carga y exploración de datos](01-carga-y-exploracion-de-datos.md) · Siguiente: *(pendiente)*
> Lab que lo acompaña: [labs/02-arquitectura-mlp](../labs/02-arquitectura-mlp/)

Con los datos ya en formato tensor, el siguiente paso es definir la red que va a aprender a
clasificarlos: un **MLP** (Multi-Layer Perceptron, perceptrón multicapa). Es la red neuronal más
simple: una sucesión de capas totalmente conectadas, sin memoria ni estructura espacial (a
diferencia de una CNN o una RNN).

## 1. De imagen a vector: `Flatten`

Una imagen de Fashion-MNIST tiene forma `[1, 28, 28]` (canal, alto, ancho). Un MLP no entiende esa
estructura 2D: necesita un vector plano. `nn.Flatten()` convierte `[1, 28, 28]` en un vector de
`784` valores (`28 × 28 = 784`), uno por píxel.

```python
self.flatten = nn.Flatten()   # [1, 28, 28] → [784]
```

## 2. Capas lineales (`nn.Linear`)

Una capa `nn.Linear(entrada, salida)` aplica una transformación afín: `y = xW + b`. Cada neurona
de salida es una combinación lineal de todas las entradas, más un sesgo (`bias`). Es la pieza
básica de cómputo de toda la red.

```python
nn.Linear(784, 128)   # 784 entradas (píxeles) → 128 neuronas de salida
```

> Analogía (C#): pensar en cada capa como una función matemática pura que recibe un array y
> devuelve otro de distinto tamaño — no guarda estado entre llamadas, solo sus pesos (`W`, `b`), que
> son los que se van ajustando al entrenar.

## 3. Por qué hace falta no-linealidad: ReLU

Si se encadenan solo capas lineales, el resultado sigue siendo una transformación lineal (una
combinación lineal de combinaciones lineales es, al final, otra combinación lineal). Eso limita
muchísimo lo que la red puede aprender: nunca podría separar clases que no sean linealmente
separables. La solución es intercalar una **función de activación no lineal** entre capas. La más
simple y habitual es **ReLU** (Rectified Linear Unit): `ReLU(x) = max(0, x)`.

```python
nn.Linear(784, 128), nn.ReLU(),
nn.Linear(128, 64),  nn.ReLU(),
nn.Linear(64, 10)   # capa de salida: sin activación, son logits
```

La capa de salida no lleva ReLU porque sus valores (**logits**) representan una puntuación por
clase, no una activación intermedia — más adelante, una función de pérdida como
`CrossEntropyLoss` los convierte en probabilidades internamente.

## 4. Definir el modelo con `nn.Module`

En PyTorch, un modelo se define como una clase que hereda de `nn.Module`, con dos partes:

- `__init__`: declara las capas que va a usar.
- `forward`: define cómo fluyen los datos por esas capas (el **forward pass**).

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)
```

`nn.Sequential` encadena capas en orden sin tener que escribir el paso a paso a mano — equivalente
a llamar cada capa una tras otra dentro de `forward`.

## 5. El forward pass y los parámetros entrenables

El **forward pass** es, simplemente, pasar una entrada por la red y obtener una salida. Con
`torch.no_grad()` se hace sin construir el grafo de gradientes (no hace falta todavía, porque no se
va a entrenar en este paso, solo comprobar que la arquitectura funciona).

Cada `nn.Linear(entrada, salida)` tiene `entrada × salida` pesos más `salida` sesgos. Sumando las
tres capas de este MLP salen los parámetros entrenables totales de la red.

## 6. Mi arquitectura real

```
MLP(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (network): Sequential(
    (0): Linear(in_features=784, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ReLU()
    (4): Linear(in_features=64, out_features=10, bias=True)
  )
)
```

| Dato | Valor |
|---|---|
| Parámetros entrenables totales | 109.386 |
| Forma de entrada (batch=1) | `[1, 1, 28, 28]` |
| Forma de salida (logits) | `torch.Size([1, 10])` |

El forward pass de prueba confirma que la red acepta una imagen real del dataset y devuelve 10
logits, uno por clase — sin haber entrenado nada todavía, los valores no significan nada (son casi
aleatorios, fruto de la inicialización de pesos).

## Lo que falta (siguiente sesión)

Esto define la arquitectura, pero una red sin entrenar no sirve de nada. Queda pendiente:

- **Función de pérdida** (`CrossEntropyLoss`) para medir qué tan mal predice la red.
- **Backpropagation** para calcular cómo cambiar cada peso.
- **Optimizador** (SGD/Adam) que aplique esos cambios.
- **Bucle de entrenamiento** por épocas, y evaluación de accuracy en el set de test.
