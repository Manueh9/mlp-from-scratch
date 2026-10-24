# Lab 02 · Arquitectura MLP y forward pass

> Anterior: [lab 01 · Carga y exploración de Fashion-MNIST](../01-carga-fashion-mnist/)
> Teoría: [apuntes/02-arquitectura-mlp-forward-pass.md](../../apuntes/02-arquitectura-mlp-forward-pass.md)

## Objetivo

Definir la arquitectura de un MLP con `nn.Module` (capas lineales + ReLU) para clasificar
Fashion-MNIST, codificar las etiquetas en one-hot, y comprobar con un forward pass de prueba que
la red acepta una imagen real y devuelve logits con la forma esperada.

## Preparación

```bash
cd ../..                       # raíz del repo
source venv/bin/activate
pip install -r requirements.txt
jupyter lab labs/02-arquitectura-mlp/02_arquitectura_mlp.ipynb
```

## Resultado real

```
Imágenes de entrenamiento: 60000
Imágenes de test: 10000
Forma de la imagen: torch.Size([1, 28, 28])
Valor mínimo: 0.000
Valor máximo: 1.000

Etiqueta original: 7
One-hot: tensor([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])

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

Total de parámetros entrenables: 109,386

Forma de la salida de la red: torch.Size([1, 10])
```

## Qué queda pendiente a propósito

- **Función de pérdida** (`CrossEntropyLoss`): medir qué tan mal predice la red sin entrenar.
- **Backpropagation y optimizador**: calcular y aplicar el gradiente de cada peso.
- **Bucle de entrenamiento** por épocas, con evaluación de accuracy en el set de test.
- Todo esto, de momento, deliberadamente fuera de este lab — es el siguiente paso del roadmap.

---

⬅️ Anterior: [01 · Carga Fashion-MNIST](../01-carga-fashion-mnist/) | ➡️ Siguiente: *(pendiente)* | 🏠 [Índice](../../README.md)
