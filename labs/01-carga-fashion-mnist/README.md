# Lab 01 · Carga y exploración de Fashion-MNIST

> Anterior: — (primer lab)
> Teoría: [apuntes/01-carga-y-exploracion-de-datos.md](../../apuntes/01-carga-y-exploracion-de-datos.md)

## Objetivo

Descargar el dataset Fashion-MNIST con `torchvision`, comprobar que las imágenes quedan
normalizadas a `[0, 1]`, y visualizar un primer vistazo de las muestras y sus etiquetas.

## Preparación

```bash
cd ../..                       # raíz del repo
source venv/bin/activate
pip install -r requirements.txt
jupyter lab labs/01-carga-fashion-mnist/01_cargar_datos.ipynb
```

La primera ejecución descarga Fashion-MNIST a `./data` (no se versiona, ver `.gitignore`).

## Resultado real

```
Total de imágenes de entrenamiento: 60000
Forma de una imagen: torch.Size([1, 28, 28])
```

Además, un grid de 2×5 con las 10 primeras imágenes de entrenamiento y su etiqueta numérica.

## Qué queda pendiente a propósito

- **One-hot encoding** de las etiquetas y **definición de la arquitectura**: es el lab siguiente.
- **Normalización avanzada** (media/desviación del dataset en vez de `[0,1]` fijo): no hace falta
  todavía, se deja para cuando el modelo lo necesite.

---

⬅️ Anterior: — | ➡️ Siguiente: [02 · Arquitectura MLP](../02-arquitectura-mlp/) | 🏠 [Índice](../../README.md)
