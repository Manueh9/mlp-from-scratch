# 🧠 mlp-from-scratch — Construyendo una red neuronal desde cero

---

## Cómo usar este repo

- **`apuntes/`** → la teoría, numerada y en orden. Empieza por el 01.
- **`labs/`** → un ejercicio práctico por sesión, cada uno con su notebook y su README (objetivo,
  pasos, resultado y qué queda pendiente a propósito).

Cada lab es reproducible: clona, crea el entorno virtual, instala `requirements.txt` y sigue su
README.

---

## Ruta de aprendizaje

El objetivo no es solo usar `nn.Linear` y listo, sino entender qué hace cada pieza de un MLP por
dentro: de los datos en crudo a una red que aprende, pasando por el forward pass, la función de
pérdida y la backpropagation.

| # | Tema | De qué va | Estado |
|---|---|---|---|
| 1 | Carga y exploración de datos | Fashion-MNIST, normalización a tensores, one-hot encoding | ✅ Hecho |
| 2 | Arquitectura MLP y forward pass | `nn.Module`, capas lineales, ReLU, conteo de parámetros | ✅ Hecho |
| 3 | Función de pérdida | `CrossEntropyLoss`: medir qué tan mal predice la red | 🔜 Pendiente |
| 4 | Backpropagation | Cómo se calcula el gradiente de cada peso | 🔜 Pendiente |
| 5 | Entrenamiento | Optimizador (SGD/Adam) y bucle de entrenamiento por épocas | 🔜 Pendiente |
| 6 | Evaluación | Accuracy en test, curvas de loss, overfitting | 🔜 Pendiente |

---

## Índice

### Apuntes

| # | Apunte | De qué va |
|---|---|---|
| 01 | [Carga y exploración de datos](apuntes/01-carga-y-exploracion-de-datos.md) | Fashion-MNIST, normalización, one-hot encoding |
| 02 | [Arquitectura MLP y forward pass](apuntes/02-arquitectura-mlp-forward-pass.md) | `nn.Module`, `nn.Linear`, ReLU, forward pass, parámetros |

### Labs

| # | Lab | Qué se practica | Estado |
|---|---|---|---|
| 01 | [Carga de Fashion-MNIST](labs/01-carga-fashion-mnist/) | Descargar, normalizar y visualizar el dataset | ✅ Hecho |
| 02 | [Arquitectura MLP](labs/02-arquitectura-mlp/) | Definir el MLP, one-hot encoding, forward pass de prueba | ✅ Hecho |

---

## Roadmap (orden previsto)

De los datos a una red que realmente aprende, sin saltarse pasos:

1. Carga y exploración de datos (Fashion-MNIST)
2. **Arquitectura MLP y forward pass** ← estoy aquí
3. Función de pérdida (`CrossEntropyLoss`)
4. Backpropagation
5. Entrenamiento (optimizador + bucle por épocas)
6. **Evaluación y accuracy en test** (la meta del track)

Un tema por sesión, sin prisa.

---

## Entorno

- Python 3.14 en entorno virtual (`venv/`, no se versiona — ver `.gitignore`).
- Dependencias en `requirements.txt`: `torch`, `torchvision`, `numpy`, `matplotlib`, `jupyterlab`.
- Todo el código vive en notebooks Jupyter, uno por lab.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

---

## Otros repos de esta serie

Este repo es uno de cuatro, montados con la misma estructura de "asignatura":

| Repo | De qué va |
|---|---|
| [`cloud-devops-lab`](https://github.com/Manueh9/cloud-devops-lab) | Docker, Kubernetes, Terraform, CI/CD, Azure, observabilidad |
| [`security-lab`](https://github.com/Manueh9/security-lab) | Seguridad web: PortSwigger Web Security Academy y OWASP Juice Shop |
| [`ml-lab`](https://github.com/Manueh9/ml-lab) | IA/ML con Python: datos, modelos y llevarlos a producción |
| `mlp-from-scratch` | Este: construir un MLP desde cero con PyTorch, entendiendo cada pieza |

---

## Licencia

[MIT](LICENSE)
