# 🇬🇧 multilayer_perceptron_42

**If you enjoy this project, feel free to give it a star ⭐️!**

## Introduction

The goal of this project is to build a multi-layer perceptron (MLP) from scratch to create a model that predicts the outcome of a breast cancer diagnosis.
---

## Multilayer perceptron
The multilayer perceptron is a feedforward network (meaning that the data
flows from the input layer to the output layer) defined by the presence of one or more
hidden layers, as well as an interconnection of all the neurons of one layer to the next.

<img width="697" height="431" alt="Screenshot from 2025-09-25 15-44-50" src="https://github.com/user-attachments/assets/ee5f8d55-e87e-4cd5-a206-e1589700a127" />

The diagram above represents a network containing 4 dense layers (also called fully
connected layers). Its inputs consist of 4 neurons and its output consists of 2 (perfect
for binary classification). The weights of one layer to the next are represented by
two-dimensional matrices noted Wlj lj+1 . The matrix Wl0l1 is of size (3, 4) for example, as
it contains the weights of the connections between layer l0 and layer l1.
The bias is often represented as a special neuron which has no inputs and an output
always equal to 1. Like a perceptron, it is connected to all the neurons of the following
layer (the bias neurons are noted blj on the diagram above). The bias is generally useful
as it allows one to “control the behavior” of a layer.

## Perceptron
The perceptron is the type of neuron that the multilayer perceptron is composed
of. It is defined by the presence of one or more input connections, an activation
function, and a single output. Each connection contains a weight (also called a
parameter) which is learned during the training phase.

<img width="595" height="266" alt="Screenshot from 2025-09-25 15-46-18" src="https://github.com/user-attachments/assets/c4947a3f-ac26-4bca-962f-dc2eb0a1bebc" />

Two steps are necessary to get the output of a neuron. The first one consists in
computing the weighted sum of the outputs of the previous layer with the weights of the
input connections of the neuron, which gives

<img width="323" height="46" alt="Screenshot from 2025-09-25 15-47-18" src="https://github.com/user-attachments/assets/6f236407-0511-44bc-8a65-0e2bcdca3599" />

The second step consists in applying an activation function on this weighted sum.
The output of this function is the output of the perceptron and can be understood as
the threshold above which the neuron is activated (activation functions can take many
forms, and you are free to choose whichever one you want depending on the model to
train. Here are some of the most frequently used ones to give you an idea: sigmoid,
hyperbolic tangent, and rectified linear unit).

## Usage

1. Clone the repository
```bash
git clone https://github.com/MatLBS/ft_linear_regression_42.git
cd ft_linear_regression_42
```
2. Create a virtual environment and install dependencies
```python
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Dataset preview

```bash
python srcs/describe.py dataset/data.csv
```

<img width="1010" height="535" alt="Screenshot from 2025-09-25 15-53-31" src="https://github.com/user-attachments/assets/884e4570-3c56-47b8-901f-2496bab916ef" />

```bash
python srcs/histogram.py dataset/data.csv
```

<img width="1420" height="1097" alt="Screenshot from 2025-09-25 15-54-11" src="https://github.com/user-attachments/assets/2a5a0f35-0a2c-4f57-98ee-bb93013937ea" />

---

# Training & Prediction

## Training of an MLP model

```bash
python srcs/mlp_training.py --file dataset/data.csv --layer 24 24 --epochs 2000 --batch_size 32 --learning_rate 0.001
```

<img width="689" height="559" alt="Screenshot from 2025-09-25 15-57-08" src="https://github.com/user-attachments/assets/97223140-3389-44db-8bd8-1aa64586cf8f" />

<img width="1220" height="597" alt="Screenshot from 2025-09-25 15-56-34" src="https://github.com/user-attachments/assets/abb43e4e-8184-40ed-bb19-93fbc70953dc" />

## Prediction and evaluation
You can predict the test data and evaluate your model by running below command.
```bash
python srcs/mlp_predict.py mlp_weights.npz mlp_topology.json dataset/data.csv
```

<img width="258" height="18" alt="Screenshot from 2025-09-25 15-59-10" src="https://github.com/user-attachments/assets/b6308c2c-1e2b-46d7-af40-34e26672cd5f" />

# 🇫🇷 multilayer_perceptron_42

**Si tu apprécies ce projet, n’hésite pas à lui donner une étoile ⭐️ !**

## Introduction

L’objectif de ce projet est de construire un perceptron multicouche (MLP) “from scratch” pour créer un modèle capable de prédire l’issue d’un diagnostic du cancer du sein.

---

## Perceptron multicouche (MLP)

Le perceptron multicouche est un réseau “feedforward” (ce qui signifie que les données circulent de la couche d’entrée à la couche de sortie) défini par la présence d’une ou plusieurs couches cachées, ainsi que par l’interconnexion de tous les neurones d’une couche à la suivante.

<img width="697" height="431" alt="Screenshot from 2025-09-25 15-44-50" src="https://github.com/user-attachments/assets/ee5f8d55-e87e-4cd5-a206-e1589700a127" />

Le schéma ci-dessus représente un réseau contenant 4 couches denses (ou “fully connected”). Ses entrées sont constituées de 4 neurones et sa sortie de 2 (parfait pour une classification binaire). Les poids d’une couche à l’autre sont représentés par des matrices notées Wlj lj+1. Par exemple, la matrice Wl0l1 est de taille (3, 4), car elle contient les poids des connexions entre la couche l0 et la couche l1.

Le biais est souvent représenté comme un neurone spécial qui n’a pas d’entrée et une sortie toujours égale à 1. Comme un perceptron, il est connecté à tous les neurones de la couche suivante (les neurones de biais sont notés blj sur le schéma ci-dessus). Le biais est généralement utile car il permet de “contrôler le comportement” d’une couche.

## Perceptron

Le perceptron est le type de neurone qui compose le perceptron multicouche. Il est défini par la présence d’une ou plusieurs connexions d’entrée, d’une fonction d’activation et d’une seule sortie. Chaque connexion possède un poids (aussi appelé paramètre) qui est appris lors de la phase d’entraînement.

<img width="595" height="266" alt="Screenshot from 2025-09-25 15-46-18" src="https://github.com/user-attachments/assets/c4947a3f-ac26-4bca-962f-dc2eb0a1bebc" />

Deux étapes sont nécessaires pour obtenir la sortie d’un neurone. La première consiste à calculer la somme pondérée des sorties de la couche précédente avec les poids des connexions d’entrée du neurone, ce qui donne :

<img width="323" height="46" alt="Screenshot from 2025-09-25 15-47-18" src="https://github.com/user-attachments/assets/6f236407-0511-44bc-8a65-0e2bcdca3599" />

La seconde étape consiste à appliquer une fonction d’activation sur cette somme pondérée. La sortie de cette fonction est la sortie du perceptron et peut être comprise comme le seuil au-delà duquel le neurone s’active (les fonctions d’activation peuvent prendre différentes formes, et tu es libre de choisir celle qui te convient selon le modèle à entraîner. Voici quelques-unes des plus utilisées : sigmoïde, tangente hyperbolique, et rectified linear unit).

## Utilisation

1. Clone le dépôt
```bash
git clone https://github.com/MatLBS/ft_linear_regression_42.git
cd ft_linear_regression_42
```
2. Crée un environnement virtuel et installe les dépendances
```python
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Aperçu du dataset

```bash
python srcs/describe.py dataset/data.csv
```

<img width="1010" height="535" alt="Screenshot from 2025-09-25 15-53-31" src="https://github.com/user-attachments/assets/884e4570-3c56-47b8-901f-2496bab916ef" />

```bash
python srcs/histogram.py dataset/data.csv
```

<img width="1420" height="1097" alt="Screenshot from 2025-09-25 15-54-11" src="https://github.com/user-attachments/assets/2a5a0f35-0a2c-4f57-98ee-bb93013937ea" />

---

# Entraînement & Prédiction

## Entraînement d’un modèle MLP

```bash
python srcs/mlp_training.py --file dataset/data.csv --layer 24 24 --epochs 2000 --batch_size 32 --learning_rate 0.001
```

<img width="689" height="559" alt="Screenshot from 2025-09-25 15-57-08" src="https://github.com/user-attachments/assets/97223140-3389-44db-8bd8-1aa64586cf8f" />

<img width="1220" height="597" alt="Screenshot from 2025-09-25 15-56-34" src="https://github.com/user-attachments/assets/abb43e4e-8184-40ed-bb19-93fbc70953dc" />

## Prédiction et évaluation

Tu peux prédire les données de test et évaluer ton modèle en lançant la commande suivante :
```bash
python srcs/mlp_predict.py mlp_weights.npz mlp_topology.json dataset/data.csv
```

<img width="258" height="18" alt="Screenshot from 2025-09-25 15-59-10" src="https://github.com/user-attachments/assets/b6308c2c-1e2b-46d7-af40-34e26672cd5f" />
