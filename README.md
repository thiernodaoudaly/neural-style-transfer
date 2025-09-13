# Transfert de Style Neuronal

## Description
Ce code propose une implémentation en **PyTorch** de l’article :  
"[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)",  
par **Leon A. Gatys et al.**  

L’application peut être exécutée localement soit via une application Streamlit (`app.py`), soit en modifiant les arguments d’entrée d’un **script Python** (`main.py`).  

## Bref descriptif
Le **Neural Style Transfer (NST)** est une technique d’apprentissage profond qui permet de générer une image combinant :  
- le **contenu** d’une image (*content image*),  
- avec le **style** d’une autre image (*style image*).  

## Installation avec Conda
Ces instructions supposent que [Anaconda](https://www.anaconda.com/products/individual) ou [Miniconda](https://docs.conda.io/en/latest/miniconda.html) soient installés sur votre machine.  

1. Ouvrez **Anaconda Prompt** et clonez ce dépôt à l’emplacement souhaité :  
   ```bash
   cd <votre_dossier>
   git clone https://github.com/thiernodaoudaly/neural-style-transfer-app
   cd neural-style-transfer-app
2. Créez l’environnement avec les dépendances fournies dans environment.yml :
   ```bash
   conda env create -f env.yml
3. Activez l’environnement :
   ```bash
   conda activate nst-env


