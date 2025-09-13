import os
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from main import neural_style_transfer

@st.cache_data
def prepare_imgs(content_im, style_im, RGB=False):
    """ Return scaled RGB images as numpy array of type np.uint8 """    
    # check sizes in order to avoid huge computation times:
    h,w,c = content_im.shape
    ratio = 1.
    if h > 512:
        ratio = 512./h
    if (w > 512) and (w>h):
        ratio = 512./w
    content_im = cv2.resize(content_im, dsize=None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_CUBIC)        
    # reshape style_im to match the content_im shape 
    # (method followed in Gatys et al. paper):
    style_im = cv2.resize(style_im, content_im.shape[1::-1], cv2.INTER_CUBIC)
    
    # pass from BGR (OpenCV) to RGB:
    if not RGB:
        content_im = cv2.cvtColor(content_im, cv2.COLOR_BGR2RGB)
        style_im   = cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB)
    
    return content_im, style_im

    
def print_info_NST():
    st.markdown("""
                Le Transfert de style d'image (*Neural Style Transfer* abregé en **NST**) est une technique combinant l'apprentissage 
                profond et la vision par ordinateur permettant de générer une image en combinant le contenu et le style de deux images différentes. 
                Regardons un exemple (dans la colonne de gauche, en haut et en bas, se trouvent respectivement 
                le *contenu* et le *style*) :
                """)

    # Show exemplar images:
    root_content = os.path.join('data', 'content-images', 'lion.jpg')
    root_style = os.path.join('data', 'style-images', 'wave.jpg')
    
    content_im = cv2.imread(root_content)
    style_im = cv2.imread(root_style)    
    im_cs, im_ss = prepare_imgs(content_im, style_im)    
    im_rs = cv2.imread(os.path.join('data', 'output-images', 'clion_swave_sample.jpg'))
    
    col1, col2 = st.columns([1,2.04])
    col1.header("Base")
    col1.image(im_cs, use_container_width=True)
    col1.image(im_ss, use_container_width=True)
    col2.header("Résultat")
    col2.image(im_rs, use_container_width=True, channels="BGR")
    
    # Information about the parameters:
    st.markdown("""
            ## Paramètres dans la barre latérale gauche
            ### Poids de la fonction de perte (*lambdas*)
            """)
    st.latex(r"""
            \mathcal{L}(\lambda_{\text{content}}, 
            \lambda_{\text{style}}, \lambda_{\text{variation}}) =
            \lambda_{\text{content}}\mathcal{L}_{\text{content}} +
            \lambda_{\text{style}}\mathcal{L}_{\text{style}} +
            \lambda_{\text{variation}}\mathcal{L}_{\text{variation}}
            """)
    st.markdown("""
            - **Content** : Une valeur plus élevée augmente l'influence de l'image de contenu,
            - **Style** : Une valeur plus élevée augmente l'influence de l'image de style, 
            - **Variation** : Une valeur plus élevée rend l'image générée plus lisse.
            """)
    st.markdown("""
            ### Nombre d'itérations  
            Sa valeur détermine la durée du processus d'optimisation.  
            Un nombre plus élevé rendra l'optimisation plus longue.  
            Ainsi, si l'image semble non optimisée, essayez d'augmenter ce nombre  
            (ou d'ajuster les poids de la fonction de perte).
            """)
    st.markdown("""
           ### Enregistrer le résultat  
            Si cette option est cochée, l'image sera enregistrée sur l'ordinateur une fois l'optimisation terminée  
            (dans le même dossier où se trouve le fichier *app.py* de ce projet).
            """)

if __name__ == "__main__":
    
    # app title and sidebar:
    st.title("Bienvenue sur Artify - Transforme tes images en chef-d'œuvre ")

    # Set parameters to tune at the sidebar:
    st.sidebar.title('Paramètres')
    # Weights of the loss function
    st.sidebar.subheader('Poids sur la Fonction de perte')
    step = 1e-1
    cweight = st.sidebar.number_input("Contenu", value=1e-3, step=step, format="%.5f")
    sweight = st.sidebar.number_input("Style", value=1e-1, step=step, format="%.5f")
    vweight = st.sidebar.number_input("Variation", value=0.0, step=step, format="%.5f")
    
    # Number of iterations
    st.sidebar.subheader('Nombre d\'itérations')
    niter = st.sidebar.number_input('Iterations', min_value=1, max_value=1000, value=20, step=1)
    
    # Save or not the image
    st.sidebar.subheader('Enregistrer ou non l\'image stylisée')
    save_flag = st.sidebar.checkbox('Enregistrer l\'image résultante')
    
    # Page d'exécution unique
    st.markdown("### Téléchargez la paire d'images à utiliser :")        
    col1, col2 = st.columns(2)
    im_types = ["png", "jpg", "jpeg"]
    
    with col1:
        file_c = st.file_uploader("Choisissez l'image de CONTENU :", type=im_types)
        imc_ph = st.empty()            
    with col2: 
        file_s = st.file_uploader("Choisissez l'image de STYLE :", type=im_types)
        ims_ph = st.empty()
    
    if all([file_s, file_c]):
        im_c = np.array(Image.open(file_c))
        im_s = np.array(Image.open(file_s))
        im_c, im_s = prepare_imgs(im_c, im_s, RGB=True)
        
        imc_ph.image(im_c, use_container_width=True)
        ims_ph.image(im_s, use_container_width=True) 
    
    st.markdown("### Démarrez la génération de l'image en appuyant sur commencer :")
    
    start_flag = st.button("Commencer", help="Start the optimization process")
    bt_ph = st.empty()
    
    if start_flag:
        if not all([file_s, file_c]):
            bt_ph.markdown("Vous devez **télécharger des images** d'abord ! :)")
        else:
            bt_ph.markdown("Optimisation ...")
            
            progress = st.progress(0.)
            res_im_ph = st.empty()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            parent_dir = os.path.dirname(__file__)
            out_img_path = os.path.join(parent_dir, "app_stylized_image.jpg")
            
            cfg = {
                'output_img_path': out_img_path,
                'style_img': im_s,
                'content_img': im_c,
                'content_weight': cweight,
                'style_weight': sweight,
                'tv_weight': vweight,
                'optimizer': 'lbfgs',
                'model': 'vgg19',
                'init_metod': 'random',
                'running_app': True,
                'res_im_ph': res_im_ph,
                'save_flag': save_flag,
                'st_bar': progress,
                'niter': niter
            }
            
            result_im = neural_style_transfer(cfg, device)
            bt_ph.markdown("Voici l'image **stylisée** résultante !")
