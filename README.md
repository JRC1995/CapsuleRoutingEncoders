### Text Classification using Capsule Routing

[Project Report link](https://github.com/JRC1995/CapsuleRoutingEncoders/blob/main/CS_521_Project_report.pdf)

#### Abstract:
In this work, we study and compare multiple capsule routing algo-rithms for text classification including dynamic routing, Heinsenrouting, and capsule-routing inspired attention-based sentence en-coding techniques like dynamic self-attention. Further, similar tosome works in computer vision, we do an ablation test of the cap-sule network where we remove the routing algorithm itself. Weanalyze the theoretical connection between attention and capsulerouting, and contrast the two ways of normalizing the routingweights. Finally, we present a new way to do capsule routing, orrather iterative refinement, using a richer attention function tomeasure agreement among output and input capsules and withhighway connections in between iterations

Code from the final project of the course CS521: Stastitical Natural Language Processing. The code was created around 2020 Quarter 1. 

### Embedding Setup
* Download [google word2vec 300 dimension embeddings](https://drive.google.com/drive/folders/15fRWD3bWSb_2VRm0Hy1OcMQ_UKqsJyex?usp=sharing)
* Put the downloaded embedding file in `embeddings/word2vec`

### Preprocessing
* Go to `process/` after downloading embeddings.
* Run both the python files in `process/'.

#### To train a model on AAPD dataset, run:
`python train_AAP.py --model=[INSERT MODEL NAME]`

#### To train a model on Reuters dataset run:
`python train_Reuters.py --model=[INSERT MODEL NAME]`

#### Similarly, for running test AAPD dataset (to evaluate a model):
`python test_AAP.py --model=[INSERT MODEL NAME]`

#### Similarly, for running test Reuters dataset (to evaluate a model):
`python test_Reuters.py--model=[INSERT MODEL NAME]`

#### Possible candidates for mdoel are:
* CNN
* CNN_att
* CNN_capsule
* CNN_heinsen_capsule
* CNN_DSA
* CNN_DSA_global
* CNN_PCaps
* CNN_custom
* CNN_custom_alpha_ablation
* CNN_custom_global
* CNN_custom2

#### Descriptions:
* CNN: Convolutional Neural Network (similar to Kim et al. version)
* CNN_att (Convolutional Neural Network + attention)
* CNN_capsule (CNN + Dynamic Capsule Routing)
* CNN_heinsen_capsule (CNN + heinsen Capsule Routing)
* CNN_DSA (CNN + Dynamic Self Attention)
* CNN_DSA_global (CNN + Dynamic Self Attention for Sentence encoding)
* CNN_PCaps (CNN + “non-routing” mechanism inspired from PCapsNet)
* CNN_custom (CNN + “new routing” + “reverse normalization”)
* CNN_custom_alpha_ablation (CNN + “new routing” + “reverse normalization”- Highway connection)
* CNN_custom_global (CNN + “new routing” + “reverse normalization” for sentence encoding)
* CNN_custom2 (CNN + “new routing”)

#### Project navigation guide:
* Hyperparameters used for each model in AAPD are in configs/AAPD_args.py
* Hyperparameters used for each model in Reuters are in configs/Reuters_args.py
* process/ directory have preprocessing codes for AAPD and Reuters. (but the data is already
* preprocessed within processed_data/. So not need to process further).
* data/ directory have the data in use
* Preprocessing requires word2vec embeddings in embeddings/word2vec directory (download google word2vec 300 dimension embeddings put it in the directory and run bin2txt.py) (word2vec downloadable from [here](https://drive.google.com/drive/folders/15fRWD3bWSb_2VRm0Hy1OcMQ_UKqsJyex?usp=sharing))
models/modules have the routing codes
* models/ have all the model codes
* utils/ have some evaluation and other utilities code.
* saved_params/ will have the saved model parameters after training.

### Credits:
* heinsen routing is based on [this](https://github.com/glassroom/heinsen_routing/blob/master/heinsen_routing.py)
* The data were downloaded from: https://github.com/castorini/hedwig (word2vec bin file, and bin2txt.py too are available through a link in their repository).
* [Hedwig library](https://github.com/castorini/hedwig) was also referenced for initial CNN implementations.
