# Proposal: Zero-Shot Visual-language feature alignment via ViT-based Contrastive LearningProblem 

DONG, Yunao ydongbd@connect.ust.hk

LIU, Xiyue xliufp@connect.ust.hk

CHEN, Hongyu hchendu@connect.ust.hk


## Investigation & Interest:
Standard classifiers suffer from rigid, pre-set labels. We mitigate this by employing a pre-trained text encoder to match label semantics with image features, enhancing both transferability and generalization. Unlike traditional closed-set classifiers, our model will recognize unseen categories by measuring the similarity between image embeddings and text-derived category prompts. 
This is highly interesting because it shifts computer vision from "learning to label" to "learning to understand concepts," enabling a model trained on general image-caption pairs to generalize to specialized tasks like ImageNet classification without any category-specific training. 
## Data:  
MicrosoftCOCO, ImageNet, etc. 
### Example:
图片名: 000000397133.jpg (ID: 397133)
对应的 5 条描述是:
  1. A man is in a kitchen making pizzas.
  2. Man in apron standing on front of oven with pans and bakeware
  3. A baker is working in the kitchen rolling dough.
  4. A person standing by a stove in a kitchen.
  5. A table with pies being made and a person standing near a wall with pots and pans hanging on the wall.

图片名: 000000037777.jpg (ID: 37777)
对应的 5 条描述是:
  1. The dining table near the kitchen has a bowl of fruit on it.
  2. A small kitchen has various appliances and a table.
  3. The kitchen is clean and ready for us to see.
  4. A kitchen and dining area decorated in white.
  5. A kitchen that has a bowl of fruit on the table.

图片名: 000000252219.jpg (ID: 252219)
对应的 5 条描述是:
  1. a person with a shopping cart on a city street
  2. City dwellers walk by as a homeless man begs for cash.
  3. People walking past a homeless man begging on a city street
  4. a homeless man holding a cup and standing next to a shopping cart on a street
  5. People are walking on the street by a homeless person.

## Methodology:
Our methodology focuses on a dual-encoder architecture. We will use a Vision Transformer (ViT) as the image encoder and a pre-trained text-encoder. Critically, we will freeze the text encoder to preserve its linguistic knowledge while actively training the ViT image encoder and a learnable linear projection layer that maps both modalities into a shared d-dimensional latent space. 
## Evaluation:
We will evaluate the results both qualitatively and quantitatively. Qualitatively, we will visualize "similarity heatmaps" and retrieve the Top-k most likely classes for sample images to inspect if the model captures correct semantic attributes. Quantitatively, we will report Zero-shot Top-1 and Top-k Accuracy on the ImageNet validation set. We will also perform an ablation study comparing our trained ViT's performance against a standard supervised image classification model in similar scale to demonstrate the advantages of the open-vocabulary approach.

## environment: 

```bash
pip install -r requirements.txt

## How to use: 

demo: 

1. run: python make_demo_data.py (this will generate demo data)
2. run: python train.py --config configs/demo.yaml (trained model will be stored at checkpoints\demo)
3. run: python evaluate.py --config configs/demo.yaml --checkpoint checkpoints/demo/best.pt (for evaluation)

COCO: need to be downloaded first to data/coco, where the annotations should be in data/coco/annotations; images should be in data/coco/train2017 and data/coco/val2017