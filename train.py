import os
from dotenv import load_dotenv

load_dotenv()

hf_hub_cache = os.getenv("HF_HUB_CACHE")
hf_ds_cache = os.getenv("HF_DATASETS_CACHE")
hf_home = os.getenv("HF_HOME")

print(f"{hf_hub_cache = }. {hf_ds_cache = }, {hf_home = }")

if hf_home is not None:
    os.environ['HF_HOME'] = hf_home
    print('Using HF_HOME')
else:
    if hf_hub_cache is not None:
        os.environ['HF_HUB_CACHE'] = hf_hub_cache
        print('Using HF_HUB_CACHE')

    if hf_ds_cache is not None:
        os.environ['HF_DATASETS_CACHE'] = hf_ds_cache
        print('Using HF_DATASETS_CACHE')

# print(f"{os.environ = }")

# PyTorch
import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
# For dislaying images
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
# Loading dataset
from datasets import load_dataset
# Transformers
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
# Matrix operations
import numpy as np
# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main(headless: bool = True):
    trainds, testds = load_dataset("cifar10", split=["train[:5000]","test[:1000]"])
    splits = trainds.train_test_split(test_size=0.1)
    trainds = splits['train']
    valds = splits['test']
    print(trainds, valds, testds)
    print(trainds.features, trainds.num_rows, trainds[0])

    # build dicts for str to int and vice versa
    itos = dict((k,v) for k,v in enumerate(trainds.features['label'].names))
    stoi = dict((v,k) for k,v in enumerate(trainds.features['label'].names))
    print(itos)

    # view an image
    index = 0
    img, lab = trainds[index]['img'], itos[trainds[index]['label']]
    print(lab)
    if not headless:
        plt.imshow(img)
        plt.show()

    # model_name = "google/vit-base-patch16-224"
    # TODO: need program args - need to configure HF TOKEN
    # model_name = "Dinov2WithRegistersForImageClassification"
    model_name = "apple/aimv2-large-patch14-224"
    processor = ViTImageProcessor.from_pretrained(model_name) 

    mu, sigma = processor.image_mean, processor.image_std #get default mu,sigma
    size = processor.size
    # for aimv2
    if 'height' not in size:
        size['height'] = size['shortest_edge']
    print(f"process info: {mu = }, {sigma = }, {size = }")

    norm = Normalize(mean=mu, std=sigma) #normalize image pixels range to [-1,1]

    # resize 3x32x32 to 3x224x224 -> convert to Pytorch tensor -> normalize
    _transf = Compose([
        Resize(size['height']),
        ToTensor(),
        norm
    ]) 

    # apply transforms to PIL Image and store it to 'pixels' key
    def transf(arg):
        arg['pixels'] = [_transf(image.convert('RGB')) for image in arg['img']]
        return arg
    
    # apply transforms to each ds
    trainds.set_transform(transf)
    valds.set_transform(transf)
    testds.set_transform(transf)

    if not headless:
        idx = 0
        ex = trainds[idx]['pixels']
        # TODO: does not work for aimv2 (imagenet mu and sigma) - will not look as expected
        ex = (ex+1)/2 #imshow requires image pixels to be in the range [0,1]
        exi = ToPILImage()(ex)
        plt.imshow(exi)
        plt.show()

    # recreate as ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(model_name)
    print(model.classifier)

    # recreate targeting the "10" cifar-10 classes
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True, id2label=itos, label2id=stoi)
    print(model.classifier)

    # trainer
    args = TrainingArguments(
        f"train-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )

    def collate_fn(examples):
        pixels = torch.stack([example["pixels"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixels, "labels": labels}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=accuracy_score(predictions, labels))
    
    trainer = Trainer(
        model,
        args, 
        train_dataset=trainds,
        eval_dataset=valds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    train_output = trainer.train()
    print(train_output)

    outputs = trainer.predict(testds)
    print(outputs.metrics)

if __name__ == '__main__':
    # TODO: add to cl args
    main(headless=False)