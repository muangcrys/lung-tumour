# Investigating Cross-Domain Transfer Learning for Lung Nodule Classification: From Videos to Medical Images

*Jetanat Sihanatkathakul*

[Dissertation](https://github.com/muangcrys/lung-tumour/releases/tag/draft)

---

This repository contains the implementations and the codebase for the dissertation with the above title.
This is a part of the MSc Artificial Intelligence degree at the University of Edinburgh. 

## Abstract
Transfer learning has become a standard approach for image models. However, applying it to 3-dimensional medical image models is difficult due to the scarcity of large-scale volumetric pretraining datasets. This dissertation investigates different pretraining strategies and datasets for lung nodule classification from low-dose computed tomography (LDCT) scans using the recently released LUNA25 dataset. Two different model families were studied: 3D convolutional neural networks (CNNs) and transformers. CNNs are represented by two residual neural network (ResNet) models, ResNet3D and MedicalNet. For transformers, we used the Video Vision Transformer (ViViT) architecture. We used ResNet3D and ViViT weights that were pretrained on video classification, while the MedicalNet weights were transferred from medical image segmentation. We also applied other transfer learning techniques commonly used by researchers: linear probing followed by full finetuning (LP-FT), and multistage transfer learning (MSTL) using LUNA16 as the intermediate dataset.

Generally, positive effects were observed with the video-pretrained models, and the best model overall was video-pretrained. We also observed that the best model, ResNet3D-18, performed significantly worse without pretraining ($p=0.003$). In contrast, medical-pretraining did not provide significant benefits to any of the models. Instead, we observed that one model performed worse with medical-pretraining ($p=0.009$). ViViT collapsed to predicting the most frequent class without pretraining. With video-pretraining, it performed significantly better ($p<0.001$), but its performance was not as good as the ResNets'. We conclude that video-pretraining helps ViViT, but it alone is not enough to bridge the gap between transformers and CNNs in a data-limited setting.

These results further suggest that the success of transfer learning depends on a multitude of factors besides the similarity between the source and target domains. One likely factor is the size of the pretraining dataset. With video datasets being orders of magnitude larger than typical medical datasets, video-pretraining is a competitive alternative to medical-pretraining.

---

## Organisation
```aiignore
./
├── checkpoints/            # Pretrained model checkpoints (video/medical-pretrained)
├── data/                   # Datasets, raw and processed
├── jobs/                   # Training and evaluating scripts for launching on the slurm environment
├── notebooks/              # Data exploration and analysis notebooks
├── output/                 # Experiment outputs and results
├── scripts/                # Utility bash scripts
│
├── src/
│   ├── evaluate/           # Model evaluation and inference
│   ├── luna_dataset/       # Dataset loading, preprocessing, transforming, and augmenting
│   ├── models/             # Model architectures + fresh model construction
│   ├── plotting/           # Training-related plotting (loss + metrics)
│   ├── pretrains/          # Pretrained model construction
│   ├── training/           # Training pipeline
│   ├── training_k_fold/    # K-fold training wrapper
│   └── utility/            # Utility functions
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Requirements
You will need a `Python 3.13.13` environment. Install the required packages by using the following command:
```bash
pip install -r requirements.txt
```

## Dataset
We used two datasets: LUNA25 and LUNA16. LUNA25 alone should be sufficient for reproducing the majority of the dissertation. The splits and the annotation data are provided for both datasets. However, the raw images are not included due to their sizes. Follow the instructions below to acquire the nodule images:

- **LUNA25**: Download the nodules from [The LUNA25 Challenge: Public Training and Development set - Imaging Data](https://zenodo.org/records/14223624). Specifically, you need to download two files: `luna25_nodule_blocks.zip.001` and `luna25_nodule_blocks.zip.002`. Extract them together and put them under `./data/LUNA/raw/image/`. The extracted files should be in `.npy` format.
- **LUNA16**: Download the nodules from [LUNA22-ISMI](https://zenodo.org/records/6559584). Download the `LIDC-IDRI_1176.zip` file and extract the nodules to the directory `./data/LUNA16/niigz/`. The extracted nodule files should be in `.nii.gz` format. The nodules must be processed to `.npy` like LUNA25 to be compatible with the pipeline. To do that, after putting the nifti image files in place, use the following command:
```bash
python -u src/luna16_process_images.py
```

## Pretrained Weights
We used pretrained weights from ResNet3D and MedicalNet. All of these weights must be present to reproduce any experiments with the pretrained model.

### ResNet3D Weights
Checkpoints for the ResNet3D models can be found in the [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch) repository. The checkpoints are stored in a [Google Drive](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M). We used the following:
- `r3d18_KM_200ep.pth`: ResNet3D-18
- `r3d34_KM_200ep.pth`: ResNet3D-34
- `r3d50_KMS_200ep.pth`: ResNet3D-50
Put all `.pth` files inside the `./checkpoints/resnet3d/` directory.

### MedicalNet Weights
Checkpoints for the MedicalNet model are found inside the [MedicalNet](https://github.com/Tencent/MedicalNet) repository. You can download the `zip` file containing the checkpoint [here](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view). Unzip the file, and put these files inside the `./checkpoints/medicalnet/` directory:
- `resnet_18_23dataset.pth`: MedicalNet-18
- `resnet_34_23dataset.pth`: MedicalNet-34
- `resnet_50_23dataset.pth`: MedicalNet-50

### ViViT Checkpoint
The code expects a local instance of ViViT weight, downloaded from Hugging Face Hub at `google/vivit-b-16x2-kinetics400`. You can run the following script to download it to the correct path:

```bash
python -u ./src/utility/downloads.py
```


## Reproduction
We mainly reported results from 4-fold cross-validation. The scripts are split into two parts: training and inference + evaluation. We launched job scripts on the School of Informatic's cluster using slurm. You can use these job script files as the base and modify them to suit your computing environment. These files are inside [`./jobs/k_fold/`](./jobs/k_fold/) directory.

### Training
K-fold training scripts are collected under three directories:
1. **Main Experiment:** under `./jobs/k_fold/train/`. These scripts will train the models on the LUNA25 dataset only, by unfreezing weights in all layers (full finetuning).
2. **LP-FT Experiment:** under `./jobs/k_fold/train2stage/`. These scripts will train the models using LP-FT technique, on the LUNA25 dataset only.
3. **MSTL Experiment:** under `./jobs/k_fold/trainluna16/`. These scripts will train the models using MLST technique. LUNA16 is used as the intermediate dataset. These jobs require the LUNA16 dataset.

For example, we trained the video-pretrained ResNet3D-18 using [`./jobs/k_fold/train/kf_train_resnet3d_18.sh`](./jobs/k_fold/train/kf_train_resnet3d_18.sh). This file uses the following script to train the model:
```bash
python -u src/k_fold_training.py \
    --model_string "resnet3d" \
    --folds 4 \
    --seed 4242 \
    --fold_annotation_dir <your k_fold annotation image directory> \
    --depth 18 \
    --train_image_dir <your train image directory> \
    --val_image_dir <your validation image directory> \
    --num_workers 2 \
    --report_frequency 5 \
    --epochs 50 \
    --batch_size 2 \
    --metric auroc \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --device "cuda"
```

*Generally*, the directory arguments are optional. The script will use the default path if so. However, we advise that they be declared explicitly to prevent unexpected errors. Most of the other arguments are also optional.

### Evaluation
Training using the pipeline above will generate training statistics and model weights in specified directories. These scripts load the *latest* model weights, run inference on the validation and test sets, and finally evaluate the model on multiple metrics. Loading the latest model is done via the bash script, which will select the latest timestamp directory.

Evaluation scripts are collected under three directories:
1. **Main Experiment:** under `./jobs/k_fold/evaluate/`.
2. **LP-FT Experiment:** under `./jobs/k_fold/evaluate2stage/`.
3. **MSTL Experiment:** under `./jobs/k_fold/evaluate_double_luna16_25/`.

Evaluation process is similar across experiments. The scripts in different directories differ only in where the target model is. For example, to evaluate the ResNet3D-18 model:

```bash
python -u src/k_fold_evaluate_model_dir.py \
    --fold_directory <your 4 fold model directory> \
    --annotation_dir <your k_fold annotation image directory> \
    --test_annotation <your test annotation directory> \
    --image_dir <your image directory> \
    --preprocessing "video_pretrained" \
    --model_type "resnet3d" \
    --depth 18 \
    --channels 3 \
    --batch_size 8 \
    --num_workers 2 \
    --device "cuda"
```

By default, the evaluation result for that model will be saved in the same directory as the weight's directory. To collate them into a `.csv` file from multiple folds, you can use this utility script:
```bash
python -u ./src/extract_kfold_results.py --source ["kfold", "kfold_2stage", "kfold_luna16"]
```

Different `--source` argument will extract results from different training strategy.

---

## Acknowledgement
We use the code for the backbone models from these repositories:
- [video-classification-3d-cnn-pytorch](https://github.com/kenshohara/video-classification-3d-cnn-pytorch): ResNet3D
- [MedicalNet](https://github.com/Tencent/MedicalNet): MedicalNet