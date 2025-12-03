Here is the `README.md` file in English, covering the project background, environment setup, dataset preparation, training/testing instructions, and the demo.

````markdown
# TVFFC (Time-Varying Flat-Field Correction)


Paper DOI: [10.1364/OE.529419](https://doi.org/10.1364/OE.529419)

## 1. Prerequisites

This project provides a complete environment configuration file: `pytorch1_9.yaml`. Please ensure you have Anaconda or Miniconda installed, then run the following commands to create and activate the virtual environment:

```bash
# Create environment
conda env create -f pytorch1_9.yaml

# Activate environment
conda activate pytorch1_9
````

## 2\. Dataset Setup

To train and test the model, please prepare your dataset using the following structure:

  * **Location**: Place the dataset folder inside the `datasets/data` directory in the project root (create this folder if it does not exist).
  * **Folder Structure**: The dataset should contain the following four subfolders:
      * `train_A`: Input images for training
      * `train_B`: Ground truth (label/target) images for training
      * `test_A`: Input images for testing
      * `test_B`: Ground truth images for testing (optional, used for metric calculation)
  * **Filename Requirement**: The filenames in `train_A` and `train_B` must correspond one-to-one.
      * Example: `train_A/000000.tif` must correspond to `train_B/000000.tif`.

## 3\. Training the Model

Use the following command to start training the network model. This command trains a mapping model from domain A to domain B.

```bash
python train.py --name project1024 --label_nc 0 --no_instance --gpu_ids 0
```

  * `--name`: Project name (experiment name). Training results and weights will be saved in `checkpoints/project1024`.
  * `--label_nc 0`: Set the number of input label channels to 0 (suitable for non-semantic segmentation tasks, such as image restoration/translation).
  * `--no_instance`: Do not use instance maps.
  * `--gpu_ids 0`: Specify the GPU ID to use.

## 4\. Testing the Model

Use the following command to test a trained network model:

```bash
python test.py --name project1024 --ngf 64 --label_nc 0 --no_instance --how_many 100
```

  * `--ngf`: The number of generator filters in the first conv layer (default is usually 64; keep consistent with training or adjust based on model architecture).
  * `--how_many`: Specify the number of images to test (e.g., 100 images).
  * The test results will be saved in the `results/project1024/` directory.

## 5\. Demo

We provide a pre-trained example containing projection image data for 100ms and 5ms scenarios. You can download the resources below to quickly verify the model's performance.

### Resources

1.  **Checkpoints (Weights)**:

      * Download Link: [Google Drive - Checkpoints](https://drive.google.com/drive/folders/1ixCHxHdSNeKZ0XGeEnsLIGbvyhBNYPQd?usp=sharing)
      * **Action**: Download and place the folder into the project's `checkpoints` directory. (Ensure the folder name is `yutou_100ms` to match the command below).

2.  **Test Dataset**:

      * Download Link: [Google Drive - Test Data](https://drive.google.com/drive/folders/17eODuKbqK5qE2_sW14MA-ff0VmOtjb85?usp=sharing)
      * **Content**: Includes 100 projection images each for 100ms and 5ms scenarios.
      * **Action**: Download and place the dataset folder into the project directory (e.g., under `datasets`, or specify the path using `--dataroot`).

### Running the Demo

Once the weights and data are in place, run the following command to generate results:

```bash
python test.py --no_instance --label_nc 0 --name yutou_100ms --gpu_ids 0
```

  * **Results**: The processed result images will be saved in the `results/yutou_100ms` folder.

## References

  * For more detailed parameter explanations and advanced usage, please refer to [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD).

<!-- end list -->

```
```
## Acknowledgments
This code borrows heavily from [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD).