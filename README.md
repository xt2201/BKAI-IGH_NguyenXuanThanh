# BKAI-IGH_NguyenXuanThanh
 Assignment_Intro_to_Deep_learning

## Requirements

To install the necessary packages, please run the following command in your terminal:
    ```
    pip install torch opencv-python numpy torchvision timm pillow matplotlib
 albumentations==1.0.3 segmentation-models-pytorch==0.3.0`
    ```

## Instructions to Run Inference

1. Clone the repository:
    ```
    git clone https://github.com/xt2201/BKAI-IGH_NguyenXuanThanh.git
    cd 'BKAI-IGH_NguyenXuanThanh'
    ```

2. Ensure the model checkpoint is in the `checkpoints/` directory:
    - Download the checkpoint from the provided [link](https://drive.google.com/file/d/1SQmjSYBMkFckl9X9VKhzDWGCmSfIEnzV/view?usp=sharing).
    - Place the `best_model.pth.tar` file in the `checkpoints/` directory of this repository.

3. Run the inference script:
    ```
    python3 infer.py --image_path image.jpeg
    ```
    For example: 
    ```
    python3 infer.py --image_path dataset_test/13dd311a65d2b46d0a6085835c525af6.jpeg
    ```    

4. The segmented image will be saved in the current working directory with the prefix segmented_. For example, segmented_image.jpeg.
