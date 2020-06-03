<p align="center">
  <img src="/ui/images/NetMD_logo.png" align="center" alt="NetMD" title="A cute kitten" width="30%" height="30%" />
</p>

# NetMD 

NetMD is an easy-to-use, easy-to-learn software that takes advantage of recent developments in the deep learning field to provide the fast and efficient recovery of compressively sampled RCM images through an intuitive user interface. On top of this, it implements a denoising step in the image recovery process to improve the quality of the reconstructed target sample, presenting users with the ability to quickly inspect recovered RCM skin tissue images for the diagnosis of skin cancer. 

## NOTE: COVID-19
Due to difficulties faced during the unexpected hit of the global pandemic cause by COVID-19 and a major setback in the last phase of the project were the development team lost access to the computing resources that were being used to train the neural networks in NetMD's backend, we were unable to train NetMD with the target dataset of RCM skin stacks. However, we are providing the necessary train files and instructions on how to train the networks to prepare the network for its intended use. 

## How to use

To run NetMD, you'll need to have python installed. You'll also need to install some depencies which you'll be prompted for. After installing dependencies, run the command:

>  `python netmd.py` 

Alternatively, as the denoiser has yet to be trained, you can run the program by running:

>  `python netmd.py --denoiser off`

Once up and running, you'll be greeted with an instructions screens that contains everything you need to know on how to use the features available to you!

*NOTE* NetMD accepts cs measurements as a .mat file with an array accessed through the key 'cs_measuremets' of shape [num_img_blocks, 1089 * cs_ratio] where cs_ratio is 10, 25 or 50 for 10%, 25% or 50%.

## How to train neural network backend

NetMD's backend consists of two neural networks: 
* ISTA-Net
* RCM Deep Denoiser (RCMDD)

ISTA-Net (https://github.com/nbansal90/ISTA-Net) performs the task of image reconstruction from compressive sensing measurements, while RCMDD is a convolutional autoencoder for denoising the reconstructed image.

Given that we were unable to train the networks on our RCM skin stack dataset we've included the necessary methods to train the networks.

### To train ISTA-Net

1. Go to the `image_processing/ista/` , inside you'll find the `train.py` file which has the following arguments available:

* --start_epoch -> epoch number from which to start training, checkpointing happens ever 5 epochs, so you can continue training from a specific checkpoint
* --end_epoch -> epoch where training ends
* --batch_size -> batch size for training
* --model_dir -> points to the directory where the different models for different datasets are stored, in our case the `model` directory
* --data_dir -> directory where different datasets are stored, in our case the `../../data` directory
* --dataset_name -> dataset directory inside data directory, in our test case the `Natural_Images`
* --log_dir -> directory where logs are saved, in our case the `log`
* --cs_ratio -> cs sampling ratio, in our case 10, 25 or 50 for 10%, 25% and 50%
* --sampling_matrix -> path to sampling matrix, i.e `sampling_matrix/phi_0_10_1089.mat`
* --initialization_matrix -> path to initialization matrix, i.e `initialization_matrix/Natural_Images/Initialization_Matrix_10.mat`

2. Example run of `train.py`:
> `python train.py --dataset_name Natural_Images --cs_ratio 10 --sampling_matrix sampling_matrix/phi_0_10_1089.mat --initalization_matrix initialization_matrix/Natural_Images/Initialization_Matrix_10.mat`

*NOTE* This files makes the follower assumptions:
* File containing training data is in either .mat or .hdf5 format
  * Inside the data file, is an accessable array through the key 'labels'
  * This array as the shape: [num_img_blocks, 1089] where num_img_blocks is the total number of randomly cropped 33x33px image blocks that have been vectorized (1089px).
 
### To train RCMDD

1. Go to the `image_processing/denoiser/` , inside you'll find the `train.py` file which has the following arguments available:

* --start_epoch -> epoch number from which to start training, checkpointing happens ever 5 epochs, so you can continue training from a specific checkpoint
* --end_epoch -> epoch where training ends
* --batch_size -> batch size for training
* --model_dir -> points to the directory where the different models for different datasets are stored, in our case the `model` directory
* --dataset_name -> dataset directory inside data directory, it's set to `RCM` as default
* --data_dir -> directory where different datasets are stored, in our case the `../../data` directory
* --log_dir -> directory where logs are saved, in our case the `log`
* --cs_ratio -> cs sampling ratio, in our case 10, 25 or 50 for 10%, 25% and 50%

2. Example run of `train.py`:
> `python train.py --dataset_name RCM --cs_ratio 10

*NOTE* This files makes the follower assumptions:
* File containing training data is in either .mat or .hdf5 format
  * Inside the data file, are two accessable arrays through the keys 'X_train' and 'y_train', where X_train are the noisy images and y_train are the original noiseless images.
  * 'X_train' and 'y_train' arrays have the shape: [num_imgs, img_width, img_height] 
 * The training data should be constructed with the train ISTA-Net models, as the noisy input images that make X_train should come from ISTA-Net.

### To simulate cs measurements from images for testing NetMD

Inside the `data` folder is a file `simulate_cs.py` that takes the following arguments:
* --image -> path to image
* --sampling_matrix -> path to sampling_matrix
* --dataset_name ->  dataset name in the data folder image is drawn from
* --cs_ratio -> cs sampling ratio to simulate

This file when run will output a .mat file to the `../cs_test_samples/ + dataset_name + /cs_ + cs_ratio + /'` folder with simulated cs measurements that can be inputted into NetMD for reconstruction.

### To replicate final report evaluation metrics of ISTA-Net
Run the `evaluation_psnr_ssim.py` file in the project's root directory
