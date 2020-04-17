# Multi-human Data Generation

This repository provides the code to generate a datasets similar to the one proposed in [Learning Multi-Human Optical Flow](https://arxiv.org/pdf/1910.11667.pdf) and [Learning to Train with Synthetic Humans](https://arxiv.org/pdf/1908.00967.pdf).
![The data generation pipeline provides ground truth for multiple modalities.](https://github.com/DavHoffmann/Multi-humanDataGeneration/blob/master/pipeline_data_v3.png)

Click the image below to see an example video of our Multi-Human Flow dataset.
[![Multi-Human Flow Dataset](https://img.youtube.com/vi/sKYQ30AecFg/0.jpg)](http://www.youtube.com/watch?v=sKYQ30AecFg)


## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](./LICENSE) and any accompanying documentation before you download and/or
use the Multi-HumanFlow/Learning to Train with synthetic humans, data, data generation code and software, ( "Data" & Software) including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Data & Software (including
downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these
terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions,
you must not download and/or use the Data & Software. Any infringement of the terms of this agreement will automatically
terminate your rights under this [License](./LICENSE).

## Download the Multi-Human Optical Flow dataset (MHOF) or Multi-Human Pose dataset.
You can download the MHOF dataset [here](https://humanflow.is.tue.mpg.de/) and the Multi-Human Pose dataset [here](https://ltsh.is.tue.mpg.de/). In order to download the dataset you need to register and accept the license terms.

## Installation for data generation
a) Clone this repository.

b) Next, you need to download resources and smpl_data from the download section of [our website](https://humanflow.is.tue.mpg.de/) and extract them in the folder ```/Muli-humanFlowGeneration/datageneration/```.

c) Download [Blender 2.79b](https://www.blender.org/download/releases/2-79/) and extract the content to ```/Muli-humanFlowGeneration/datageneration/```.
Blender ships with a custom python environment, however we need to add a few packages. You can first install [pip](https://pip.pypa.io/en/stable/installing/) ```cd ./blender-2.79-linux-glibc219-x86_64/2.79/python/bin/``` and run```./python3.5m get-pip.py```. Next, run ``` python3.5m pip install h5py scipy==0.16.0 opencv-python```. However, this method fails sometimes. As a workaround, you can create a temporary virtualenv with python 3.5. Lets call it blendPy_tmp.
After activating blendPy_tmp pip install all the dependencies. cd to ```/lib/python3.5/site-packages``` and copy the folders "cv2, h5py, numpy, scipy, chumpy" and "six.py" to ```./Muli-humanFlowGeneration/blender-2.79-linux-glibc219-x86_64/2.79/python/lib/python3.5/site-packages```. The virtualenv blendPy_tmp is not needed anymore and can be deleted.

d) Create an account and download [SMPL-H](http://mano.is.tue.mpg.de/). You can find the necessary files at Model & Code in the download section. Extract the folder in
```/smpl_data/ ```.
Copy the file ```./datagenerateion/smpl_handpca_wrapper_multiHuman.py``` to ```./datagenerateion/smpl_data/mano_v1_2/webuser```. Now follow the installation instruction for SMPL-H as detailed in ```./datagenerateion/smpl_data/mano_v1_2/webuser/README.txt```. Download the .fbx models from the [SMPL-H](http://mano.is.tue.mpg.de/) website and save them in ```./datageneration/smpl_data/```. It can be found in the "Tools for rendering synthetic data" section. Finally, download the hand poses from the same section on the MANO website and place them in ```./datageneration/smpl_data/handPoses/per_SEQ___bodyHands```.

e) Setup a virtualenv with python 3 called collisionCheck and activate your virtualenv

f) ```pip install chumpy cython``` in this virtualenv

g) cd to /Muli-humanDataGeneration/ and clone the [Mesh](https://github.com/MPI-IS/mesh) repository.
Follow the installation instruction.

h) Install CGAL. To do so ```cd mesh/mesh/thirdparty``` and extract CGAL-4.7.tar.gz to ```/Multi-humanFlowGeneration/cgal```. CGAL should be ready to use. In case you encounter any issues the [CGAL installation instructions](https://doc.cgal.org/4.7/Manual/installation.html) might be of help.

i) Install the mesh intersection package. To do so, first install the boost library system wide with ```sudo apt-get install libboost-all-dev```. Next, ```cd /Multi-humanFlowGeneration/datageneration/meshIsect``` and run ```python setup.py build_ext --inplace```.

j) Open ```./Multi-humanFlowGeneration/datageneration/meshIsect/collision_detect_cgal_meshIsect.py``` and set the absolute path to your installation of the Mesh package.

k) Dependent on the type of data you want to generate open Generate_multiHumanFlow.py, generate_multiHumanPose.py or generate_mpii_mixed.py and set the working directory to the folder on your system ```/Muli-humanFlowGeneration/datageneration/```. Similarly, set the the path to the root folder of the collisionCheck virtualenvionment.

l) Download MoShed mocap data from [AMASS](https://amass.is.tue.mpg.de/) and generate a file containing mocap sequences of the selected dataset. For example, to generate a file with sequences from CMU and HumanEva run ```python generateMocapFile.py --amass_root=./ --datasets=CMU --datasets=HumanEva```.

m) Download the [SUN397](https://groups.csail.mit.edu/vision/SUN/) dataset. Set the variable ``lsun_base_path`` in Generate_multiHumanFlow.py, generate_multiHumanPose.py. Other background images are possible, but you need to specify a new train/test/valid split.

n) Set ```PATH_out``` and ```PATH_tmp``` in whichever script of Generate_multiHumanFlow.py, generate_multiHumanPose.py and generate_mpii_mixed.py you want to run. Set ```mocapDataName``` to the name you have chosen when running ```generateMocapFile.py```.

## Multi-human Flow Generation

Run the script using ```../blender-2.79-linux-glibc219-x86_64/blender -b -P Generate_multiHumanFlow.py -- 0 0 3```. The first argument determines the MoCap sequence for the first synthetic human, the second argument determines the train/valid/test split for 0/1/2, respectively. The last argument should be 0 per default and can be used to set the number of synthetic humans.

## Multi-human pose data

![Purely synthetic dataset](https://github.com/DavHoffmann/Multi-humanDataGeneration/blob/master/surreal_multi_small.png)

Run the script using ```../blender-2.79-linux-glibc219-x86_64/blender -b -P generate_multiHumanPose.py -- 0 0 3```. The first argument determines the MoCap sequence for the first synthetic human, the second argument determines the train/valid/test split for 0/1/2, respectively. The last argument should be 0 per default and can be used to set the number of synthetic humans.

## MPII-mixed dataset

![MPII pose estimation dataset augmented with synthetic humans](https://github.com/DavHoffmann/Multi-humanDataGeneration/blob/master/mixed3_sm.png)

To use this scrip you need to download [MPII pose estimation dataset](http://human-pose.mpi-inf.mpg.de/). Set the variable ```img_base_path``` to the directory in which you saved the MPII images. From the same website download the annotations and place ```annolist_dataset_v12.mat``` in ```/Muli-humanFlowGeneration/datageneration/resources/.```

Run the script with ```../blender-2.79-linux-glibc219-x86_64/blender -b -P generate_mpii_mixed.py -- 0 0 3```.


## Citation

If you find this software or dataset useful in your research we would kindly ask you to cite the respective paper (Learning Multi-Human Optical Flow for sequences and Learning to Train with Synthetic Humans for frame and mixed datasets/code)

```
@article{multihumanflow,
  title = {Learning Multi-Human Optical Flow},
  author = {Ranjan, Anurag and Hoffmann, David T and Tzionas, Dimitrios and Tang, Siyu and Romero, Javier and Black, Michael J},
  journal = {International Journal of Computer Vision (IJCV)},
  month = jan,
  year = {2020},
  url = {http://humanflow.is.tue.mpg.de },
  month_numeric = {1}
}

@inproceedings{Hoffmann:GCPR:2019,
  title = {Learning to Train with Synthetic Humans},
  author = {Hoffmann, David T. and Tzionas, Dimitrios and Black, Michael J. and Tang, Siyu},
  booktitle = {German Conference on Pattern Recognition (GCPR)},
  month = sep,
  year = {2019},
  url = {https://ltsh.is.tue.mpg.de},
  month_numeric = {9}
}

```

## Support
We only support the default settings. Some options to change flags are left in the code, but should only be used with caution and might require changes to the code.

## Supported Software

This code was tested with Blender2.79b only. We did not test for compatibility with newer Blender versions. We tested the code on Ubuntu 16.04 and 18.04.


## Contact
For questions, please contact [david.hoffmann@tuebingen.mpg.de](mailto:david.hoffmann@tuebingen.mpg.de).

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).
