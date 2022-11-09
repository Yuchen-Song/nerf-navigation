# NeRF-Navigation

## Abstract

NeRFs have recently emerged as a powerful paradigm for the representation of natural, complex 3D scenes. NeRFs represent continuous volumetric density and RGB values in a neural network, and generate photo-realistic images from unseen camera viewpoints through ray tracing.  We propose an algorithm for navigating a robot through a 3D environment represented as a NeRF using only an on-board RGB camera for localization.  We assume the NeRF for the scene has been pre-trained offline, and the robot's objective is to navigate through unoccupied space in the NeRF to reach a goal pose.  We introduce a trajectory optimization algorithm that avoids collisions with high-density regions in the NeRF based on a discrete time version of differential flatness that is amenable to constraining the robot's full pose and control inputs.  We also introduce an optimization based filtering method to estimate 6DoF pose and velocities for the robot in the NeRF given only an onboard RGB camera.  We combine the trajectory planner with the pose filter in an online replanning loop to give a vision-based robot navigation pipeline.  We present simulation results with a quadrotor robot navigating through a jungle gym environment, the inside of a church, and Stonehenge using only an RGB camera. We also demonstrate an omnidirectional ground robot navigating through the church, requiring it to reorient to fit through the narrow gap. Videos of this work can be found at [this link](https://mikh3x4.github.io/nerf-navigation/)

---

## Code Structure

For more infomation on the paper see the [paper page](https://mikh3x4.github.io/nerf-navigation/). 

# torch-NGP


[NeRF](http://www.matthewtancik.com/nerf) (Neural Radiance Fields) is a method that achieves state-of-the-art results for synthesizing novel views of complex scenes.

[Instant-NGP](https://github.com/NVlabs/instant-ngp) is an extension that grants enormous performance boosts in inference and training. This repository for navigation is built off of the PyTorch version of NGP.

## Installation
It is recommended to go to [torch-ngp](https://github.com/ashawkey/torch-ngp) page and install its dependencies there, as our code is an application of their code. If you can begin training without any issues in a conda environment, then you should be able to run our code just fine. This repo includes not only the navigation code, but also the code necessary to train the models (i.e. the repository is self-sufficient). As our dependencies are exactly the same as theirs, you can create a virtual environment straight from our repository by:

```
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
```

## How To Run?

### Training

Run NeRF training. Make sure your training data (from Blender) is located in ```data/nerf_synthetic/model_name```. This 
format should be identical to most NeRF repositories. The command to train on Blender scenes is: 

```python main_nerf.py data/nerf_synthetic/model_name --workspace model_name_nerf -O --bound x.x --scale 1.0 --dt_gamma 0```

It is imperative you set ```scale``` to 1.0, so that torch-NGP does not resize the scene dimensions and cause a mismatch between the 
scale of the model dynamics and that of the NeRF. Set ```bound``` to be the bounding box of your Blender mesh. For example, for 
the Stonehenge scene, we used ```--bound 2.0```. For the Stonehenge scene data and model, please see the pretrained models section below. 

### Validation

Once training has finished or you've achieved satisfactory results, the checkpoint will be in the ```model_name_nerf``` folder. You can see our pretrained Stonehenge model as a point of comparison.

### Setting up Blender

Make sure to download the latest version of Blender. We will use Blender as our simulation environment. 
First, open a new terminal in the folder where Blender is located and open blender (i.e. ```./blender``` in the terminal). This 
allows the user to break out of hanging scripts through the terminal. Once Blender is open, navigate to the Scripting tab, 
and open ```visualize.py``` if it is not already in the tab.

Note: Make sure there is a Camera object in the scene. 

### Running

Create a ```sim_img_cache``` folder if it is not already there. This is where ```visualize.py``` will read in poses of the robot 
and return an observation image that ```simulate.py``` will perform pose estimation on.

Run ```visualize.py``` in Blender by pressing the run button.

Run the planning and estimation loop (in a terminal different than the one ```visualize.py``` is on) using the command:

```python simulate.py data/nerf_synthetic/model_name --workspace model_name_nerf -O --bound x.x --scale 1.0 --dt_gamma 0```

It is imperative that the parameters you pass in are the same as those used to train the NeRF (i.e. ```--bound```, ```--scale```, ```--dt_gamma```).
All tunable configs (e.g. noise, initial and final conditions) are in ```simulate.py```. 

---

### Datasets
Following the canonical data format for NeRFs, your training data from Blender should look like the following:

```                                                                                                                              
├── model_name                                                                                                  
│   ├── test      #Contains test images      
│   │   └── r_0.png           
│   │   └── ...                                                                                                    
│   ├── train                                                                                  
│   ├── val  
│   └── transforms_test.json  
│   └── transforms_train.json
│   └── transforms_val.json
```

Place this folder here:
```                                                                                                                                                                 ├── data                                       
│   ├── nerf_synthetic                                                                                                  
│   │   └── model_name                                                                                                                             
```

---

### Pre-trained Models

Our results are primarily from the Stonehenge scene. The training data, pre-trained model, and Blender mesh can be found [here](https://drive.google.com/drive/folders/104v_ehsK8joFHpPFZv_x31wjt-FUOe_Y?usp=sharing).

## Citation
Remember to cite the original NeRF authors for their work:
```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

and those of Instant-NGP:

```
@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}
```

and those from torch-NGP:
```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}
```