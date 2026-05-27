# Learning Mario: 1-1 Using a Genetic Algorithm (CS480 Final Project)

### Authors: 

Roan Morgan, Hayk Chaloyan, and Jonathan Martin

### Environment:

- you need to have > Visual Studio 2015 with "Desktop Development with C++" package to install nes_py python package
```
pip install -r requirements.txt
```

python version 3.12.3

Specific Version Requirements

- `gym==0.25.1` 
- `numpy==1.26.4`
- `nes-py==8.2.1`
- `gym-super-mario-bros==7.4.0`
- `pyglet==1.5.21`

gym-super-mario-bros not compatible with 0.26+
gym not compatible with numpy 2.0+

### Project Framework

Three Options:

- Reflex Agent (Very easy, baseline)

- Transition model (
        Maybe difficult due to keeping track of future gamestates,
        Don't have enemy positions,
        Search type problem
    )

- Markov decision process (Hard but most likely best model for project, fits deep learning model)

#### Resources
https://www.v7darwin.com/blog/neural-networks-activation-functions


### TODO:

- Scaffold project and figure out how to split up work


- 