# Video Surveillance for Road Traffic Monitoring

## Abstract

### Week 1

In this week we setup the different metrics used and get used to the dataset
we used in the following weeks. 

### Week 2

The main goal of this week is to estimate the background in a video sequence
using different statistical models and compute the metrics to compare them.


## Code

The code is structured in the following way:

- `src` folder with the sources code. Inside this folder, you will find:
    - `weeks` package, with one package created for each week.
    - `utils` package, with different modules of useful functions (*Work in progress*).
    - `metrics` module, with different metrics to measure the performance of
    our experimentss (*Work in progress*).

- `data` folder contains the data provided with the codes (*Work in progress*).

- `output` folder contains the output of each week. Each subfolder corresponds
to each week.

To run each task you need to uncomment the line from the `src/main.py` file.

## Setup

To run the files, you need to install the dependencies listed in the 
`requirements.txt` file:


```
$ pip install -r requirements.txt
```

Or you could create a virtual environment and install them on it:

```
$ mkvirtualenv -p python2.7 m6
(m6) $ pip install -r requirements.txt
```


## Team members

|      Member     |           Email          |
|:---------------:|:------------------------:|
| Ferrín, Facundo | facundo.ferrin@gmail.com |
|     Mor, Noa    |    noamor87@gmail.com    |
|  Pose, Agustina |    aguupose@gmail.com    |