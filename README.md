## Reducing Emissions in Car Carriers: An AI-Based Alert System
We have developed an AI-based alert system to reduce emissions in car carriers. The system uses a CNN model to detect engine idling and can issue a warning in situations where the engines should be turned off, such as in underground parking lots. For more details check out the project on [Matthias'](https://matthiasbressan.github.io/sounds_classification.html) or [Giacomo's](https://giacomo-lab.github.io/sounds_classification.html) profile.

### Installation and usage
To use this project you will need to clone the repository, and install the libraries contained in _requirements.txt_, download and unzip the [UrbanSound8k dataset](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) from Kaggle. 

The main part of the code is in the Jupyter Notebook *idling_detection.ipynb*. To keep the notebook as compact as possible the functions used in the notebook are in the file *idling_detection_functions.py*.
The *history.pickle* file contains the info about the training of the tuned model. This information can be obtained by rerunning the training or by loading our result (just to save time).

### Credits
This project was a collaborative effort between Matthias Bressan and Giacomo Labbri


