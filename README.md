This project is focused on detecting snoring sounds from audio recordings using machine learning techniques. The dataset includes labeled audio samples of snoring and non-snoring sounds, and the project involves preprocessing the audio data, extracting features, training a machine learning model, and evaluating its performance.

Project Structure
The project is structured as follows:

```
├── data/
│   ├── raw/  
│   │   ├── 0/ # Folder containing non snoring audio files
│   │   └── 1/ Folder containing snoring audio files
│   ├── inference/  
│   ├── train/  # Training data (created by the script)
│   │   ├── snoring/
│   │   └── non-snoring/
│   └── test/  # Testing data (created by the script)
│       ├── snoring/
│       └── non-snoring/
├── models/  # Saved models
├── docs 
├── src/
│   ├── data/ # Data preprocessing, augmention, etc
│   ├── features/    # Feature extraction, transformations, etc 
│   ├── models/ 
│   ├── train.py  # Model training script
│   ├── evaluate.py  # Model evaluation script
│   └── config.py # Configuration settings
├── README.md  # Project documentation
└── requirements.txt
```

### Dataset


In this project the dataset consists of two folders:

- `data/raw/1/`: Contains 500 snoring audio files, each 1 second long.
- `data/raw/0/`: Contains 500 non-snoring audio files, each 1 second long. These include background sounds like baby crying, clock ticking, door opening, etc.
Dataset Sources


The snoring sounds were downloaded from the following online sources:

- Download from Kaggle: [Dataset Link](https://www.kaggle.com/code/mpwolke/snoring-wav/input)

### Project Workflow



-----------------------------------------------------------------------------------

By Hassan K (lordyhas)


