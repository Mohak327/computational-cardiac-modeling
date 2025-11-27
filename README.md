# Computational Cardiac Modeling

This project implements a computational model for simulating healthy and pathological ECG signals. The project is organized as follows:

- `notebooks/`: Jupyter notebooks for analysis and simulation.
- `src/`: Source code for models and utilities.
- `tests/`: Unit tests for the project.
- `data/`: Raw and processed data files.
- `results/`: Generated results, including plots and logs.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the notebooks in the `notebooks/` directory to reproduce results in one file or the main.ipynb to use the modular structure.

## Directory Structure

computational-cardiac-modeling/   
│
├── notebooks/   
│   ├── main.ipynb  # One-shot notebook for the project  
│   ├── standalone_base_model.py                # BaseModel class 
│
├── src/  
│   ├── models/
│   │   ├── base_model.py                # BaseModel class  
│   │   ├── ecg_model.py                 # ECG_Model class
│   │   ├── ecg_ode.py                  
│   │   └── __init__.py                  # Makes the folder a Python package  
│   │  
│   ├── utils/  
│   │   ├── plotting.py                  # Plotting utilities   
│   │   ├── analysis.py                  # Analysis utilities  
│   │   └── __init__.py                  # Makes the folder a Python package  
│   │  
│   └── __init__.py                      # Makes the src folder a Python package  
│  
├── tests/  
│   ├── test_models.py                   # Unit tests for models  
│   ├── test_utils.py                    # Unit tests for utilities  
│   └── __init__.py                      # Makes the folder a Python package  
│  
├── data/  
│   ├── raw/                             # Raw data files  
│   ├── processed/                       # Processed data files  
│   └── README.md                        # Description of the data structure  
│  
├── results/  
│   ├── figures/                         # Generated plots and figures  
│   ├── logs/                            # Logs from simulations  
│   └── README.md                        # Description of the results structure  
│  
├── main.ipynb                           # Main file that uses the modular structure
├── .gitignore                           # Git ignore file  
├── README.md                            # Project overview and instructions  
├── requirements.txt                     # Python dependencies  
