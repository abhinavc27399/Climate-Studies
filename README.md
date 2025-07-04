# Climate Studies

This Python project is focused on analyzing and studying climate data. It provides tools and scripts for data processing, visualization, and modeling to support climate research.

## Features

- Data ingestion and preprocessing
- Climate data analysis and visualization
- Statistical modeling and forecasting
- Modular and extensible codebase

## Getting Started

1. **Clone the repository:**
    ```
    git clone https://github.com/your-username/Climate-Studies.git
    cd Climate-Studies
    ```
    
2. **Create virtual environment:**
    ```
    python -m venev env
    ```
    
3. **Activate the virtual environment:**
   
    For Linux/macOS:
    ```   
    source env/bin/activate
    ```
    For Windows:
    ```
    env\Scripts\activate 
    ```

5. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

6. **Run the scripts:**
   
    Using bash:
   
    1) Make it executable:
    ```
    chmod +x run.sh
    ```
    2) Run it:
    ```
    ./run.sh
    ```
    Using powershell or command prompt:
    ```
    powershell -File run.ps1
    ```
    or
    ```
    powershell -ExecutionPolicy Bypass -File run.ps1
    ```

## Project Structure
```
Climate-Studies/
├── CO2-Concentration-Forecast/                 # each sub project directory contains a mian.py
│   ├── main.py
.
.
.
.
├── datasets/                        # conatains data files
├── run.sh/run.ps1                   # platform specific run scripts for the project
├── requirements.txt                 # package dependencies list
└── README.md
```
