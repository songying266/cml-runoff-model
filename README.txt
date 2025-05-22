This project includes Python code for rainfall-runoff modeling using CML and RG data.

1. Place your data files in the ./data/ folder:
    - model_CML.csv
    - model_rain.csv
    - model_CML_mean.csv
    - model_rain_mean.csv

2. Install dependencies:
    - Using conda: conda env create -f environment.yml
    - Using pip: pip install -r requirements.txt

3. Run the main script:
    cd src
    python main_script.py

4. Output files will be saved in ./results/cross-validation/
For demonstration, only partial validation results (2 iterations per setting) are included.  
Full results (210 CSVs) are available upon request.


Note:
- Ensure the datasets contain a column named "event" for grouping.
- Update paths in the script if you move files.