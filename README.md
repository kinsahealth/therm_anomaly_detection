# therm_anomaly_detection

Steps:
To run these reproducible examples, you will need `pipenv` installed to generate the correct dependencies. This can be installed as follows:

    brew install pipenv

You can then navigate to the directory and run the following set of commands to rerun the examples. This assumes you also have python 3.6 installed.

    cd therm_anomaly_detection  # Change to this directory
    pipenv install              # Load the required dependencies
    pipenv shell                # Activate virtual env
    
    # Now, create a jupyter kernel that can be used to run the examples
    python -m ipykernel install --user --name=therm_anomaly_detection
    
    jupyter notebook            # Activate jupyter notebooks

Once you load the jupyter notebook up, you can toggle the kernel (`Kernel > Change Kernel`) to run using the dependencies installed via pipenv. This only matters if you want to run the code locally. Otherwise, you should be able to view output of the Jupyter notebook. The kernel you want to switch to is `therm_anomaly_detection`.
