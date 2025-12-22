# hyperparameter_nni_entry.py
from optimization.hyperparameter_tuner import nni_trial
import nni

def main():
    params = nni.get_next_parameter()  # NNI injects trial parameters
    result = nni_trial(params)         # run one trial
    nni.report_final_result(result)    # report back to NNI

if __name__ == "__main__":
    main()
