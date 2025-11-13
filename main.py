from experiment_controller import ExperimentController

if __name__ == "__main__":
    print("Starting RNN Architecture Comparison")
    print("Note: Plot generation is disabled for faster results")
    print("Only results table will be generated\n")
    
    controller = ExperimentController()
    controller.run_experiments()  # Remove the num_experiments parameter