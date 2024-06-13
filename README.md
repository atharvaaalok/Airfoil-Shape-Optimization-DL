## Project Plan

### Initial Plan
- [ ] Prepare Data
  - [x] Get airfoil data files.
  - [x] Import data into a numpy array and plot airfoils using matplotlib. Check coordinate ordering.
  - [x] Generate new airfoil from a given airfoil using random noise.
  - [x] Work on high, medium and low variance airfoil generation. Prepare directory structure.
  - [x] Work on getting Xfoil working.
  - [x] Prepare the data generation pipeline.
  - [x] Use Xfoil to generate only valid airfoils.
  - [x] Find and save the L/D ratio for generated airfoils.
  - [x] Generate H, M, L variance airfoils. Look at plots to decide on noise levels.
  - [x] Estimate time to generate a H, M, L data point on average.
  - [x] See if data can be generated through multi-processing in parallel.
  - [x] Decide H, M, L count depending on time taken and given that I am willing to spend D days on just data generation.
  - [x] Need for using parametric representation and smooth surface airfoil generation after adding noise.
  - [ ] TE and LE at (1, 0) and (0, 0) and setting angle of attack of generated airfoils to 0.
- [ ] Work on neural network training pipeline.
  - [ ] Get data from dataset using Dataset and Dataloader in pytorch.
  - [x] Define Neural Network class.
  - [x] Define the hyperparameters.
  - [x] Complete the training loop.
  - [x] Work on checkpoints saving.
- [x] Work on input optimization given the neural network.
  - [x] Fix L/D
  - [x] Change Loss function.
- [ ] Work on getting the entire neural network pipeline working.

### Move to Google Cloud Platform
- [x] Move code to GCP. Connect to GCP via VSCode and SSH.
- [ ] Make sure all code works on GCP.
- [ ] Generate actual data.
- [ ] Decide where to store the data.
- [ ] Train the neural network.
- [ ] Run the two optimization processes.



## References
- [For airfoil database](https://github.com/npuljc/Airfoil_preprocessing?tab=readme-ov-file)