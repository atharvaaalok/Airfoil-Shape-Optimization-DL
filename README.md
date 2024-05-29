## Project Plan

### Initial Plan
- [ ] Prepare Data
  - [ ] Get airfoil data files.
  - [ ] Import data into a numpy array and plot airfoils using matplotlib. Check coordinate ordering.
  - [ ] Generate new airfoil from a given airfoil using random noise.
  - [ ] Work on high, medium and low variance airfoil generation. Prepare directory structure.
  - [ ] Work on getting Xfoil working.
  - [ ] Prepare the data generation pipeline.
  - [ ] Use Xfoil to generate only valid airfoils.
  - [ ] Find and save the L/D ratio for generated airfoils.
  - [ ] Generate H, M, L variance airfoils. Look at plots to decide on noise levels.
  - [ ] Estimate time to generate a H, M, L data point on average.
  - [ ] See if data can be generated through multi-processing in parallel.
  - [ ] Decide H, M, L count depending on time taken and given that I am willing to spend D days on just data generation.
  - [ ] Need for using parametric representation and smooth surface airfoil generation after adding noise.
  - [ ] TE and LE at (1, 0) and (0, 0) and setting angle of attack of generated airfoils to 0.
- [ ] Work on neural network training pipeline.
  - [ ] Get data from dataset using Dataset and Dataloader in pytorch.
  - [ ] Define Neural Network class.
  - [ ] Define the hyperparameters.
  - [ ] Complete the training loop.
  - [ ] Work on checkpoints saving.
- [ ] Work on input optimization given the neural network.
  - [ ] Fix L/D
  - [ ] Change Loss function.
- [ ] Work on getting the entire neural network pipeline working.

### Move to Google Cloud Platform
- [ ] Move code to GCP. Connect to GCP via VSCode and SSH.
- [ ] Make sure all code works on GCP.
- [ ] Generate actual data.
- [ ] Decide where to store the data.
- [ ] Train the neural network.
- [ ] Run the two optimization processes.