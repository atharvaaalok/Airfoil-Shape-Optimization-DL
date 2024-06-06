import random

# Get a list of all airfoil names
with open('airfoil_database/airfoil_names.txt', 'r') as f:
    # Remove the trailing \n from each line
    airfoil_names = f.readlines()
    total_airfoils = len(airfoil_names)


all_airfoils_set = set(airfoil_names)

# Select the dev-test samples 335 in total out of a total of 1235 airfoils
total_dev_samples = 170
total_test_samples = 165
total_dev_test_samples = total_dev_samples + total_test_samples
# Get dev test combined samples
dev_test_set = set(random.sample(list(all_airfoils_set), total_dev_test_samples))
# Get the other airfoils into the training set
train_set = all_airfoils_set - dev_test_set

# Split the dev test combined samples
dev_set = set(random.sample(list(dev_test_set), total_dev_samples))
test_set = dev_test_set - dev_set

# Make list of all airfoils
train_set = sorted(list(train_set))
dev_set = sorted(list(dev_set))
test_set = sorted(list(test_set))

# Write the train, dev and test airfoil names to corresponding files
with open('airfoil_database/airfoil_train.txt', 'w') as f:
    f.writelines(train_set)

with open('airfoil_database/airfoil_dev.txt', 'w') as f:
    f.writelines(dev_set)

with open('airfoil_database/airfoil_test.txt', 'w') as f:
    f.writelines(test_set)