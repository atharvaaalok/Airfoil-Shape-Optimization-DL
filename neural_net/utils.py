import torch


# Set colors that will be used to highlight the print output
color_end = '\033[0m' # Reset terminal color
red = '\033[0;31m'
cyan = '\033[0;36m'
green = '\033[0;92m'
yellow = '\033[0;93m'
blue = '\033[0;94m'


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set the learning rate for all parameter groups in an optimizer.
    
    Args:
        optimizer: The optimizer whose learning rate is to be set.
        learning_rate: The new learning rate to be set.
    """

    for group in optimizer.param_groups:
        group['lr'] = learning_rate


def print_net_performance(epochs: int, epoch: int, J_train: float, J_val: float = None) -> None:
    """Utility function for formatted printing of training and validation losses with epoch
    count.
    
    Args:
        epochs: Total epochs in the training run.
        epoch: Current training epoch.
        J_train: Training error.
        J_val: Validation error.
    """
    
    # Get number of digits in the number 'epochs' for formatted printing
    num_digits = len(str(epochs))

    # Print the current performance
    print_performance = (
        f'{red}Epoch:{color_end} [{epoch:{num_digits}}/{epochs}].  ' +
        f'{cyan}Train Cost:{color_end} {J_train:11.6f}.  '
    )
    if J_val is not None:
        print_performance += f'{cyan}Val Cost:{color_end} {J_val:11.6f}'
    
    print(print_performance)



def train_loop(X_train, Y_train, B, model, loss_fn, optimizer, verbose = False):

    N = X_train.shape[0]
    # Ensure B perfectly divides N
    assert N % B == 0, "B should perfectly divide N"

    # Generate a random permutation of indices from 0 to N-1
    perm = torch.randperm(N)
    # Index the tensor with the permutation to shuffle the rows
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # Figure out the number of batches and the batch indices
    num_batches = N // B

    idx = torch.arange(N)
    idx = idx.view(num_batches, B)
    # Convert each row of the reshaped tensor to a separate tensor and store them in a list
    batch_indices = [idx[i] for i in range(num_batches)]


    num_digits = len(str(num_batches))

    # Set the model to training mode
    model.train()

    # Run the training loop
    for batch in range(num_batches):
        X, Y = X_train[batch_indices[batch], :], Y_train[batch_indices[batch], :]

        # Run the forward pass
        Y_pred = model(X)

        # Compute the loss
        loss = loss_fn(Y_pred, Y)

        # Run the backward pass and calculate the gradients
        loss.backward()

        # Take an update step and then zero out the gradients
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            if batch % (num_batches // 5) == 0:
                loss = loss.item()
                print(f'{cyan}Train Loss:{color_end} [{batch + 1:{num_digits}}/{num_batches}] {loss:20.6f}')
    if verbose:
        print()


def dev_loop(X_val, Y_val, B, model, loss_fn, verbose = False):
    N = X_val.shape[0]
    # Ensure B perfectly divides N
    assert N % B == 0, "B should perfectly divide N"

    # Generate a random permutation of indices from 0 to N-1
    perm = torch.randperm(N)
    # Index the tensor with the permutation to shuffle the rows
    X_val = X_val[perm]
    Y_val = Y_val[perm]

    # Figure out the number of batches and the batch indices
    num_batches = N // B

    idx = torch.arange(N)
    idx = idx.view(num_batches, B)
    # Convert each row of the reshaped tensor to a separate tensor and store them in a list
    batch_indices = [idx[i] for i in range(num_batches)]


    num_digits = len(str(num_batches))
    test_loss = 0

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model with torch.no_grad() to ensure no gradients are computed
    with torch.no_grad():
        for batch in range(num_batches):
            X, Y = X_val[batch_indices[batch], :], Y_val[batch_indices[batch], :]

            # Run the forward pass
            Y_pred = model(X)

            # Compute the loss
            loss = loss_fn(Y_pred, Y)
            test_loss += loss.item()
    
    test_loss = test_loss / num_batches
    if verbose:
        print(f'{cyan}Valid Loss:{color_end} {test_loss:20.6f}')
        print()
    
    return test_loss


def train_loop_old(dataloader, model, loss_fn, optimizer, verbose = False):
    num_batches = len(dataloader)
    num_digits = len(str(num_batches))

    # Set the model to training mode
    model.train()

    # Run the training loop
    for batch, (X, Y) in enumerate(dataloader):
        # Run the forward pass
        Y_pred = model(X)

        # Compute the loss
        loss = loss_fn(Y_pred, Y)

        # Run the backward pass and calculate the gradients
        loss.backward()

        # Take an update step and then zero out the gradients
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            if batch % (num_batches // 5) == 0:
                loss = loss.item()
                print(f'{cyan}Train Loss:{color_end} [{batch + 1:{num_digits}}/{num_batches}] {loss:20.6f}')
    if verbose:
        print()


def dev_loop_old(dataloader, model, loss_fn, verbose = False):
    num_batches = len(dataloader)
    test_loss = 0

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model with torch.no_grad() to ensure no gradients are computed
    with torch.no_grad():
        for X, Y in dataloader:
            Y_pred = model(X)
            test_loss += loss_fn(Y_pred, Y).item()
    
    test_loss = test_loss / num_batches
    if verbose:
        print(f'{cyan}Valid Loss:{color_end} {test_loss:20.6f}')
        print()
    
    return test_loss