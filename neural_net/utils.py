import torch


# Set colors that will be used to highlight the print output
color_end = '\033[0m' # Reset terminal color
red = '\033[0;31m'
cyan = '\033[0;36m'
green = '\033[0;92m'
yellow = '\033[0;93m'
blue = '\033[0;94m'



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
    train_loss = 0

    # Set the model to training mode
    model.train()

    print_cost_every = 1 if num_batches // 5 == 0 else num_batches // 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run the training loop
    for batch in range(num_batches):
        X, Y = X_train[batch_indices[batch], :], Y_train[batch_indices[batch], :]

        # Use BF16 for faster compute
        with torch.autocast(device_type = device, dtype = torch.bfloat16):
            # Run the forward pass
            Y_pred = model(X)

            # Compute the loss
            loss = loss_fn(Y_pred, Y)
        
        train_loss += loss.item()

        # Run the backward pass and calculate the gradients
        loss.backward()

        # Take an update step and then zero out the gradients
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            if batch % print_cost_every == 0:
                loss = loss.item()
                print(f'{cyan}Train Loss:{color_end} [{batch + 1:{num_digits}}/{num_batches}] {loss:20.6f}')
    
    train_loss = train_loss / num_batches
    if verbose:
        print(f'{cyan}Avg. Train Loss:{color_end} {train_loss:20.6f}')
        print()
    
    return train_loss


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model with torch.no_grad() to ensure no gradients are computed
    with torch.no_grad():
        for batch in range(num_batches):
            X, Y = X_val[batch_indices[batch], :], Y_val[batch_indices[batch], :]

            # Use BF16 for faster compute
            with torch.autocast(device_type = device, dtype = torch.bfloat16):
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