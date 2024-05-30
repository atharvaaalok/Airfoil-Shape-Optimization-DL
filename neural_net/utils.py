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