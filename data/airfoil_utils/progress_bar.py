import sys


def print_progress_bar(iteration, total_iterations, length = 20):
    progress = int((iteration / (total_iterations - 1)) * length)
    print_bar = '#' * progress + '-' * (length - progress)
    percent_completed = int((iteration / (total_iterations - 1)) * 100)
    sys.stdout.write(f'\r Progress: [{print_bar}] {percent_completed}%')
    sys.stdout.flush()
    if iteration == (total_iterations - 1):
        print()