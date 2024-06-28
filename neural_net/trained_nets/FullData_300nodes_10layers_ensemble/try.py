from os import listdir


path = 'checkpoints3/'
filenames = listdir(path)


my_list = [name.split('_') for name in filenames]

# for list in my_list:
#     print(list)


my_new_list = []

for list in my_list:
    J_val = float(list[6][:-4])
    a = [list[0], list[1], list[2], list[3], list[4], list[5], J_val]
    my_new_list.append(a)


# for new_list in my_new_list:
#     print(new_list)

sorted_data = sorted(my_new_list, key = lambda x: x[-1])
# for sorted_list in sorted_data:
#     print(sorted_list)

n = 40
first_n = sorted_data[:n]
for l in first_n:
    print(l)