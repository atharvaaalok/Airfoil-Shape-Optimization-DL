import numpy as np


fname = 'original_LV'
original_LV = np.loadtxt(f'generated_airfoils/train/{fname}.log')

fname = 'original_MV'
original_MV = np.loadtxt(f'generated_airfoils/train/{fname}.log')

fname = 'original_HV'
original_HV = np.loadtxt(f'generated_airfoils/train/{fname}.log')

fname = 'original_MV_LV'
original_MV_LV = np.loadtxt(f'generated_airfoils/train/{fname}.log')

fname = 'original_HV_LV'
original_HV_LV = np.loadtxt(f'generated_airfoils/train/{fname}.log')

fname = 'original_HV_MV'
original_HV_MV = np.loadtxt(f'generated_airfoils/train/{fname}.log')

fname = 'original_HV_MV_LV'
original_HV_MV_LV = np.loadtxt(f'generated_airfoils/train/{fname}.log')


generated_samples = 0

# Print statistics for each
print('original_LV')
b = np.array(original_LV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
generated_samples += np.sum(b)
print()

print('original_MV')
b = np.array(original_MV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
generated_samples += np.sum(b)
print()


print('original_HV')
b = np.array(original_HV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
generated_samples += np.sum(b)
print()

print('original_MV_LV')
b = np.array(original_MV_LV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
generated_samples += np.sum(b)
print()

print('original_HV_LV')
b = np.array(original_HV_LV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
generated_samples += np.sum(b)
print()

print('original_HV_MV')
b = np.array(original_HV_MV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
generated_samples += np.sum(b)
print()

print('original_HV_MV_LV')
b = np.array(original_HV_MV_LV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
generated_samples += np.sum(b)
print()


N = 900
L, M, H = 5, 5, 5
D = N * (1 + L + M + H + M * L + H * L + H * M + H * M * L)
print('Actual samples   :', D)
generated_samples += N
print('Generated samples: ', generated_samples)