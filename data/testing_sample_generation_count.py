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


# Print statistics for each
print('original_LV')
b = np.array(original_LV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
print()

print('original_MV')
b = np.array(original_MV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
print()


print('original_HV')
b = np.array(original_HV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
print()

print('original_MV_LV')
b = np.array(original_MV_LV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
print()

print('original_HV_LV')
b = np.array(original_HV_LV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
print()

print('original_HV_MV')
b = np.array(original_HV_MV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
print()

print('original_HV_MV_LV')
b = np.array(original_HV_MV_LV)
print('average', np.mean(b))
print('max    ', np.max(b))
print('max 10 ', np.flip(np.sort(b))[:10])
print()