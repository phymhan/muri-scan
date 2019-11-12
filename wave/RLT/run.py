import os

kernel_sizes = [11, 21, 31, 41]
init_methods = ['normal, xavier, kaiming, orthogonal'] 
activations  = ['relu', 'lrelu', 'prelu', 'elu', 'softplus']

#for init in ['xavier, kaiming, orthogonal']:
for activation in ['prelu', 'elu']:
	for L in [11, 21, 31, 41]:
  		os.system("python train.py --init_method=orthogonal --activation {} --kernel_size {}".format(activation, L))