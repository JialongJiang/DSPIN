import numpy as np

num_spin = 30

net_class = [[14, 15, 19, 16, 18, 17], [23, 22, 24, 25], [13, 10, 11, 12],  [28, 26, 27, 29, 0], [9, 3, 2, 4], [7, 8, 21], [6, 1, 5], [20]]

spin_order = [spin for cur_list in net_class for spin in cur_list]
net_class_len = [len(cur_list) for cur_list in net_class]

spin_name = ['Antigen presentation', 'Inflammatory cytokines', 'M2 macrophage', 'Monocyte', 'Myeloid growth', 'Metallothionein', 'Pathogen response', 'Inflammatory macrophage', 'Complement/chemokine', 'Myeloid cell', 'T-cell cytotoxicity', 'Effector CD8 T-cell', 'NK cell', 'Granzyme K T-cell', 'Naive T-cell', 'T-cell maintenance', 'Lymphocyte', 'T-cell', 'Resting maintainence', 'Housekeeping', 'Stress response', 'Dendritic cell', 'Histone', 'T-cell activation', 'Regulatory T-cell', 'T-cell exhaustion', 'B-cell activation', 'Naive B-cell', 'B-cell', 'Resting B-cell']

spin_pname = ['P' + str(spin_order.index(ii) + 1) for ii in range(num_spin)]
spin_name_extend = ['P' + str(spin_order.index(ii) + 1) + '-' + spin_name[ii] for ii in range(num_spin)]