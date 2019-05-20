# Experiments
1. GPU CPU speedup: Var: img_size, batch_size, e1 network, 3 episodes, 3 runs
2. GPU CPU speedup:  e2 network bigger than e1, 3 episodes, 3 runs
3. GPU CPU speedup:  e2 network, 3 episodes, 3 runs, pin memory in dataloader off
4. Compare random background vs default.
5. Squeezenet speedup GPU, 3 runs 3 episodes, batchsize 256 due to cuda memory errors at512
RuntimeError: CUDA out of memory. 
Tried to allocate 546.75 MiB (GPU 0; 4.00 GiB total capacity; 2.47 GiB already allocated; 437.99 MiB free; 1.18 MiB cached)


federated speed - 50 episodes SGD lr e-1, momentum=0.75, 2-4 worker, 3 runs

speedup_kd_tree 200x200 num_points = [10, 33, 100, 333]


