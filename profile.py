import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--metric', action='store_true', help='get the metric of major kernels')
args = parser.parse_args()


metrics = ['flop_count_sp', 'flop_count_sp_special']#'all']
kernels = ['fusedL2Normv2', 'fusedL2Normb', 'bn_fw', 'bn_bw']


for i in range(4):
    if args.metric:
        cmd = "nvprof -o compare%d_m.nvvp -f --profile-from-start off --metrics all" % i

    else:
        cmd = "nvprof -o compare%d_selu.nvvp -f --profile-from-start off" % i
    
    cmd += ' python benchmark.py --bm %d' % i
    
    print(cmd)
    os.system(cmd)
    