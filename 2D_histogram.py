import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm



def get_csv_data_file(dirpath):
    files = os.listdir(dirpath)
    files = [x for x in files if x.endswith('Summary_2.csv')]
    return files

def read_csv_file(file, dirpath):
    ccsoutfile = open(os.path.join(dirpath, file), 'r+')
    ccsoutfile = ccsoutfile.read().splitlines()

    fname = []
    ccs_solution = []
    ccs_gas = []

    for line in ccsoutfile:
        filename, ccs_solution_PA, ccs_solution_TM, ccs_gas_PA, ccs_gas_TM, compaction = line.split(',')
        ccs_solution_TM = float(ccs_solution_TM)
        ccs_gas_TM = float(ccs_gas_TM)
        ccs_solution.append(ccs_solution_TM)
        ccs_gas.append(ccs_gas_TM)
        fname.append(filename)


    return fname, ccs_solution, ccs_gas


def make_ccs_distribution_plot(ccs_solution, ccs_gas, dirpath):
    plt.xlabel('CCS (Å²)')
    plt.ylabel('number of structures')
    plt.xlim(7000,15000)
    plt.ylim(0,30)
    # n, bins, patches = plt.hist(ccs, bins=15, color='blue', alpha=0.75)
    # mu, sigma = norm.fit(ccs)
    # best_fit_line = norm.pdf(mu, sigma)
    # plt.plot(best_fit_line)
    plt.hist(ccs_solution, bins=15, color='blue', alpha=0.75)
    plt.savefig(os.path.join(dirpath, 'ccs_solution_dist.pdf'))
    plt.close()

    plt.xlabel('CCS (Å²)')
    plt.ylabel('number of structures')
    plt.xlim(7000, 15000)
    plt.ylim(0, 30)
    plt.hist(ccs_gas, bins=15, color='blue', alpha=0.75)
    plt.savefig(os.path.join(dirpath, 'ccs_gas_dist.pdf'))
    plt.close()

    # with open(os.path.join(dirpath, "corrected_ccs_gaussian_info.csv"), "w") as csvfile:
    #     for item in zip(bins, n):
    #         csvfile.write('{}, {}\n'.format(item[0], item[1]))
    #     csvfile.close()
    # all_ccs = [ccs_solution, ccs_gas]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for z in [100, 0]:
    #     hist = np.histogram(all_ccs, bins=15)
    #     xs = range(7000, 15000)
    #     ax.bar(xs, hist, zs=z, color='blue', alpha=0.75)
    #
    # ax.set_xlabel('CCS (Å²)')
    # ax.set_zlabel('number of structures')
    # plt.show()
    # plt.savefig(os.path.join(dirpath, "2d_ccs_histogram.pdf"))
    # plt.close()

if __name__ == '__main__':
    dirpath = r"\\lsa-research02.m.storage.umich.edu\lsa-research02\bruotolo\Mengdao\antibody"
    ccs_files = get_csv_data_file(dirpath)
    for file in ccs_files:
        fname, ccs_solution, ccs_gas = read_csv_file(file, dirpath)
        read_csv_file(file, dirpath)
        make_ccs_distribution_plot(ccs_solution, ccs_gas, dirpath)