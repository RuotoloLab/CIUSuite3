import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm



def get_csv_data_file(dirpath):
    files = os.listdir(dirpath)
    files = [x for x in files if x.endswith('corrected.csv')]
    return files

def read_csv_file(files, dirpath):
    ccsoutfile = open(os.path.join(dirpath, file), 'r+')
    ccsoutfile = ccsoutfile.read().splitlines()

    fname = []
    ccs_pa = []
    ccs_tm_corrected = []

    for line in ccsoutfile:
        filename, ccspa, ccstmcoll = line.split(',')
        ccspa = float(ccspa)
        ccstmcoll = float(ccstmcoll)
        ccs_pa.append(ccspa)
        ccs_tm_corrected.append(ccstmcoll)
        fname.append(filename)


    return fname, ccs_pa, ccs_tm_corrected
    #

def make_ccs_distribution_plot(ccs, dirpath):
    plt.xlabel('CCS (Å²)')
    plt.ylabel('number of structures')
    plt.xlim(7000,10000)
    plt.ylim(0,1200)
    n, bins, patches = plt.hist(ccs, bins=15, color='red', alpha=0.5)
    # mu, sigma = norm.fit(ccs)
    # best_fit_line = norm.pdf(mu, sigma)
    # plt.plot(best_fit_line)
    plt.savefig(os.path.join(dirpath, 'corrected_ccs_distribution.pdf'))
    plt.close()

    with open(os.path.join(dirpath, "corrected_ccs_gaussian_info.csv"), "w") as csvfile:
        for item in zip(bins, n):
            csvfile.write('{}, {}\n'.format(item[0], item[1]))
        csvfile.close()

    # (mu, sigma) = norm.fit(ccs)
    # _, bins, _ = plt.hist(ccs)
    # y = norm.pdf(bins, mu, sigma)
    # l = plt.plot(bins, y)
    # plt.xlabel('CCS (Å²)')
    # plt.ylabel('number of structures')
    # plt.savefig(os.path.join(dirpath, 'corrected_ccs_distribution.pdf'))

if __name__ == '__main__':
    dirpath = r"\\lsa-research02.m.storage.umich.edu\lsa-research02\bruotolo\Mengdao\antibody\output_narrowfiltering\filtered_struct"
    ccs_files = get_csv_data_file(dirpath)
    for file in ccs_files:
        fname, ccs_pa, ccs_tm_corrected = read_csv_file(file, dirpath)
        make_ccs_distribution_plot(ccs_tm_corrected, dirpath)