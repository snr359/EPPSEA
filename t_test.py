# This script reads a csv file of two samples and performs a two-tailed t-test on them

import scipy.stats
import csv
import statistics
import sys

def main(csvFilePath):
    with open(csvFilePath, newline='') as csvFile:
        reader = csv.reader(csvFile)
        readerLines = list(reader)
        sample1 = list(float(s) for s in readerLines[0])
        sample2 = list(float(s) for s in readerLines[1])
        sample1Mean = statistics.mean(sample1)
        sample2Mean = statistics.mean(sample2)
        t, p = scipy.stats.ttest_rel(sample1, sample2)

        print(sample1Mean, sample2Mean, t, p)

if __name__ == '__main__':
    csvFilePath = sys.argv[1]
    main(csvFilePath)