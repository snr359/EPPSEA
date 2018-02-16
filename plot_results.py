import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import csv

# this script plots the results in a csv file, and saves the plot
if __name__ == '__main__':
    csv_file_path = sys.argv[1]
    save_location = sys.argv[2]
    symlog = sys.argv[3]
    functions = sys.argv[4:]

    if symlog == '1':
        plt.yscale('symlog')
    with open(csv_file_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        rows = []
        for line in reader:
            rows.append(float(n) for n in line)

        for i in range(len(rows)//2):
            x = list(rows[2*i])
            y = list(rows[2*i+1])
            function_name = functions[i]

            plt.plot(x, y, label=function_name)


    plt.xlabel('Evaluations')
    plt.ylabel('Best Fitness')

    plt.legend()
    plt.savefig('{0}/figure.png'.format(save_location))