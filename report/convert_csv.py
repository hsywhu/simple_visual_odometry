with open('result_gpu.csv') as infile, open('result_gpu_1.csv', 'w') as outfile:
    for line in infile:
        outfile.write(" ".join(line.split()).replace(' ', ','))
        outfile.write("\n") # trailing comma shouldn't matter
