import pandas as pd

domain_list = ["Art", "Clipart", "Product", "RealWorld"]

for domain_name in domain_list:
    domain_disc_dict = {label: 0 for label in range(65)}
    for line in pd.read_csv("%s_%s.csv" % (domain_name, domain_name), header=None).values:
        domain_disc_dict[int(line[-1])] += 1
    record_file = open("data_detail/%s.txt" % domain_name, mode="w")
    for k, v in domain_disc_dict.items():
        print("%d" % (v), file=record_file)
        pass