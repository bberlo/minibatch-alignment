techniques = ["std", "domain_class", "wigrunt", "minibatch", "fido"]
dataset = ["widar", "signfi"]
dataset_types = ["dfs", "gaf"]

backbones = ["efficientnet", "resnet", "vgg"]
domain_type = {"widar": ["user", "position", "orientation"], "signfi": ["user", "environment"]}

# techniques = ['fido']
dataset = ["widar"]
domain_type["widar"] = ["orientation"]


def print_bash_hpc(extra_suffix:str):
    count = 0
    for techn in techniques:
        print(f"# Domain techniques: {techn}")
        for data in dataset:
            for type in dataset_types:
                for back in backbones[:1]:
                    for domain in domain_type[data]:
                        count += 1
                        print(f"./run_experiment_slurm.sh {techn} testing-leave-out 0 {data} {type} {back} {domain}{extra_suffix}")
        print("\n")

    print("total: ", count)


def print_bash_snellius():
    count = 0
    for techn in techniques:
        print(f"# Domain techniques: {techn}")
        for data in dataset:
            for type in dataset_types:
                for back in backbones[:1]:
                    for domain in domain_type[data]:
                        count += 1
                        print(f"./run_experiment_snellius.sh {techn} testing-leave-out 0 {data} {type} {back} {domain}")
        print("\n")

    print("total: ", count)


def print_cancel():
    for job in range(11362565, 11362569 + 1):
        print(f"scancel {job}")


# print_bash_snellius()
print_bash_hpc(' 2')
