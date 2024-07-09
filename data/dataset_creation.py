import torch


def dataset_creation(tokens, config, seq_len):
    x = []
    y = []
    temp = torch.zeros(seq_len, dtype=torch.int64)

    for i in range(len(tokens) - 1):
        x.append(temp)
        y.append(tokens[i])
        temp = temp[1:]
        temp = torch.cat((temp, torch.tensor([tokens[i]], dtype=torch.int64)))


    # print(x)
    # print(y)
    # exit()
    x = torch.stack(x).to(torch.int64)
    y = torch.tensor(y).to(torch.int64)
    data_dir = config["data_dir"]
    torch.save((x, y), f"{data_dir}/data.pt")
    print("Total tokens:", len(tokens))
    return x, y
