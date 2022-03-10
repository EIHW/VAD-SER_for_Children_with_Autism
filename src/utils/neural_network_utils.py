import torch
from scipy.spatial.distance import cosine

def calcualate_cosine_distance(layer1, layer2):
    layer1 = torch.flatten(layer1).cpu().detach().numpy()
    layer2 = torch.flatten(layer2).cpu().detach().numpy()
    return cosine(layer1, layer2)
    #return 1 - torch.dot(layer1, layer2)/(torch.norm(layer1)*torch.norm(layer2))

def calculate_cosine_distance_CNN_layers(model1, model2):
    names = []
    cosine_distances = []
    i = 1
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(),model2.named_parameters()):
        if ".conv1" in name1 or ".conv2" in name2:
            names.append("c{}".format(i))
            cosine_distances.append(calcualate_cosine_distance(param1, param2).item())
            i += 1
    return names, cosine_distances
