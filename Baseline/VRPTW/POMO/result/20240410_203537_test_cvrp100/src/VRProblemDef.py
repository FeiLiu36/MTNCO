
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand

def get_random_problems_VRPTW(batch_size, problem_size): # mix tsp and cvrp, tsp has the same form as cvrp with zero demand

    # for test plot only
    seed = 2
    torch.manual_seed(seed)

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 70
    else:
        raise NotImplementedError
        
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)



    node_serviceTime = torch.rand(size=(batch_size, problem_size)) * 0.05 +0.55
    # shape: (batch, problem)
    # range: (0.15,0.2) for T=4.6


    node_lengthTW = torch.rand(size=(batch_size, problem_size)) * 0.05 +0.55
    # shape: (batch, problem)
    # range: (0.15,0.2) for T=4.6 
    # difference TW lengths (0.0,0.05),(0.05,0.1),(0.1,0.15),(0.15,0.2),(0.2,0.25)

    d0i = ((node_xy - depot_xy.expand(size=(batch_size,problem_size,2)))**2).sum(2).sqrt()
    # shape: (batch, problem)


    ei = torch.rand(size=(batch_size, problem_size)).mul((torch.div((4.6*torch.ones(size=(batch_size, problem_size)) - node_serviceTime - node_lengthTW),d0i) - 1)-1)+1
    # shape: (batch, problem)
    # default velocity = 1.0

    node_earlyTW = ei.mul(d0i)
    # shape: (batch, problem)
    # default velocity = 1.0

    node_lateTW = node_earlyTW + node_lengthTW
    # shape: (batch, problem)



    return depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_serviceTime



def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data