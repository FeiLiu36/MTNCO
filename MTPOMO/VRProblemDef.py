
import torch
import numpy as np


###-----
# if problem_type is 'unified', it is trained on 20% CVRP, 20% OVRP, 20% VRPB, 20% VRPTW, 20% VRPL
###----

def get_random_problems_mixed(batch_size, problem_size, problem_type):


    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)


    # if size > 50, demand_scaler = 30 + size/5
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200: 
        demand_scaler = 70
    elif problem_size == 500:
        demand_scaler = 130
    elif problem_size == 1000:
        demand_scaler = 230
    else:
        raise NotImplementedError
        
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)


    node_serviceTime = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)
    # zeros 

    node_lengthTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    node_earlyTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)
    # default velocity = 1.0

    node_lateTW = node_earlyTW + node_lengthTW
    # shape: (batch, problem)

    route_length_limit = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    route_open = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    seed = np.random.rand()
    if ((problem_type == 'unified' and seed>=0.2 and seed <0.4) or 'L' in problem_type): # problem_type is 'unified' or there is 'L' in the problem_type 
        if problem_size == 20:
            route_length_limit = 3.0*torch.ones(size=(batch_size, problem_size))
            # shape: (batch, problem)   
        elif problem_size == 50:
            route_length_limit = 3.0*torch.ones(size=(batch_size, problem_size))
            # shape: (batch, problem)   
        elif problem_size == 100:
            route_length_limit = 3.0*torch.ones(size=(batch_size, problem_size))
            # shape: (batch, problem)   

    if ((problem_type == 'unified' and seed>=0.4 and seed <0.6) or 'TW' in problem_type): # problem_type is 'unified' or there is 'TW' in the problem_type 

        node_serviceTime = torch.rand(size=(batch_size, problem_size)) * 0.05 +0.15
        # shape: (batch, problem)
        # range: (0.15,0.2) for T=4.6 

        node_lengthTW = torch.rand(size=(batch_size, problem_size)) * 0.05 +0.15
        # shape: (batch, problem)
        # range: (0.15,0.2) for T=4.6 

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

    if ((problem_type == 'unified' and seed>=0.6 and seed <=0.8) or 'O' in problem_type): # problem_type is 'unified' or there is 'O' in the problem_type 

        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
        # shape: (batch, problem)

        route_open = torch.ones(size=(batch_size, problem_size))
        # shape: (batch, problem)   

    if ((problem_type == 'unified' and seed>=0.8) or 'B' in problem_type): # problem_type is 'unified' or there is 'B' in the problem_type 

        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
        # shape: (batch, problem)
        linehaul = int(0.8*problem_size)
        node_demand[:,linehaul:] = -node_demand[:,linehaul:]
        # shape: (batch, problem)

    return depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_serviceTime, route_open, route_length_limit



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