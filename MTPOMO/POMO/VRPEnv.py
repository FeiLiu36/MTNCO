
from dataclasses import dataclass
import torch

from MTPOMO.VRProblemDef import get_random_problems_mixed, augment_xy_data_by_8_fold


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_earlyTW: torch.Tensor = None
    # shape: (batch, problem)
    node_lateTW: torch.Tensor = None
    # shape: (batch, problem)
    # route_open: torch.Tensor = None
    # # shape: (batch, problem)
    # length: torch.Tensor = None
    # # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    time: torch.Tensor = None
    # shape: (batch, pomo)
    route_open: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class VRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.problem_type = env_params['problem_type']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_earlyTW = None
        # shape: (batch, problem+1)
        self.depot_node_lateTW = None
        # shape: (batch, problem+1)
        self.depot_node_servicetime = None
        # shape: (batch, problem+1)
        self.length = None
        # shape: (batch, pomo)

        ##################################
        self.attribute_c = False
        self.attribute_tw = False
        self.attribute_o = False
        self.attribute_b = False # currently regard as CVRP with negative demand 
        self.attribute_l = False


        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.time = None
        # shape: (batch, pomo)
        self.route_open= None
        # shape: (batch, pomo)
        self.length= None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_node_earlyTW = loaded_dict['node_earlyTW']
        self.saved_node_lateTW = loaded_dict['node_lateTW']
        self.saved_node_servicetime = loaded_dict['node_serviceTime']
        self.saved_route_open = loaded_dict['route_open']
        self.saved_route_length = loaded_dict['route_length_limit']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_servicetime, route_open, route_length_limit = get_random_problems_mixed(batch_size, self.problem_size, self.problem_type)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            node_earlyTW = self.saved_node_earlyTW[self.saved_index:self.saved_index+batch_size]
            node_lateTW = self.saved_node_lateTW[self.saved_index:self.saved_index+batch_size]
            node_servicetime = self.saved_node_servicetime[self.saved_index:self.saved_index+batch_size]
            route_open = self.saved_route_open[self.saved_index:self.saved_index+batch_size]
            route_length_limit = self.saved_route_length[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                node_earlyTW= node_earlyTW.repeat(8,1)
                node_lateTW = node_lateTW.repeat(8,1)
                node_servicetime = node_servicetime.repeat(8,1)
                route_open = route_open.repeat(8,1)
                route_length_limit = route_length_limit.repeat(8,1)
            else:
                raise NotImplementedError
        
        self.route_open = route_open
        # shape: (batch,pomo)
        self.length = route_length_limit
        # shape: (batch,pomo)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        depot_earlyTW = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        depot_lateTW = 4.6*torch.ones(size=(self.batch_size, 1)) # the lenght of time windows should be normalized into [0,1] not 4.6
        # shape: (batch, 1)
        depot_servicetime = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_earlyTW = torch.cat((depot_earlyTW, node_earlyTW), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_lateTW = torch.cat((depot_lateTW, node_lateTW), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_servicetime = torch.cat((depot_servicetime, node_servicetime), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_earlyTW = node_earlyTW
        self.reset_state.node_lateTW = node_lateTW
        

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        if (node_demand.sum()>0):
            self.attribute_c = True
        else:
            self.attribute_c = False
        if (node_lateTW.sum()>0):
            self.attribute_tw = True
        else:
            self.attribute_tw = False
        if (route_open.sum()>0):
            self.attribute_o = True
        else:
            self.attribute_o = False
        if (route_length_limit.sum()>0):
            self.attribute_l = True
        else:
            self.attribute_l = False


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.time = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.length = 3.0*torch.ones(size=(self.batch_size, self.pomo_size))
        # # shape: (batch, pomo)
        self.route_open = torch.zeros((self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time = self.time
        self.step_state.route_open = self.route_open
        self.step_state.length = self.length.clone()

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################

        self.at_the_depot = (selected == 0)

        #### update load information ###

        demand_list = self.depot_node_demand[:, None, :].expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot


        #### mask nodes if load exceed ###

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.000001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)
        
        #### update time&distance information ###

        servicetime_list = self.depot_node_servicetime[:, None, :].expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        selected_servicetime = servicetime_list.gather(dim=2,index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        earlyTW_list = self.depot_node_earlyTW[:, None, :].expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        selected_earlyTW = earlyTW_list.gather(dim=2,index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)


        xy_list = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1,-1)
        # shape: (batch, pomo, problem+1, 2)
        gathering_index = selected[:, :, None,None].expand(-1,-1,-1,2)
        # shape: (batch, pomo, 1, 2)
        selected_xy = xy_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo, 2)

        if self.selected_node_list.size()[2] == 1:
            gathering_index_last = self.selected_node_list[:, :, -1][:,:,None,None].expand(-1,-1,-1,2)
            # shape: (batch, pomo, 1,2)
        else:
            gathering_index_last = self.selected_node_list[:, :, -2][:,:,None,None].expand(-1,-1,-1,2)
            # shape: (batch, pomo, 1,2)           
        last_xy = xy_list.gather(dim=2, index=gathering_index_last).squeeze(dim=2)
        # shape: (batch, pomo, 2)
        selected_time = ((selected_xy - last_xy)**2).sum(dim=2).sqrt()
        # shape: (batch, pomo)


        # update time window attribute if it is used
        if (self.attribute_tw):
            #print(selected_time)
            #selected_time += selected_servicetime
            self.time = torch.max((self.time + selected_time), selected_earlyTW)
            self.time += selected_servicetime
            # shape: (batch, pomo)
            self.time[self.at_the_depot] = 0 # refill time at the depot

            time_to_next = ((selected_xy[:,:,None,:].expand(-1,-1,self.problem_size+1,-1) - xy_list)**2).sum(dim=3).sqrt()
            # shape: (batch, pomo, problem+1)
            # time_to_depot = ((xy_list[:,:,0,:].expand(-1,-1,self.problem_size+1,-1)  - xy_list)**2).sum(dim=3).sqrt()
            # shape: (batch, pomo, problem+1)
            time_too_late = self.time[:, :, None] + time_to_next > self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size, -1)
            # shape: (batch, pomo, problem+1)
            time_too_late[self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size, -1) == 0]= 0 
            # unmask the the zero late TW      

            self.ninf_mask[time_too_late] = float('-inf')
            # shape: (batch, pomo, problem+1)

        # update route duration (length) attribute if it is used
        if (self.attribute_l):
            self.step_state.length -= selected_time
            # shape: (batch, pomo)
            self.step_state.length[self.at_the_depot] = self.length[0][0] # refill length at the depot
            # shape: (batch, pomo)
            length_to_next = ((selected_xy[:,:,None,:].expand(-1,-1,self.problem_size+1,-1) - xy_list)**2).sum(dim=3).sqrt()
            # shape: (batch, pomo, problem+1)
            depot_xy = xy_list[:,:,0,:]
            next_to_depot =  ((depot_xy[:,:,None,:].expand(-1,-1,self.problem_size+1,-1)  - xy_list)**2).sum(dim=3).sqrt()
            # shape: (batch, pomo, problem+1)

            # if open attribute is used, the distance return to depot is not counted
            if self.attribute_o:
                length_too_small = self.step_state.length[:, :, None] - round_error_epsilon < length_to_next 
                # shape: (batch, pomo, problem+1)
            else:
                length_too_small = self.step_state.length[:, :, None] - round_error_epsilon < (length_to_next + next_to_depot )
                # print(self.step_state.length)
                # print(length_to_next + next_to_depot)
                # print("length_too_large",length_too_large)
            # shape: (batch, pomo, problem+1)
            self.ninf_mask[length_too_small] = float('-inf')
            self.ninf_mask[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot
            # shape: (batch, pomo, problem+1)


        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)


        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        # if the target problem is open route the distance return to depot will be set as 0       
        if self.attribute_o:
            segment_lengths[self.selected_node_list.roll(dims=2, shifts=-1)==0] = 0

        #print(segment_lengths[0][0])
        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def get_node_seq(self):

        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        return gathering_index,ordered_seq
