
import torch

import os
from logging import getLogger

from VRPEnv import VRPEnv as Env
from VRPModel import VRPModel as Model

from utils.utils import *


class VRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Output Result
        ###############################################                    
        # #print(state.current_node[0])
        # indexList, coordinateList = self.env.get_node_seq()
        # nodesID = torch.Tensor.cpu(indexList[0][0]).numpy()[:,0]
        # nodesCoordinate = torch.Tensor.cpu(coordinateList[0][0]).numpy()      
        # depotCoordinate = torch.Tensor.cpu(self.env.depot_node_xy[0][0]).numpy()
        # demandList = torch.Tensor.cpu(self.env.depot_node_demand[0]).numpy()[nodesID]
        # # earlyTWList = torch.Tensor.cpu(self.env.depot_node_earlyTW[0]).numpy()[nodesID]
        # # lateTWList = torch.Tensor.cpu(self.env.depot_node_lateTW[0]).numpy()[nodesID]
        # # servicetimeList = torch.Tensor.cpu(self.env.depot_node_servicetime[0]).numpy()[nodesID]
        # # print(servicetimeList)

        # print('demandlist',demandList)
        # #instanceName = "./CVRPInstances100/instance"+str(episode)
        # #self._plot_TSP(nodesCoordinate)
        # #input()
        # #solution,cost = self._call_split(nodesCoordinate,depotCoordinate,demandList)
        # #print(instanceName,' ',solution,' ',cost)
        # solution = []
        # #print(nodesID)
        
        # for i in range (nodesID.size):
        #     if nodesID[i] == 0:
        #         solution.append(i+1)
        # #print(solution)
        # self._plot_CVRP(nodesCoordinate,depotCoordinate,demandList,solution)
        # # input()
        # #self.splitCost += cost
        # #self._write_cvrplib(instanceName,nodesCoordinate,depotCoordinate,demandList)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()

    def _plot_TSP(self,nodesCoordinate):
        
        #print("nodesCoordinate = ",nodesCoordinate)
        
        plt.plot(nodesCoordinate[:,0],nodesCoordinate[:,1])
        plt.show()

    def _plot_CVRP(self,nodesCoordinate,depotCoordinate,demands,result):
        
        #print("nodesCoordinate = ",nodesCoordinate)
        
        plt.scatter(depotCoordinate[0],depotCoordinate[1],marker='*',s=160,c='r')
        
        for i in range(len(result)-1):
            if (self.env.problem_type == "OVRP" or self.env.problem_type == "OVRPTW" ):
                xlist = nodesCoordinate[int(result[i]-1):int(result[i+1]-1),0]
                ylist = nodesCoordinate[int(result[i]-1):int(result[i+1]-1),1]
                demandlist = demands[int(result[i]-1):int(result[i+1]-1)]
                # earlytwlist = earlytw[int(result[i]-1):int(result[i+1]-1)]
                # latetwlist = latetw[int(result[i]-1):int(result[i+1]-1)]

            else:
                xlist = nodesCoordinate[int(result[i]-1):int(result[i+1]),0]
                ylist = nodesCoordinate[int(result[i]-1):int(result[i+1]),1]
                demandlist = demands[int(result[i]-1):int(result[i+1])]
                # earlytwlist = earlytw[int(result[i]-1):int(result[i+1])]
                # latetwlist = latetw[int(result[i]-1):int(result[i+1])]
            # timecurrent = 0
            # visittimelist = np.zeros(xlist.size)
            # for j in range(xlist.size):
            #     if j >0:
            #         visittimelist[j] = timecurrent+((xlist[j]-xlist[j-1])**2+(ylist[j]-ylist[j-1])**2)**0.5
            #         visittimelist[j] = max(visittimelist[j],earlytwlist[j])
            #         timecurrent = visittimelist[j]+servicetime[j]

            #print(earlytwlist)
            # for x,y,d,e,l,v in zip(xlist,ylist,demandlist,earlytwlist,latetwlist,visittimelist):
            #    plt.text(x,y,'(%.2f, %.2f, %.2f, %.2f)' % (d,e,l,v),fontdict={'fontsize':10})
            plt.plot(xlist,ylist,marker='o')

        if (self.env.problem_type == "OVRP" or self.env.problem_type == "OVRPTW" ):
            xlist = nodesCoordinate[int(result[len(result)-1]-1):self.env.problem_size-1,0]
            ylist = nodesCoordinate[int(result[len(result)-1]-1):self.env.problem_size-1,1]
        else:
            xlist = nodesCoordinate[int(result[len(result)-1]-1):self.env.problem_size-1,0]
            ylist = nodesCoordinate[int(result[len(result)-1]-1):self.env.problem_size-1,1]
        # demandlist = demands[int(result[len(result)-1]-1):self.env.problem_size]
        # earlytwlist = earlytw[int(result[len(result)-1]-1):self.env.problem_size]
        # latetwlist = latetw[int(result[len(result)-1]-1):self.env.problem_size]
        # timecurrent = 0
        # for j in range(xlist.size):
        #     if j >0:
        #         visittimelist[j] = timecurrent+((xlist[j]-xlist[j-1])**2+(ylist[j]-ylist[j-1])**2)**0.5
        #         visittimelist[j] = max(visittimelist[j],earlytwlist[j])
        #         timecurrent = visittimelist[j]+servicetime[j]        
        # for x,y,d,e,l,v in zip(xlist,ylist,demandlist,earlytwlist,latetwlist,visittimelist):
        #     plt.text(x,y,'(%.2f, %.2f, %.2f, %.2f)' % (d,e,l,v),fontdict={'fontsize':10})

        # for x,y,d in zip(xlist,ylist,demandlist):
        #     plt.text(x,y,'%.0f' % d,fontdict={'fontsize':14})
        plt.plot(xlist,ylist,marker='d')
        plt.axis('off')
        plt.show()