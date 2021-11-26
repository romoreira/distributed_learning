import socket
import pickle
import time
import threading
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset  
from model import VGG, ClassNet, CIFARNet
from dataset import ImageDataset, CSVWriter
import csv
import threading



class FedServerThread(threading.Thread):
    
    
    def __init__(self, connection, client_info, global_model, queue, data_loader, clients, evaluator=None, \
                 iteration=None, delta=None, lock=None, dictionary=None, scenario=0, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout
        self.client_models = []
        self.evaluator = evaluator
        self.queue = queue
        self.clients = clients
        self.isStart = False
        self.global_model = global_model
        self.test_loader = data_loader
        self.scenario = scenario
        self.iteration = iteration
        self.lock = lock
        self.delta = delta
        self.shared_dict = dictionary
        self.file = open("logs.txt", "a") 
        self.startup()


    def startup(self):
        '''
        Performs initialized activities
        '''
        self.update_dictionary(self.client_info, 0.0)


    # function to process recieved data
    def recv_packets(self, the_socket, buffer_size=8192, timeout=2):
        #make socket non blocking
        the_socket.setblocking(0)
        
        #total data partwise in an array
        total_data=[];
        data='';
        
        #beginning time
        begin=time.time()
        while 1:
            #if you got some data, then break after timeout
            if total_data and time.time()-begin > timeout:
                break
            
            #if you got no data at all, wait a little longer, twice the timeout
            elif time.time()-begin > timeout*2:
                break
            
            #recv something
            try:
                data = the_socket.recv(buffer_size)
                if data:
                    total_data.append(data)
                    #change the beginning time for measurement
                    begin=time.time()
                else:
                    #sleep for sometime to indicate a gap
                    time.sleep(0.1)
            except:
                pass
        
        #join all parts to make final string
        return b''.join(total_data)


    def init_weights(self, m):
        '''
        Initialize model parameters from a uniform distribution
        '''
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def compute_aggregates(self, client_models):
        '''
        This function has aggregation method 'mean'
        '''
        global_dict = self.global_model.state_dict()

        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i][k].float() for i in range(len(client_models))], 0).mean(0)
        
        return global_dict


    def update_dictionary(self, key, value):
        self.lock.acquire()
        try:
            self.shared_dict[key] = value
        finally:
            self.lock.release()


    def get_dictionary(self, key):
        result = self.shared_dict.get(key)
        return float(result)


    def run(self):
        
        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:

            #######################################################
            ###### Broadcast global model to workers ##############
            #######################################################
            if self.isStart == False:
                data = {"sub": "model", "data": self.global_model.state_dict() }
                response = pickle.dumps(data)
                self.connection.sendall(response)
                
                self.isStart = True


            #######################################################
            ###### Receiving  data from the clients ###############
            #######################################################
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
            
            # receive packets
            packets_rcv = self.recv_packets(the_socket=self.connection, 
                                        buffer_size=4096, 
                                        timeout=10)
            if len(packets_rcv) > 0:
                received_data = pickle.loads(packets_rcv)

                # process received data
                if (type(received_data) is dict):
                    if (("data" in received_data.keys()) and ("sub" in received_data.keys())):
                        subject = received_data["sub"]
                        if subject == "model":
                            received_model = received_data["data"]
                            self.queue.put(received_model)
                            with self.lock:
                                self.iteration = received_data["it"]
                            
                
                if self.queue.qsize() == len(self.clients):
                    print("Processing ...")

                    recv_models = list(self.queue.queue)
                    with self.queue.mutex:
                        self.queue.queue.clear()

                    ##############################################
                    # +++++++++++ DEFENCE MECHANISM  +++++++++++ #
                    ##############################################
                    file = open("logs.txt", "a") 

                    # implement delta (0, 10, 40)
                    # if self.iteration >= 40:
                    #     self.scenario = 1

                    if self.scenario == 1:
                        valid_models = []
                        print("Implementing defence algorithm on federated averages.")
                        temp_model = ClassNet()
                        for each_model in recv_models:
                            temp_model.load_state_dict(each_model)
                            curr_acc = self.evaluator.test_model(temp_model, self.test_loader)
                            prev_acc = self.get_dictionary(self.client_info)
                            
                            if prev_acc < curr_acc:
                                valid_models.append(each_model)
                            else:
                                pass

                            # update accuracy of each worker
                            self.update_dictionary(self.client_info, curr_acc)
                        
                        print('{} of {} models validated for defence strategy at delta {}'.format(len(valid_models), len(recv_models), self.iteration))
                        fedavg_models = self.compute_aggregates(client_models=valid_models)
                        self.global_model.load_state_dict(fedavg_models)
                        global_acc, precision, recall = self.evaluator.test_precision(self.global_model, self.test_loader)
                        print("Accuracy on iteration {} is {}%".format(self.iteration, global_acc))
                        file.write('{}, {}, {}, {} \n'.format(self.iteration, global_acc, precision, recall))
                    else:
                        print("Aggregating with no defence strategy")
                        fedavg_models = self.compute_aggregates(client_models=recv_models)
                        self.global_model.load_state_dict(fedavg_models)
                        global_acc = self.evaluator.test_model (self.global_model, self.test_loader)
                        print("Accuracy on iteration {} is {}%".format(self.iteration, global_acc))
                        file.write('{}, {}, {}, {} \n'.format(self.iteration, global_acc, 0.0, 0.0))

                    file.close()

                    print("Sending new global weights to clients")
                    for cl in self.clients:
                        data = {"sub": "model", "data": self.global_model.state_dict() }
                        response = pickle.dumps(data)
                        cl.sendall(response)

                    
                
                    


