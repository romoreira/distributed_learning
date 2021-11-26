import socket
import pickle
import time
import threading
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset  
from model import VGG, ClassNet, CIFARNet
from dataset import ImageDataset
import sys
import argparse
from random import randint


class FedClient(object):

    def __init__(self, server, port, scenario=0, dataset='MNIST'):
        self.local_model = ClassNet()
        # self.local_model = CIFARNet()
        self.host = server
        self.port = port
        self.attack = scenario
        self.evaluator = ImageDataset(dataset)
        self.train_loader, self.test_loader = self.evaluator.load_data()


    def connect(self):
        '''
        Sets connection to listening server
        '''
        client = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        print("Socket Created.\n")

        try:
            client.connect((self.host, self.port))
            print("Successful Connection to the Server.\n")
        except socket.error as e:
            print("Error Connecting to the Server: {msg}".format(msg=e))
            client.close()
            print("Socket Closed.")

        print("Receiving global model updates from chief")
        iteration = 1
        while True:
            received_data = self.recv_packets(the_socket=client, 
                                        buffer_size=4096, 
                                        timeout=10)

            if len(received_data) > 0:
                pickle_data = pickle.loads(received_data)

                self.process_server_data(pickle_data, iteration) # process received data

                # Send local updates to server
                data = {"sub": "model", "it":iteration, "data": self.local_model.state_dict()}
                data_byte = pickle.dumps(data)
                client.sendall(data_byte)
                print("Local model updates successfully sent to the chief.")

                iteration += 1

        client.close()
        print("Socket Closed.\n")


    def init_weights(self, m):
        '''
        Initialize model parameters from a uniform distribution
        '''
        if type(m) == nn.Linear:
            # torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.normal_(m.weight, mean=0, std=1)
            # torch.nn.init.uniform_(tensor, a=0, b=1)
            m.bias.data.fill_(0.01)


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


    def process_server_data(self, received_data, iteration):
        '''
        This function receives global model from chief
        Updates the local model by the global model
        '''
        subject = received_data["sub"]
        if subject == "model":
            model_updates = received_data["data"]
        elif subject == "done":
            print("Model is trained.")
        else:
            print("Unrecognized message type.")

        print("Updating local model with average global weights ... \n")
        self.local_model.load_state_dict(model_updates) # Update the client model with global updates 

        print("Begin training of local model ... \n")
        if self.attack==0:
            self.evaluator.train_model(self.local_model, self.train_loader, iteration=iteration)
        elif self.attack==1:
            print("Training attack model 1")
            self.local_model.apply(self.init_weights) # apply normal weights permanently
            self.evaluator.train_attack(self.local_model, self.train_loader, iteration=iteration)
        elif self.attack==2:
            print("Training attack model 2")
            rnd = randint(0, 10)
            if (rnd % 2) == 0:      # when rnd is even, initialize with normal weights
                self.local_model.apply(self.init_weights)
                self.evaluator.train_attack(self.local_model, self.train_loader, iteration=iteration)
            else:
                self.evaluator.train_model(self.local_model, self.train_loader, iteration=iteration)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=int, nargs="?", default=0, help="Attack scenario")
    parser.add_argument("--data", type=str, nargs="?", default='MNIST', help="Dataset")
    args = parser.parse_args()

    # cl = FedClient(server="host.docker.internal", port=5000, scenario=args.attack, dataset=args.data)
    cl = FedClient(server="localhost", port=5000, scenario=args.attack, dataset=args.data)
    cl.connect()