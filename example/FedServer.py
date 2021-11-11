import socket
import pickle
import time
import threading
from _thread import *
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset  
from model import VGG, ClassNet, CIFARNet
from dataset import ImageDataset
from FedServerThread import FedServerThread
from queue import Queue
import argparse


class FedServer(object):

    def __init__(self, host, port, scenario=None, dataset = 'MNIST', delta=None):

        self.global_model = ClassNet() if dataset == 'MNIST' else CIFARNet()
        self.host = host
        self.port = port
        self.evaluator = ImageDataset(dataset)
        self.test_loader = None
        self.scenario = scenario
        self.isStart = False
        self.delta = delta
        self.wait_time = 30
        self._lock = threading.Lock()
        self.startup()


    def init_weights(self, m):
        '''
        Initialize model parameters from a uniform distribution
        '''
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def startup(self):
        '''
        Performs initialized activities
        '''
        self.global_model.apply(self.init_weights) # apply random values to model parameters
        print("Global model initialized with parameters of uniform distribution")

        _, self.test_loader = self.evaluator.load_data()  # downloads the test data
        print("Copies of testing data cached.")


    def evaluate_global_model(self):
        '''
        Executes prediction on the global model
        Returns: Accuracy of prediction
        '''
        test_accuracy = self.evaluator.test_model(self.global_model, self.test_loader)

        return test_accuracy



    def connect(self):
        '''
        Estabish socket connections for multiple clients
        '''
        serverSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        connected_clients = 0

        try:
            serverSocket.bind((self.host, self.port))
        except socket.error as e:
            print(str(e))

        print("Chief listening for incoming connection ...")
        serverSocket.listen(12)
        serverSocket.settimeout(self.wait_time)
        end_time = time.time() + self.wait_time
        
        threads = []
        clients = []
        q = Queue()
        shared_dict = {}
        iteration_counter = 1
        while time.time() < end_time:   # accept multiple connections 
            try:
                client, address = serverSocket.accept()
                print('Connected to: ' + address[0] + ':' + str(address[1]))
                connected_clients += 1
                print("{workers} workers connected to the server.".format(workers=connected_clients))
                clients.append(client)

                
                socket_thread = FedServerThread(connection=client,
                                    client_info=address, 
                                    global_model=self.global_model,
                                    queue=q,
                                    clients=clients,
                                    dictionary=shared_dict,
                                    data_loader=self.test_loader,
                                    evaluator=self.evaluator,
                                    scenario=self.scenario,
                                    lock=self._lock,
                                    delta=self.delta,
                                    iteration=iteration_counter,
                                    buffer_size=4096,
                                    recv_timeout=20)

                socket_thread.start()
                
            except socket.error as socketerror:
                print("Socket error occurred: {}".format(socketerror))


        print("Initial global weights sent to clients.")

        # for thread in threads:  # wait for threads to complete
        #     thread.join()

        # serverSocket.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--defence", type=int, nargs="?", default=0, help="Defence scenario")
    parser.add_argument("--data", type=str, nargs="?", default='MNIST', help="Dataset")
    parser.add_argument("--delta", type=int, nargs="?", default=0, help="Iteration delta value")
    args = parser.parse_args()
    mServer = None
    if args.defence == 1:
        mServer = FedServer(host="localhost", port=5000, scenario=args.defence, dataset=args.data)
    else:
        mServer = FedServer(host="localhost", port=5000, dataset=args.data, delta=args.delta)

    mServer.connect()   

