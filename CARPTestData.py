import math
import time
from collections import deque

from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import random
from torch.utils.data import Dataset, DataLoader
import pickle




class CARPTestDataset(Dataset):
    """Simulated Dataset Generator
    This class can generate random points in euclidean space for
    training and testing the reinforcement learning agent.

    ...

    Parameters
    ----------
    num_samples : int
        number of training/testing examples to be generated
    vertex_size  : int
        number of nodes to be generated in each training example
    edge_size  : int
        number of edges to be generated in each training example
    max_load    : int
        maximum load that a vehicle can carry
    max_demand  : int
        maximum demand that a edge can have
    seed        : int
        random seed for reproducing results

    Methods
    -------
    __len__()
        To be used with class instances. class_instance.len returns the num_samples value

    __getitem__(idx)
        returns the specific example at the given index (idx)

    """
    def __init__(self, num_samples, vertex_size, edge_size, device, max_demand=10, max_dhcost=10, seed=None):

        super(CARPTestDataset, self).__init__()

        self.depot_features = None
        # shape: (batch, 8)
        self.customer_features = None
        # shape: (batch, edge_size, 8)
        self.graph_dynamic = None
        # shape: (batch, edge_size)
        self.graph_info = None
        # shape: (batch, 5, edge_size+1)
        self.D = None
        # shape: (batch, vertex_size, vertex_size)


        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.vertex_size = vertex_size
        self.edge_size = edge_size
        self.num_samples = num_samples
        self.device = device

    def SyntheticDataload(self,file_path):

        if self.edge_size == 20:
            self.max_load = 30
        elif self.edge_size == 50:
            self.max_load = 50
        elif self.edge_size == 100:
            self.max_load = 100
        else:
            raise NotImplementedError

        print(file_path)
        path = os.path.join(file_path, 'graph_dynamic.txt')
        self.graph_dynamic = torch.load(path, map_location=self.device)
        path = os.path.join(file_path, 'graph_info.txt')
        self.graph_info = torch.load(path, map_location=self.device)

        # 初始化 node_features
        node_features = torch.zeros((self.num_samples, self.edge_size + 1, 8))
        graph_info_ori = torch.zeros((self.num_samples, self.edge_size + 1, 5))
        D_relabelled_tensor = torch.zeros((self.num_samples, self.vertex_size, self.vertex_size))
        for sample in tqdm(range(self.num_samples), desc="Processing graphs"):
            G = nx.Graph()
            total_dhcost = 0
            for i in range(self.edge_size + 1):
                if i == 0:
                    depot = int(self.graph_info[sample, i, 1])
                else:
                    node1 = int(self.graph_info[sample, i, 1])
                    node2 = int(self.graph_info[sample, i, 2])
                    dhcost = self.graph_info[sample, i, 3].item()
                    demand = self.graph_info[sample, i, 4].item()
                    total_dhcost += dhcost
                    G.add_edge(node1, node2, dhcost=dhcost, demand=demand)

            bfs_order = BFS(G, depot)
            node_mapping = {old_label: new_label for new_label, old_label in enumerate(bfs_order)}
            G_relabelled = nx.relabel_nodes(G, node_mapping)
            depot = 0

            D_relabelled, _ = floyd(G_relabelled)
            D_relabelled = torch.tensor(D_relabelled)
            D_relabelled_tensor[sample, :D_relabelled.shape[0], :D_relabelled.shape[1]] = D_relabelled

            # 假设每个节点的特征是一个向量，这里随机生成节点特征
            i = 0
            node_feature = [1, 1, depot, depot, 0, 0, 0, 0]
            graph_info_ = [i, depot, depot, 0, 0]
            node_features[sample, i, :] = torch.tensor(node_feature)
            graph_info_ori[sample, i, :] = torch.tensor(graph_info_)
            i += 1
            for node_ori_1, node_ori_2, attributes in G_relabelled.edges(data=True):
                edge_dhcost = attributes['dhcost']
                edge_demand = attributes['demand']
                f_node_ori_1 = 1 if node_ori_1 == depot else 0
                f_node_ori_2 = 1 if node_ori_2 == depot else 0

                node_feature = [f_node_ori_1,
                                f_node_ori_2,
                                node_ori_1,
                                node_ori_2,
                                D_relabelled[depot][node_ori_1],
                                D_relabelled[depot][node_ori_2],
                                edge_dhcost / total_dhcost,
                                edge_demand / self.max_load
                                ]
                graph_info_ = [i, node_ori_1, node_ori_2, edge_dhcost, edge_demand]
                node_features[sample, i, :] = torch.tensor(node_feature)
                graph_info_ori[sample, i, :] = torch.tensor(graph_info_)
                i += 1

        self.D = D_relabelled_tensor

        node_ori_1 = graph_info_ori[:, :, 1].long()
        node_ori_2 = graph_info_ori[:, :, 2].long()
        # 广播扩展维度，形成所有可能的节点对
        node_ori_1_exp = node_ori_1.unsqueeze(2).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        node_ori_2_exp = node_ori_2.unsqueeze(2).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        targetnode_ori_1_exp = node_ori_1.unsqueeze(1).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        targetnode_ori_2_exp = node_ori_2.unsqueeze(1).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)

        D_expanded = torch.zeros((self.num_samples, self.edge_size + 1, self.edge_size + 1))
        node_size = self.D.size(1)
        # 将 self.D 的值复制到新的张量中
        D_expanded[:, :node_size, :node_size] = self.D
        # 计算所有节点对之间的距离，并取最小值
        distances_1 = D_expanded.gather(2, targetnode_ori_1_exp).gather(1, node_ori_1_exp)
        distances_2 = D_expanded.gather(2, targetnode_ori_2_exp).gather(1, node_ori_1_exp)
        distances_3 = D_expanded.gather(2, targetnode_ori_1_exp).gather(1, node_ori_2_exp)
        distances_4 = D_expanded.gather(2, targetnode_ori_2_exp).gather(1, node_ori_2_exp)
        # 取四者的最小值作为 edge_distance
        edge_distance = torch.min(torch.min(distances_1, distances_2), torch.min(distances_3, distances_4))

        self.edge_distance = edge_distance

        self.graph_info = graph_info_ori.permute(0, 2, 1)  # [num_samples ,5, edge_size + 1]

        self.depot_features = node_features[:, 0, :] # [num_samples, _, num_features]
        self.customer_features = node_features[:, 1:, :] # [num_samples , edge_size, num_features]
        self.graph_dynamic = self.graph_dynamic[:, 1:] # [num_samples, edge_size, 1]


    def StandardDatasetload(self, file_path):
        G, depot, total_cost, total_demand, capacity  = parse_StandardDataset(file_path)

        D_origin, _ = floyd(G)

        edge_size = G.number_of_edges()
        vertex_size = G.number_of_nodes()
        self.graph_dynamic = torch.zeros((1, edge_size + 1))
        self.graph_info = torch.zeros((1, edge_size + 1, 5))
        self.D = torch.zeros((1, vertex_size, vertex_size))

        node_features = torch.zeros((1, edge_size + 1, 8), dtype=torch.float32)

        bfs_order = BFS(G, depot)
        node_mapping = {old_label: new_label for new_label, old_label in enumerate(bfs_order)}
        self.node_mapping = node_mapping
        G_relabelled = nx.relabel_nodes(G, node_mapping)
        depot = 0

        D_relabelled, _ = floyd(G_relabelled)
        self.D[0, :, :] = torch.tensor(D_relabelled)

        i = 0
        node_feature = [1, 1, depot, depot, 0, 0, 0, 0]
        dynamic_np = [0]
        graph_info_ = [i, depot, depot, 0, 0]

        node_features[0, i, :] = torch.tensor(node_feature)
        self.graph_dynamic[0, i] = torch.tensor(dynamic_np)
        self.graph_info[0, i, :] = torch.tensor(graph_info_)

        i += 1

        for node_ori_1, node_ori_2, attributes in G_relabelled.edges(data=True):
            edge_dhcost = attributes['dhcost']
            edge_demand = attributes['demand']
            f_node_ori_1 = 1 if node_ori_1 == depot else 0
            f_node_ori_2 = 1 if node_ori_2 == depot else 0

            node_feature = [f_node_ori_1,
                            f_node_ori_2,
                            node_ori_1,
                            node_ori_2,
                            D_relabelled[depot][node_ori_1],
                            D_relabelled[depot][node_ori_2],
                            edge_dhcost / total_cost,
                            edge_demand / capacity
                            ]


            dynamic_np = [edge_demand / capacity]
            graph_info = [i, node_ori_1, node_ori_2, edge_dhcost, edge_demand]

            node_features[0, i, :] = torch.tensor(node_feature)
            self.graph_dynamic[0, i] = torch.tensor(dynamic_np)
            self.graph_info[0, i, :] = torch.tensor(graph_info)
            i += 1

        node_ori_1 = self.graph_info[:, :, 1].long()
        node_ori_2 = self.graph_info[:, :, 2].long()
        # 广播扩展维度，形成所有可能的节点对
        node_ori_1_exp = node_ori_1.unsqueeze(2).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        node_ori_2_exp = node_ori_2.unsqueeze(2).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        targetnode_ori_1_exp = node_ori_1.unsqueeze(1).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)
        targetnode_ori_2_exp = node_ori_2.unsqueeze(1).expand(self.num_samples, self.edge_size + 1, self.edge_size + 1)

        D_expanded = torch.zeros((self.num_samples, self.edge_size + 1, self.edge_size + 1), device=self.device)
        node_size = self.D.size(1)
        # 将 self.D 的值复制到新的张量中
        D_expanded[:, :node_size, :node_size] = self.D
        # 计算所有节点对之间的距离，并取最小值

        distances_1 = D_expanded.gather(2, targetnode_ori_1_exp).gather(1, node_ori_1_exp)
        distances_2 = D_expanded.gather(2, targetnode_ori_2_exp).gather(1, node_ori_1_exp)
        distances_3 = D_expanded.gather(2, targetnode_ori_1_exp).gather(1, node_ori_2_exp)
        distances_4 = D_expanded.gather(2, targetnode_ori_2_exp).gather(1, node_ori_2_exp)
        # 取四者的最小值作为 edge_distance
        edge_distance = torch.min(torch.min(distances_1, distances_2), torch.min(distances_3, distances_4))


        self.graph_info = self.graph_info.permute(0, 2, 1)  # [num_samples ,4, edge_size + 1]
        self.depot_features = node_features[:, 0, :] # [num_samples, _, num_features]
        self.customer_features = node_features[:, 1:, :] # [num_samples , edge_size, num_features]
        self.graph_dynamic = self.graph_dynamic[:, 1:] # [num_samples, edge_size, 1]
        self.edge_distance = edge_distance

    def __len__(self):
        """Returns the number of examples being trained/tested on"""
        return self.num_samples

    def __getitem__(self, idx):
        """Returns the specific example at the given index (idx)
        Parameters
        ----------
        idx : int
            index for which the example has to be returned.
        """
        return ((self.depot_features[idx], self.customer_features[idx], self.graph_dynamic[idx]), self.graph_info[idx], self.D[idx],self.edge_distance[idx])



def parse_StandardDataset(file_path):
    """解析 gdb.dat 文件并构建无向图"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 初始化变量
    vertices = 0
    depot = None
    capacity = 0
    total_cost = 0
    total_demand = 0
    edges = []
    in_edge_list = False

    id = 0
    # 逐行解析文件内容
    for line in lines:
        line = line.strip()
        if line.startswith("VERTICES"):
            vertices = int(line.split(":")[1].strip())

        elif line.startswith("CAPACIDAD"):
            capacity = int(line.split(":")[1].strip())

        elif line.startswith("COSTE_TOTAL_REQ"):
            total_cost = int(line.split(":")[1].strip())

        elif line.startswith("LISTA_ARISTAS_REQ"):
            in_edge_list = True
        elif line.startswith("DEPOSITO"):
            in_edge_list = False
            depot = int(line.split(":")[1].strip())-1
        elif in_edge_list and line.startswith("("):
            # 解析边的信息
            parts = line.split("coste")
            edge_part = parts[0].strip().strip("() ")
            cost_part = parts[1].split("demanda")[0].strip()
            demand_part = parts[1].split("demanda")[1].strip()
            u, v = map(int, edge_part.split(","))
            cost = int(cost_part)
            demand = int(demand_part)
            total_demand += demand
            id += 1
            edges.append((u-1, v-1, {"dhcost": cost, "demand": demand, "id": id }))
    # 创建无向图
    G = nx.Graph()
    G.add_nodes_from(range(vertices))
    G.add_edges_from(edges)

    return G, depot, total_cost, total_demand, capacity

def edge2vertex(edge_graph, depot):
    G = nx.Graph()
    edge_info_list = []
    edge_info_list.append((depot, depot, 0, 0))

    G.add_node(0, demand=0, dhcost=0,node_ori=(depot,depot))
    i = 1
    for node1, node2, edge_data in edge_graph.edges(data=True):
        demand = edge_data['demand']
        dhcost = edge_data['dhcost']
        G.add_node(i, demand=demand, dhcost=dhcost,node_ori=(node1,node2) )
        i += 1
        edge_info_list.append((node1, node2, demand, dhcost))


    for index1, (index1_node1, index1_node2, index1_demand, index1_dhcost) in enumerate(edge_info_list):
        for index2 in range(index1 + 1, len(edge_info_list)):
            data_index2 = edge_info_list[index2]
            index2_node1 = data_index2[0]
            index2_node2 = data_index2[1]
            if (index1_node1 == index2_node1 or index1_node1 == index2_node2 or index1_node2 == index2_node1 or index1_node2 == index2_node2):
                G.add_edge(index1, index2)

    return G


def floyd(G):
    num_nodes = G.number_of_nodes()
    # 初始化邻接矩阵为无穷大
    adj_matrix = np.full((num_nodes, num_nodes), np.inf)
    # 将有边特征的边存入邻接矩阵
    for node in G.nodes():
        adj_matrix[node, node] = 0
    for node1, node2, edge_data in G.edges(data=True):
        dhcost = edge_data['dhcost']
        adj_matrix[node1, node2] = dhcost
        adj_matrix[node2, node1] = dhcost

    num_nodes = len(adj_matrix)

    # 初始化距离矩阵
    distance_matrix = np.copy(adj_matrix)

    # 初始化路径矩阵，用于记录最短路径的中间节点
    path_matrix = np.ones((num_nodes, num_nodes), dtype=int) * -1

    # Floyd-Warshall 算法
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, k] + distance_matrix[k, j] < distance_matrix[i, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]
                    path_matrix[i, j] = k

    return distance_matrix, path_matrix

def generate_graph_degree_BFS(vertex_size, edge_size, max_dhcost, max_demand, min_degree, max_degree):
    errorNum = 0
    while True:
        if errorNum == 100:
            raise ValueError("节点与度的数量设置不合理")

        total_cost, total_demand= 0,0

        # 生成地图 根据点的度生成
        G = nx.Graph()
        G.add_nodes_from(range(vertex_size))

        # 为每个节点添加边，确保图是连通的
        for node in G.nodes():

            # 获取当前节点的度
            degree = G.degree(node)
            degree = random.randint(max(degree,min_degree), max_degree)

            select_nodes = [n for n in G.nodes() if G.degree(n) < max_degree and n != node]

            if len(select_nodes) < max(degree - G.degree(node), 0):
                errorNum += 1
                break

            # 随机选择与当前节点相邻的其他节点
            neighbors = random.sample(select_nodes, max(degree - G.degree(node), 0))

            # 添加边
            for neighbor in neighbors:
                dhcost = random.randint(1, max_dhcost)
                total_cost += dhcost
                #demand = random.randint(1, max_demand) if random.random() < 0.7 else 0
                demand = random.randint(1, max_demand)
                total_demand += demand
                G.add_edge(node, neighbor, dhcost=dhcost,demand=demand)

        is_done = False
        iteration_count = 0
        while not is_done and iteration_count < 100:
            iteration_count += 1
            if G.number_of_edges() < edge_size:
                add_nodes = [n for n in G.nodes() if G.degree(n) < max_degree]
                if not add_nodes:
                    break
                add_node = random.sample(add_nodes,k=1)[0]
                select_nodes = [n for n in G.nodes() if
                                G.degree(n) < max_degree and n != add_node and not G.has_edge(n, add_node)]
                if not select_nodes:
                    break
                neighbor = random.sample(select_nodes,k=1)[0]
                dhcost = random.randint(1, max_dhcost)
                total_cost += dhcost
                demand = random.randint(1, max_demand)
                total_demand += demand
                G.add_edge(add_node, neighbor, dhcost=dhcost,demand=demand)

            elif G.number_of_edges() > edge_size:
                delete_nodes = [n for n in G.nodes() if G.degree(n) > min_degree]
                if not delete_nodes:
                    break
                delete_node = random.sample(delete_nodes,k=1)[0]
                select_nodes = [n for n in G.neighbors(delete_node) if G.degree(n) > min_degree]
                if not select_nodes:
                    break
                neighbor = random.sample(select_nodes,k=1)[0]
                edge_data = G.get_edge_data(delete_node, neighbor)
                if edge_data and 'weight' in edge_data:
                    total_cost -= edge_data['dhcost']
                    total_demand -= edge_data['demand']
                G.remove_edge(delete_node, neighbor)

            if G.number_of_edges() == edge_size:
                is_done = True

        # 随机将一个节点设为depot
        depot = random.choice(list(G.nodes))

        if nx.is_connected(G) and G.number_of_edges() == edge_size:
            # 根据广度优先搜索重新将点进行编号
            bfs_order = BFS(G, depot)
            node_mapping = {old_label: new_label for new_label, old_label in enumerate(bfs_order)}
            G_relabelled = nx.relabel_nodes(G, node_mapping)
            depot = 0
            return G_relabelled, depot, total_cost, total_demand

def BFS(G, start_node):
    visited = set()
    queue = deque([start_node])
    bfs_order = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            bfs_order.append(node)
            visited.add(node)
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)
    return bfs_order


if __name__ == '__main__':
    USE_CUDA = True
    if USE_CUDA:
        cuda_device_num = 0
        torch.cuda.set_device(cuda_device_num)
        device = torch.device('cuda', cuda_device_num)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')


    train_dataset = CARPTestDataset(10000, 50, 100, device, 10, 10,None)
    train_dataset.DataGenerate()
    # train_dataset.Dataload()
    # train_loader = DataLoader(train_data, 10, True, num_workers=0)
    # for batch_idx, batch in enumerate(train_loader):
    #     x = batch
    #     print(x[0][0].shape)
    #     print(x[0][1].shape)
    #     print(x[0][2].shape)


