from torch.nn import Linear, ModuleList, Sequential, LeakyReLU
from model.networks import BaseNetwork
from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool as gap, global_add_pool as gadp
from icecream import ic


class crysgraphconv(BaseNetwork):

    def __init__(self, opt, n_node_features:int, n_edge_features:int):
        super().__init__(opt, n_node_features)

        self._name = "crysgraphconv"
        self.n_edge_features = n_edge_features
        self.batch_norm = opt.batch_norm
        self.pooling = opt.pooling

        # Expand the number of features to the embedding dimension
        self.linear = Linear(n_node_features, 
                             self.embedding_dim)

        # Convolutions
        self.convolutions = ModuleList([])
        for _ in range(self.n_convolutions):
            self.convolutions.append(CGConv(channels = self.embedding_dim, 
                                            dim = self.n_edge_features,
                                            batch_norm = self.batch_norm))
        
        # Graph embedding size is the same as the node embedding size
        graph_embedding = self.embedding_dim
        
        # Readout layers
        self.readout = ModuleList([])
        for _ in range(self.readout_layers-1):
            reduced_dim = int(graph_embedding/2)
            self.readout.append(Sequential(Linear(graph_embedding, 
                                                  reduced_dim), 
                                           LeakyReLU()))
            graph_embedding = reduced_dim

        # Final readout layer
        self.readout.append(Linear(int(self.embedding_dim/2), 
                                   self._n_classes))
        
        self.float()
        # Create loss, optimizer and scheduler
        self._make_loss(opt.problem_type)
        self._make_optimizer(opt.optimizer, opt)
        self._make_scheduler(scheduler=opt.scheduler, step_size = opt.step_size, gamma = opt.gamma, min_lr=opt.min_lr)


    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch

        # Expand the number of features to the embedding dimension
        x = self.linear(x)

        # Convolutions
        for conv in self.convolutions:
            x = conv(x, edge_index, edge_attr)

        # Apply pooling
        if self.pooling == "gap":
            x = gap(x, batch)
        elif self.pooling == "gadp":
            x = gadp(x, batch)
        else:
            raise ValueError(f"Pooling {self.pooling} not implemented")

        # Readout layers
        for layer in self.readout:
            x = layer(x)

        return x
    

    def get_intermediate_embedding(self, data):

        self.eval()

        x, edge_index, edge_attr, batch = data.x.float(), data.edge_index, data.edge_attr.float(), data.batch

        # Expand the number of features to the embedding dimension
        x = self.linear(x)

        # Convolutions
        for conv in self.convolutions:
            x = conv(x, edge_index, edge_attr)

        return x
    
    def get_final_prediction(self, graph):
        self.eval()

        # Apply pooling
        if self.pooling == "gap":
            x = gap(graph.x, graph.batch)
        elif self.pooling == "gadp":
            x = gadp(graph.x, graph.batch)
        else:
            raise ValueError(f"Pooling {self.pooling} not implemented")
        
        # Readout layers
        for layer in self.readout:
            x = layer(x)

        return x

