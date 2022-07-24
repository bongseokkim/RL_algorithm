import torch 
import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self,
                input_dim : int,
                output_dim : int, 
                num_neurons: list = [32,32],
                hidden_activation : str = "ReLU",
                out_activation : str  = 'Identity'):
        super(MLP, self).__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_activation = getattr(nn,hidden_activation)()
        self.out_activation = getattr(nn,out_activation)()

        input_dims = [input_dim]+num_neurons
        output_dims = num_neurons +[output_dim]
        # ex) input_dims : [4,64,64]
        # ex) output_dims :[64,64,1]
        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims,output_dims)) :
            done = True if i == (len(input_dims)-1) else False 
            self.layers.append(nn.Linear(in_dim,out_dim)) 

            if done :
                self.layers.append(self.out_activation)
            else : 
                self.layers.append(self.hidden_activation)
        
    def forward(self, x):
        for layer in self.layers : 
            x = layer(x)
        return x 

if __name__ == '__main__':
    net = MLP(input_dim=10, 
              output_dim=1, 
            num_neurons=[20, 12],
            hidden_activation ='ReLU',
            out_activation= 'Identity')
    print(net)
    x = torch.randn(size=(12, 10))
    print(f'x :{x}')
    y = net(x)
    print(f'y :{y}')
    
    """"
    under line is output of mlp.py

    MLP(
  (hidden_activation): ReLU()
  (out_activation): Identity()
  (layers): ModuleList(
    (0): Linear(in_features=10, out_features=20, bias=True)
    (1): ReLU()
    (2): Linear(in_features=20, out_features=12, bias=True)
    (3): ReLU()
    (4): Linear(in_features=12, out_features=1, bias=True)
    (5): Identity()
  )
)
x :tensor([[-0.5661, -0.7113,  0.0939,  0.2973, -0.7764, -1.4990,  0.2274,  0.1882,
         -0.2340,  0.0749],
        [-1.8956, -1.0811, -0.8864,  0.7020,  1.0509, -2.0381,  0.0850,  1.9735,
          1.2359, -0.7633],
        [-0.1027, -1.8215,  1.4963,  0.0178,  0.6066,  1.0719,  0.9610, -0.6209,
         -0.1187, -0.6445],
        [ 0.8463, -0.1480, -0.3716,  0.9969, -0.2456,  0.2593,  1.4460,  0.1346,
          0.6716, -0.1796],
        [ 0.0069, -0.0562, -0.9022,  0.2645,  0.3511,  0.3981,  0.8543,  0.1884,
         -1.4852, -0.2144],
        [ 1.8906,  1.1638,  0.6622,  1.3415, -0.6947,  0.1306, -0.5540, -1.2805,
         -0.8596,  1.7506],
        [ 0.2482,  1.4253, -1.4781,  0.1038,  0.3397, -0.1164, -0.1740,  0.1230,
         -0.6724, -1.4970],
        [ 1.6218,  1.9952, -1.5223,  0.1825,  0.1505,  0.4471,  1.4832,  0.2442,
         -0.8824,  0.4381],
        [-1.2831,  1.1850, -0.2390, -1.9865,  0.4187,  0.4755,  1.1815, -0.4320,
          2.2365,  0.1777],
        [ 2.1403,  0.3784,  1.4720,  0.1210,  1.6033, -0.1659, -1.3103, -0.0239,
          0.5986,  0.4655],
        [ 0.3020, -1.6067, -0.1535,  2.5739,  0.1063,  0.2272,  0.1568,  0.0030,
          0.3136, -0.2252],
        [-0.2117, -0.5910, -1.0954,  0.1880, -2.0215, -1.6315, -1.8700,  0.2735,
          0.7596,  2.5738]])
y :tensor([[ 0.2030],
        [ 0.1678],
        [ 0.1278],
        [ 0.1879],
        [ 0.1789],
        [ 0.1646],
        [ 0.1515],
        [ 0.0783],
        [-0.0079],
        [ 0.1533],
        [ 0.2301],
        [ 0.1908]], grad_fn=<AddmmBackward>)
    

    """