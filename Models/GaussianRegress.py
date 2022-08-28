import torch
import torch.nn as nn

class pred_net(nn.Module):
    def __init__(self, obs_len, pred_len):
        super(pred_net, self).__init__()
        self.input_size = 1
        self.embed_size2 = 32
        self.output_size = 1

        self.obs_len = obs_len
        self.pred_len = pred_len
        # self.lstm = nn.LSTM(2*self.embed_size2, self.embed_size2, num_layers=self.num_layers , bidirectional=True, batch_first=True)
        self.rnn = nn.GRU(1*self.embed_size2, self.embed_size2, batch_first=True)
        # self.rnn = nn.GRUCell(self.embed_size2, self.embed_size2)

        # self.embed_obs = nn.Linear(self.input_size, int(0.5*self.embed_size2))
        self.embed_obs1 = nn.Linear(self.input_size, int(0.5*self.embed_size2))

        # self.embed_obs_ = nn.Linear(int(0.5*self.embed_size2), self.embed_size2)
        self.embed_obs1_ = nn.Linear(int(0.5*self.embed_size2), self.embed_size2)
        self.output = nn.Linear( 1 * self.embed_size2, self.obs_len * self.output_size)

        self.embed_mlp_obs = nn.Linear(int(self.obs_len*self.input_size*2), self.embed_size2)
        self.output_mlp = nn.Linear(1 * self.embed_size2, self.obs_len * self.output_size)

        # self.get_w = nn.Linear(self.input_size*2, self.input_size*2)

        self.gelu = nn.Sigmoid()#nn.GELU()
        self.dropout = nn.Dropout(0.00)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, input):
        # input = input.type(torch.FloatTensor)
        #Mlp
        # f_input = torch.cat((input[:, :, 1:], input[:, :, 0:1]), 2)
        # f_input = f_input.reshape(f_input.size()[0], f_input.size()[1]*f_input.size()[2])
        # f_input = self.gelu(self.embed_mlp_obs(f_input))
        # output_mlp = self.output_mlp(f_input)
        # output_mlp = output_mlp.reshape(output_mlp.size()[0], output_mlp.size()[1], 1)
        #return output_sf

        # rnn
        # f_ii = self.dropout(self.gelu(self.embed_obs_(self.gelu(self.embed_obs(input[:, :, 1:])))))
        f_i = self.dropout(self.gelu(self.embed_obs1_(self.gelu(self.embed_obs1(input[:, :, 0:1])))))
        # f_j = self.dropout(self.gelu(self.embed_obs1_(self.gelu(self.embed_obs1(input[:, :, 1:])))))
        # f_i = torch.cat((f_ii, f_j), 2)

        # h0 = torch.randn(input.size()[0], self.embed_size2)
        # for i in range(input.size()[1]):
        #     hx = self.rnn(f_i[:, i, :], h0)
        # output_sf = []
        # for j in range(self.pred_len):
        #     output_sf.append(self.output(hx))
        #     f_i = self.dropout(self.gelu(self.embed_obs_(self.gelu(self.embed_obs(self.output(hx))))))
        #     hx = self.rnn(f_i, hx)
        #
        # output_sf = torch.stack(output_sf, 1)


        h0 = torch.randn(1, f_i.size()[0], self.embed_size2)
        input_obs, h0 = self.rnn(f_i, h0)
        # input_obs = f_j + input_obs
        # input_obs = input_obs.reshape(input_obs.size()[0], input_obs.size()[1]*input_obs.size()[2])
        output_sf = self.output(h0)
        output = output_sf.reshape(output_sf.size()[0], output_sf.size()[2], 1)


        # output_w = self.gelu(self.get_w(torch.cat((output_mlp, output_sf), 2)))
        # output_w = torch.cat((output_mlp, output_sf), 2)
        # output_w = self.softmax(output_w)
        # output = torch.sum(output_w*torch.cat((output_mlp, output_sf), 2), 2)
        # output = torch.unsqueeze(output, 2)

        return output #, output_bias, output_sf_sigma, output_bias_sigma, output_pi



