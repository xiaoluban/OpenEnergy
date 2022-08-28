import torch
import torch.nn as nn

class pred_net(nn.Module):
    def __init__(self):
        super(pred_net, self).__init__()
        self.input_size = 1
        self.embed_size2 = 16
        self.output_size = 1
        # self.lstm = nn.LSTM(2*self.embed_size2, self.embed_size2, num_layers=self.num_layers , bidirectional=True, batch_first=True)
        self.rnn = nn.GRU(self.embed_size2, self.embed_size2, batch_first=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1*self.embed_size2, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)


        self.embed_obs = nn.Linear(self.input_size, int(0.5*self.embed_size2))
        self.embed_obs1 = nn.Linear(self.input_size, int(0.5*self.embed_size2))

        self.embed_obs_ = nn.Linear(int(0.5*self.embed_size2), self.embed_size2)
        self.embed_obs1_ = nn.Linear(int(0.5*self.embed_size2), self.embed_size2)

        self.output = nn.Linear(1*self.embed_size2, self.output_size)

        self.gelu = nn.Sigmoid()#nn.GELU()
        self.dropout = nn.Dropout(0.05)


    def forward(self, input):
        # input = torch.unsqueeze(input, 2)
        # time_input = torch.unsqueeze(time_input, 2)
        # input = input.type(torch.FloatTensor)
        f_i = self.dropout(self.embed_obs_(self.embed_obs(input)))
        f_i = self.transformer_encoder(f_i.transpose(0, 1))
        f_i = f_i.transpose(0, 1)


        # h0 = torch.randn(1, f_i.size()[0], f_i.size()[2])
        # input_obs, h0 = self.rnn(f_i, h0)
        output_sf = self.output(f_i)


        return output_sf #, output_bias, output_sf_sigma, output_bias_sigma, output_pi



