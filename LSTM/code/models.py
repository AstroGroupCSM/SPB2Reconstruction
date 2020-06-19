__author__ = 'Connor Heaton'

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class MyRNNBlock(nn.Module):
    def __init__(self, args):
        super(MyRNNBlock, self).__init__()
        self.args = args

        self.dim = getattr(self.args, 'rnn_dim', 512)
        self.input_dropout_prob = getattr(self.args, 'rnn_input_dropout_prob', 0.2)
        self.output_dropout_prob = getattr(self.args, 'rnn_output_dropout_prob', 0.2)

        self.cell = nn.LSTMCell(self.dim, self.dim)
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.output_dropout = nn.Dropout(self.output_dropout_prob)

    def forward(self, x_t, prev_h, prev_c):
        prev_c = self.input_dropout(prev_c)
        state = (prev_h, prev_c)
        x_t = self.input_dropout(x_t)

        output, hidden_state = self.cell(x_t, state)

        output = self.output_dropout(output)
        # new_h, new_c = hidden_state

        return output, hidden_state


class MyConvBlock(nn.Module):
    def __init__(self, args):
        super(MyConvBlock, self).__init__()
        self.args = args

        self.conv_kernel_size = getattr(self.args, 'conv_kernel_size', 3)
        self.conv_stride = getattr(self.args, 'conv_stride', 1)
        self.pool_kernel_size = getattr(self.args, 'pool_kernel_size', 3)
        self.pool_stride = getattr(self.args, 'pool_stride', 1)
        self.dropout_prob = getattr(self.args, 'conv_dropout_prob', 0.2)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 6, self.conv_kernel_size, stride=self.conv_stride, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 6, self.conv_kernel_size, stride=self.conv_stride, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        self.bn2 = nn.BatchNorm2d(6)

        self.conv3 = nn.Conv2d(6, 6, self.conv_kernel_size, stride=self.conv_stride, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        self.bn3 = nn.BatchNorm2d(6)

        self.conv4 = nn.Conv2d(6, 1, self.conv_kernel_size, stride=self.conv_stride, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        self.bn4 = nn.BatchNorm2d(1)

        # self.conv5 = nn.Conv2d(6, 6, self.conv_kernel_size, stride=self.conv_stride, padding=0)
        # self.pool5 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        #
        # self.conv6 = nn.Conv2d(6, 1, self.conv_kernel_size, stride=self.conv_stride, padding=0)
        # self.pool6 = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)

    def forward(self, x):
        # output1 = self.act(self.pool1(self.conv1(x)))
        output1 = self.act(self.bn1(self.pool1(self.conv1(x))))
        # print('output1.shape: {}'.format(output1.shape))
        # output2 = self.act(self.pool2(self.conv2(output1)))
        output2 = self.act(self.bn2(self.pool2(self.conv2(output1))))
        # print('output2.shape: {}'.format(output2.shape))
        # output3 = self.act(self.pool3(self.conv3(output2)))
        output3 = self.act(self.bn3(self.pool3(self.conv3(output2))))
        # print('output3.shape: {}'.format(output3.shape))
        # output4 = self.act(self.pool4(self.conv4(output3)))
        output4 = self.act(self.bn4(self.pool4(self.conv4(output3))))
        # print('output4.shape: {}'.format(output4.shape))

        # output5 = self.act(self.pool5(self.conv5(output4)))
        # print('output5.shape: {}'.format(output5.shape))
        # output6 = self.act(self.pool6(self.conv6(output5)))
        # print('output6.shape: {}'.format(output6.shape))

        # print('output1.shape: {}'.format(output1.shape))
        # print('output2.shape: {}'.format(output2.shape))
        # print('output3.shape: {}'.format(output3.shape))
        # print('output4.shape: {}'.format(output4.shape))
        # print('output5.shape: {}'.format(output5.shape))
        # print('output6.shape: {}'.format(output6.shape))

        return output4


class ResModule(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, n_layers=3):
        super(ResModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_layers = n_layers

        self.act = nn.ReLU(inplace=True)

        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride,
        #                            padding=self.padding)
        # self.res_pool1 = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride,
        #                               padding=self.padding)
        # self.res_bn1 = nn.BatchNorm2d(self.out_channels)
        #
        # self.res_conv2 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride,
        #                            padding=self.padding)
        # self.res_pool2 = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride,
        #                               padding=self.padding)
        # self.res_bn2 = nn.BatchNorm2d(self.out_channels)

        for i_layer in range(self.n_layers):
            setattr(self, 'res_conv{}'.format(i_layer), nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size,
                                                                  stride=self.stride, padding=self.padding))
            setattr(self, 'res_pool{}'.format(i_layer), nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride,
                                                                     padding=self.padding))
            setattr(self, 'res_bn{}'.format(i_layer), nn.BatchNorm2d(self.out_channels))

    def forward(self, x):
        identity = x
        # output = x

        # res_output1 = self.act(self.res_bn1(self.res_pool1(self.res_conv1(x))))
        # res_output2 = self.res_bn2(self.res_pool2(self.res_conv2(res_output1)))

        for i_layer in range(self.n_layers):
            conv_i = getattr(self, 'res_conv{}'.format(i_layer))
            pool_i = getattr(self, 'res_pool{}'.format(i_layer))
            bn_i = getattr(self, 'res_bn{}'.format(i_layer))
            x = self.act(bn_i(pool_i(conv_i(x))))

        output = self.act(x + identity)

        return output


class ConvModule(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride,
                               padding=self.padding)
        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        self.bn1 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        output = self.act(self.bn1(self.pool1(self.conv1(x))))
        return output


def update_conv_dim(in_dim, kernel_size, padding, stride):
    out_dim = (in_dim - kernel_size + 2 * padding) / stride + 1
    out_dim = int(out_dim)

    out_dim = (out_dim - kernel_size + 2 * padding) / stride + 1
    out_dim = int(out_dim)

    return out_dim


class MyResConvBlock(nn.Module):
    def __init__(self, args):
        super(MyResConvBlock, self).__init__()
        self.args = args

        self.res_conv_kernel_size = getattr(self.args, 'res_conv_kernel_size', 3)
        self.res_conv_stride = getattr(self.args, 'res_conv_stride', 1)
        self.res_pool_kernel_size = getattr(self.args, 'res_pool_kernel_size', 3)
        self.res_pool_stride = getattr(self.args, 'res_pool_stride', 1)

        self.conv_kernel_size = getattr(self.args, 'conv_kernel_size', 3)
        self.conv_stride = getattr(self.args, 'conv_stride', 1)
        self.pool_kernel_size = getattr(self.args, 'pool_kernel_size', 3)
        self.pool_stride = getattr(self.args, 'pool_stride', 1)
        self.dropout_prob = getattr(self.args, 'conv_dropout_prob', 0.2)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.act = nn.ReLU()
        self.res_act = nn.ReLU(inplace=True)
        self.n_conv_layers = getattr(self.args, 'n_conv_layers', 6)
        self.second_res_layer = int(self.n_conv_layers * 0.333)
        self.third_res_layer = int(self.n_conv_layers * 0.666)

        self.channels = self.args.conv_channels
        self.input_x_dim = 48
        self.input_y_dim = 144

        self.curr_x_dim = self.input_x_dim
        self.curr_y_dim = self.input_y_dim

        self.res_conv_padding = int(self.res_conv_kernel_size // 2) if int(self.res_conv_kernel_size // 2) != 0 else 1
        self.res_pool_padding = int(self.res_pool_kernel_size // 2) if int(self.res_pool_kernel_size // 2) != 0 else 1

        # self.res_conv1 = nn.Conv2d(1, 1, self.res_conv_kernel_size, stride=self.res_conv_stride,
        #                            padding=self.res_conv_padding)
        # self.res_pool1 = nn.MaxPool2d(kernel_size=self.res_pool_kernel_size, stride=self.res_pool_stride,
        #                               padding=self.res_pool_padding)
        # self.res_bn1 = nn.BatchNorm2d(1)

        self.res_mod1 = ResModule(in_channels=1, out_channels=1, kernel_size=self.res_conv_kernel_size,
                                  stride=self.res_conv_stride, padding=self.res_conv_padding)

        self.res_mod2 = ResModule(in_channels=self.channels, out_channels=self.channels,
                                  kernel_size=self.res_conv_kernel_size,
                                  stride=self.res_conv_stride, padding=self.res_conv_padding)

        self.res_mod3 = ResModule(in_channels=self.channels, out_channels=self.channels,
                                  kernel_size=self.res_conv_kernel_size,
                                  stride=self.res_conv_stride, padding=self.res_conv_padding)

        for i in range(self.n_conv_layers):
            setattr(self, 'conv_mod{}'.format(i), ConvModule(in_channels=1 if i == 0 else self.channels,
                                                             out_channels=1 if i == self.n_conv_layers - 1
                                                             else self.channels,
                                                             kernel_size=self.conv_kernel_size,
                                                             stride=self.conv_stride, padding=0))
            self.curr_x_dim = update_conv_dim(in_dim=self.curr_x_dim,
                                              kernel_size=self.conv_kernel_size,
                                              padding=0, stride=self.conv_stride)
            self.curr_y_dim = update_conv_dim(in_dim=self.curr_y_dim,
                                              kernel_size=self.conv_kernel_size,
                                              padding=0, stride=self.conv_stride)

        self.output_dim = self.curr_x_dim * self.curr_y_dim

        print('*** MyResConvBlock ***')
        print('N conv layers: {}'.format(self.n_conv_layers))
        print('Output dim: {}'.format(self.output_dim))

    def forward(self, x):
        output = self.res_mod1(x)

        for i in range(self.n_conv_layers):
            if i == self.second_res_layer:
                output = self.res_mod2(output)
            if i == self.third_res_layer:
                output = self.res_mod3(output)
            output = getattr(self, 'conv_mod{}'.format(i))(output)

        return output


class ConvRNNModel(nn.Module):
    def __init__(self, args, device):
        super(ConvRNNModel, self).__init__()
        self.args = args
        self.device = device
        self.model_type = getattr(self.args, 'model_type', 'ConvRNN')

        if self.model_type == 'ResConvRNN':
            self.conv_block = MyResConvBlock(self.args)
            # self.cnn_output_dim = 1296
            self.cnn_output_dim = self.conv_block.output_dim
        else:
            self.conv_block = MyConvBlock(self.args)
            self.cnn_output_dim = 1

        self.rnn_block1 = MyRNNBlock(self.args)
        self.rnn_block2 = MyRNNBlock(self.args)

        # self.cnn_output_dim = 2880  # 832
        self.rnn_dim = getattr(self.args, 'rnn_dim', 512)
        self.linear_projection = nn.Linear(self.cnn_output_dim, self.rnn_dim, bias=True)
        self.clf = nn.Linear(2 * self.rnn_dim, 2)

        self.dropout_prob = getattr(self.args, 'linear_projection_dropout_prob', 0.2)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.use_frame_loss = getattr(self.args, 'use_frame_loss', False)
        self.frame_loss_weight = getattr(self.args, 'frame_loss_weight', 0.5)
        if self.use_frame_loss:
            self.frame_linear = nn.Linear(self.rnn_dim, 2)

        # self.noise_loss_weight = getattr(self.args, 'noise_loss_weight', 1.0)
        # self.signal_loss_weight = getattr(self.args, 'signal_loss_weight', 1.0)
        # self.loss_weight = torch.tensor([self.noise_loss_weight, self.signal_loss_weight]).to(self.device)
        self.loss_weight = None

    def set_label_weights(self, weights):
        self.loss_weight = torch.tensor(weights).to(self.device)

    def forward(self, x, y=None, x_raw=None, batch_xent_weight=None):
        """
        Add loss for identifying signal in a frame?
        """
        batch_size = x.shape[0]
        n_timesteps = x.shape[1]

        prev_h1 = torch.zeros(batch_size, self.rnn_dim).to(self.device)
        prev_c1 = torch.zeros(batch_size, self.rnn_dim).to(self.device)
        prev_h2 = torch.zeros(batch_size, self.rnn_dim).to(self.device)
        prev_c2 = torch.zeros(batch_size, self.rnn_dim).to(self.device)

        conv_outputs = []

        if self.use_frame_loss:
            frame_logits = []

        for t in range(n_timesteps):
            x_t = x[:, t, :, :].unsqueeze(1)
            conv_t = self.conv_block(x_t).view(batch_size, -1)
            conv_t = self.dropout(conv_t)
            proj_t = self.linear_projection(conv_t)

            conv_outputs.append(proj_t)

            # proj_t = self.dropout(proj_t)     INCLUDED IN RNN BLOCK
            # prev_h1, prev_c1 = self.rnn_block1(proj_t, prev_h1, prev_c1)
            # prev_h2, prev_c2 = self.rnn_block2(proj_t, prev_h2, prev_c2)
            # if self.use_frame_loss:
            #     frame_logit = self.frame_linear(prev_h2)
            #     frame_logits.append(frame_logit)

        # logits = self.clf(prev_h2)
        # if self.use_frame_loss:
        #     frame_logits = torch.cat(frame_logits, dim=0)

        # forward RNN pass
        for conv_out in conv_outputs:
            prev_h1, prev_c1 = self.rnn_block1(conv_out, prev_h1, prev_c1)

        # backward RNN pass
        for conv_out in conv_outputs[::-1]:
            prev_h2, prev_c2 = self.rnn_block2(conv_out, prev_h2, prev_c2)

        comb_h = torch.cat((prev_h1, prev_h2), dim=-1)
        logits = self.clf(comb_h)

        outputs = (logits,)
        if y is not None:
            loss_fct = CrossEntropyLoss(weight=batch_xent_weight)
            loss = loss_fct(logits, y)

            if self.use_frame_loss:
                # print('y.shape: {}'.format(y.shape))
                y_frame = torch.repeat_interleave(y.view(-1, 1), n_timesteps, dim=1)
                # print('y_frame.shape: {}'.format(y_frame.shape))
                x_frame = x_raw.view(batch_size, n_timesteps, -1)
                # print('x_frame.shape: {}'.format(x_frame.shape))
                x_max = torch.max(x_frame, dim=-1)[0]
                # print('x_max: {}'.format(x_max))
                # print('x_max.shape: {}'.format(x_max.shape))
                y_frame[(y_frame == 1) & (x_max < 10)] = 0
                # print('y_frame: {}'.format(y_frame))
                # print('y_frame.shape: {} mean: {}'.format(y_frame.shape, y_frame.double().mean()))

                frame_logits = torch.cat(frame_logits, dim=0)
                # print('frame_logits.shape: {}'.format(frame_logits.shape))
                frame_logits = frame_logits.view(-1, 2)
                # print('frame_logits.shape: {}'.format(frame_logits.shape))
                y_frame = y_frame.view(-1)
                # print('y_frame.shape: {}'.format(y_frame.shape))
                frame_loss = loss_fct(frame_logits, y_frame)

                loss += self.frame_loss_weight * frame_loss

                # input('okty')

            outputs = (loss,) + outputs

        return outputs


class ConvModel(nn.Module):
    def __init__(self, args, device):
        super(ConvModel, self).__init__()
        self.args = args
        self.device = device

        self.dropout_p = getattr(self.args, 'frame_clf_dropout_p', 0.2)
        self.dropout = nn.Dropout(self.dropout_p)

        self.conv_block = MyResConvBlock(self.args)
        self.conv_output_dim = self.conv_block.output_dim
        self.clf_head = nn.Linear(self.conv_output_dim, 2)

    # def set_label_weights(self, weights):
    #     self.loss_weight = torch.tensor(weights).to(self.device)

    def forward(self, x, y=None, x_raw=None, batch_xent_weight=None):
        batch_size = x.shape[0]

        x = x.unsqueeze(1)
        conv_out = self.conv_block(x)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.view(batch_size, -1)
        logits = self.clf_head(conv_out)
        outputs = (logits,)

        if y is not None:
            y = y.view(-1)

            loss_fct = CrossEntropyLoss(weight=batch_xent_weight)
            loss = loss_fct(logits, y)

            outputs = (loss, logits)

        return outputs
