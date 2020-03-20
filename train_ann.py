from torch import optim
import torch.utils.data as data
from setting import *


class Trainer:
    def __init__(self, inverse, record):
        self.inverse_net = inverse
        self.records = record
        self.iteration = record['iterations'][-1] if load else 0
        self.iter_num = record['iterations']
        self.losses = {'err_train': record['err_train'],
                       'err_test': record['err_test'],
                       'loss_train': record['loss_train'],
                       'loss_test': record['loss_test']}
        self.interval = 100
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.epoch = 100
        self.threshold = record['err_test'][-1] if load else 1  # 0 if not update

        train_numpy = np.array(np.load('./data/AI_train.npy'), dtype=np.float32)
        train_input = torch.FloatTensor(train_numpy[:, 0:p1]).to(device)
        train_label = torch.FloatTensor(train_numpy[:, p1:p2]).to(device)
        train_dataset = data.TensorDataset(train_input, train_label)
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        test_numpy = np.array(np.load('./data/AI_test.npy'), dtype=np.float32)
        self.test_input = torch.FloatTensor(test_numpy[:, 0:p1]).to(device)
        self.test_label = torch.FloatTensor(test_numpy[:, p1:p2]).to(device)

    def train(self):
        optimizer = optim.Adam(self.inverse_net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss(reduction='mean')

        for t in range(self.epoch):
            for step, item in enumerate(self.train_loader):
                # train
                train_input, train_label = item
                train_predict = self.inverse_net(train_input)
                loss_train = criterion(train_predict, train_label)

                if self.iteration % self.interval == 0:
                    end = time.time()

                    # test
                    test_predict = self.inverse_net(self.test_input)
                    loss_test = criterion(test_predict, self.test_label)
                    self.losses['loss_train'].append(to_numpy(loss_train))
                    self.losses['loss_test'].append(to_numpy(loss_test))

                    # compute and print the absolute error
                    forward_net = Forward_Net().to(device)
                    checkpoint_forward = torch.load('checkpoint_forward.pth')
                    forward_net.load_state_dict(checkpoint_forward['state_dict'])
                    train_out = forward_net(train_predict) - train_input
                    train_error = np.abs(to_numpy(train_out)).mean()
                    test_out = forward_net(test_predict) - self.test_input
                    test_error = np.abs(to_numpy(test_out)).mean()
                    self.losses['err_train'].append(train_error)
                    self.losses['err_test'].append(test_error)

                    print('iteration: {}, time: {}'.format(self.iteration, end-start))
                    print('train_loss: {:.4}, test_loss: {:.4}'.
                          format(loss_train, loss_test))
                    print('train_error: {:.4}, test_error: {:.4}'.
                          format(train_error, test_error))

                    self.iter_num.append(self.iteration)
                    self.records['err_train'] = self.losses['err_train']
                    self.records['err_test'] = self.losses['err_test']
                    self.records['loss_train'] = self.losses['loss_train']
                    self.records['loss_test'] = self.losses['loss_test']
                    self.records['iterations'] = self.iter_num
                    torch.save({'state_dict': self.inverse_net.state_dict(),
                                'records': self.records,
                                'time': time.time() - start},
                               'check_ann.pth')

                    if self.threshold > test_error:
                        self.threshold = test_error
                        # save the model
                        torch.save({'state_dict': self.inverse_net.state_dict(),
                                    'records': self.records,
                                    'time': time.time() - start},
                                   'checkpoint_ann.pth')

                # update parameters
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                self.iteration += 1


if __name__ == '__main__':
    load = False

    inverse_model = Inverse_Net().to(device)

    if load:
        checkpoint = torch.load('checkpoint_ann.pth')
        inverse_model.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        start = time.time()
        records = {'err_train': [], 'err_test': [],
                   'loss_train': [], 'loss_test': [],
                   'iterations': []}

    trainer = Trainer(inverse_model, records)
    trainer.train()

