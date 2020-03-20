from torch import optim
import torch.utils.data as data
from setting import *
import time


class Trainer:
    def __init__(self, model, forward_net, record):
        self.model = model
        self.forward_net = forward_net
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
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.epoch):
            for step, item in enumerate(self.train_loader):
                # train
                train_input, train_label = item
                pi0, sigma0, mu0 = self.model(train_input)
                loss_train = mdn_loss(pi0, sigma0, mu0, train_label)

                if self.iteration % self.interval == 0:
                    end = time.time()
                    # test
                    pi1, sigma1, mu1 = self.model(self.test_input)
                    loss_test = mdn_loss(pi1, sigma1, mu1, self.test_label)
                    self.losses['loss_train'].append(to_numpy(loss_train))
                    self.losses['loss_test'].append(to_numpy(loss_test))

                    # compute and print the absolute error
                    train_predict = sample(pi0, sigma0, mu0)
                    train_diff = self.forward_net(train_predict) - train_input
                    train_error = np.abs(to_numpy(train_diff)).mean()
                    test_predict = self.forward_net(sample(pi1, sigma1, mu1))
                    test_diff = test_predict - self.test_input
                    test_error = np.abs(to_numpy(test_diff)).mean()
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
                    torch.save({'state_dict': self.model.state_dict(),
                                'records': self.records,
                                'time': time.time() - start},
                               'check_pdn.pth')

                    if self.threshold > test_error:
                        self.threshold = test_error
                        # save the model
                        torch.save({'state_dict': self.model.state_dict(),
                                    'records': self.records,
                                    'time': time.time() - start},
                                   'checkpoint_pdn.pth')

                # update parameters
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                self.iteration += 1
                scheduler.step()


if __name__ == '__main__':
    torch.manual_seed(0)
    f = Forward_Net().to(device)
    checkpoint_forward = torch.load('checkpoint_forward.pth')
    f.load_state_dict(checkpoint_forward['state_dict'])
    m = PDN().to(device)

    load = False
    train = True

    if load:
        checkpoint = torch.load('checkpoint_pdn.pth')
        m.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'err_train': [], 'err_test': [],
                   'loss_train': [], 'loss_test': [],
                   'iterations': []}
        start = time.time()

    trainer = Trainer(m, f, records)
    trainer.train()





