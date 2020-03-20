from torch import optim
import torch.utils.data as data
from setting import *
# torch.manual_seed(0)


class Trainer:
    def __init__(self, model, iterations, iter_number, loss):
        self.forward_model = model
        self.iteration = iterations
        self.interval = 100
        self.losses = loss
        self.batch_size = 256
        self.lr = 1e-4
        self.epoch = 200
        self.iter_num = iter_number
        self.threshold = loss['err_test'][-1] if load else 1  # 0 if not update

        train_numpy = np.array(np.load('./data/AI_train.npy'), dtype=np.float32)
        train_input = torch.FloatTensor(train_numpy[:, p1:p2]).to(device)
        train_label = torch.FloatTensor(np.log(train_numpy[:, 0:p1])).to(device)
        train_dataset = data.TensorDataset(train_input, train_label)
        self.loader = data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        test_numpy = np.array(np.load('./data/AI_test.npy'), dtype=np.float32)
        self.test_input = torch.FloatTensor(test_numpy[:, p1:p2]).to(device)
        self.test_label = torch.FloatTensor(np.log(test_numpy[:, 0:p1])).to(device)

    def train(self):
        optimizer = optim.Adam(self.forward_model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction='mean')

        for t in range(self.epoch):
            for step, item in enumerate(self.loader):
                # train
                train_input, train_label = item
                train_predict = self.forward_model(train_input)
                loss_train = criterion(train_predict, train_label)

                if self.iteration % self.interval == 0:
                    # test
                    test_predict = self.forward_model(self.test_input)
                    loss_test = criterion(test_predict, self.test_label)
                    self.losses['loss_train'].append(to_numpy(loss_train))
                    self.losses['loss_test'].append(to_numpy(loss_test))

                    # compute and print the absolute error
                    train_out = train_predict - train_label
                    train_error = np.abs(to_numpy(train_out)).mean()
                    test_out = test_predict - self.test_label
                    test_error = np.abs(to_numpy(test_out)).mean()
                    self.losses['err_train'].append(train_error)
                    self.losses['err_test'].append(test_error)

                    print('iteration: {}'.format(self.iteration))
                    print('train_loss: {:.4}, test_loss: {:.4}'.
                          format(loss_train, loss_test))
                    print('train_error: {:.4}, test_error: {:.4}'.
                          format(train_error, test_error))

                    self.iter_num.append(self.iteration)

                    # save the model
                    if self.threshold > test_error:
                        self.threshold = test_error
                        torch.save({'iteration': self.iteration,
                                    'iter_num': self.iter_num,
                                    'state_dict': self.forward_model.state_dict(),
                                    'loss': self.losses,
                                    'time': time.time() - start},
                                   'checkpoint_forward_log.pth')

                # update parameters
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                self.iteration += 1


if __name__ == '__main__':
    load = False

    forward_model = Forward_Log_Net().to(device)

    if load:
        checkpoint_forward = torch.load('checkpoint_forward_log.pth')
        forward_model.load_state_dict(checkpoint_forward['state_dict'])
        iteration = checkpoint_forward['iteration']
        iter_num = checkpoint_forward['iter_num']
        losses = checkpoint_forward['loss']
        start = time.time() - checkpoint_forward['time']
    else:
        iteration = 0
        iter_num = []
        losses = {'err_train': [], 'err_test': [], 'loss_train': [], 'loss_test': []}
        start = time.time()

    trainer = Trainer(forward_model, iteration, iter_num, losses)
    trainer.train()
    # trainer.inference()


