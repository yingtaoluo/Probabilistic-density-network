from torch import optim
import torch.utils.data as data
from setting import *


class Trainer:
    def __init__(self, forward, inverse, records):
        self.forward_net = forward
        self.inverse_net = inverse
        self.records = records
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

        for p in self.forward_net.parameters():
            p.requires_grad = False

        for p in self.inverse_net.parameters():
            p.requires_grad = True

        for t in range(self.epoch):
            for step, item in enumerate(self.train_loader):
                # train
                train_input, train_label = item
                train_predict = self.forward_net(self.inverse_net(train_input))
                loss_train = criterion(train_predict, train_input)

                if self.iteration % self.interval == 0:
                    end = time.time()

                    # test
                    test_predict = self.forward_net(self.inverse_net(self.test_input))
                    loss_test = criterion(test_predict, self.test_input)
                    self.losses['loss_train'].append(to_numpy(loss_train))
                    self.losses['loss_test'].append(to_numpy(loss_test))

                    # compute and print the absolute error
                    train_out = train_predict - train_input
                    train_error = np.abs(to_numpy(train_out)).mean()
                    test_out = test_predict - self.test_input
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
                               'check_tnn.pth')

                    # save the model
                    if self.threshold > test_error:
                        self.threshold = test_error
                        torch.save({'state_dict': self.inverse_net.state_dict(),
                                    'records': self.records,
                                    'time': time.time() - start},
                                   'checkpoint_tnn.pth')

                # update parameters
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                self.iteration += 1

    def inference(self, c):
        spectrum_input = self.test_input[c:c+1]
        intermediate = self.inverse_net(spectrum_input)
        structure_predict = to_numpy(intermediate.view(-1)) * np.pi
        spectrum_predict = to_numpy(self.forward_net(intermediate).view(-1)) * 100
        spectrum_label = to_numpy(self.test_input[c]) * 100
        structure_label = to_numpy(self.test_label[c]) * np.pi

        plt.figure(1)
        spectra = np.linspace(1, 340, 340) * 50
        plt.title('Comparison of Transmission Spectrum')
        plt.plot(spectra, spectrum_predict, color=color1, label='Prediction')
        plt.plot(spectra, spectrum_label, color=color2, label='Simulation')
        plt.legend(loc='upper right')
        plt.xlabel('Frequency (HZ)')
        plt.ylabel('Intensity (%)')
        plt.savefig('./figures/tandem/spec/' + str(c) + '.jpg')
        plt.close()

        plt.figure(2)
        spectra = np.linspace(1, 5, 5)
        plt.title('Comparison of Metamaterial Structure')
        plt.plot(spectra, structure_predict, color=color1, label='Prediction', marker='o')
        plt.plot(spectra, structure_label, color=color2, label='Simulation', marker='o')
        plt.legend(loc='upper right')
        plt.xlabel('Layer')
        plt.ylabel('Width (cm)')
        plt.savefig('./figures/tandem/des/' + str(c) + '.jpg')
        plt.close()


if __name__ == '__main__':
    load = False
    train = True

    forward_model = Forward_Net().to(device)
    checkpoint_forward = torch.load('checkpoint_forward.pth')
    forward_model.load_state_dict(checkpoint_forward['state_dict'])
    inverse_model = Inverse_Net().to(device)

    if load:
        checkpoint = torch.load('checkpoint_tnn.pth')
        inverse_model.load_state_dict(checkpoint['state_dict'])
        record = checkpoint['records']
        start = time.time() - checkpoint['time']
    else:
        record = {'err_train': [], 'err_test': [],
                  'loss_train': [], 'loss_test': [],
                  'iterations': []}
        start = time.time()

    trainer = Trainer(forward_model, inverse_model, record)

    if train:
        trainer.train()
    else:
        for i in range(100):
            trainer.inference(i)


