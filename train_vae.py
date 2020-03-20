from torch import optim
import torch.utils.data as data
from botorch.utils.transforms import standardize, normalize, unnormalize
from setting import *


class Trainer:
    def __init__(self, model, forward_net, record):
        self.model = model.to(device)
        self.F = forward_net.to(device)
        self.record = record
        self.iteration = record['iterations'][-1] if load else 0
        self.iter_num = record['iterations']
        self.loss = {'err_train': record['err_train'],
                     'err_test': record['err_test'],
                     'loss_train': record['loss_train'],
                     'loss_test': record['loss_test']}
        self.interval = 100
        self.batch_size = 256
        self.epoch = 100
        self.threshold = record['err_test'][-1] if load else 1  # 0 if not update

        # loading train data
        train_numpy = np.array(np.load('./data/AI_train.npy'), dtype=np.float32)
        train_numpy[:, :-5] /= 1.45  # to use BCE loss
        train_label = torch.FloatTensor(train_numpy[:, 0:p1]).to(device)
        train_data = torch.FloatTensor(train_numpy[:, p1:p2]).to(device)
        train_dataset = data.TensorDataset(train_data, train_label)
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        # loading test data
        test_numpy = np.array(np.load('./data/AI_test.npy'), dtype=np.float32)
        test_numpy[:, :-5] /= 1.45  # to use BCE loss
        self.test_label = torch.FloatTensor(test_numpy[:, 0:p1]).to(device)
        self.test_data = torch.FloatTensor(test_numpy[:, p1:p2]).to(device)

    def train_loop(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_model
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.optim import optimize_acqf
        from botorch.acquisition.monte_carlo import qExpectedImprovement
        from botorch.sampling.samplers import SobolQMCNormalSampler

        seed = 1
        torch.manual_seed(seed)
        dt, d = torch.float32, 3
        lb, ub = [1e-4, 0.1, 0.1], [3e-3, 1-1e-3, 1-1e-3]
        bounds = torch.tensor([lb, ub], dtype=dt)

        def gen_initial_data():
            # auto
            # x = unnormalize(torch.rand(1, 3, dtype=dt), bounds=bounds)
            # manual
            x = torch.tensor([[1e-3, 0.9, 0.999]])
            print('BO Initialization: \n')
            print('Initial Hyper-parameter: ' + str(x))
            obj = self.train(x.view(-1))
            print('Initial Error: ' + str(obj))
            return x, obj.unsqueeze(1)

        def get_fitted_model(x, obj, state_dict=None):
            # initialize and fit model
            fitted_model = SingleTaskGP(train_X=x, train_Y=obj)
            if state_dict is not None:
                fitted_model.load_state_dict(state_dict)
            mll = ExactMarginalLogLikelihood(fitted_model.likelihood, fitted_model)
            mll.to(x)
            fit_gpytorch_model(mll)
            return fitted_model

        def optimize_acqf_and_get_observation(acq_func):
            """Optimizes the acquisition function,
            and returns a new candidate and a noisy observation"""
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([
                    torch.zeros(d, dtype=dt),
                    torch.ones(d, dtype=dt),
                ]),
                q=1,
                num_restarts=10,
                raw_samples=200,
            )

            x = unnormalize(candidates.detach(), bounds=bounds)
            print('Hyper-parameter: ' + str(x))
            obj = self.train(x.view(-1)).unsqueeze(-1)
            print(print('Error: ' + str(obj)))
            return x, obj

        N_BATCH = 500
        MC_SAMPLES = 2000
        best_observed = []
        train_x, train_obj = gen_initial_data()  # (1,3), (1,1)
        best_observed.append(train_obj.view(-1))

        print(f"\nRunning BO......\n ", end='')
        state_dict = None
        for iteration in range(N_BATCH):
            # fit the model
            model = get_fitted_model(
                normalize(train_x, bounds=bounds),
                standardize(train_obj),
                state_dict=state_dict,
            )

            # define the qNEI acquisition module using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
            qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_obj).max())

            # optimize and get new observation
            new_x, new_obj = optimize_acqf_and_get_observation(qEI)

            # update training points
            train_x = torch.cat((train_x, new_x))
            train_obj = torch.cat((train_obj, new_obj))

            # update progress
            best_value = train_obj.max().item()
            best_observed.append(best_value)

            state_dict = model.state_dict()
            print(".", end='')

        print(best_observed)

    def train(self, hp):
        iteration = self.iteration
        # optimization strategies
        opt = optim.Adam(self.model.parameters(), lr=hp[0], betas=(hp[1], hp[2]))
        criterion = nn.BCELoss(reduction='mean')

        for t in range(self.epoch):
            for step, item in enumerate(self.train_loader):
                # acquire next batch data
                train_data, train_label = item

                # acquire loss
                train_predict, latent_loss = self.model(train_data, train_label)
                loss_train = criterion(train_predict, train_data) + latent_loss

                if iteration % self.interval == 0:
                    end = time.time()

                    # test
                    test_predict, latent_loss = self.model(self.test_data, self.test_label)
                    loss_test = criterion(test_predict, self.test_data) + latent_loss
                    self.loss['loss_train'].append(to_numpy(loss_train))
                    self.loss['loss_test'].append(to_numpy(loss_test))

                    # compute and print the absolute error
                    train_out = self.F(train_predict) - train_label
                    train_error = np.abs(to_numpy(train_out)).mean()
                    test_out = self.F(test_predict) - self.test_label
                    test_error = np.abs(to_numpy(test_out)).mean()
                    self.loss['err_train'].append(train_error)
                    self.loss['err_test'].append(test_error)

                    print('iteration: {}, time: {}'.format(iteration, end-start))
                    # print('train_loss: {:.4}, test_loss: {:.4}'.
                    #       format(to_numpy(loss_train), loss_test))
                    # print('latent_loss: {:.4}'.format(latent_loss))
                    print('train_error: {:.4}, test_error: {:.4}'.
                          format(train_error, test_error))

                    self.iter_num.append(iteration)
                    self.record['err_train'] = self.loss['err_train']
                    self.record['err_test'] = self.loss['err_test']
                    self.record['loss_train'] = self.loss['loss_train']
                    self.record['loss_test'] = self.loss['loss_test']
                    self.record['iterations'] = self.iter_num
                    torch.save({'state_dict': self.model.state_dict(),
                                'records': self.record,
                                'time': time.time() - start},
                               'check_vae.pth')

                    # save the model
                    if self.threshold > test_error:
                        self.threshold = test_error
                        torch.save({'state_dict': self.model.state_dict(),
                                    'records': self.record,
                                    'time': time.time() - start},
                                   'checkpoint_vae.pth')

                # update parameters
                opt.zero_grad()
                loss_train.backward()
                opt.step()

                iteration += 1

        # the best result obtained
        return torch.tensor([self.threshold]).to(device)


if __name__ == '__main__':
    f = Forward_Net()
    checkpoint_forward = torch.load('checkpoint_forward.pth')
    f.load_state_dict(checkpoint_forward['state_dict'])

    hyper = {'ld': 20, 'd': 100}
    err = np.zeros(shape=(10, 10))

    encoder = Encoder(hyper['d'])
    decoder = Decoder(hyper['ld'], hyper['d'])
    m = VAE(encoder, decoder, hyper['ld'])

    load = False

    if load:
        checkpoint = torch.load('checkpoint_vae.pth')
        m.load_state_dict(checkpoint['state_dict'])
        records = checkpoint['records']
        start = time.time() - checkpoint['time']
    else:
        records = {'err_train': [], 'err_test': [],
                   'loss_train': [], 'loss_test': [],
                   'iterations': []}
        start = time.time()

    x = torch.tensor([1e-3, 0.9, 0.999])
    trainer = Trainer(m, f, records)
    trainer.train(x)


