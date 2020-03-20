from torch import nn, optim
import torch.utils.data as data
from torch.autograd import Variable, grad
from botorch.utils.transforms import standardize, normalize, unnormalize
from setting import *


class Trainer:
    def __init__(self, generator, discriminator, forward_net, records):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.F = forward_net.to(device)
        self.iteration = records['iterations'][-1] if load else 0
        self.iter_num = records['iterations']
        self.loss = {'err_train': records['err_train'],
                     'err_test': records['err_test'],
                     'G': records['G'], 'D': records['D'],
                     'grad_norm': records['grad_norm'],
                     'GP': records['GP'], 'distance': records['distance']}
        self.records = records
        self.lamb = 10
        self.critic = 4
        self.interval = 20
        self.batch_size = 256
        self.learning_rate = 1e-4
        self.epoch = 100
        self.threshold = record['err_test'][-1] if load else 1  # 0 if not update

        # loading train data
        train_numpy = np.array(np.load('./data/AI_train.npy'), dtype=np.float32)
        train_label = torch.FloatTensor(train_numpy[:, 0:p1]).to(device)
        train_real = torch.FloatTensor(train_numpy[:, p1:p2]).to(device)
        train_dataset = data.TensorDataset(train_label, train_real)
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        # loading test data
        test_numpy = np.array(np.load('./data/AI_test.npy'), dtype=np.float32)
        self.test_label = torch.FloatTensor(test_numpy[:, 0:p1]).to(device)
        self.test_real = torch.FloatTensor(test_numpy[:, p1:p2]).to(device)

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

    def gradient_penalty(self, real_data, fake_data, label_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)
        interpolates = Variable(interpolates, requires_grad=True)

        # Calculate probability of interpolated examples
        disc_interpolates = self.D(interpolates, label_data)

        # Calculate gradients of probabilities w.r.t examples
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                         create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height)
        # Flatten gradients to easily take norm per example in batch
        gradients = gradients.view(self.batch_size, -1)
        self.loss['grad_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, thus manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gp = self.lamb * ((gradients_norm - 1) ** 2).mean()

        return gp

    def train(self):
        # set another iteration index for-loop (not self.iteration)
        i = self.iteration * (self.critic + 1)

        # optimization strategies
        g_opt = optim.Adam(self.G.parameters(), lr=self.learning_rate)
        d_opt = optim.Adam(self.D.parameters(), lr=self.learning_rate)

        for t in range(self.epoch):
            for step, item in enumerate(self.train_loader):
                i += 1

                noise_data = np.random.uniform(high=1, low=-1, size=(self.batch_size, 10))
                noise_tensor = torch.FloatTensor(noise_data).to(device)

                # acquire next batch data
                train_label, train_real = item  # 170-label, 5-real
                train_fake = self.G(noise_tensor, train_label)

                # update D network
                if i % (self.critic + 1) != 0:
                    for p in self.D.parameters():
                        p.requires_grad = True

                    # empty gradients in discriminator
                    self.D.zero_grad()

                    # calculate probabilities on data
                    d_out_real = self.D(train_real, train_label)
                    d_real = d_out_real.mean()
                    d_out_fake = self.D(train_fake, train_label)
                    d_fake = d_out_fake.mean()

                    # get and record gradient penalty
                    gp = self.gradient_penalty(train_real.data, train_fake.data, train_label.data)
                    self.loss['GP'].append(gp.item())

                    # get total discriminator loss and wasserstein distance
                    d_loss = d_fake - d_real + gp

                    self.loss['D'].append(d_loss.item())
                    w_distance = d_real - d_fake
                    self.loss['distance'].append(w_distance.item())

                    # optimize
                    d_loss.backward()
                    d_opt.step()

                # update G network
                else:
                    for p in self.D.parameters():
                        p.requires_grad = False

                    # empty gradients in generator
                    self.G.zero_grad()

                    # calculate probabilities on fake data
                    d_fake = self.D(train_fake, train_label)
                    d_fake = d_fake.mean()

                    # get loss, record, and optimize
                    g_loss = -d_fake
                    self.loss['G'].append(g_loss.item())
                    g_loss.backward()
                    g_opt.step()

                # recording the training process
                per = (self.critic + 1) * self.interval
                if i % per == 0:
                    end = time.time()
                    self.iteration = int(i)

                    # compute train error
                    spectrum_real = to_numpy(self.F(train_real))
                    spectrum_fake = to_numpy(self.F(train_fake))
                    err_train = np.abs(spectrum_real - spectrum_fake).mean()

                    # compute test error
                    noise_data = np.random.uniform(high=1, low=-1, size=(self.test_label.shape[0], 10))
                    noise_tensor = torch.FloatTensor(noise_data).to(device)
                    test_fake = self.G(noise_tensor, self.test_label)
                    spectrum_real = to_numpy(self.F(self.test_real))
                    spectrum_fake = to_numpy(self.F(test_fake))
                    err_test = np.abs(spectrum_real - spectrum_fake).mean()

                    self.loss['err_train'].append(err_train)
                    self.loss['err_test'].append(err_test)
                    self.iter_num.append(self.iteration)

                    # output loss
                    print("iteration: {}, time: {}".format(self.iteration, end-start))
                    print("Train error: {}".format(err_train))
                    print("Test error: {}".format(err_test))
                    print("D loss: {}".format(self.loss['D'][-1]))
                    print("G loss: {}".format(self.loss['G'][-1]))
                    # print("GP: {}".format(self.loss['GP'][-1]))
                    # print("Gradient norm: {}".format(self.loss['grad_norm'][-1]))
                    # print("Wasserstein distance: {}".format(self.loss['distance'][-1]))

                    self.records['err_train'] = self.loss['err_train']
                    self.records['err_test'] = self.loss['err_test']
                    self.records['G'] = self.loss['G']
                    self.records['D'] = self.loss['D']
                    self.records['GP'] = self.loss['GP']
                    self.records['grad_norm'] = self.loss['grad_norm']
                    self.records['distance'] = self.loss['distance']
                    self.records['iterations'] = self.iter_num
                    torch.save({'G_state_dict': self.G.state_dict(),
                                'D_state_dict': self.D.state_dict(),
                                'time': time.time() - start,
                                'records': self.records},
                               'check_gan.pth')

                    # save model
                    if self.threshold > err_test:
                        self.threshold = err_test
                        torch.save({'G_state_dict': self.G.state_dict(),
                                    'D_state_dict': self.D.state_dict(),
                                    'time': time.time() - start,
                                    'records': self.records},
                                   'checkpoint_gan.pth')

        return torch.tensor([self.threshold]).to(device)


if __name__ == '__main__':
    f = Forward_Net()
    checkpoint_forward = torch.load('checkpoint_forward.pth')
    f.load_state_dict(checkpoint_forward['state_dict'])

    load = False

    if load:
        g = Generator()
        d = Discriminator()
        checkpoint = torch.load('checkpoint_gan.pth')
        g.load_state_dict(checkpoint['G_state_dict'])
        d.load_state_dict(checkpoint['D_state_dict'])
        record = checkpoint['records']
        start = time.time() - checkpoint['time']
    else:
        g = Generator()
        d = Discriminator()
        record = {'G': [], 'D': [], 'GP': [], 'grad_norm': [], 'distance': [],
                  'err_train': [], 'err_test': [], 'iterations': []}
        start = time.time()

    trainer = Trainer(g, d, f, record)
    trainer.train()


