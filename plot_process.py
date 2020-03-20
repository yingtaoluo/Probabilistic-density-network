from setting import *
import matplotlib.pyplot as plt


def plotting():
    plt.figure(1)
    plt.plot(records['iterations'], records['err_train'], label='error per hundred iterations')
    min_index = int(np.argmin(records['err_train']))
    mini = records['err_train'][min_index]
    plt.plot(min_index * 100, mini, color=color1, marker='o',
             markersize=5, label='error minimum: ' + str(np.round(mini, 5)))
    plt.title(choice.upper() + '  Train Error', fontsize=font2)
    plt.legend(loc='upper right')
    plt.xlabel('Number of iterations', fontsize=font3)
    plt.ylabel('Error value', fontsize=font3)
    plt.tight_layout()
    plt.savefig('./figures/process/' + str(choice) + '_train.png')
    # plt.show()
    plt.close()

    plt.figure(2)
    plt.plot(records['iterations'], records['err_test'], label='error per hundred iterations')
    min_index = int(np.argmin(records['err_test']))
    mini = records['err_test'][min_index]
    plt.plot(min_index * 100, mini, color=color1, marker='o',
             markersize=5, label='error minimum: ' + str(np.round(mini, 5)))
    plt.title(choice.upper() + '  Test Error', fontsize=font2)
    plt.legend(loc='upper right')
    plt.xlabel('Number of iterations', fontsize=font3)
    plt.ylabel('Error value', fontsize=font3)
    plt.tight_layout()
    plt.savefig('./figures/process/' + str(choice) + '_test.png')
    # plt.show()
    plt.close()

    # np.savetxt('./ann.csv', records['err_test'], delimiter=',')

    print('time:{}'.format(start))
    print('iteration:{}'.format(records['iterations'][-1]))
    print('err_train:{}, err_test:{}'.format(records['err_train'], records['err_test']))
    # print('loss_train:{}, loss_test:{}'.format(records['loss_train'], records['loss_test']))
    print(min_index * 100)


if __name__ == '__main__':
    choice = 'pdn'
    checkpoint = torch.load('check_' + choice + '.pth')
    start = checkpoint['time']
    records = checkpoint['records']

    plotting()

