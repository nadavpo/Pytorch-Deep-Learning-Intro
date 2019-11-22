import matplotlib.pyplot as plt


def plot_train_results(best_acc, best_loss, accs_train, losses_train, accs_val, losses_val):
    fig, axs = plt.subplots(2)
    axs[0].plot(losses_train)
    axs[0].plot(losses_val)
    axs[0].set_title('running  loss')
    axs[0].legend(['train', 'validation'])
    axs[1].plot(accs_train)
    axs[1].plot(accs_val)
    axs[1].set_title('running accuracy')
    axs[1].legend(['train', 'validation'])
    # plt.ion()
    plt.show()