import matplotlib.pyplot as plt 
import wandb 

def plot_result(result_dict, train=True):
    images = [] 
    epi_reward = result_dict['rewards']
    epi_action = result_dict['actions']
    epi_state  = result_dict['states']
    for k in range(10):
        vc_lst = [] 
        il_lst = [] 
        vc_trg_lst = [] 
        action_lst = [] 

        epi_num = epi_reward.index(max(epi_reward))+k-10

        for i in range(len(epi_state[epi_num])):
            v_in,vc,il,tg = epi_state[epi_num][i]
            vc_lst.append(vc)
            il_lst.append(il)
            vc_trg_lst.append(tg)


        for j in range(len(epi_action[epi_num])):
            action_lst.append(epi_action[epi_num][j])


        fig, ax = plt.subplots(nrows=3,ncols=1, figsize=(18,10))
        ax[0].plot(vc_lst, color='black',linewidth=1.1, label='$vc_t$')
        ax[0].plot(vc_trg_lst, color='red', label='target', linewidth=1.1, linestyle='--')
        ax[0].set_title("$vc_t$ : voltage at capacitor at time $t$")
        ax[0].grid()
        ax[0].set_xlabel("$time_t$")
        ax[0].set_ylabel("vc")
        ax[0].legend()

        ax[1].plot(il_lst, color='blue',linewidth=1.1, label='$IL_t$')
        ax[1].set_title("$IL_t$ : inductor current at time $t$")
        ax[1].grid()
        ax[1].set_xlabel("$time_t$")
        ax[1].set_ylabel("$IL$")
        ax[1].legend()



        ax[2].plot(action_lst, color='red',linewidth=1.1, label='$d_t$')
        ax[2].set_title("$a_t$ : action at time $t$")
        ax[2].grid()
        ax[2].set_xlabel("$time_t$")
        ax[2].set_ylabel("$d_t$")
        ax[2].legend()


        plt.subplots_adjust(left=0.125,
                            bottom=0.1, 
                            right=0.9, 
                            top=0.9, 
                            wspace=0.4, 
                            hspace=0.5)

        if train :
            file_name = f'training_result_{k}_sample'
        else : 
            file_name = f'testing_result_{k}_sample'

        print(f"#### Saving plot ##### :{k}")
        plt.close('all') 
        wandb.log({file_name: wandb.Image(fig)})