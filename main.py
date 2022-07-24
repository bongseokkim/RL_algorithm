import argparse
from model.td3 import TD3
import wandb 
import os 
import numpy
import pathlib
import os.path
from plecsSubModule.plecsEnvSetting import plecsEnvSetting
from stable_baselines3.common.monitor import Monitor
from plecsSubModule.interfacePLECS import interfacePLECS
from plecsSubModule.plecsRLCallback import plecsRLCallback
from utils.experiment_setting import set_experiment_dir
from utils.plot_result import plot_result
from model.td3 import TD3


def main():
    # plecs env setting 
    plecsEnvParam = plecsEnvSetting.plecsEnvParameter(
    plecsDirPath= plecsEnvSetting.plecsDirPath(
      plecsVer= "PLECS 4.4",
      index= 0, mix= False),
    timeStep= 200.e-6,
    simTime = 100.e-3,
    nState  = 4,
    nAction = 1,
    actionHi= [+4.5],
    actionLo= [-4.5])
    modelDict = dict({})
    env_class=interfacePLECS(plecsSetup= plecsEnvParam, modelDict= modelDict)
    env = Monitor(env_class)


    # experiment parameter setting  
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", default=1, type=int)
    parser.add_argument("--train_timesteps", default=7e3,type=int)
    parser.add_argument("--test_timesteps", default=7e3,type=int)
    parser.add_argument("--batch_size", default=128,type=int)
    parser.add_argument("--actor_lr", default=0.0001,type=float)
    parser.add_argument("--critic_lr", default=0.0001,type=float)
    parser.add_argument("--buffer_size", default=1000000,type=int)
    parser.add_argument("--update_interval", default=5,type=int)
    parser.add_argument("--actor_layer_size", default=[8,8,8],type=list)
    parser.add_argument("--critic_layer_size", default=[128,128],type=list)
    parser.add_argument("--exp_noise", default=0.1,type=float)
    parser.add_argument("--pol_noise", default=0.2,type=float)

    args = parser.parse_args()

    # experiment setting 
    name = 'test'
    dir_path  = set_experiment_dir(name)
    wandb.init( project='plecs-test2',name = f'experiment_{dir_path}')
    wandb.config.update(args)

    model = TD3(env=env, actor_lr=args.actor_lr, critic_lr=args.critic_lr, input_dim=4, output_dim=1, tau=0.01, pol_noise=args.pol_noise, exp_noise=args.exp_noise,
                chkpt_dir=dir_path, actor_num_nuerons=[8,8,8], critic_num_nuerons=[128,128], batch_size=args.batch_size, 
                max_size=args.buffer_size, warmup=500,  update_actor_interval=args.update_interval)
    
    # train agent, show result  
    train_result_dict = model.learn(args.train_timesteps, window=args.window_size, wandb_log=True,)
    plot_result(train_result_dict, train=True)
    model.load_models()

    # test agent, show result  
    test_result_dict = model.test(total_timesteps=args.test_timesteps, window=args.window_size, wandb_log=True,)
    plot_result(test_result_dict, train=False)
    wandb.finish()

if __name__=="__main__":
    main()
