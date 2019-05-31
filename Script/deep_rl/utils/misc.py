#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime
import torch
import time
import psutil
import json
import pickle
from .torch_utils import *
try:
    # python >= 3.5
    from pathlib import Path
except:
    # python == 2.7
    from pathlib2 import Path

def run_steps(agent):
    config = agent.config
    agent_name = config.name
    t0 = time.time()
    loss = 0
    try:
        f = open('data/result-%s-%s-%s.json' % (agent_name, config.task_name, config.tag), "r")
        config.logger.writer.scalar_dict = json.load(f)
        episode_num = config.logger.writer.scalar_dict['./log/run time/'+agent_name+config.task_name+config.tag][-1][1]
    except Exception as e:
        print("no json found", e)
        episode_num = 0

    while True:      
        loss += agent.step()
        if config.log_interval and not agent.total_steps % config.log_interval and len(agent.episode_rewards):
            rewards = agent.episode_rewards
            old_ep_num = episode_num
            episode_num += len(agent.episode_rewards)
            # config.logger.info('total steps %d, \
            # returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), \
            # %.2f steps/s' % (agent.total_steps, np.mean(rewards), \
            # np.median(rewards), np.min(rewards), np.max(rewards),config.log_interval \
            # / (time.time() - t0)))
            mem = psutil.virtual_memory()
            episode_time = time.time()-t0
            config.logger.writer.add_scalars("loss", {
                        agent_name+config.task_name+config.tag: loss,            
                        }, episode_num
                    )
            config.logger.writer.add_scalars("run time", {
                        agent_name+config.task_name+config.tag: episode_time/len(agent.episode_rewards),            
                        }, episode_num
                    )
            for ep in range(old_ep_num, episode_num):
                config.logger.writer.add_scalars("reward", {
                            agent_name+config.task_name+config.tag: rewards[ep-old_ep_num],            
                            }, ep
                        )
            config.logger.writer.add_scalars("available memory", {
                        agent_name+config.task_name+config.tag: mem.available/100.0,            
                        }, episode_num
                    )

            print('eps %d, total steps %d, returns %.2f/%.2f/%.2f/%.2f (mean/median/min/max), \
                %.2f steps/s' % (episode_num, agent.total_steps, np.mean(rewards), \
                np.median(rewards), np.min(rewards), np.max(rewards),\
                config.log_interval/(time.time() - t0)))

            agent.episode_rewards = []
            t0 = time.time()
            loss = 0

        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()

        if config.save_interval and not agent.total_steps % config.save_interval:
            print("saving model!")
            agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
            temp_scalar = config.logger.writer.scalar_dict
            config.logger.writer.export_scalars_to_json('data/result-%s-%s-%s.json' % (agent_name, config.task_name, config.tag))
            config.logger.writer.scalar_dict = temp_scalar

        if (config.max_steps and agent.total_steps >= config.max_steps) \
            or (config.max_eps and episode_num>=config.max_eps):
            # config.logger.writer.export_scalars_to_json("./"+agent_name+".json")
            # config.logger.writer.close()
            agent.close()
            break

def train_curr(agent):
    config = agent.config
    sub_goals = config.sub_goals
    agent_name = config.name
    max_len = len(sub_goals)
    task_rewards = np.zeros(len(sub_goals))
    task_times = np.ones(len(sub_goals))
    task_avg_rewards = np.zeros(len(sub_goals))
    task_tot_rewards = np.zeros(len(sub_goals))
    task_tot_times = np.ones(len(sub_goals))

    filename = "data/"+agent_name+config.task_name+config.tag+".misc"
    t = 0
    task_len_start = 1

    try:
        with open(filename, 'rb') as f:
            pick_load = pickle.load(f)
            t = pick_load[0]
            task_len_start = pick_load[1]
            task_tot_rewards = pick_load[2]
            task_tot_times = pick_load[3]
            task_rewards = pick_load[4]
            task_times = pick_load[5]
            task_avg_rewards = pick_load[6]

    except Exception as e:
        print(e)

    print(task_len_start)
    print(max_len+1)
    for task_len in range(task_len_start, max_len+1):
        r_min = -1
        all_tasks = sub_goals[0:task_len]
        all_probs = 1.0/task_len*np.ones(task_len)
        print("max task length", task_len)
        while r_min < config.reward_thres or (task_len==max_len and t < config.max_eps):
            if config.reward_batch and t % config.reward_batch == 0:
                task_rewards = config.reward_batch*task_rewards/task_times
                task_times = config.reward_batch*np.ones(len(sub_goals))

            task_choice = np.random.choice(a=range(1,task_len+1), p=all_probs)

            if task_avg_rewards[task_choice-1] > 0.95:
                continue

            print("task_choice ", task_choice-1)
            task = all_tasks[0:task_choice]
            s_g_step = task[-1]
            step_idx = len(task)-1

            # only update last subplicy
            for roll_idx in range(step_idx):
                s_g_roll = task[roll_idx]
                agent.roll(s_g_roll)
                task_rewards[roll_idx] += agent.roll_rewards
                task_times[roll_idx] += 1
                task_tot_rewards[roll_idx] = agent.roll_rewards
                task_tot_times[roll_idx] = 1
                print("finish rollout", roll_idx, "reward", agent.roll_rewards)
                if agent.roll_rewards<0:
                    agent.states = agent.task.reset()
                    break

            if agent.roll_rewards > 0 or step_idx == 0:
                agent.step(s_g_step)
                task_rewards[step_idx] += agent.step_rewards
                task_times[step_idx] += 1
                task_tot_rewards[step_idx] = agent.step_rewards
                task_tot_times[step_idx] = 1
                agent.states = agent.task.reset()
                t += 1

            # # update all subplicy
            # for roll_idx in range(step_idx+1):
            #     s_g_roll = task[roll_idx]
            #     agent.step(s_g_roll)
            #     task_rewards[roll_idx] += agent.step_rewards
            #     task_times[roll_idx] += 1
            #     task_tot_rewards[roll_idx] = agent.step_rewards
            #     task_tot_times[roll_idx] = 1
            #     print("finish update", roll_idx, "reward", agent.step_rewards)
                
            # agent.states = agent.task.reset()

            # update average reward for each task
            task_avg_rewards = 1.0*task_rewards/task_times
            task_len_rewards = task_avg_rewards[0:task_len]
            print("task average reward",task_len_rewards)
            # update prob for each task
            z = np.sum(np.exp(-2*task_len_rewards))
            all_probs = (1.0/z)*np.exp(-2*task_len_rewards)
            r_min = np.min(task_len_rewards)

            for i in range(len(task_avg_rewards)):
                config.logger.writer.add_scalars("avg_rewards", {
                        agent_name+config.task_name+config.tag+str(i): task_avg_rewards[i]
                        }, t)

            for i in range(len(task_avg_rewards)):
                config.logger.writer.add_scalars("rewards", {
                        agent_name+config.task_name+config.tag+str(i): 1.0*task_tot_rewards[i]/task_tot_times[i]
                        }, t)

            for i in range(len(task_len_rewards)):
                config.logger.writer.add_scalars("avg_probs", {
                        agent_name+config.task_name+config.tag+str(i): all_probs[i]
                        }, t)

            

            if config.save_interval and t % config.save_interval == 0:
                print("saving model!")
                agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
                temp_scalar = config.logger.writer.scalar_dict
                config.logger.writer.export_scalars_to_json('data/result-%s-%s-%s.json' % (agent_name, config.task_name, config.tag))
                config.logger.writer.scalar_dict = temp_scalar

                with open(filename, 'wb') as f:
                    pickle.dump([t, task_len, task_tot_rewards, task_tot_times, task_rewards, task_times, task_avg_rewards], f)


    print("saving model!")
    agent.save('data/model-%s-%s-%s.bin' % (agent_name, config.task_name, config.tag))
    temp_scalar = config.logger.writer.scalar_dict
    config.logger.writer.export_scalars_to_json('data/result-%s-%s-%s.json' % (agent_name, config.task_name, config.tag))
    config.logger.writer.scalar_dict = temp_scalar
    with open(filename, 'wb') as f:
        pickle.dump([t, task_len, task_tot_rewards, task_tot_times, task_rewards, task_times, task_avg_rewards], f)


def eval_curr(agent):
    config = agent.config
    sub_goals = config.sub_goals
    agent_name = config.name
    eval_eps = config.eval_eps
    task_rewards = 0
    task_times = 0

    for t in range(eval_eps):
        for s_g in sub_goals:
            agent.eval(s_g)
        task_rewards += agent.eval_rewards
        task_times += 1

    print("average reward for eval",1.0*task_rewards/task_times)

            
def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()

class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
