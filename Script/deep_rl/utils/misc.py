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
    random_seed()
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
        loss = agent.step()

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
