#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
from .lang_encoder import *
from .state_encoder import *
from .fusion import *

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std.data)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        # print logits
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v

class SubGoalActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 dict_size=45,
                 lang_embed_dim=64,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 latent_dim=256):
        super(SubGoalActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.lang_embed_dim = lang_embed_dim
        self.dict_size = dict_size
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.latent_dim = latent_dim
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        self.lang_embeds = LangEncoderBoW(self.dict_size, self.lang_embed_dim)
        self.fc1 = layer_init(nn.Linear(2*self.lang_embed_dim, self.phi_body.feature_dim), 1e-3)
        self.fusion_cat = FusionCat(self.phi_body.feature_dim, self.phi_body.feature_dim)
        self.fc2 = layer_init(nn.Linear(self.phi_body.feature_dim, self.phi_body.feature_dim), 1e-3)
        # self.lstm = nn.LSTMCell(self.phi_body.feature_dim, self.latent_dim)
        # self.hidden = self.init_hidden()

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        self.to(Config.DEVICE)

    def init_hidden(self):
        return (torch.rand(1, self.latent_dim).to(Config.DEVICE), \
            torch.rand(1, self.latent_dim).to(Config.DEVICE))

    def forward(self, obs, subgoal):
        task_embed = self.lang_embeds(subgoal[0].view(1,1))
        ingred_embed = self.lang_embeds(subgoal[1].view(1,1))
        phi = self.phi_body(obs)
        t_i_cat = torch.cat((task_embed, ingred_embed), dim=-1)
        t_i_cat = F.relu(self.fc1(t_i_cat))
        t_i_im_cat = self.fusion_cat(phi, t_i_cat)
        phi_cat = F.relu(self.fc2(t_i_im_cat))
        # self.hidden = self.lstm(phi_cat, self.hidden)

        phi_a = self.actor_body(phi_cat)
        phi_v = self.critic_body(phi_cat)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        # print logits
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v