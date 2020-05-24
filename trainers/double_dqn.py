from trainers.dqn import DQNTrainer


class DoubleDQNTrainer(DQNTrainer):
    def __init__(self, memory_size, batch_size, gamma, eps_start, eps_end, eps_decay, target_update, env, model=None, optimizer_klass=None, optimizer_params=None, loss_f=None, loss_params=None, is_ipython=False, log_dir=None):
        super().__init__(memory_size, batch_size, gamma, eps_start, eps_end, eps_decay, target_update, env, model=model, optimizer_klass=optimizer_klass,
                         optimizer_params=optimizer_params, loss_f=loss_f, loss_params=loss_params, is_ipython=is_ipython, log_dir=log_dir)

    def next_state_action(self, non_final_next_states):
        return self.policy_net(non_final_next_states).max(1)[0].detach()
