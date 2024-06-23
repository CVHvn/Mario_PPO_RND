class Memory():
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.logits = []
        self.values = []
        self.values_int = []

    def save(self, state, action, reward, next_state, done, logit, value, value_int):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logits.append(logit)
        self.values.append(value)
        self.values_int.append(value_int)

    def reset(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.logits = []
        self.values = []
        self.values_int = []

    def get_data(self):
        return self.states, self.actions, self.next_states, self.rewards, self.dones, self.logits, self.values, self.values_int