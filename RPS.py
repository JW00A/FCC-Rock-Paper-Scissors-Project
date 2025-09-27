# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
import math

def player(prev_play,
           opponent_history=[],
           self_history=[],
           mod_trigger=[random.randint(1,100)],
           model_state=[None]):

    K = 4
    input_per_move = 3
    input_size = (K * 2) * input_per_move
    hidden_size = 16
    lr = 0.06
    choices = ['R', 'P', 'S']
    idx = {'R':0, 'P':1, 'S':2}
    inv = {0:'R', 1:'P', 2:'S'}
    cases = {'R':'P', 'P':'S', 'S':'R'}

    def dot_mat_vec(mat, vec):
        return [sum(m*v for m,v in zip(row, vec)) for row in mat]

    def add_vec(a, b):
        return [x+y for x,y in zip(a,b)]

    def tanh_vec(v):
        return [math.tanh(x) for x in v]

    def softmax(logits):
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps)
        return [e/s for e in exps]

    def one_hot_move(m):
        v = [0.0, 0.0, 0.0]
        if m in idx:
            v[idx[m]] = 1.0
        return v

    def build_input(op_hist, self_hist):
        vec = []
        for i in range(K):
            j = len(op_hist) - K + i
            if j >= 0:
                vec.extend(one_hot_move(op_hist[j]))
            else:
                vec.extend([0.0, 0.0, 0.0])
        for i in range(K):
            j = len(self_hist) - K + i
            if j >= 0:
                vec.extend(one_hot_move(self_hist[j]))
            else:
                vec.extend([0.0, 0.0, 0.0])
        return vec

    if model_state[0] is None:
        W1 = [[(random.random() - 0.5) * 0.2 for _ in range(input_size)] for _ in range(hidden_size)]
        b1 = [0.0] * hidden_size
        W2 = [[(random.random() - 0.5) * 0.2 for _ in range(hidden_size)] for _ in range(3)]
        b2 = [0.0] * 3
        model_state[0] = {
            'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
            'lr': lr,
            'last_input': None
        }

    model = model_state[0]

    if prev_play in choices:
        opponent_history.append(prev_play)

    if model['last_input'] is not None and prev_play in choices:
        x = model['last_input']
        z1 = add_vec(dot_mat_vec(model['W1'], x), model['b1'])
        h = tanh_vec(z1)
        z2 = add_vec(dot_mat_vec(model['W2'], h), model['b2'])
        probs = softmax(z2)
        t = [0.0, 0.0, 0.0]
        t[idx[prev_play]] = 1.0
        dz2 = [p - ti for p, ti in zip(probs, t)]
        dW2 = [[dz2[r] * h_c for h_c in h] for r in range(3)]
        db2 = dz2[:]
        dh = [0.0] * hidden_size
        for r in range(3):
            for j in range(hidden_size):
                dh[j] += model['W2'][r][j] * dz2[r]
        dh_raw = [dh_j * (1 - h_j*h_j) for dh_j, h_j in zip(dh, h)]
        dW1 = [[dh_raw[j] * x_i for x_i in x] for j in range(hidden_size)]
        db1 = dh_raw[:]
        lr_m = model['lr']
        for r in range(3):
            for j in range(hidden_size):
                model['W2'][r][j] -= lr_m * dW2[r][j]
            model['b2'][r] -= lr_m * db2[r]
        for j in range(hidden_size):
            for i_in in range(input_size):
                model['W1'][j][i_in] -= lr_m * dW1[j][i_in]
            model['b1'][j] -= lr_m * db1[j]

    curr_input = build_input(opponent_history, self_history)

    z1 = add_vec(dot_mat_vec(model['W1'], curr_input), model['b1'])
    h = tanh_vec(z1)
    z2 = add_vec(dot_mat_vec(model['W2'], h), model['b2'])
    probs = softmax(z2)

    model['last_input'] = curr_input

    pred_idx = max(range(3), key=lambda i: probs[i])
    predicted_opp = inv[pred_idx]

    primary_guess = cases[predicted_opp]

    options = [['R','P','S'],['R','S','P'],['P','R','S'],
               ['P','S','R'],['S','R','P'],['S','P','R'],
               ['R','R','S'],['R','S','R'],['S','R','R'],
               ['R','R','P'],['R','P','R'],['P','R','R'],
               ['S','S','R'],['S','R','S'],['R','S','S'],
               ['P','P','R'],['R','P','P'],['P','R','P'],
               ['S','S','P'],['S','P','S'],['P','S','S'],
               ['P','P','S'],['S','P','P'],['P','S','P'],
               ['P','P','P'],['P','P','P'],['P','P','P'],
               ['S','S','S'],['S','S','S'],['S','S','S'],
               ['R','R','R'],['R','R','R'],['R','R','R']
    ]
    if len(self_history) < 3:
        guess = random.choice(['R','P','S'])
    else:
        confidence = probs[pred_idx]
        if random.random() < confidence * 0.9:
            guess = primary_guess
        else:
            freq = {m: opponent_history.count(m) for m in choices}
            top = max(freq, key=freq.get)
            bait = cases[top]
            if random.random() < 0.6:
                guess = bait
            else:
                guess = random.choice(random.choice(options))

    if len(self_history) >= 1 and len(self_history) % mod_trigger[0] == 0:
        guess = cases[self_history[-1]]

    self_history.append(guess)
    return guess