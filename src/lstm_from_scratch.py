import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.001, clip_value=1.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.clip_value = clip_value

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.1
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.1

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))

        self.caches = []
        self.h_last = None
        self.y_hat = None

    def forward(self, seq):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        self.caches = []

        for x in seq:
            x = x.reshape(self.input_size, 1)
            z = np.vstack((h, x))

            f = self.sigmoid(self.Wf @ z + self.bf)
            i = self.sigmoid(self.Wi @ z + self.bi)
            o = self.sigmoid(self.Wo @ z + self.bo)
            c_hat = np.tanh(self.Wc @ z + self.bc)

            c_new = f * c + i * c_hat
            h_new = o * np.tanh(c_new)

            self.caches.append({
                "z": z,
                "f": f,
                "i": i,
                "o": o,
                "c_hat": c_hat,
                "c": c_new,
                "c_prev": c
            })

            h = h_new
            c = c_new

        self.h_last = h
        self.y_hat = self.Wy @ h + self.by
        return self.y_hat

    def backward(self, y_true):
        dy = 2 * (self.y_hat - y_true)

        dWy = dy @ self.h_last.T
        dby = dy

        dh_next = self.Wy.T @ dy
        dc_next = np.zeros_like(self.caches[0]["c"])

        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWc = np.zeros_like(self.Wc)

        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)

        for t in reversed(range(len(self.caches))):
            cache = self.caches[t]
            z = cache["z"]
            f = cache["f"]
            i = cache["i"]
            o = cache["o"]
            c_hat = cache["c_hat"]
            c = cache["c"]
            c_prev = cache["c_prev"]

            tanh_c = np.tanh(c)

            dh = dh_next
            dc = dc_next + dh * o * (1 - tanh_c ** 2)

            do = dh * tanh_c
            do_pre = do * o * (1 - o)

            df = dc * c_prev
            df_pre = df * f * (1 - f)

            di = dc * c_hat
            di_pre = di * i * (1 - i)

            dc_hat = dc * i
            dc_hat_pre = dc_hat * (1 - c_hat ** 2)

            dWf += df_pre @ z.T
            dWi += di_pre @ z.T
            dWo += do_pre @ z.T
            dWc += dc_hat_pre @ z.T

            dbf += df_pre
            dbi += di_pre
            dbo += do_pre
            dbc += dc_hat_pre

            dz = (self.Wf.T @ df_pre +
                  self.Wi.T @ di_pre +
                  self.Wo.T @ do_pre +
                  self.Wc.T @ dc_hat_pre)

            dh_next = dz[:self.hidden_size, :]
            dc_next = dc * f

        for grad in [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWy, dby]:
            np.clip(grad, -self.clip_value, self.clip_value, out=grad)

        self.Wf -= self.learning_rate * dWf
        self.Wi -= self.learning_rate * dWi
        self.Wo -= self.learning_rate * dWo
        self.Wc -= self.learning_rate * dWc

        self.bf -= self.learning_rate * dbf
        self.bi -= self.learning_rate * dbi
        self.bo -= self.learning_rate * dbo
        self.bc -= self.learning_rate * dbc

        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

    def predict(self, seq):
        return self.forward(seq)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
