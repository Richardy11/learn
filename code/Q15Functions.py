import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Q15Functions:
    def __init__(self):
        pass

    def convert(self, f):

        y_clamped = np.clip(f, -1.0, float.fromhex("0x0.fffe"))
        y_fixed = np.multiply(y_clamped, 32768).astype(np.int16)
        y_fixed_ = np.multiply(y_fixed, 1/32768)

        return y_fixed_#/(2**15)

    def clipQ15(self, f):

        '''y_fxp = Fxp(f, signed=True, n_word=16, n_frac=15)

        return y_fxp'''

        y_clamped = np.clip(f, -1.0, float.fromhex("0x0.fffe"))

        return y_clamped

    def fit(self, proj, components):

        self.components = components

        self.stdscaler = MinMaxScaler((-1,1))
        self.stdscaler_post = MinMaxScaler((-1,1))
        self.proj_scaled = self.stdscaler.fit_transform(proj)
        self.proj_scaled = self.convert(self.proj_scaled)


    def transform(self, x, save = False):

        x_scaled = self.stdscaler.transform(x)
        x_scaled = self.convert(x_scaled)

        x_projected = np.dot(x_scaled, self.proj_scaled)

        '''if save:
            self.stdscaler_post.fit(x_projected)

        x_projected = self.stdscaler_post.transform(x_projected)'''

        return x_projected[:, :self.components]
    

if __name__ == '__main__':
    
    q15 = Q15Functions()

    f = np.random.rand(3,2)
    print(f)

    print(f - q15.convert(f))