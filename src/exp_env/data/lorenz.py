import numpy as np
from exp_env.data.data_maker_base import DataMaker, DataMakerSpec

class LorenzTaskType:
    parameter_regression = 'parameter_regression'
    random_parameter_time_series = 'random_parameter_time_series'
    random_initial_time_series = 'random_initial_time_series'
    random_all_time_series = 'random_all_time_series'
    lyapunov_regression = 'lyapunov_regression'
    all_regression = 'all_regression'
    sigma_regression = 'sigma_regression'
    rho_regression = 'rho_regression'
    beta_regression = 'beta_regression'
    x_lyapunov_regression = 'x_lyapunov_regression'
    y_lyapunov_regression = 'y_lyapunov_regression'
    z_lyapunov_regression = 'z_lyapunov_regression'
    xyz_lyapunov_regression = 'xyz_lyapunov_regression'
    rebuild_and_estimate_parameters = 'rebuild_and_estimate_parameters'
    estimate_parameters_next_step= 'estimate_parameters_next_step'
    rebuild_vs= 'rebuild_vs'

class LorenzSpec(DataMakerSpec):
    def __init__(self, max_T, dt, how_many, task_type: str,cut_off=0):
        super().__init__()
        self.max_T = max_T
        self.dt = dt
        self.how_many = how_many
        self.task_type = task_type
        self.cut_off = cut_off
    
class LorenzMaker(DataMaker):
    def __init__(self, spec: LorenzSpec):
        self.spec = spec
    
    def lorentz(self, x, y, z, sigma=10., rho=28., beta=8/3):
        # dx = sigma * (y - x)
        # dy = x * (rho - z) - y
        # dz = x * y - beta * z
        X=np.array([[x],[y],[z]])
        J=self.jacobi(x,y,z,sigma,rho,beta)
        dX=J@X
        dx, dy, dz = dX[0,0], dX[1,0], dX[2,0]
        return dx, dy, dz

    def lorenz_auto(self, x, y, z, sigma=10., rho=28., beta=8/3):
        import torch
        X=torch.tensor([x,y,z])
        ts=np.arange(0,self.spec.max_T,self.spec.dt)
        for t in ts:
            J=self.jacobi(X[0],X[1],X[2],sigma,rho,beta,is_torch=True)
            dX=J@X
            X+=dX*self.spec.dt
        return X


    def jacobi(self, x, y, z, sigma=10., rho=28., beta=8/3,is_torch=False):
        if is_torch:
            import torch
            return torch.tensor([
                [-sigma, sigma, 0],
                [rho - z, -1, -x],
                [y, x, -beta]
            ])
        return np.array([
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ])

    def runge_kutta(self, x, y, z, dt, sigma=10., rho=28., beta=8/3):
        dx1, dy1, dz1 = self.lorentz(x, y, z, sigma, rho, beta)
        dx2, dy2, dz2 = self.lorentz(x + dx1 * dt / 2, y + dy1 * dt / 2, z + dz1 * dt / 2, sigma, rho, beta)
        dx3, dy3, dz3 = self.lorentz(x + dx2 * dt / 2, y + dy2 * dt / 2, z + dz2 * dt / 2, sigma, rho, beta)
        dx4, dy4, dz4 = self.lorentz(x + dx3 * dt, y + dy3 * dt, z + dz3 * dt, sigma, rho, beta)
        return x + (dx1 + 2 * dx2 + 2 * dx3 + dx4) * dt / 6, y + (dy1 + 2 * dy2 + 2 * dy3 + dy4) * dt / 6, z + (dz1 + 2 * dz2 + 2 * dz3 + dz4) * dt / 6
    
    def lyapunov_qr(self, x, y, z, sigma=10., rho=28., beta=8/3):
        T = self.spec.max_T
        dt = self.spec.dt
        ts = np.arange(0, T+dt, dt)
        xs = np.empty((len(ts), 3))
        lambda_ = np.empty((len(ts), 3))
        Q = np.identity(3)
        for i, t in enumerate(ts):
            J = self.jacobi(x, y, z)
            M = np.identity(3) + J * dt
            Q,R = np.linalg.qr(np.dot(M, Q))
            lambda_[i] = np.squeeze(np.asarray(np.log(np.abs(R.diagonal()))))
            x, y, z = self.runge_kutta(x, y, z, dt, sigma, rho, beta)
        lambda_ = lambda_.cumsum(axis=0) / (np.expand_dims(ts, axis=1) + dt)
        return lambda_[-1]

    def make(self):
        T = self.spec.max_T
        dt = self.spec.dt
        how_many = self.spec.how_many
        xs = np.empty((how_many, int(T//dt+2), 3))
        objectives = []
        parameters = []
        x0=[]
        for i in range(how_many):
            sigma, rho, beta = 10., 28., 8/3
            x, y, z =np.random.uniform(-10, 10, 3)
            if self.spec.task_type == LorenzTaskType.sigma_regression:
                sigma, rho, beta = np.random.uniform(9, 11), 28, 8/3
                objectives.append(sigma/20)
            elif self.spec.task_type == LorenzTaskType.rho_regression:
                sigma, rho, beta = 10, np.random.uniform(0, 40), 8/3
                objectives.append(rho/40)
            elif self.spec.task_type == LorenzTaskType.beta_regression:
                sigma, rho, beta = 10, 28, np.random.uniform(0, 4)
                objectives.append(beta/4)
            
            elif self.spec.task_type == LorenzTaskType.parameter_regression:
                sigma, rho, beta = np.random.uniform(0, 20), np.random.uniform(20, 30), np.random.uniform(2, 3)
                objectives.append([sigma, rho, beta])

            
            elif self.spec.task_type == LorenzTaskType.x_lyapunov_regression:
                sigma, rho, beta = np.random.uniform(0, 20), np.random.uniform(0, 40), np.random.uniform(0, 4)
                objectives.append(self.lyapunov_qr(sigma, rho, beta)[0])
            elif self.spec.task_type == LorenzTaskType.y_lyapunov_regression:
                sigma, rho, beta = np.random.uniform(0, 20), np.random.uniform(0, 40), np.random.uniform(0, 4)
                objectives.append(self.lyapunov_qr(sigma, rho, beta)[1])
            elif self.spec.task_type == LorenzTaskType.z_lyapunov_regression:
                sigma, rho, beta = np.random.uniform(0, 20), np.random.uniform(0, 40), np.random.uniform(0, 4)
                objectives.append(self.lyapunov_qr(sigma, rho, beta)[2])
            elif self.spec.task_type == LorenzTaskType.xyz_lyapunov_regression:
                sigma, rho, beta = np.random.uniform(0, 40), np.random.uniform(0, 40), np.random.uniform(0, 40)
                objectives.append(self.lyapunov_qr(sigma, rho, beta))
            

            elif self.spec.task_type == LorenzTaskType.rebuild_and_estimate_parameters:
                sigma, rho, beta = np.random.uniform(9, 11), np.random.uniform(20, 30), np.random.uniform(2, 3)
                parameters.append([sigma, rho, beta])
                x0.append([x,y,z])
            elif self.spec.task_type == LorenzTaskType.estimate_parameters_next_step:
                sigma, rho, beta = np.random.uniform(0, 20), np.random.uniform(20, 30), np.random.uniform(2, 3)
                parameters.append([sigma, rho, beta])
            elif self.spec.task_type == LorenzTaskType.rebuild_vs:
                sigma, rho, beta = np.random.uniform(9, 11), np.random.uniform(20, 30), np.random.uniform(2, 3)
            # delta=np.random.uniform(-1, 1, 3)
            # x, y, z = x+delta[0], y+delta[1], z+delta[2]
            ts = np.arange(0, T+dt, dt)
            t_length=ts.shape[0]
            if self.spec.task_type == LorenzTaskType.estimate_parameters_next_step:
                ts = np.arange(0, (T+dt)*2, dt)
            obj_est_nxt=[]
            for j, t in enumerate(ts):
                new_x, new_y, new_z = self.lorentz(x, y, z, sigma, rho, beta)
    
                x, y, z = x+new_x*dt, y+new_y*dt, z+new_z*dt
                
                if j>=t_length:
                    obj_est_nxt.append([x,y,z])
                else:
                    xs[i, j] = x, y, z
            if self.spec.task_type == LorenzTaskType.estimate_parameters_next_step:
                objectives.append(obj_est_nxt)
        xs=xs[:,int(self.spec.cut_off/self.spec.dt):,:]
        if self.spec.task_type == LorenzTaskType.random_initial_time_series:
            objectives = xs[:,1:,:]
            xs=xs[:,:-1,:]
        if self.spec.task_type == LorenzTaskType.rebuild_and_estimate_parameters:
            xs=xs[:,:-1,:]
            objectives=(parameters,x0)
        if self.spec.task_type == LorenzTaskType.rebuild_vs:
            objectives = xs[:,1:,:]
            xs=xs[:,:-1,:]
        return xs, objectives

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    spec = LorenzSpec(1, 0.025, 3, LorenzTaskType.random_initial_time_series)
    maker = LorenzMaker(spec)
    x, y, z = 1, 5, -100
    xs,y2 = maker.make()
    plt.plot(xs[0, :, 0], xs[0, :, 1])
    #save to image
    plt.savefig('lorenz.png')
    # print(y2)
    # print(maker.lyapunov_qr(x, y, z))
    # print(maker.lyapunov_qr(x+10, y+10, z+10,rho=y2[0]))

