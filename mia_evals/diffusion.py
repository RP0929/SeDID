import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon', 'eps_xt_xt-1']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.register_buffer(
            'sqrt_recip_alphas',
            1. / torch.sqrt(alphas)
        )
        self.register_buffer(
            'eps_coef',
            self.betas / torch.sqrt(1 - alphas_bar)
        )

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        # 1 / sqrt(alpha^{bar}) * x_t - 1 / sqrt(alpha^{bar}) - 1 * eps
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'eps_xt_xt-1':
            eps = self.model(x_t, t)
            model_mean, _ = self.p_prev_eps_xt(eps, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        if self.mean_type != 'eps_xt_xt-1':
            x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def p_prev_eps_xt(self, eps, x_t, t):
        sqrt_recip_alphas = extract(self.sqrt_recip_alphas, t=t, x_shape=x_t.shape)
        eps_coef = extract(self.eps_coef, t=t, x_shape=x_t.shape)
        result = sqrt_recip_alphas * (x_t - eps_coef * eps)
        return result, None


    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, eta, n_step=None, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon', 'eps_xt_xt-1']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.n_step = n_step
        # control the randomness of sampling
        # sampling will be deterministic if eta is 0
        self.eta = eta

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        # alphas_prev = torch.cat([alphas[0:1], alphas[:-1]])
        alphas = torch.cumprod(alphas, dim=0)
        alphas_prev = torch.cat([alphas[0:1], alphas[:-1]])
        # alphas_prev = F.pad(alphas, [1, 0], value=1)[:T]

        self.alphas = alphas.cuda()

        # for DDIM
        self.register_buffer(
            'eta_sigma',
            self.eta * torch.sqrt((1 - alphas_prev) / (1 - alphas)) * torch.sqrt(1 - (alphas / alphas_prev))
        )
        self.register_buffer(
            'sqrt_alphas_prev',
            torch.sqrt(alphas_prev)
        )
        self.register_buffer(
            'sqrt_1m_alphas',
            torch.sqrt(1 - alphas)
        )
        self.register_buffer(
            'sqrt_alphas',
            torch.sqrt(alphas)
        )
        self.register_buffer(
            'sqrt_1m_alphas_prev_eta_sigma2',
            torch.sqrt(1 - alphas_prev - self.eta_sigma ** 2)
        )
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_prev) / (1. - alphas))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))

    @torch.no_grad()
    def deterministic_sample(self, x_t, t, prev_t):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        eps = self.model(x_t, t)

        alphas = extract(self.alphas, t=t, x_shape=x_t.shape)
        sqrt_alphas = alphas.sqrt()
        alphas_prev = extract(self.alphas, t=prev_t, x_shape=x_t.shape)
        sqrt_alphas_prev = alphas_prev.sqrt()
        sqrt_1m_alphas = (1 - alphas).sqrt()
        eta_sigma2 = self.eta * ((1 - alphas_prev) / (1 - alphas)).sqrt() * (1 - (alphas / alphas_prev)).sqrt()
        sqrt_1m_alphas_prev_eta_sigma2 = (1 - alphas_prev - eta_sigma2 ** 2).sqrt()
        mean = sqrt_alphas_prev * ((x_t - sqrt_1m_alphas * eps) / (sqrt_alphas)) + sqrt_1m_alphas_prev_eta_sigma2 * eps

        return mean, model_log_var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        sample_results = []
        if isinstance(self.n_step, list):
            time_steps = self.n_step[0]
            prev_time_steps = self.n_step[1]
        else:
            if self.n_step is None:
                prev_time_steps = list(reversed(range(self.T)))
                # time_steps = list(reversed(range(self.T)))
            else:
                # time_steps = list(reversed(list(range(self.T))[::self.T // self.n_step]))
                prev_time_steps = list(reversed(list(range(self.T))[(self.T // self.n_step) - 1::self.T // self.n_step]))
            time_steps = prev_time_steps[:1] + prev_time_steps[:-1]
        # print(time_steps, prev_time_steps)
        if not isinstance(x_T, list):
            x_t = x_T.cuda()
        for idx, (time_step, prev_time_step) in enumerate(zip(time_steps, prev_time_steps)):
            if isinstance(x_T, list):
                x_t = x_T[len(x_T) - 1 - idx].cuda()
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            prev_t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * prev_time_step
            mean, _ = self.deterministic_sample(x_t=x_t, t=t, prev_t=prev_t)
            noise = torch.randn_like(x_t)
            x_t = mean + noise * extract(self.eta_sigma, t=t, x_shape=x_t.shape)
            sample_results.append(x_t.detach().clone())
        x_0 = x_t
        return torch.clip(x_0, -1, 1), sample_results
