import torch
# from types import FunctionType
# from functools import wraps
# from types import FunctionType
# from typing import Any, Callable, Dict, Tuple
from typing import Callable


# def dont_decorate(f: Callable) -> Callable:
#     """Decorator to exclude methods from autodecoration."""
#     f._decorate = False  # type: ignore[attr-defined]
#     return f


# def no_grad(f: Callable) -> Callable:
#     """Dummy decorator which prints before and after the function it decorates."""

#     @wraps(f)
#     def wrapper(*args: Any, **kwargs: Any) -> Any:
#         """Wraps provided function and prints before and after."""
#         # print(f"Calling decorated function {f.__name__}")
#         with torch.no_grad():
#             return_val = f(*args, **kwargs)
#         # print(f"Called decorated function {f.__name__}")
#         return return_val

#     return wrapper


# def decorate_all(decorator: Callable) -> type:
#     """Decorate all instance methods (unless excluded) with the same decorator."""

#     class DecorateAll(type):
#         """Decorate all instance methods (unless excluded) with the same decorator."""

#         @classmethod
#         def do_decorate(cls, attr: str, value: Any) -> bool:
#             """Checks if an object should be decorated."""
#             return (
#                 ("__" not in attr or attr == "__call__")
#                 and isinstance(value, FunctionType)
#                 and getattr(value, "_decorate", True)
#             )

#         def __new__(
#             cls, name: str, bases: Tuple[type, ...], dct: Dict[str, Any]
#         ) -> type:
#             for attr, value in dct.items():
#                 if cls.do_decorate(attr, value):
#                     dct[attr] = decorator(value)
#             return super().__new__(cls, name, bases, dct)

#         def __setattr__(self, attr: str, value: Any) -> None:
#             if self.do_decorate(attr, value):
#                 value = decorator(value)
#             super().__setattr__(attr, value)

#     return DecorateAll


class EpsGetter:
    def __init__(self, model):
        self.model = model

    def __call__(self, xt: torch.Tensor, condition: torch.Tensor = None, noise_level=None, t: int = None) -> torch.Tensor:
        raise NotImplementedError


# class Attacker(metaclass=decorate_all(decorator=no_grad)):
class Attacker:
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize: Callable = None, denormalize: Callable = None):
        self.eps_getter = eps_getter
        self.betas = betas
        self.noise_level = torch.cumprod(1 - betas, dim=0).float()
        self.interval = interval
        self.attack_num = attack_num
        self.normalize = normalize
        self.denormalize = denormalize
        self.T = len(self.noise_level)

    def __call__(self, x0, xt, condition):
        raise NotImplementedError

    def get_xt_coefficient(self, step):
        return self.noise_level[step] ** 0.5, (1 - self.noise_level[step]) ** 0.5

    def get_xt(self, x0, step, eps):
        a_T, b_T = self.get_xt_coefficient(step)
        return a_T * x0 + b_T * eps

    def _normalize(self, x):
        if self.normalize is not None:
            return self.normalize(x)
        return x

    def _denormalize(self, x):
        if self.denormalize is not None:
            return self.denormalize(x)
        return x


class DDIMAttacker(Attacker):
    def get_y(self, x, step):
        return (1 / self.noise_level[step] ** 0.5) * x

    def get_x(self, y, step):
        return y * self.noise_level[step] ** 0.5

    def get_p(self, step):
        return (1 / self.noise_level[step] - 1) ** 0.5

    def __call__(self, x0, condition=None):
        x0 = self._normalize(x0)
        intermediates = self.ddim_reverse(x0, condition)
        intermediates_denoise = self.ddim_denoise(x0, intermediates, condition)
        return torch.stack(intermediates), torch.stack(intermediates_denoise)

    def ddim_reverse(self, x0, condition):
        raise NotImplementedError

    def ddim_denoise(self, x0, intermediates, condition):
        raise NotImplementedError


class IterDDIMAttacker(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        x = x0
        intermediates.append(x0)

        for step in range(0, terminal_step, self.interval):
            y_next = self.eps_getter(x, condition, self.noise_level, step) * (self.get_p(step + self.interval) - self.get_p(step)) + self.get_y(x, step)
            x = self.get_x(y_next, step + self.interval)
            intermediates.append(x)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        ternimal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(self.interval, ternimal_step + self.interval, self.interval), 1):
            x = intermediates[idx]
            y_prev = self.eps_getter(x, condition, self.noise_level, step) * (self.get_p(step - self.interval) - self.get_p(step)) + self.get_y(x, step)
            x_prev = self.get_x(y_prev, step - self.interval)
            x = x_prev
            intermediates_denoise.append(x_prev)

            if idx == len(intermediates) - 1:
                del intermediates[-1]
        return intermediates_denoise

    def get_prev_from_eps(self, x0, eps_x0, eps, t):
        t = t + self.interval
        xta1 = self.get_xt(x0, t, eps_x0)

        y_prev = eps * (self.get_p(t - self.interval) - self.get_p(t)) + self.get_y(xta1, t)
        x_prev = self.get_x(y_prev, t - self.interval)
        return x_prev


class DDIM(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        x = x0
        intermediates.append(x0)

        # for step in range(0, terminal_step, self.interval):
        #     y_next = self.eps_getter(x, condition, self.noise_level, step) * (self.get_p(step + self.interval) - self.get_p(step)) + self.get_y(x, step)
        #     x = self.get_x(y_next, step + self.interval)
        #     intermediates.append(x)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        # ternimal_step = self.interval * self.attack_num
        x = torch.randn_like(x0)
        for idx, step in enumerate(range(0, self.T - 1, self.interval), 1):
            step = self.T - step
            if step == self.T:
                step -= 1
            # x = intermediates[idx]
            y_prev = self.eps_getter(x, condition, self.noise_level, step) * (self.get_p(step - self.interval) - self.get_p(step)) + self.get_y(x, step)
            x_prev = self.get_x(y_prev, step - self.interval)
            x = x_prev
            intermediates_denoise.append(x_prev)

            # if idx == len(intermediates) - 1:
            #     del intermediates[-1]
        return intermediates_denoise

    def get_prev_from_eps(self, x0, eps_x0, eps, t):
        t = t + self.interval
        xta1 = self.get_xt(x0, t, eps_x0)

        y_prev = eps * (self.get_p(t - self.interval) - self.get_p(t)) + self.get_y(xta1, t)
        x_prev = self.get_x(y_prev, t - self.interval)
        return x_prev


class GroundtruthAttacker(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, self.noise_level, 0)
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(0, terminal_step, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise


class GroundtruthAbsAttacker(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, self.noise_level, 0)
        eps = eps / eps.abs().mean(list(range(1, eps.ndim)), keepdim=True) * (2 / torch.pi) ** 0.5
        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(0, terminal_step, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise


class NaiveAttacker(DDIMAttacker):
    def ddim_reverse(self, x0, condition):
        intermediates = []
        # x = x0
        terminal_step = self.interval * self.attack_num
        for _ in reversed(range(0, terminal_step, self.interval)):
            eps = torch.randn_like(x0)
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(0, terminal_step, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise


class NaivePGDAttacker(DDIMAttacker):
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize = None, denormalize = None, iteration=4, epsilon=0.04):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        self.iteration = iteration
        self.epsilon = epsilon

    def ddim_reverse(self, x0, condition):
        intermediates = []
        # x = x0
        terminal_step = self.interval * self.attack_num
        for step in reversed(range(0, terminal_step, self.interval)):
            eps = torch.randn_like(x0)

            with torch.enable_grad():
                for _ in range(self.iteration):
                    eps.requires_grad = True
                    # x =
                    x = self.get_xt(x0, step, eps)
                    grad = torch.autograd.grad((((eps - self.eps_getter(x, condition, self.noise_level, step)) ** 2).flatten(2).sum(-1).sqrt().mean()), eps)[0]

                    eps = torch.tensor(eps - self.epsilon * (grad.detach().sign()))

            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(0, terminal_step, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise


class GroundtruthPGDAttacker(DDIMAttacker):
    def __init__(self, betas, interval, attack_num, eps_getter: EpsGetter, normalize = None, denormalize = None, iteration=4, epsilon=0.04):
        super().__init__(betas, interval, attack_num, eps_getter, normalize, denormalize)
        self.iteration = iteration
        self.epsilon = epsilon

    def ddim_reverse(self, x0, condition):
        intermediates = []
        terminal_step = self.interval * self.attack_num
        eps = self.eps_getter(x0, condition, self.noise_level, 0)
        with torch.enable_grad():
            for _ in range(self.iteration):
                eps.requires_grad = True
                # x =
                x = self.get_xt(x0, 0, eps)
                grad = torch.autograd.grad((((eps - self.eps_getter(x, condition, self.noise_level, 0)) ** 2).flatten(2).sum(-1).sqrt().mean()), eps)[0]

                eps = torch.tensor(eps - self.epsilon * (grad.detach().sign()))

        for _ in reversed(range(0, terminal_step, self.interval)):
            intermediates.append(eps)

        return intermediates

    def ddim_denoise(self, x0, intermediates, condition):
        intermediates_denoise = []
        terminal_step = self.interval * self.attack_num

        for idx, step in enumerate(range(0, terminal_step, self.interval)):
            eps = intermediates[idx]

            eps_back = self.eps_getter(self.get_xt(x0, step, eps), condition, self.noise_level, step)
            intermediates_denoise.append(eps_back)
        return intermediates_denoise
