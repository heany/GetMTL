import copy
import random
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """
        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :
        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )
        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class GetMTL(WeightMethod):
    def __init__(self, cfg, scales, n_tasks, device: torch.device):
        super().__init__(n_tasks, device=device)
        self.cfg = cfg
        self.c = self.cfg.exp['c']
        self.s = torch.Tensor(scales)

    def get_weighted_loss(
            self,
            losses,
            shared_parameters,
            **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), self.n_tasks).to(self.device)

        for i in range(self.n_tasks):
            if i < self.n_tasks:
                losses[i].backward(retain_graph=True)
            else:
                losses[i].backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, w = self._getmtl(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims)

        return w

    def _getmtl(self, grads, alpha=0.15, rescale=1):
        GtG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GtG.mean() + 1e-8).sqrt()  # norm of the average gradient

        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}

        A = GtG.numpy()
        b = x_start.copy()
        cg0 = (alpha * g0_norm + 1e-8).item()
        cg02 = (alpha * g0_norm * g0_norm + 1e-8).item()
        rho = (
                1 - np.square(alpha * g0_norm)
        ).item()

        def objfn(x):
            return (
                    np.square(
                        (cg0 + 1) * np.sqrt(
                                x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                                + 1e-8
                        )
                ) / (2 * alpha * np.sqrt(x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                                        + 1e-8) * rho)
                    - np.sqrt(x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1)) + 1e-8) / 2 * alpha
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        wv = torch.Tensor(w_cpu) * self.s
        ww = torch.nn.functional.softmax(wv, dim=0)
        ww = torch.Tensor(ww).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmda = (gw_norm + 1e-8) / cg02
        d = grads.mean(1) / rho + gw / rho * lmda
        if rescale == 0:
            return d, w_cpu
        elif rescale == 1:
            return d / (1 + alpha ** 2), w_cpu
        else:
            return d / (1 + alpha), w_cpu

    @staticmethod
    def grad2vec(shared_params, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        # for mm in m.shared_modules():
        #     for p in mm.parameters():

        for param in shared_params:
            grad = param.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[: cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0

        # for mm in m.shared_modules():
        #     for param in mm.parameters():
        for param in shared_parameters:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
            **kwargs,
    ):
        w = self.get_weighted_loss(losses, shared_parameters)
        return w  # NOTE: to align with all other weight methods



if __name__ == '__main__':

    from .min_norm_solvers import MinNormSolver, gradient_normalizers


    gn = gradient_normalizers(grads, losses, 'loss+') # grads is the gradient dict of all tasks
    grads = {t: grads[t][0] / gn[t] for t in grads}
    # Frank-Wolfe iteration to compute scales.
    sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
    scales = {t: sol[i] for i, t in enumerate(tasks)}
    losses_ = {}

    opt_g.zero_grad()
    opt_c.zero_grad()
    for t in tasks:
        if n == 0:
            output_s.append(scales[t])
        y_ = model[t](model['rep'](xs[t]))
        loss_t = criterion(y_, ys[t])
        losses_[t] = loss_t

    ls_ = torch.stack([losses_[t] for t in tasks])
    shared_parameters, task_specific_parameters = get_parameters(model)

    # weight method
    getmtl = GetMTL(cfg, list(scales.values()), n_tasks=len(tasks), device=device)

    w = getmtl.backward(
        losses=ls_,
        shared_parameters=shared_parameters,
        # task_specific_parameters=task_specific_parameters,
    )
    opt_g.step()
    opt_c.step()
