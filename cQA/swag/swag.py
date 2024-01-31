import torch

from .utils import flatten, set_weights
from .subspaces import Subspace
from ._assess_dimension import _infer_dimension_


class SWAG(torch.nn.Module):

    def __init__(self, base_model, subspace_type,
                 subspace_kwargs=None, var_clamp=1e-6):
        super(SWAG, self).__init__()

        self.base_model = base_model
        #total sum of all weights
        # self.num_parameters = sum(param .numel() for param in self.base_model.parameters())
        self.collected_layer = ['pooler.','W1.','W2.','out.']
        # self.sampled_layer = ['pooler.','W1.','W2.','out.']
        self.num_parameters = sum(param.numel() for name, param in self.base_model.named_parameters() for layer in self.collected_layer if layer in name)
        #register_buffer is not model parameters but immediate parameters used to calculate
        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))

        # Initialize subspace
        if subspace_kwargs is None:
            subspace_kwargs = dict()
        #subspace used to 
        self.subspace = Subspace.create(subspace_type, num_parameters=self.num_parameters,
                                        **subspace_kwargs)

        self.var_clamp = var_clamp

        self.cov_factor = None
        self.model_device = 'cpu'
        
    # dont put subspace on cuda?
    def cuda(self, device=None):
        self.model_device = 'cuda'
        self.base_model.cuda(device=device)

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        print(torch._C._nn._parse_to(*args, **kwargs))
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        self.model_device = device.type
        self.subspace.to(device=torch.device('cpu'), dtype=dtype, non_blocking=non_blocking)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def collect_model(self, base_model, *args, **kwargs):
        # need to refit the space after collecting a new model
        self.cov_factor = None

        # w = flatten([param.detach().cpu() for param in base_model.parameters()])
        w = flatten([param.detach().cpu() for name, param in base_model.named_parameters() for layer in self.collected_layer if layer in name])
        # first moment
        # a = torch.tensor(self.n_models.item() / (self.n_models.item() + 1.0))
        a = self.n_models.item() / (self.n_models.item() + 1.0)
        self.mean.mul_(a)
        
        b = w / (self.n_models.item() + 1.0)
        self.mean.add_(b)

        # second moment
        self.sq_mean.mul_(a)

        b = w ** 2 / (self.n_models.item() + 1.0)
        self.sq_mean.add_(b)

        dev_vector = w - self.mean
        # dev_vector = w - self.mean.detach().cpu().numpy()
        # for what?
        self.subspace.collect_vector(dev_vector, *args, **kwargs)
        self.n_models.add_(1)

    def _get_mean_and_variance(self):
        # variance = self.sq_mean - self.mean ** 2
        variance = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp)
        return self.mean, variance

    def fit(self):
        if self.cov_factor is not None:
            return
        # [num_rank,  num_params]
        self.cov_factor = self.subspace.get_space()

    def set_swa(self):
        ## assign new parameters
        set_weights(self.base_model, self.mean, self.collected_layer, self.model_device)

    def sample(self, scale=0.5, diag_noise=True):
        self.fit()
        mean, variance = self._get_mean_and_variance()
        eps_low_rank = torch.randn(self.cov_factor.size()[0])
        # [params, rank], [rank, 1] z: [params, 1], merging params of all ranks                                    
        z = self.cov_factor.t() @ eps_low_rank
        if diag_noise:
            z += variance * torch.randn_like(variance)
        # 
        z *= scale ** 0.5
        sample = mean + z

        # apply to parameters
        set_weights(self.base_model, sample, self.collected_layer, self.model_device)
        return sample

    def get_space(self, export_cov_factor=True):
        mean, variance = self._get_mean_and_variance()
        if not export_cov_factor:
            return mean.clone(), variance.clone()
        else:
            self.fit()
            return mean.clone(), variance.clone(), self.cov_factor.clone()

    def infer_dimension(self, update_max_rank=True, use_delta=True):
        if use_delta:
            delta = self.subspace.delta

        _, var, subspace = self.get_space()
        subspace /= (self.n_models.item() - 1) ** 0.5
        tr_sigma = var.sum()

        spectrum, _ = torch.eig(subspace @ subspace.t())
        spectrum, _ = torch.sort(spectrum[:,0], descending=True)
        
        new_max_rank, ll, _ = _infer_dimension_(spectrum.numpy(), tr_sigma.numpy(),
                                                self.n_models.item(), self.num_parameters, delta)

        if new_max_rank + 1 == self.subspace.max_rank and update_max_rank:
            self.subspace.max_rank += 1
