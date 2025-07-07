import torch
from torch.nn.functional import pad
import numpy as np
import pdb

HADAMARD_MAX_RANDOM_DIMENSION = 2 ** 29
hadamard_random_vec_pool = {}
for i in range(8):
    hadamard_random_vec_pool["cuda:{}".format(i)] = None
hadamard_random_vec_pool["cpu"] = None

class Hadamard:
    def __init__(self, dim, seed, device, paritial_times=0):
        self.d = dim
        self.device = device
        self.prng = torch.Generator(device=device)
        self.prng.manual_seed(seed)
        if hadamard_random_vec_pool[device] == None:
            hadamard_random_vec_pool[device] = 2 * torch.bernoulli(torch.ones(size=(HADAMARD_MAX_RANDOM_DIMENSION,), device=device) / 2, generator=self.prng) - 1
        randl = torch.randint(0, HADAMARD_MAX_RANDOM_DIMENSION - dim + 1, (1,)).item()
        randr = randl + self.d
        self.is_random_diagonal_refreshed = 0
        self.partial_rotation_times = paritial_times
        if self.d & (self.d - 1) != 0:
            raise Exception("input numel must be a power of 2")
            
    def hadamard(self, vec):
        d = vec.numel()
        assert(d == self.d)
        iterations = int(np.log2(d))
        if self.partial_rotation_times > 0:
            iterations = min(self.partial_rotation_times, iterations)
            h = 2 ** (int(np.log2(d)) - iterations + 1)
        else:
            h = 2
        while h <= d:
            hf = h//2
            vec = vec.view(d//h,h)
            vec[:,:hf]  = vec[:,:hf] + vec[:,hf:2*hf]
            vec[:,hf:2*hf] = vec[:,:hf] - 2*vec[:,hf:2*hf]
            h *= 2
        vec /= np.sqrt(2 ** iterations)
        return vec.view(-1)

    def rht(self, vec):
        dim = vec.numel()
        randl = torch.randint(0, HADAMARD_MAX_RANDOM_DIMENSION - self.d + 1, (1,)).item()
        randr = randl + self.d
        random_diagonal = hadamard_random_vec_pool[self.device][randl: randr]
        self.random_diagonal_randrange = (randl, randr)
        vec = vec * random_diagonal
        vec = self.hadamard(vec)
        return vec
        
    def irht(self, vec):
        randl, randr = self.random_diagonal_randrange
        random_diagonal = hadamard_random_vec_pool[self.device][randl: randr]
        vec = self.hadamard(vec)
        vec = vec * random_diagonal
        return vec

class UHCCompressor(object):
    def __init__(self, params):
        self.device = params.get('device', 'cuda') 
        self.ds = params['d']
        self.original_size = params["size"]
        self.seed = params.get('seed', 42)
        self.rotation = params.get("rotation", True)
        self.prng = torch.Generator(device=self.device)
        self.prng.manual_seed(self.seed)
        self.ef = params.get('ef', True)
        self.quantization_levels = params.get('quantization_levels', 16)
        self.partial_rotation_times = params.get('partial_rotation_times', 0)
        self.hadamards = dict()
    
        for name_idx in self.ds:
            self.hadamards[name_idx] = Hadamard(self.ds[name_idx], self.seed, self.device, self.partial_rotation_times)

        self.compress_vec = dict()
        for name_idx in self.ds:
            self.compress_vec[name_idx] = torch.zeros(self.ds[name_idx], dtype=torch.float32, device=self.device)

        if self.ef:
            self.errors = {}

        ### sender ###########################################################      
        self.sender_prng = torch.Generator(device=self.device)
        
         ### receiver #########################################################
        self.receiver_prng = torch.Generator(device=self.device)
       
        self.normal_dist_clamp = 3.2971933456919635
        self.nclients = params.get('nclients', 16)
        self.max_norm_dict = dict()

            
    def rvec_compress(self, tensor, max_norm, dim):
        max_coordinate = self.normal_dist_clamp * max_norm / np.sqrt(dim)
        min_coordinate = -max_coordinate
        delta = (max_coordinate - min_coordinate) / (self.quantization_levels - 1) + 1e-18
        tensor.sub_(min_coordinate).div_(delta)
        tensor = torch.clamp_(tensor, min=0, max=self.quantization_levels - 1)
        tensor2 = tensor.clone()
        tensor = tensor.floor_()
        p = tensor2.sub_(tensor)
        tensor = tensor.add_(torch.bernoulli(p, generator=self.sender_prng))
        return tensor.type(torch.int8), tensor, min_coordinate, delta

    """compression."""
    def compress(self, tensor, name, max_norm): # name: (index_id, partition_id)
        """Returns the tensor unmodified."""

        orig_size = dim = padded_dim = self.original_size[name]
        self.max_norm_dict[name] = max_norm
        if not dim & (dim - 1) == 0:
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            paddings = self.compress_vec[name]
            paddings[:dim].copy_(tensor)
            assert(paddings[dim] < 1e-9)
            tensor = paddings
        
        if self.ef:
            self.errors[name] = tensor + self.errors.get(name, 0)
            # compress gradient
            if self.rotation:
                temp1 = self.hadamards[name].rht(self.errors[name])
            else:
                temp1 = self.errors[name]
            tensor, float_tensor, min_coordinate, delta = self.rvec_compress(temp1, max_norm, padded_dim)
            # update the error
            temp2 = min_coordinate + float_tensor * delta
            if self.rotation:
                self.errors[name] -= self.hadamards[name].irht(temp2)
            else:
                self.errors[name] -= temp2
        else:
            if self.rotation:
                temp1 = self.hadamards[name].rht(tensor)
            else:
                temp1 = tensor
            tensor, _, _, _ = self.rvec_compress(temp1, max_norm, padded_dim)
        return tensor

    """Uncompress the tensor."""
    def decompress(self, tensor, name, max_norm=None):
        """Returns the tensor unmodified."""
        if max_norm == None:
            max_norm = self.max_norm_dict[name]
        max_coordinate = self.normal_dist_clamp * max_norm / np.sqrt(self.ds[name])
        min_coordinate = -max_coordinate
        delta = (max_coordinate - min_coordinate) / (self.quantization_levels - 1) + 1e-18
        tensor = tensor.float()
        tensor.mul_(delta / self.nclients).add_(min_coordinate)
        if self.rotation:
            tensor = self.hadamards[name].irht(tensor)
        return tensor[:self.original_size[name]]