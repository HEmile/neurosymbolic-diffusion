from typing import List, Optional
from tap import Tap
import torch
import os

os.environ["WANDB_INIT_TIMEOUT"] = "300"


class Arguments(Tap):
    use_cuda: bool = True
    use_mps: bool = True
    use_wandb: bool = True
    send_conf_matrix: bool = False
    DEBUG: bool = False
    amt_samples_test: int = 128
    model: str = "mnist"
    epochs: int = 1000
    test: bool = False
    log_per_epoch: int = 10
    run: int = 0


class AbsArguments(Arguments):
    entropy_weight: float = 0.01
    w_denoise_weight: float = 0.01
    Z_weight: float = 0.01
    lr: float = 5e-4
    beta: float = 5.0
    batch_size: int = 16

    # Only simple model is used in paper
    simple_model: bool = True
    # Not used in paper
    denoising_entropy: bool = False

    # Number of samples of \tilde{w}_0 for computing the loss with RLOO
    loss_S: int = 1024
    # Number of samples of \tilde{w}_0 for rejection sampling when simulating the model.
    # This is more expensive since it needs to be done variational_T times. This is called K in the paper
    variational_K: int = 1024
    # Same but for test_T times
    test_K: int = 1024
    # Number of samples of w from the variational distribution q(w_0|x, y_0). Currently, only 1 is supported. Unused in the paper
    variational_J: int = 1
    # Number of samples of w and y for testing (using majority vote)
    test_L: int = 8
    # Number of timesteps for diffusion sampler. If None, uses the first-hitting exact sampler
    variational_T: Optional[int] = 8
    test_T: Optional[int] = None
    

    # Whether to use the exact variant of the model
    # Only turn this on for small problems
    entropy_variant: str = "unconditional" # [unconditional, exact_conditional, boia]


class MNISTArguments(Tap):
    N: int = 4
    op: str = "sum"
    batch_size: int = 16
    batch_size_test: int = 16
    test_every_epochs: int = 2
    layers: int = 1
    embedding_size: int = 64
    config_file: str = None

   


class PathPlanningArguments(AbsArguments):
    epochs: int = 40
    grid_size: int = 12
    # Train size: 10000 maps
    batch_size: int = 50
    # Test size: 1000 maps
    batch_size_test: int = 50 
    val_every_epochs: int = 5
    model: str = "CombResnet18"
    use_ray: bool = True
    loss_S: int = 16
    variational_K: int = 4
    test_K: int = 4
    variational_T: int = 20
    test_T: int = 20
    save_model: bool = True
    model_dir: str = "models/path_planning"
    wandb_resume_id: Optional[str] = None

    # Tuned hyperparameters 
    beta: float = 12.0
    lr: float = 0.0005
    entropy_weight: float = 0.002
    w_denoise_weight: float = 0.00001
    # Only for complex model 
    Z_weight: float = 0.0000031

    # Whether to embed the partially masked path into the model.
    #  Can be set to False to prevent the model from predicting costs that 'carve out' the path
    optimizer: str = "RAdam"
    y_embed: bool = False
    costs: List[float] = [0.8, 1.2, 5.3, 7.7, 9.2]

class PathPlanningEvalArguments(PathPlanningArguments):
    eval_at_epoch: List[int] 
    run_ids: List[str] 
    test: bool = True
    eval_name: str


class MNISTAbsorbingArguments(AbsArguments, MNISTArguments):
    model: str = "mnist"
    
    epochs: int = 100
    entropy_weight: float = 0.01
    w_denoise_weight: float = 0.00002
    lr: float = 3e-4
    beta: float = 20.0 
    Z_weight: float = 0.0001

class RSBenchArguments(AbsArguments):
    dataset: str = "halfmnist" # [halfmnist, shortcutmnist, boia]
    c_sup: int = 1
    joint: bool = False
    which_c: List[int] = [-1]
    model: str = "nesydiffusion"
    task: str = "addition"
    boia_ood_knowledge: bool = False
    save_model: bool = True
    run_id: str = ""
    epochs: int = 500

    lr: float = 0.00009
    batch_size: int = 16
    epochs: int = 500
    entropy_weight: float = 1.6
    w_denoise_weight: float = 0.0000015
    beta: float = 10
    entropy_variant: str = "exact_conditional" # [unconditional, exact_conditional]
    entropy_epoch_increase: float = 0.0 # Starts at the value of entropy_weight and increases linearly with this value every epoch. Not used in paper. 

    backbone: str = "disentangled" # [disentangled, fullentangled, partialentangled]

    test_every_epochs: int = 10
    ECE_bins: int = 10
    # Number of samples of w and y for testing (using majority vote)
    # Majority voting samples much higher in RSbench to get accurate ECE estimates and because of small size
    test_L: int = 1000

class RSBenchEvalArguments(RSBenchArguments):
    run_ids: List[str]
    test: bool = True
    model_dir: str = "models"
    eval_name: str
