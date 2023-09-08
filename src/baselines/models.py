from src.baselines.RCGAN import RCGANTrainer
from src.baselines.TimeGAN import TIMEGANTrainer
from src.baselines.TimeVAE import TimeVAETrainer
from src.baselines.networks.discriminators import LSTMDiscriminator
from src.baselines.networks.generators import LSTMGenerator
from src.baselines.networks.TimeVAE import VariationalAutoencoderConvInterpretable
import torch
from src.evaluations.test_metrics import get_standard_test_metrics
from src.utils import loader_to_tensor, loader_to_cond_tensor

#TODO: add to config
GENERATORS = {'LSTM': LSTMGenerator}
VAES = {'TimeVAE': VariationalAutoencoderConvInterpretable}
DISCRIMINATORS = {'LSTM': LSTMDiscriminator}


def get_generator(generator_type, input_dim, output_dim, **kwargs):
    return GENERATORS[generator_type](input_dim=input_dim, output_dim=output_dim, **kwargs)


def get_discriminator(discriminator_type, input_dim, **kwargs):
    return DISCRIMINATORS[discriminator_type](input_dim=input_dim, **kwargs)


def get_test_metrics(config, train_dl, test_dl):
    model_name = "%s_%s" % (config.dataset, config.algo)
    if config.conditional:
        config.update({"G_input_dim": config.G_input_dim + config.num_classes}, allow_val_change=True)
        x_real_train = torch.cat([loader_to_tensor(
            train_dl), loader_to_cond_tensor(train_dl, config)], dim=-1).to(config.device)
        x_real_test = torch.cat([loader_to_tensor(
            test_dl), loader_to_cond_tensor(test_dl, config)], dim=-1).to(config.device)
    else:
        x_real_train = loader_to_tensor(train_dl).to(config.device)
        x_real_test = loader_to_tensor(test_dl).to(config.device)
    print(model_name)

    # Compute test metrics for train and test set
    test_metrics_train = get_standard_test_metrics(x_real_train)
    test_metrics_test = get_standard_test_metrics(x_real_test)
    return {'train': test_metrics_train, 'test':test_metrics_test}


def model2trainers_GAN(generator, discriminator, 
                 test_metrics_train, test_metrics_test, 
                 train_dl, config):
    
    trainer_RCGAN = RCGANTrainer(G=generator, D=discriminator,
                                test_metrics_train=test_metrics_train, test_metrics_test=test_metrics_test,
                                train_dl=train_dl, batch_size=config.batch_size, n_gradient_steps=config.steps,
                                config=config)
    
    trainer_TIMEGAN = TIMEGANTrainer(G=generator, gamma=1,
                                    test_metrics_train=test_metrics_train,
                                    test_metrics_test=test_metrics_test,
                                    train_dl=train_dl, batch_size=config.batch_size,
                                    n_gradient_steps=config.steps, 
                                    config=config)
    # train
    model2trainer = {
        "ROUGH_RCGAN": trainer_RCGAN,
        "ROUGH_TimeGAN":  trainer_TIMEGAN,
        "AR1_RCGAN": trainer_RCGAN,
        "AR1_TimeGAN":  trainer_TIMEGAN,
        "GBM_RCGAN": trainer_RCGAN,
        "GBM_TimeGAN": TIMEGANTrainer(G=generator, gamma=1,
                                    test_metrics_train=test_metrics_train,
                                    test_metrics_test=test_metrics_test,
                                    train_dl=train_dl, batch_size=config.batch_size,
                                    n_gradient_steps=config.steps, 
                                    config=config)
                } #TODO: why dataset matters here?
    return model2trainer



def get_trainer_GAN(config, train_dl, test_dl):
    model_name = "%s_%s" % (config.dataset, config.algo)
    print(model_name)

    # Compute test metrics for train and test set
    test_metrics = get_test_metrics(config, train_dl, test_dl)
    test_metrics_train = test_metrics['train']
    test_metrics_test = test_metrics['test']

    # trainer
    D_out_dim = 1
    return_seq = False
    if config.algo == 'RCGAN':
        D_out_dim = 1
        return_seq = True

    generator = GENERATORS[config.generator](
        input_dim=config.G_input_dim, 
        hidden_dim=config.G_hidden_dim, 
        output_dim=config.input_dim, 
        n_layers=config.G_num_layers, 
        init_fixed=config.init_fixed
        )
    
    discriminator = DISCRIMINATORS[config.discriminator](
        input_dim=config.input_dim, 
        hidden_dim=config.D_hidden_dim, 
        out_dim=D_out_dim, 
        n_layers=config.D_num_layers, 
        return_seq=return_seq
        )
    
    print('GENERATOR:', generator)
    print('DISCRIMINATOR:', discriminator)

    trainers_map = model2trainers_GAN(generator, discriminator,
                                      test_metrics_train, test_metrics_test,
                                      train_dl, config)
    trainer = trainers_map[model_name]

    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    # Required for multi-GPU
    torch.backends.cudnn.benchmark = True
    return trainer

def get_trainer_VAE(config, train_dl, test_dl):
    model_name = "%s_%s" % (config.dataset, config.algo)
    print(model_name)

    # Compute test metrics for train and test set
    test_metrics = get_test_metrics(config, train_dl, test_dl)
    test_metrics_train = test_metrics['train']
    test_metrics_test = test_metrics['test']

    # train
    vae = VAES[config.model](hidden_layer_sizes=config.hidden_layer_sizes,
                                 trend_poly=config.trend_poly,
                                 num_gen_seas=config.num_gen_seas,
                                 custom_seas=config.custom_seas,
                                 use_scaler=config.use_scaler,
                                 use_residual_conn=config.use_residual_conn,
                                 n_lags=config.n_lags,
                                 input_dim=config.input_dim,
                                 latent_dim=config.latent_dim,
                                 reconstruction_wt=config.reconstruction_wt)

    print('VAE:', vae)
    trainer = {model_name: TimeVAETrainer(G=vae,
                                            test_metrics_train=test_metrics_train, test_metrics_test=test_metrics_test,
                                            train_dl=train_dl, batch_size=config.batch_size, n_gradient_steps=config.steps,
                                            config=config)}[model_name]
    
    
    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    # Required for multi-GPU
    torch.backends.cudnn.benchmark = True
    return trainer


def get_trainer(config, train_dl, test_dl):
    # print(config)
    print(config.algo)
    model_name = "%s_%s" % (config.dataset, config.algo)
    if config.model_type.upper() == "GAN":
        trainer = get_trainer_GAN(config, train_dl, test_dl)
    elif config.model_type.upper() == "VAE":
        trainer = get_trainer_VAE(config, train_dl, test_dl)
    else:
        raise NotImplementedError("only GAN and VAE are supported.")
    return trainer
