from transformer_lens import utils

global_cfg = {

}

training_cfg = {
    "learning_rate": 1e-3,  # Learning rate for the optimizer
    "epochs": 100,
    "epochs_per_checkpoint": 10,
    "batch_size": 64,
    "wandb_project": "autoencoder",
    "wandb_entity": "collingray",
}

n = 10  # Dimension of the input
m = 5  # Dimension of the hidden layer
lambda_reg = 0.01  # Regularization strength
learning_rate = 0.001
batch_size = 64  # Batch size for training
epochs = 50  # Number of training epochs
dataset_size = 1000  # Assuming we have 1000 samples

layer = 31

autoencoder_cfg = {
    "layer": 0,
    "n_dim": 10,
    "m_dim": 1000,
    "seed": 777,
    "device": "cuda",
    "lambda_reg": 0,
    "dtype": "fp16",  # todo: try using different dtypes for the buffer and the autoencoder, also try fp8
}

buffer_cfg = {
    "act_size": 4096*4,  # the width of the activations vectors, should match the size of the mlp layer of the model
    "act_name": utils.get_act_name("mlp_out", layer),  # the name of the activations to grab from the model. todo
    "buffer_size": 256,  # the size of the buffer, in number of activations
    "buffer_batches": 128,  # the number of batches to read in whenever the buffer is refreshed
    "device": "cuda",
    "enc_dtype": "fp16",
    "layer": layer,  # which layer to grab activations from
    "model_name": "Llama-2-7b-hf",  # which model to use. todo: find way to auto pull hyper-params from the specified model
    "model_path": "./weights/llama-2-7b/",
    "model_batch_size": 8,  # how many tokens to process at a time
}
