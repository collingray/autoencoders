import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from utils import *


class ActivationsBuffer:
    """
    A data buffer to store MLP activations for training the autoencoder.

    Adapted from code by Neel Nanda: https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py

    Cleaned up and modified to move all state inside the class, including the model reference, in order to simplify
    composition. The original design which reloads the buffer once it dips below 1/2 of capacity remains, although I'm
    unclear why this is done, probably just a hack to avoid having to worry about overflows
    """

    def __init__(self, cfg):
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["act_size"]), dtype=torch.bfloat16).to(cfg["device"])
        self.cfg = cfg
        # pointer to read/write location in the buffer, reset to 0 when refresh is called
        self.buffer_pointer = 0
        # pointer to the start of the next token sequence in the dataset, incremented in steps of model_batch_size
        self.token_pointer = 0
        # the number of batches to read in whenever the buffer is refreshed, buffer_batches on the first refresh and
        # buffer_batches // 2 on subsequent refreshes
        self.num_batches = self.cfg["buffer_batches"]

        # load the model, using the local path if provided
        if "model_path" in cfg.keys():
            # load into cpu first, to avoid gpu memory issues
            hf_model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], torch_dtype="auto").to(device="cpu", dtype=DTYPES[cfg["enc_dtype"]])

            self.model = HookedTransformer.from_pretrained_no_processing(model_name=cfg["model_name"], hf_model=hf_model, device=cfg["device"], dtype=DTYPES[cfg["enc_dtype"]])

            # the tokenizer isn't used, but errors are thrown if it's not present
            self.model.tokenizer = LlamaTokenizerFast.from_pretrained(TOKENIZER_PATH)

            # delete the cpu copy to free up memory
            del hf_model
        else:
            self.model = HookedTransformer.from_pretrained(model_name=cfg["model_name"], device=cfg["device"], dtype=DTYPES[cfg["enc_dtype"]])

        self.refresh()

    @torch.no_grad()
    def refresh(self):
        self.buffer_pointer = 0

        model_batch_size = self.cfg["model_batch_size"]
        layer = self.cfg["layer"]
        act_name = self.cfg["act_name"]
        act_size = self.cfg["act_size"]

        with torch.autocast("cuda", torch.bfloat16):
            # iterates through the dataset in order to fill up the buffer with activations from the model after
            # passing in model_batch_size tokens at a time
            for _ in range(0, self.num_batches, model_batch_size):
                # grab next model_batch_size tokens from the dataset
                tokens = data[self.token_pointer:self.token_pointer + model_batch_size]
                # run the model on the tokens, stopping at the specified layer and grabbing the activations for the
                # specified layer
                _, cache = self.model.run_with_cache(tokens, stop_at_layer=layer + 1, names_filter=act_name)
                # reshape the activations to be 2D, with each row being a single activation vector of size act_size
                acts = cache[act_name, layer].reshape(-1, act_size)

                # write the activations to the buffer, overwriting previous values
                self.buffer[self.buffer_pointer: self.buffer_pointer + acts.shape[0]] = acts
                # increment the buffer pointer by the number of activations written
                self.buffer_pointer += acts.shape[0]
                # increment the token pointer by the number of tokens processed
                self.token_pointer += model_batch_size

        self.num_batches = self.cfg["buffer_batches"] // 2
        self.buffer_pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])]

    @torch.no_grad()
    def next(self):
        batch_size = self.cfg["batch_size"]
        out = self.buffer[self.buffer_pointer:self.buffer_pointer + batch_size]
        self.buffer_pointer += batch_size
        if self.buffer_pointer > self.buffer.shape[0] // 2 - batch_size:
            # print("Refreshing the buffer!")
            self.refresh()
        return out
