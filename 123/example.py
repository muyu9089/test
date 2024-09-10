from weak2strong import Weak2StrongExplanation, load_exp_data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# CUDA_VISIBLE_DEVICES=0
# if use local model, model_root_path should be a root path; 
# or the model is loaded from hugging face, this should be owner, like "meta-llama/"
# model_root_path = "./"
# load_exp_data needs correct model_name so that load conversation template from fastchat.
# model_name = "Llama-2-7b-chat-hf"
# model_path = "/home/users/panjia/Llama-2-7b-chat-hf"

model_name = "mistral-7b-sft-beta"
model_path = "/home/users/panjia/decoding-time-realignment-main/mistral-7b-sft-beta"

normal, malicious, jailbreak = load_exp_data(use_conv=True, model_name=model_name)

test = Weak2StrongExplanation(model_path, layer_nums=32, return_report=False, return_visual=True)

test.explain({"norm":normal, "mali":malicious}, classifier_list=["svm", "mlp"], accuracy=True)

