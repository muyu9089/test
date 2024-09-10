from load_data import get_data, load_conv
from load_model import get_model, step_forward
import numpy as np
import os
from w2s_utils import get_layer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import accelerate
from visualization import topk_intermediate_confidence_heatmap, accuracy_line
from sklearn.metrics import accuracy_score
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



norm_prompt_path = './exp_data/normal_prompt.csv'
jailbreak_prompt_path = './exp_data/jailbreak_prompt.csv'
malicious_prompt_path = './exp_data/malicious_prompt.csv'

# jailbreak_prompt_path = './exp_data/jailbreak_prompt.csv'
# norm_prompt_path = './exp_data/normal_one.csv'
# malicious_prompt_path = './exp_data/malicious_one.csv'


def load_exp_data(shuffle_seed=None, use_conv=False, model_name=None):
    normal_inputs = get_data(norm_prompt_path, shuffle_seed)
    malicious_inputs = get_data(malicious_prompt_path, shuffle_seed)
    if os.path.exists(jailbreak_prompt_path):
        jailbreak_inputs = get_data(jailbreak_prompt_path, shuffle_seed)
    else:
        jailbreak_inputs = None
    if use_conv and model_name is None:
        raise ValueError("please set model name for load")
    if use_conv:
        normal_inputs = [load_conv(model_name, _) for _ in normal_inputs]
        malicious_inputs = [load_conv(model_name, _) for _ in malicious_inputs]
        jailbreak_inputs = [load_conv(model_name, _) for _ in jailbreak_inputs] if jailbreak_inputs is not None else None
    return normal_inputs, malicious_inputs, jailbreak_inputs


class Weak2StrongClassifier:
    def __init__(self, return_report=True, return_visual=False):
        self.return_report = return_report
        self.return_visual = return_visual

    @staticmethod
    def _process_data(forward_info):
        features = []
        labels = []
        for key, value in forward_info.items():
            for hidden_state in value["hidden_states"]:
                features.append(hidden_state.flatten())
                labels.append(value["label"])

        features = np.array(features)
        labels = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        all_indices = np.arange(len(features))
        train_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=42)
        
        print("--X_train--")
        print(X_train)
        print("--y_train--")
        print(y_train)
        print("--X_test--")
        print(X_test)
        print("--y_test--")
        print(y_test)


        print("--train_indices--")
        print(train_indices)
        print("--test_indices--")
        print(test_indices)
        

        return X_train, X_test, y_train, y_test, train_indices, test_indices

    def svm(self, forward_info):
        X_train, X_test, y_train, y_test, train_indices, test_indices = self._process_data(forward_info)
        svm_model = SVC(kernel='linear', probability=True)
        svm_model.fit(X_train, y_train)
        classes = svm_model.classes_ 

        # print("--svm--classes--")
        # print(classes)

        y_pred = svm_model.predict(X_test)
        y_proba = svm_model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        unsafe_score_dict = {}
        for i in test_indices:
            unsafe_score_dict[i] = {"ori":forward_info[i]["ori"], "label":forward_info[i]["label"], "unsafe_score": 0.0}

        return X_test, y_test, y_pred, y_proba, accuracy, unsafe_score_dict
    
    def mlp(self, forward_info):
        X_train, X_test, y_train, y_test, train_indices, test_indices = self._process_data(forward_info)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.01,
                            solver='adam', verbose=0, random_state=42,
                            learning_rate_init=.01)

        mlp.fit(X_train_scaled, y_train)

        classes = mlp.classes_ 
        # print("--mlp--classes--")
        # print(classes)
        
        y_pred = mlp.predict(X_test_scaled)
        y_proba = mlp.predict_proba(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        unsafe_score_dict = {}
        for i in test_indices:
            unsafe_score_dict[i] = {"ori":forward_info[i]["ori"], "label":forward_info[i]["label"], "unsafe_score": 0.0}

        return X_test, y_test, y_pred, y_proba, accuracy, unsafe_score_dict
    
    # def svm(self, forward_info):
    #     X_train, X_test, y_train, y_test = self._process_data(forward_info)
    #     svm_model = SVC(kernel='linear')
    #     svm_model.fit(X_train, y_train)

    #     y_pred = svm_model.predict(X_test)

    #     report = None
    #     if self.return_report:
    #         print("SVM Test Classification Report:")
    #         print(classification_report(y_test, y_pred, zero_division=0.0))
    #     if self.return_visual:
    #         report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)
    #     return X_test, y_pred, report
    

    # def mlp(self, forward_info):
    #     X_train, X_test, y_train, y_test = self._process_data(forward_info)
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)

    #     mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.01,
    #                         solver='adam', verbose=0, random_state=42,
    #                         learning_rate_init=.01)

    #     mlp.fit(X_train_scaled, y_train)
    #     y_pred = mlp.predict(X_test_scaled)
    #     report = None
    #     if self.return_report:
    #         print("MLP Test Classification Report:")
    #         print(classification_report(y_test, y_pred, zero_division=0.0))
    #     if self.return_visual:
    #         report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)
    #     return X_test, y_pred, report
    


class Weak2StrongExplanation:
    def __init__(self, model_path, layer_nums=32, return_report=True, return_visual=True):
        self.model, self.tokenizer = get_model(model_path)
        self.model_name = model_path.split("/")[-1]
        self.layer_sums = layer_nums + 1
        self.forward_info = {}
        self.return_report = return_report
        self.return_visual = return_visual

    def get_forward_info(self, inputs_dataset, class_label, debug=True):
        offset = len(self.forward_info)
        for _, i in enumerate(inputs_dataset):
            if debug and _ > 499:
                break
            list_hs, tl_pair = step_forward(self.model, self.tokenizer, i)
            # print("----list_hs:---") # 当前层的隐藏状态 array形式
            # print(list_hs)
            # print("----tl_pair:---") # top-k个可能的下一个token
            # print(tl_pair)
            print("--class_label--")
            print(class_label)
            last_hs = [hs[:, -1, :] for hs in list_hs]
            self.forward_info[_ + offset] = {"hidden_states": last_hs, "top-value_pair": tl_pair, "label": class_label, "ori": i}
            # print("---_ + offset---")
            # print(_ + offset)
            # print   

    def explain(self, datasets, classifier_list=None, debug=True, accuracy=True):
        self.forward_info = {}
        if classifier_list is None:
            classifier_list = ["svm", "mlp"]
        forward_info = {}
        if isinstance(datasets, list):
            for class_num, dataset in enumerate(datasets):
                self.get_forward_info(dataset, class_num, debug=debug)
                # print("--list class_key--")
                # print(class_key)
                # print("--list dataset--")
                # print(dataset)
        elif isinstance(datasets, dict):
            for class_key, dataset in datasets.items():
                self.get_forward_info(dataset, class_key, debug=debug)
                # print("--dict class_key--")
                # print(class_key)
                # print("--dict dataset--")
                # print(dataset)
        
        classifier = Weak2StrongClassifier(self.return_report, self.return_visual)


        rep_dict = {}
        accuracy_dict = {}
        

        if "svm" in classifier_list:

            unused_X_test, unused_y_test, unused_y_pred, unused_y_proba, unused_accuracy, unsafe_score_dict = classifier.svm(get_layer(self.forward_info, 0)) # 只是为了获取 unsafe_score_dict
            # print("--unsafe_score_dict--")
            # print(unsafe_score_dict)
            # print(unsafe_score_dict.keys())
            # print(unsafe_score_dict[17])

            # length = 10  # 列表的长度
            # value = 0.0  # 初始化值
            # score_list = [value] * length

            rep_dict["svm"] = {}
            accuracy_dict["svm"] = {}
            for _ in range(0, self.layer_sums):
                # X_test, y_pred, y_proba, _ = classifier.svm(get_layer(self.forward_info, _))
                X_test, y_test, y_pred, y_proba, accuracy, unused_dict = classifier.svm(get_layer(self.forward_info, _))
                # print("--layer--")
                # print(_)
                # print("--y_test--")
                # print(y_test)
                # print("--y_pred--")
                # print(y_pred)
                # print("--y_proba--")
                # print(y_proba)
                
                rep_dict["svm"][_] = y_proba
                accuracy_dict["svm"][_] = accuracy

                if _ in range(6,15): #左闭右开
                    print("--layer--")
                    print(_)
                    for num, key in enumerate(unsafe_score_dict.keys()):
                        # print("--sample-key--")
                        # print(num)
                        # print(type(num))

                        # print("--unsafe_score_dict[17]--")

                        # print(unsafe_score_dict[17])
                        # print("--unsafe_score_dict.get[num]--")

                        # print(unsafe_score_dict.get[num])

                        # print("all-proba")
                        # print(y_proba[num]) 

                        # print("y_proba_mail")
                        # print(y_proba[num][0])
                        # print(unsafe_score_dict[num]["unsafe_score"])
                        # x = unsafe_score_dict[num]["unsafe_score"] + y_proba[num][0] # y_proba[num][0]预测为mail的概率
                        unsafe_score_dict[key]["unsafe_score"] += y_proba[num][0]
                        # print(unsafe_score_dict[num]["unsafe_score"])
                        # unsafe_score_dict[num]["unsafe_score"] = x

                # print("len--X_test")
                # print(len(X_test))
            svm_df = pd.DataFrame.from_dict(unsafe_score_dict, orient='index')
            svm_df.to_csv('svm.csv', index=False) 

        if "mlp" in classifier_list:

            unused_X_test, unused_y_test, unused_y_pred, unused_y_proba, unused_accuracy, unsafe_score_dict = classifier.mlp(get_layer(self.forward_info, 0)) # 只是为了获取 unsafe_score_dict

            rep_dict["mlp"] = {}
            accuracy_dict["mlp"] = {}
            for _ in range(0, self.layer_sums):
                # X_test, y_pred, y_proba, _ = classifier.mlp(get_layer(self.forward_info, _))
                X_test, y_test,y_pred, y_proba, accuracy, unused_dict = classifier.mlp(get_layer(self.forward_info, _))
                rep_dict["mlp"][_] = y_proba
                accuracy_dict["mlp"][_] = accuracy

                if _ in range(6,15): #左闭右开
                    for num, key in enumerate(unsafe_score_dict.keys()):
                        # print("--sample-num--")
                        # print(num)

                        # print("all-proba")
                        # print(y_proba[num]) 

                        # print("y_proba_mail")
                        # print(y_proba[num][0])
                        # print(unsafe_score_dict[num]["unsafe_score"])
                        # x = unsafe_score_dict[num]["unsafe_score"] + y_proba[num][0] # y_proba[num][0]预测为mail的概率
                        unsafe_score_dict[key]["unsafe_score"] += y_proba[num][0]
                        # print(unsafe_score_dict[num]["unsafe_score"])
                        # unsafe_score_dict[num]["unsafe_score"] += y_proba[num][0] # y_proba[num][0]预测为mail的概率

            mlp_df = pd.DataFrame.from_dict(unsafe_score_dict, orient='index')
            mlp_df.to_csv('mlp.csv', index=False) 

                # print("--layer--")
                # print(_)
                # print("--y_test--")
                # print(y_test)
                # print("--y_pred--")
                # print(y_pred)
                # print("--y_proba--")
                # print(y_proba)

        # print("---rep_dict[svm]---")
        # print(rep_dict["svm"])
        # print("---accuracy_dict[svm]---")
        # print(accuracy_dict["svm"])

        # print("---self.forward_info---")
        # print(len(self.forward_info))
        # print(self.forward_info)

        # print("--self.forward_info[1]--")
        # print(self.forward_info[1])
        # print("--self.forward_info[0]--")
        # print(self.forward_info[0])



        # if not self.return_visual:
        #     return
        
        # if accuracy and classifier_list != []:
        #     accuracy_line(rep_dict, self.model_name)



    

    def vis_heatmap(self, dataset, left=0, right=33, debug=True, model_name=""):
        self.forward_info = {}
        self.get_forward_info(dataset, 0, debug=debug)
        topk_intermediate_confidence_heatmap(self.forward_info, left=left, right=right,model_name=model_name)
            

if __name__ == '__main__':
    model_name = "mistral-7b-sft-beta"
    model_path = "/home/users/panjia/decoding-time-realignment-main/mistral-7b-sft-beta"

    normal, malicious, jailbreak = load_exp_data(use_conv=True, model_name=model_name)

    test = Weak2StrongExplanation(model_path, layer_nums=32, return_report=False, return_visual=True)

    test.explain({"norm":normal, "mali":malicious}, classifier_list=["svm", "mlp"], accuracy=True)