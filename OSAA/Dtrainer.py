import pdb

import torch
import torch.nn.functional as F
import time
import os
import wandb
import pandas as pd
import numpy as np
from dataloader.dataloader import data_generator, data_inter_generator, generator_percentage_of_data,few_shot_data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class

from utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from utils import calc_dev_risk, calculate_risk
import warnings

import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import collections
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cudnn.benchmark = True  # to fasten TCN


class cross_domain_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        self.num_sweeps = args.num_sweeps

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description
        # sweep parameters
        self.is_sweep = args.is_sweep
        self.sweep_project_wandb = False
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method,
            'parameters': {**sweep_alg_hparams[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)  # Training with sweep

        # resuming sweep
        # wandb.agent('8wkaibgr', self.train, count=25,project='HHAR_SA_Resnet', entity= 'iclr_rebuttal' )

    def train(self):
        if self.is_sweep:
            wandb.init(config=self.default_hparams)
            run_name = f"sweep_{self.dataset}"
        else:
            run_name = f"{self.run_description}"
            wandb.init(config=self.default_hparams, mode="offline", name=run_name)  #online

        self.hparams = wandb.config
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        copy_Files(self.exp_log_dir)  # save a copy of training files:

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.

        self.metrics = {'accuracy': [], 'f1_score': [], 'src_risk': [], 'few_shot_trg_risk': [],
                        'trg_risk': [], 'dev_risk': []}

        for i in scenarios:
            src_id = i[0]
            trg_id = i[1]

            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                #fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)

                # Load data
                self.load_data(src_id, trg_id)

                # get algorithm
                algorithm_class = get_algorithm_class(self.da_method)
                #pdb.set_trace()
                backbone_fe = get_backbone_class(self.backbone)
                backbone_decoder = get_backbone_class(self.backbone+'Decoder')

                algorithm = algorithm_class(backbone_fe,backbone_decoder, self.dataset_configs, self.hparams, self.device)
                algorithm.to(self.device)

                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # training..
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!aaa define first
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    # print(time.time()-aaa)
                    # aaa = time.time()
                    joint_loaders = enumerate(zip(self.src_train_dl, self.inter_dl, self.trg_train_dl))
                    len_dataloader = min(len(self.src_train_dl), len(self.inter_dl), len(self.trg_train_dl))
                    algorithm.train()

                    for step, ((src_x, src_y), (inter_x),  (trg_x, _)) in joint_loaders:
                      #  pdb.set_trace()
                        src_x, src_y, inter_x, trg_x = src_x.float().to(self.device), src_y.long().to(self.device), inter_x[0].float().to(self.device), trg_x.float().to(self.device)
                        if self.da_method == "DANN" or self.da_method == "DIFFDANN" or self.da_method == "CoDATS" :
                            losses = algorithm.update(src_x, src_y, trg_x, step, epoch, len_dataloader)
                        elif self.da_method == "DISTANT":
                            inter_x = inter_x.squeeze(1)
                            losses = algorithm.update(src_x, src_y, inter_x, trg_x, epoch, len_dataloader)
                        else:
                            losses = algorithm.update(src_x, src_y, trg_x)

                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, src_x.size(0))

                    # logging
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    for key, val in loss_avg_meters.items():
                        self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                    self.logger.debug(f'-------------------------------------')

                self.algorithm = algorithm

                save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                                self.scenario_log_dir, self.hparams)
               # self.after_run()
                self.evaluate()
                #self.feature_space_tsne_visualization(scenario_id=i)
                self.calc_results_per_run()

        # logging metrics
        self.calc_overall_results()
        average_metrics = {metric: np.mean(value) for (metric, value) in self.metrics.items()}
        wandb.log(average_metrics)
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log({'avg_results': wandb.Table(dataframe=self.averages_results_df, allow_mixed_types=True)})
        wandb.log({'std_results': wandb.Table(dataframe=self.std_results_df, allow_mixed_types=True)})
        
    def after_run(self):
        """
        Additional run to collect the training features for NCE classifier/TSNE visualization
        """
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        feature_extractor.eval()

        # Collect all the features and labels
        self.src_all_features = np.array([])
        self.src_all_labels = np.array([])

        with torch.no_grad():
            for bs_idx, (data, labels) in enumerate(self.src_train_dl):
                data = data.float().to(self.device) 
                labels = labels.view((-1)).long().to(self.device)
                # forward pass
                features = F.normalize(feature_extractor(data)[0].detach())

                if bs_idx == 0:
                    self.src_all_features = features.cpu().numpy()
                else:
                    self.src_all_features = np.concatenate((self.src_all_features, features.cpu().numpy()), axis=0)

                self.src_all_labels = np.append(self.src_all_labels, labels.data.cpu().numpy())

    def evaluate(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        feature_extractor.eval()

        # FC layer
        classifier = self.algorithm.classifier.to(self.device)
        classifier.eval()

       

        # ############################# Testing #############################
        total_loss_ = []
        self.trg_all_features = np.array([])
        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])


        with torch.no_grad():
            for bs_idx, (data, labels) in enumerate(self.trg_test_dl):
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features, _ = feature_extractor(data)

                # Collect the features for TSNE
                if bs_idx == 0:
                    self.trg_all_features = F.normalize(features.detach()).cpu().numpy()
                else:
                    self.trg_all_features = np.concatenate((self.trg_all_features, F.normalize(features.detach()).cpu().numpy()), axis=0)

                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())

        self.trg_loss = torch.tensor(total_loss_).mean()     

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs,
                                                             self.hparams)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs,
                                                             self.hparams)
        self.few_shot_dl = few_shot_data_generator(self.trg_test_dl)

        self.inter_dl = data_inter_generator(self.data_path, src_id+trg_id, self.dataset_configs,
                                                             self.hparams)


    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calc_results_per_run(self):
        '''
        Calculates the acc, f1 and risk values for each cross-domain scenario
        '''

        self.acc, self.f1 = _calc_metrics(self.trg_pred_labels, self.trg_true_labels, self.scenario_log_dir,
                                          self.home_path,
                                          self.dataset_configs.class_names)
        if self.is_sweep:
            self.src_risk = calculate_risk(self.algorithm, self.src_test_dl, self.device)
            self.trg_risk = calculate_risk(self.algorithm, self.trg_test_dl, self.device)
            self.few_shot_trg_risk = calculate_risk(self.algorithm, self.few_shot_dl, self.device)
            self.dev_risk = calc_dev_risk(self.algorithm, self.src_train_dl, self.trg_train_dl, self.src_test_dl,
                                          self.dataset_configs, self.device)

            run_metrics = {'accuracy': self.acc,
                           'f1_score': self.f1,
                           'src_risk': self.src_risk,
                           'few_shot_trg_risk': self.few_shot_trg_risk,
                           'trg_risk': self.trg_risk,
                           'dev_risk': self.dev_risk}

            df = pd.DataFrame(columns=["acc", "f1", "src_risk", "few_shot_trg_risk", "trg_risk", "dev_risk"])
            df.loc[0] = [self.acc, self.f1, self.src_risk, self.few_shot_trg_risk, self.trg_risk,
                         self.dev_risk]
        else:
            run_metrics = {'accuracy': self.acc, 'f1_score': self.f1}
            df = pd.DataFrame(columns=["acc", "f1"])
            df.loc[0] = [self.acc, self.f1]

        for (key, val) in run_metrics.items(): self.metrics[key].append(val)

        scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, "scores.xlsx")
        df.to_excel(scores_save_path, index=False)
        self.results_df = df

    def calc_overall_results(self):
        exp = self.exp_log_dir

        # for exp in experiments:
        if self.is_sweep:
            results = pd.DataFrame(
                columns=["scenario", "acc", "f1", "src_risk", "few_shot_trg_risk", "trg_risk", "dev_risk"])
        else:
            results = pd.DataFrame(columns=["scenario", "acc", "f1"])

        scenarios_list = os.listdir(exp)
        scenarios_list = [i for i in scenarios_list if "_to_" in i]
        scenarios_list.sort()

        unique_scenarios_names = [f'{i}_to_{j}' for i, j in self.dataset_configs.scenarios]

        for scenario in scenarios_list:
            scenario_dir = os.path.join(exp, scenario)
            scores = pd.read_excel(os.path.join(scenario_dir, 'scores.xlsx'))
            scores.insert(0, 'scenario', '_'.join(scenario.split('_')[:-2]))
            results = pd.concat([results, scores])

        avg_results = results.groupby('scenario').mean()
        std_results = results.groupby('scenario').std()

        avg_results.loc[len(avg_results)] = avg_results.mean()
        avg_results.insert(0, "scenario", list(unique_scenarios_names) + ['mean'], True)
        std_results.insert(0, "scenario", list(unique_scenarios_names), True)

        report_save_path_avg = os.path.join(exp, f"Average_results.xlsx")
        report_save_path_std = os.path.join(exp, f"std_results.xlsx")

        self.averages_results_df = avg_results
        self.std_results_df = std_results
        avg_results.to_excel(report_save_path_avg)
        std_results.to_excel(report_save_path_std)
    def feature_space_tsne_visualization(self, scenario_id):
        # Concatenate features and domain labels
        src_trg_features = np.concatenate((self.src_all_features, self.trg_all_features), axis=0)
        src_trg_class_labels = np.concatenate((self.src_all_labels.astype('int32'), self.trg_true_labels.astype('int32')), axis=0)
        domain_labels = np.concatenate((np.zeros(self.src_all_features.shape[0], dtype=int),
                                        np.ones(self.trg_all_features.shape[0], dtype=int)))

        # TSNE visualization
        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        tsne_result = tsne.fit_transform(src_trg_features)

        df = pd.DataFrame(tsne_result, columns=['d1', 'd2'])
        df['domain'] = domain_labels
        df['class'] = src_trg_class_labels
        plt.figure(figsize=(6, 6), dpi=128)

        g1 = sns.scatterplot(
            x="d1", y="d2",
            hue="class",
            style="domain",
            palette="deep",
            s=20,
            data=df,
            legend="full",
            alpha=1
        )

        g1.set(xticklabels=[])  # remove the tick labels
        g1.set(xlabel=None)  # remove the axis label
        g1.set(yticklabels=[])  # remove the tick labels
        g1.set(ylabel=None)  # remove the axis label
        g1.tick_params(bottom=False, left=False)  # remove the ticks
        path = 'tsne_{}'.format(scenario_id)
        plt.savefig(path, bbox_inches='tight')
        plt.show()


def extract_samples_according_to_labels(x, y, target_ids):
    """
    Extract corresponding samples from x and y according to the labels
    :param x: data, np array
    :param y: labels, np array
    :param target_ids: list of labels
    :return:
    """
    # get the indices
    inds = list(map(lambda x: x in target_ids, y))
    x_extracted = x[inds]
    y_extracted = y[inds]

    return x_extracted, y_extracted