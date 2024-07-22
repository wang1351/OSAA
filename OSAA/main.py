import os
import argparse
import warnings
import Dtrainer
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='distantDA',               type=str, help='Name of your experiment (EEG, HAR, HHAR_SA, ')
parser.add_argument('--run_description',        default='test',                     type=str, help='name of your runs')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='DISTANT',               type=str, help='DSAN, DANN, Deep_Coral, DSAN, MMDA,  DIRT, CDAN, HoMM, CoDATS, advskm, adda')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'.',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='PU',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - encoder)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda:0',                   type=str, help='cpu or cuda')

# ======== sweep settings =====================
parser.add_argument('--is_sweep',               default=False,                      type=bool, help='singe run or sweep')
parser.add_argument('--num_sweeps',             default=1,                         type=str, help='Number of sweep runs')

parser.add_argument('--sweep_project_wandb',    default='distant',       type=str, help='Project name in Wandb')
parser.add_argument('--wandb_entity',           type=str, help='Entity name in Wandb (can be left blank if there is a default entity)')
parser.add_argument('--hp_search_strategy',     default="random",               type=str, help='The way of selecting hyper-parameters (random-grid-bayes). in wandb see:https://docs.wandb.ai/guides/sweeps/configuration')
parser.add_argument('--metric_to_minimize',     default="src_risk",             type=str, help='select one of: (src_risk - trg_risk - few_shot_trg_risk - dev_risk)')



args = parser.parse_args()

if __name__ == "__main__":
    trainer = Dtrainer.cross_domain_trainer(args)

    if args.is_sweep:
        trainer.sweep()
    else:
        trainer.train()
