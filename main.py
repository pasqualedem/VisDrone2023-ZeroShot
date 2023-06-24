import argparse
import json

from ezdl.utils.utilities import update_collection, load_yaml

parser = argparse.ArgumentParser(description='Train and test models')
parser.add_argument('action',
                    help='Choose the action to perform: '
                         'preprocess, padding, test_dataset, preprocess, manipulate, app',
                    default="experiment", type=str)
parser.add_argument('--resume', required=False, action='store_true',
                    help='Resume the experiment', default=False)
parser.add_argument('-d', '--dir', required=False, type=str,
                    help='Set the local tracking directory', default=None)
parser.add_argument('-m', '--mode', required=False, type=str,
                    help='Preprocessing mode: default / tiled', default="default")
parser.add_argument('-f', '--file', required=False, type=str,
                    help='Set the config file', default=None)
parser.add_argument("--grid", type=int, help="Select the first grid to start from")
parser.add_argument("--run", type=int, help="Select the run in grid to start from")

parser.add_argument('--subset', type=str, help="Subset chosen for preprocessing and augmentation")

if __name__ == '__main__':
    args = parser.parse_args()

    track_dir = args.dir
    action = args.action

    if action == "experiment":
        from ezdl.experiment.experiment import experiment
        param_path = args.file or 'parameters.yaml'
        settings = load_yaml(param_path)
        settings['experiment'] = update_collection(settings['experiment'], args.resume, key='resume')
        settings['experiment'] = update_collection(settings['experiment'], args.grid, key='start_from_grid')
        settings['experiment'] = update_collection(settings['experiment'], args.run, key='start_from_run')
        settings['experiment'] = update_collection(settings['experiment'], track_dir, key='tracking_dir')
        experiment(settings)