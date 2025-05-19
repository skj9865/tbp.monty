# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
from dataclasses import asdict

import numpy as np

from benchmarks.configs.names import MyExperiments
from tbp.monty.frameworks.config_utils.config_args import (
	LoggingConfig,
	MotorSystemConfigInformedNoTransStepS1,
    MotorSystemConfig,
    MotorSystemConfigNaiveScanSpiral,
    MotorSystemConfigSurface,
	PatchAndViewMontyConfig,
	PretrainLoggingConfig,
    CSVLoggingConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
	ExperimentArgs,
	OmniglotDataloaderArgs,
    OmniglotTrainDataloaderArgs,
    OmniglotEvalDataloaderArgs,
	OmniglotDatasetArgs,
    get_omniglot_train_dataloader,
    get_omniglot_eval_dataloader,
    MnistDatasetArgs,
    MnistDataloaderArgs,
    MnistEvalDataloaderArgs,
    get_mnist_train_dataloader,
    get_mnist_eval_dataloader
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.experiments import (
	MontyObjectRecognitionExperiment,
	MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.models.evidence_matching import (
	EvidenceGraphLM,
	MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.sensor_modules import (
	DetailedLoggingSM,
	HabitatDistantPatchSM,  

)

from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler


omniglot_sensor_module_config = dict(
	sensor_module_0=dict(
    	sensor_module_class=HabitatDistantPatchSM,
    	sensor_module_args=dict(
        	sensor_module_id="patch",
        	features=[                
            	"pose_vectors",
            	"pose_fully_defined",
            	"on_object",
            	"principal_curvatures_log",
        	],
        	save_raw_obs=False,
        	# Need to set this lower since curvature is generally lower
        	pc1_is_pc2_threshold=1,
    	),
	),
	sensor_module_1=dict(
    	sensor_module_class=DetailedLoggingSM,
    	sensor_module_args=dict(
        	sensor_module_id="view_finder",
        	save_raw_obs=False,
    	),
	),
)

monty_models_dir = os.getenv("MONTY_MODELS")

pretrain_dir = os.path.expanduser(os.path.join(monty_models_dir, "omniglot"))

omniglot_training = dict(
	experiment_class=MontySupervisedObjectPretrainingExperiment,
	experiment_args=ExperimentArgs(
    	n_train_epochs=1,
    	do_eval=False,
	),
	logging_config=PretrainLoggingConfig(
    	output_dir=pretrain_dir,
	),
	monty_config=PatchAndViewMontyConfig(
    	# Take 1 step at a time, following the drawing path of the letter
    	motor_system_config=MotorSystemConfigInformedNoTransStepS1(),
    	sensor_module_configs=omniglot_sensor_module_config,
	),
	dataset_class=ED.EnvironmentDataset,
	dataset_args=OmniglotDatasetArgs(),
	train_dataloader_class=ED.OmniglotDataLoader,
	# Train on the first version of each character (there are 20 drawings for each
	# character in each alphabet, here we see one of them). The default
	# OmniglotDataloaderArgs specify alphabets = [0, 0, 0, 1, 1, 1] and
    # characters = [1, 2, 3, 1, 2, 3]) so in the first episode we will see version 1
	# of character 1 in alphabet 0, in the next episode version 1 of character 2 in
	# alphabet 0, and so on.
	train_dataloader_args=OmniglotDataloaderArgs(versions=[1, 1, 1, 1, 1, 1]),
    #train_dataloader_args=OmniglotTrainDataloaderArgs(),
    #train_dataloader = get_omniglot_train_dataloader(alphabet_ids=[0,1,2], num_versions=2),
)

omniglot_inference = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=ExperimentArgs(
        model_name_or_path=pretrain_dir + "/omniglot_training/pretrained/",
        do_train=False,
        n_eval_epochs=1,
    ),
    logging_config=LoggingConfig(),
    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    # xyz values are in larger range so need to increase mmd
                    max_match_distance=5,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                            "pose_vectors": np.ones(3) * 45,
                        }
                    },
                    # Point normal always points up, so they are not useful
                    feature_weights={
                        "patch": {
                            "pose_vectors": [0, 1, 0],
                        }
                    },
                    # We assume the letter is presented upright
                    initial_possible_poses=[[0, 0, 0]],
                ),
            )
        ),
        sensor_module_configs=omniglot_sensor_module_config,
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=OmniglotDatasetArgs(),
    eval_dataloader_class=ED.OmniglotDataLoader,
    # Using version 1 means testing on the same version of the character as trained.
    # Version 2 is a new drawing of the previously seen characters. In this small test
    # setting these are 3 characters from 2 alphabets.
    eval_dataloader_args=OmniglotDataloaderArgs(versions=[1, 1, 1, 1, 1, 1]),
    #eval_dataloader_args=OmniglotDataloaderArgs(versions=[2, 2, 2, 2, 2, 2]),
    #eval_dataloader_args=OmniglotEvalDataloaderArgs(),
    #eval_dataloader_args = get_omniglot_eval_dataloader( alphabet_ids=[0,1,2], start_at_version=0,num_versions=4),
)


# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)

mnist_pretrain_dir = os.path.expanduser(os.path.join(monty_models_dir, "mnist"))

mnist_sensor_module_config = dict(
	sensor_module_0=dict(
    	sensor_module_class=HabitatDistantPatchSM,
    	sensor_module_args=dict(
        	sensor_module_id="patch",
        	features=[ 
                "rgba",               
            	"pose_vectors",
            	"pose_fully_defined",
            	"on_object",
            	"principal_curvatures_log",
        	],
        	save_raw_obs=False,
        	# Need to set this lower since curvature is generally lower
        	pc1_is_pc2_threshold=1,
    	),
	),
	sensor_module_1=dict(
    	sensor_module_class=DetailedLoggingSM,
    	sensor_module_args=dict(
        	sensor_module_id="view_finder",
        	save_raw_obs=False,
    	),
	),
)


mnist_training = dict(
	experiment_class=MontySupervisedObjectPretrainingExperiment,
	experiment_args=ExperimentArgs(
    	n_train_epochs=1,
    	do_eval=False,
	),
    logging_config=CSVLoggingConfig(
                output_dir="mnist/log",
                monty_log_level="BASIC",
                monty_handlers=[BasicCSVStatsHandler],                 
            ),

	monty_config=PatchAndViewMontyConfig(
    	# Take 1 step at a time, following the drawing path of the letter
    	#motor_system_config=MotorSystemConfigInformedNoTransStepS1(),
        motor_system_config=MotorSystemConfigNaiveScanSpiral(),
        #motor_system_config = MotorSystemConfigSurface(),
    	sensor_module_configs=mnist_sensor_module_config,
	),
	dataset_class=ED.EnvironmentDataset,
	dataset_args=MnistDatasetArgs(),
	train_dataloader_class=ED.MnistDataLoader,
	train_dataloader_args=MnistDataloaderArgs(),
    #train_dataloader_args = get_mnist_train_dataloader(start_at_version = 0, number_ids = np.arange(0,10), num_versions=30)
)

mnist_inference = dict(
    experiment_class=MontyObjectRecognitionExperiment,
    experiment_args=ExperimentArgs(
        #model_name_or_path=pretrain_dir + "/mnist_training/",
        model_name_or_path = "mnist/log/mnist_training/pretrained",
        do_train=False,
        n_eval_epochs=1,
    ),
    #logging_config=LoggingConfig(),
    logging_config=CSVLoggingConfig(
            output_dir="mnist/log",
            monty_log_level="BASIC",
            monty_handlers=[BasicCSVStatsHandler],                 
        ),

    monty_config=PatchAndViewMontyConfig(
        monty_class=MontyForEvidenceGraphMatching,
        learning_module_configs=dict(
            learning_module_0=dict(
                learning_module_class=EvidenceGraphLM,
                learning_module_args=dict(
                    # xyz values are in larger range so need to increase mmd
                    max_match_distance=5,
                    tolerances={
                        "patch": {
                            "principal_curvatures_log": np.ones(2),
                            "pose_vectors": np.ones(3) * 45,
                        }
                    },
                    # Point normal always points up, so they are not useful
                    feature_weights={
                        "patch": {
                            "pose_vectors": [0, 1, 0],
                        }
                    },
                    # We assume the letter is presented upright
                    initial_possible_poses=[[0, 0, 0]],
                ),
            )
        ),
        sensor_module_configs=mnist_sensor_module_config,
    ),
    dataset_class=ED.EnvironmentDataset,
    dataset_args=MnistDatasetArgs(),
    eval_dataloader_class=ED.MnistDataLoader,
    eval_dataloader_args=MnistEvalDataloaderArgs(),
    #eval_dataloader_args = get_mnist_eval_dataloader(start_at_version = 0, number_ids = np.arange(0,10), num_versions=60)
)


experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    omniglot_training=omniglot_training,
	omniglot_inference=omniglot_inference,
    mnist_training = mnist_training,
    mnist_inference = mnist_inference,
)
CONFIGS = asdict(experiments)
