from typing import Iterator, Tuple, Any, Union

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from pubsub_server.messages.commons import DataEntry, Image, Text, Audio
from teleop.data_classes import TargetPosition
from pubsub_server.messages.base import DataModel 

import copy 

from pydantic import ValidationError, BaseModel 
from logging import getLogger

from PIL import Image as PILImage 
import io

import sys
from logging import getLogger
 


class ExampleDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        # 'wrist_image': tfds.features.Image(
                        #     shape=(64, 64, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Wrist camera RGB observation.',
                        # ),
                        'state': tfds.features.Tensor(
                            shape=(10,),
                            dtype=np.float32,
                            doc='Robot state, consists of [x, y, theta, arm, lift, '
                                'wrist_yaw, wrist_pitch, wrist_roll, stretch_gripper, head_pan].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(10,),
                        dtype=np.float32,
                        doc='Robot state, consists of [x, y, theta, arm, lift, '
                            'wrist_yaw, wrist_pitch, wrist_roll, stretch_gripper, head_pan].',
                            
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='../../stretch_teleop_server/sweeping_green_cubes/sweep_*.jsonl'),
            }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        logger = getLogger(__name__)

        
        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset

            data_model = Union[Text, Image, TargetPosition, Audio]

            # assemble episode 
            episode = []
            
            first_step = True 
            time_diff = []
            
            # compute Kona language embedding
            language_embedding = self._embed(['clean'])[0].numpy()


            #set up template for each step 
            empty_step = {
                'observation': {
                    'image': None,
                    #'wrist_image': None,
                    'state': None,
                },
                'action': None,
                'discount': 1.0,
                'reward': None,
                'is_first': first_step,
                'is_last': None,
                'is_terminal': None,
                'language_instruction': 'clean',
                'language_embedding': language_embedding,
            }
            
            curr_step = copy.deepcopy(empty_step)
            
            # img_time = None 
            # action_time = None             

            with open(episode_path, "r") as f: 
                for lineno, line in enumerate(f.readlines()):
                    try:
                        #removed datamodel since i'm tired bro 
                        data_entry = DataEntry[data_model].model_validate_json(line)  # type: ignore[valid-type]
                    except ValidationError as e:
                        logger.error(
                            f"Validation error at Line {lineno}: {e}. Skipping this line."
                        )
                        continue

                    #save image  
                    if data_entry.channel == 'meta_2_head_cam' and curr_step['observation']['image'] is None: 
                        #convert bytes into 64 by 64 image 
                        image = PILImage.open(io.BytesIO(data_entry.data.image))
                        resized_image = image.resize((64, 64))
                        resized_image = np.array(resized_image, dtype=np.uint8)

                        curr_step['observation']['image'] = resized_image
                        img_time = data_entry.timestamp 
                    elif data_entry.channel == 'stretch_status': 
                        #Robot state, consists of [x, y, theta, arm, lift, wrist_yaw, wrist_pitch, wrist_roll, stretch_gripper, head_pan]
                        curr_step['observation']['state'] = [data_entry.data.x, data_entry.data.y, data_entry.data.theta, 
                                                            data_entry.data.arm, data_entry.data.lift, data_entry.data.wrist_yaw, 
                                                            data_entry.data.wrist_pitch, data_entry.data.wrist_roll, 
                                                            data_entry.data.stretch_gripper, data_entry.data.head_pan]

                    elif data_entry.channel == 'quest_control' and curr_step['observation']['image'] is not None: 
                        curr_step['action'] = [data_entry.data.x, data_entry.data.y, data_entry.data.theta, 
                                                            data_entry.data.arm, data_entry.data.lift, data_entry.data.wrist_yaw, 
                                                            data_entry.data.wrist_pitch, data_entry.data.wrist_roll, 
                                                            data_entry.data.stretch_gripper, data_entry.data.head_pan]
                        
                        action_time = data_entry.timestamp 

                    if curr_step['observation']['image'] is not None and curr_step['observation']['state'] != None and curr_step['action'] != None: 

                        if first_step: 
                            curr_step['is_first'] = first_step
                            first_step = False
                        else: 
                            curr_step['is_first'] = first_step
                        
                        curr_step['is_last'] = False
                        curr_step['is_terminal'] = False   
                        
                        #for saving time difference inbetween frames
                        # time_between = data_entry.timestamp
                        # if action_time != None and img_time != None: 
                        #     diff = (action_time-img_time).total_seconds() 
                        #     if diff < 0.03:  
                        #         time_diff.append(diff)
                        #     action_time = None 
                        #     img_time = None 

                        episode.append(curr_step)
                        curr_step = copy.deepcopy(empty_step) 
            

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        #breakpoint() 

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)
        

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

