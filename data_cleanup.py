
import array
import collections

from typing import Dict, List, Optional, Text, Tuple

import numpy as np
import tensorflow as tf


def _create_feature_dict() -> Dict[Text, List[tf.Tensor]]:
  """Helper function for creating an empty feature dict for defaultdict."""
  return {"race_index": [], "place": []}

def _sample_list(
    feature_lists: Dict[Text, List[tf.Tensor]],
    num_examples_per_list: int,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Function for sampling a list example from given feature lists."""
  if random_state is None:
    random_state = np.random.RandomState()

  sampled_indices = random_state.choice(
      range(len(feature_lists["race_index"])),
      size=num_examples_per_list,
      replace=False,
  )
  sampled_race_indexs = [
      feature_lists["race_index"][idx] for idx in sampled_indices
  ]
  sampled_ratings = [
      feature_lists["place"][idx]
      for idx in sampled_indices
  ]

  return (
      tf.stack(sampled_race_indexs, 0),
      tf.stack(sampled_ratings, 0),
  )

def sample_listwise(
    rating_dataset: tf.data.Dataset,
    num_list_per_user: int = 10,
    num_examples_per_list: int = 10,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
  """Function for converting the MovieLens 100K dataset to a listwise dataset.

  Args:
      rating_dataset:
        The MovieLens ratings dataset loaded from TFDS with features
        "race_index", "horse_name", and "place".
      num_list_per_user:
        An integer representing the number of lists that should be sampled for
        each user in the training dataset.
      num_examples_per_list:
        An integer representing the number of movies to be sampled for each list
        from the list of movies rated by the user.
      seed:
        An integer for creating `np.random.RandomState`.

  Returns:
      A tf.data.Dataset containing list examples.

      Each example contains three keys: "horse_name", "race_index", and
      "place". "horse_name" maps to a string tensor that represents the user
      id for the example. "race_index" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.string. It represents the list
      of candidate movie ids. "place" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.float32. It represents the
      rating of each movie in the candidate list.
  """
  random_state = np.random.RandomState(seed)

  example_lists_by_user = collections.defaultdict(_create_feature_dict)

  race_index_vocab = set()
  for example in rating_dataset:
    horse_name = example["horse_name"].numpy()
    example_lists_by_user[horse_name]["race_index"].append(
        example["race_index"])
    example_lists_by_user[horse_name]["place"].append(
        example["place"])
    race_index_vocab.add(example["race_index"].numpy())

  tensor_slices = {"horse_name": [], "race_index": [], "place": []}

  for horse_name, feature_lists in example_lists_by_user.items():
    for _ in range(num_list_per_user):

      # Drop the user if they don't have enough ratings.
      if len(feature_lists["race_index"]) < num_examples_per_list:
        continue

      sampled_race_indexs, sampled_ratings = _sample_list(
          feature_lists,
          num_examples_per_list,
          random_state=random_state,
      )
      tensor_slices["horse_name"].append(horse_name)
      tensor_slices["race_index"].append(sampled_race_indexs)
      tensor_slices["place"].append(sampled_ratings)

  return tf.data.Dataset.from_tensor_slices(tensor_slices)