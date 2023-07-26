# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #
from typing import *

from munch import Munch


# MARK: - Get label

def get_label(
	labels   : List[Dict],
	name     : Optional[str] = None,
	id       : Optional[int] = None,
	train_id : Optional[int] = None,
) -> Optional[Dict]:
	"""
	"""
	for label in labels:
		if (name is not None) and hasattr(label, "name") and (name == label.name):
			return label
		
		if (id is not None) and hasattr(label, "id") and (id == label.id):
			return label
		
		if (train_id is not None) and hasattr(label, "train_id") and (train_id == label.train_id):
			return label
			
	return None


def get_majority_label(object_labels: List[Dict]) -> Dict:
	"""Get the most popular label of the road_objects.
	"""
	# NOTE: Count number of appearance of each label.
	unique_labels = Munch()
	label_voting  = Munch()
	for label in object_labels:
		key   = label.id
		value = label_voting.get(key)
		if value:
			label_voting[key]  = value + 1
		else:
			unique_labels[key] = label
			label_voting[key]  = 1
			
	# NOTE: get key (label's id) with max value
	max_id = max(label_voting, key=label_voting.get)
	return unique_labels[max_id]
