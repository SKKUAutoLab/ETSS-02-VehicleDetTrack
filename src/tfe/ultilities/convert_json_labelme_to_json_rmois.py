# python
import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse



def load_and_group(path: Path):
	with path.open('r', encoding='utf-8') as f:
		data = json.load(f)

	rois   = []
	mois   = []
	shapes = []
	for s in data.get('shapes', []):

		# Get from original format
		label = s.get('label') or None
		entry = {
			'label' : label,
			'shape_type': s.get('shape_type'),
			'points': [list(pt) for pt in s.get('points', [])],
			'group_id': s.get('group_id'),
			'description': s.get('description'),
			'flags': s.get('flags'),
		}
		shapes.append(entry)

		# Convert to new format
		if label in "roi":
			rois.append({
				"uuid"      : 1,
				"points"    : entry["points"],
				"shape_type": entry["shape_type"]
			})
		elif label is not None:
			mois.append({
				"uuid"      : int(label),
				"points"    : entry["points"],
				"shape_type": entry["shape_type"],
				"offset"    : 0
			})

	original = {
		'imagePath': data.get('imagePath'),
		'imageWidth': data.get('imageWidth'),
		'imageHeight': data.get('imageHeight'),
		'shapes': shapes,
	}

	result = {
		'roi' : rois,
		'moi': mois
	}

	return original, result

def main(argv=None):
	INPUT_PATH  = Path('/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/Korea_cctv/rmois_description/30_SEOUL.json')
	OUTPUT_PATH = Path('/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/Korea_cctv/rmois/30_SEOUL.json')

	parser = argparse.ArgumentParser(description='Convert LabelMe JSON into grouped dict by label.')
	parser.add_argument('-i', '--input', nargs='?', type=Path, default=INPUT_PATH, help='input JSON path')
	parser.add_argument('-o', '--output', type=Path, default=OUTPUT_PATH, help='output JSON path (if omitted prints to stdout)')
	args = parser.parse_args(argv)

	try:
		original, result = load_and_group(args.input)
	except Exception as e:
		print(f"Failed to read {args.input}: {e}", file=sys.stderr)
		return 1

	if args.output:
		try:
			args.output.parent.mkdir(parents=True, exist_ok=True)
			with args.output.open('w', encoding='utf-8') as f:
				json.dump(result, f, indent=2, ensure_ascii=False)
		except Exception as e:
			print(f"Failed to write {args.output}: {e}", file=sys.stderr)
			return 1
	else:
		json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
		print()  # newline

	return 0

if __name__ == '__main__':
	raise SystemExit(main())
