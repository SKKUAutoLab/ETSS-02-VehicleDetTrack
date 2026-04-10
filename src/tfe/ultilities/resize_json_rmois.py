import json
from pathlib import Path

input_path = Path("/media/sugarubuntu/DataSKKU3/3_Workspace/traffic_surveillance_system/ETSS-02-VehicleDetTrack/src/tfe/data/Korea_cctv/rmois/23_SUWON_720x1280.json")
output_path = input_path.with_name('23_SUWON.json')

orig_w, orig_h     = 1280.0, 720.0
target_w, target_h = 960.0, 540.0
scale_x = target_w / orig_w
scale_y = target_h / orig_h

def scale_point(pt):
	x, y = pt
	return [round(x * scale_x, 6), round(y * scale_y, 6)]

def rescale_data(data):
	for key in ('roi', 'moi'):
		if key in data and isinstance(data[key], list):
			for item in data[key]:
				if 'points' in item and isinstance(item['points'], list):
					item['points'] = [scale_point(p) for p in item['points']]
	return data

with input_path.open('r', encoding='utf-8') as f:
	data = json.load(f)

rescaled = rescale_data(data)

with output_path.open('w', encoding='utf-8') as f:
	json.dump(rescaled, f, indent=2, ensure_ascii=False)

print(f"Rescaled saved to {output_path}")
