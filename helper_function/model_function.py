import sys
import math

NMS_UNSPECIFIED_OVERLAP_TYPE = 0
NMS_JACQUARD = 1
NMS_MODIFIED_JACCARD = 2
NMS_INTERSECTION_OVER_UNION = 3

NMS_DEFAULT = 0
NMS_WEIGHTED = 1

NUM_KEYPOINTS_PER_BOX = 6
NUM_COORDS_PER_KEYPOINT = 2


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
  if num_strides == 1:
    return (min_scale + max_scale) * 0.5
  else:
    return (min_scale +
      (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0))

def generate_anchors(options):
  anchors = []
  if len(options["strides"]) != options["num_layers"]:
    raise Exception("Strides count (%d) doesn't equal num_layers (%d)." %
                    (len(options["strides"]), options["num_layers"]))
  layer_id = 0
  while layer_id < options["num_layers"]:
    anchor_height = []
    anchor_width = []
    aspect_ratios = []
    scales = []
    last_same_stride_layer = layer_id

    while (last_same_stride_layer < len(options["strides"]) and
          options["strides"][last_same_stride_layer] == options["strides"][layer_id]):
      scale = calculate_scale(options["min_scale"], options["max_scale"],
        last_same_stride_layer, len(options["strides"]))
      if (last_same_stride_layer == 0) and (options["reduce_boxes_in_lowest_layer"]):
        aspect_ratios.append(1.0)
        aspect_ratios.append(2.0)
        aspect_ratios.append(0.5)
        scales.append(0.1)
        scales.append(scale)
        scales.append(scale)
      else:
        for aspect_ratio in options["aspect_ratios"]:
          aspect_ratios.append(aspect_ratio)
          scales.append(scale)
        if options["interpolated_scale_aspect_ratio"] > 0.0:
          if last_same_stride_layer == len(options["strides"]):
            scale_next = 1.0
          else:
            scale_next = calculate_scale(options["min_scale"],
              options["max_scale"], last_same_stride_layer + 1,
              len(options["strides"]))
            aspect_ratios.append(options["interpolated_scale_aspect_ratio"])
            scales.append(math.sqrt(scale * scale_next))
      last_same_stride_layer += 1

    for i, aspect_ratio in enumerate(aspect_ratios):
      ratio_sqrts = math.sqrt(aspect_ratio)
      anchor_height.append(scales[i] / ratio_sqrts)
      anchor_width.append(scales[i] * ratio_sqrts)
    
    stride = options["strides"][layer_id]
    feature_map_height = math.ceil(1.0 * options["input_size_height"] / stride)
    feature_map_width = math.ceil(1.0 * options["input_size_width"] / stride)

    for y in range(feature_map_height):
      for x in range(feature_map_width):
        for anchor_id in range(len(anchor_height)):
          x_center = (x + options["anchor_offset_x"]) * 1.0 / feature_map_width
          y_center = (y + options["anchor_offset_y"]) * 1.0 / feature_map_height
          new_anchor = {}
          new_anchor["x"] = x_center
          new_anchor["y"] = y_center
          if options["fixed_anchor_size"]:
            new_anchor["w"] = 1.0
            new_anchor["h"] = 1.0
          else:
            new_anchor["w"] = anchor_width[anchor_id]
            new_anchor["h"] = anchor_height[anchor_id]
          anchors.append(new_anchor)
    layer_id = last_same_stride_layer

  return anchors

def decode_boxes(options, coords, scores, anchors):
  boxes = [None] * options["num_boxes"] * options["num_coords"]
  for i in range(options["num_boxes"]):
    box_offset = (i * options["num_coords"]) + options["box_coord_offset"]
    if options["reverse_output_order"]:
      x_center = coords[box_offset]
      y_center = coords[box_offset + 1]
      w = coords[box_offset + 2]
      h = coords[box_offset + 3]
    else:
      y_center = coords[box_offset]
      x_center = coords[box_offset + 1]
      h = coords[box_offset + 2]
      w = coords[box_offset + 3]
    
    x_center = x_center / options["x_scale"] * anchors[i]["w"] + anchors[i]["x"]
    y_center = y_center / options["y_scale"] * anchors[i]["h"] + anchors[i]["y"]

    if options["apply_exponential_on_box_size"]:
      h = math.exp(h / options["h_scale"]) * anchors[i]["h"]
      w = math.exp(w / options["w_scale"]) * anchors[i]["w"]
    else:
      h = h / options["h_scale"] * anchors[i]["h"]
      w = w / options["w_scale"] * anchors[i]["w"]
        
    ymin = y_center - h / 2.0
    xmin = x_center - w / 2.0
    ymax = y_center + h / 2.0
    xmax = x_center + w / 2.0

    boxes[i * options["num_coords"] + 0] = xmin
    boxes[i * options["num_coords"] + 1] = ymin
    boxes[i * options["num_coords"] + 2] = xmax
    boxes[i * options["num_coords"] + 3] = ymax

    for k in range(options["num_keypoints"]):
      offset = (i * options["num_coords"] + options["keypoint_coord_offset"] +
        k * options["num_values_per_keypoint"])
      
      if options["reverse_output_order"]:
        keypoint_x = coords[offset]
        keypoint_y = coords[offset + 1]
      else:
        keypoint_y = coords[offset]
        keypoint_x = coords[offset + 1]

      boxes[offset] = keypoint_x / options["x_scale"] * anchors[i]["w"] + anchors[i]["x"]
      boxes[offset + 1] = keypoint_y / options["y_scale"] * anchors[i]["h"] + anchors[i]["y"]

  return boxes

def rect_is_empty(rect):
  return rect["min_x"] > rect["max_x"] or rect["min_y"] > rect["max_y"]

def rect_intersects(a, b):
    return not (rect_is_empty(a) or 
                rect_is_empty(b) or 
                b["max_x"] < a["min_x"] or 
                a["max_x"] < b["min_x"] or
                b["max_y"] < a["min_y"] or
                a["max_y"] < b["min_y"])

def rect_empty():
  return {
    "min_x": sys.float_info.max,
    "min_y": sys.float_info.max,
    "max_x": sys.float_info.min,
    "max_y": sys.float_info.min,
  }

def rect_intersect(a, b):
  result = {
    "min_x": max(a["min_x"], b["min_x"]),
    "min_y": max(a["min_y"], b["min_y"]),
    "max_x": min(a["max_x"], b["max_x"]),
    "max_y": min(a["max_y"], b["max_y"]),
  }
  if result["min_x"] > result["max_x"] or result["min_y"] > result["max_y"]:
    return rect_empty()
  else:
    return result

def rect_union(a, b):
  return {
    "min_x": min(a["min_x"], b["min_x"]),
    "min_y": min(a["min_y"], b["min_y"]),
    "max_x": max(a["max_x"], b["max_x"]),
    "max_y": max(a["max_y"], b["max_y"]),
  }

def rect_width(rect):
  return rect["max_x"] - rect["min_x"]

def rect_height(rect):
  return rect["max_y"] - rect["min_y"]

def rect_area(rect):
  return rect_width(rect) * rect_height(rect)

def rect_from_coords(coords):
  return {
    "min_x": coords[0],
    "min_y": coords[1],
    "max_x": coords[2],
    "max_y": coords[3],
  }

def overlap_similarity(overlap_type, rect1, rect2):
  if not rect_intersects(rect1, rect2):
    return 0.0
  
  intersection = rect_intersect(rect1, rect2)
  intersection_area = rect_area(intersection)
  if overlap_type == NMS_JACQUARD:
    normalization = rect_area(rect_union(rect1, rect2))
  elif overlap_type == NMS_MODIFIED_JACCARD:
    normalization = rect_area(rect2)
  elif overlap_type == NMS_INTERSECTION_OVER_UNION:
    normalization = rect_area(rect1) + rect_area(rect2) - intersection_area

  if normalization < 0.0:
    return 0.0
  else:
    return intersection_area / normalization

def unweighted_non_max_suppression(options, indexed_scores, coords, max_num_detections):
  detections = []
  for indexed_score in indexed_scores:
    candidate_coords_offset = indexed_score[0] * options["num_coords"]
    offset = candidate_coords_offset + options["box_coord_offset"]
    candidate_box_coords = coords[offset:offset+4]
    candidate_rect = rect_from_coords(candidate_box_coords)
    candidate_detection = {
      "rect": candidate_rect,
      "score": indexed_score[1],
    }
    if options["min_score_threshold"] > 0.0 and candidate_detection["score"] < options["min_score_threshold"]:
      break
    suppressed = False
    for existing_detection in detections:
      similarity = overlap_similarity(
                options["overlap_type"],
                existing_detection["rect"],
                candidate_detection["rect"])
      if similarity > options["min_suppression_threshold"]:
        suppressed = True
        break
  
    if not suppressed:
      detections.append(candidate_detection)
    if len(detections) >= max_num_detections:
      break
  return detections

def weighted_non_max_suppression(options, indexed_scores, coords, max_num_detections):
  num_keypoints = options["num_keypoints"]
  remained_indexed_scores = indexed_scores
  detections = []
  while len(remained_indexed_scores) > 0:
    indexed_score = remained_indexed_scores[0]
    original_indexed_scores_size = len(remained_indexed_scores)
    candidate_coords_offset = indexed_score[0] * options["num_coords"]
    offset = candidate_coords_offset + options["box_coord_offset"]
    candidate_box_coords = coords[offset:offset+4]
    candidate_rect = rect_from_coords(candidate_box_coords)
    candidate_keypoints_offset = candidate_coords_offset + options["keypoint_coord_offset"]
    candidate_keypoints_offset_end = candidate_keypoints_offset + num_keypoints * NUM_COORDS_PER_KEYPOINT
    candidate_keypoints_coords = coords[candidate_keypoints_offset:candidate_keypoints_offset_end]
    candidate_detection = {
      "rect": candidate_rect,
      "score": indexed_score[1],
      "keypoints": candidate_keypoints_coords,
    }
    if options["min_score_threshold"] > 0.0 and candidate_detection["score"] < options["min_score_threshold"]:
      break
    remained = []
    candidates = []
    candidate_location = candidate_detection["rect"]
    for remained_indexed_score in remained_indexed_scores:
      remained_coords_offset = remained_indexed_score[0] * options["num_coords"]
      remained_offset = remained_coords_offset + options["box_coord_offset"]
      remained_box_coords = coords[remained_offset:remained_offset+4]
      remained_rect = rect_from_coords(remained_box_coords)
      similarity = overlap_similarity(options["overlap_type"], remained_rect, candidate_rect)
      if similarity > options["min_suppression_threshold"]:
        candidates.append(remained_indexed_score)
      else:
        remained.append(remained_indexed_score)
    if len(candidates) == 1:
      weighted_detection = candidate_detection
    else:
      keypoints = [0.0] * NUM_KEYPOINTS_PER_BOX * NUM_COORDS_PER_KEYPOINT
      w_xmin = 0.0
      w_ymin = 0.0
      w_xmax = 0.0
      w_ymax = 0.0
      total_score = 0.0
      for sub_indexed_score in candidates:
        sub_score = sub_indexed_score[1]
        total_score += sub_score
        sub_coords_offset = sub_indexed_score[0] * options["num_coords"]
        sub_offset = sub_coords_offset + options["box_coord_offset"]
        sub_box_coords = coords[sub_offset:sub_offset+4]
        sub_rect = rect_from_coords(sub_box_coords)
        w_xmin += sub_rect["min_x"] * sub_score
        w_ymin += sub_rect["min_y"] * sub_score
        w_xmax += sub_rect["max_x"] * sub_score
        w_ymax += sub_rect["max_y"] * sub_score

        sub_keypoints_offset = sub_coords_offset + options["keypoint_coord_offset"]
        sub_keypoints_offset_end = sub_keypoints_offset + num_keypoints * NUM_COORDS_PER_KEYPOINT
        sub_keypoints_coords = coords[sub_keypoints_offset:sub_keypoints_offset_end]
        for k in range(num_keypoints):
          keypoints[k * 2] += sub_keypoints_coords[k * 2] * sub_score
          keypoints[(k * 2) + 1] += sub_keypoints_coords[(k * 2) + 1] * sub_score
      
      weighted_detection = {
        "rect": {
          "min_x": w_xmin / total_score,
          "min_y": w_ymin / total_score,
          "max_x": w_xmax / total_score,
          "max_y": w_ymax / total_score,
          },
        "score": indexed_score[1],
      }
      weighted_detection["keypoints"] = [None] * num_keypoints * NUM_COORDS_PER_KEYPOINT
      for k in range(num_keypoints):
        weighted_detection["keypoints"][k * 2] = keypoints[k * 2] / total_score
        weighted_detection["keypoints"][(k * 2) + 1] = keypoints[(k * 2) + 1] / total_score

    detections.append(weighted_detection)
    if original_indexed_scores_size == len(remained):
      break
    else:
      remained_indexed_scores = remained

  return detections

def non_max_suppression(options, scores, coords, NMS_WEIGHTED):
  indexed_scores = []
  for i in range(options["num_boxes"]):
    indexed_scores.append((i, scores.flatten()[i]))
  indexed_scores.sort(key = lambda x: x[1], reverse=True)
  if options["max_num_detections"] < 0:
    max_num_detections = options["num_boxes"]
  else:
    max_num_detections = options["max_num_detections"]
  if options["algorithm"] == NMS_WEIGHTED:
    return weighted_non_max_suppression(options, indexed_scores, coords, max_num_detections)
  else:
    return unweighted_non_max_suppression(options, indexed_scores, coords, max_num_detections)