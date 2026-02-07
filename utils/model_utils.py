import torch
import json
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode # Explicit import for clarity
import PIL.Image as Image
from PIL import ImageFilter, ImageOps # Added ImageOps for potential future use, PIL.Image is used
import random
import os
# Make sure the 'clip' library is correctly installed and importable
try:
    from clip import clip
except ImportError:
    print("Warning: 'clip' library not found. Please install it (e.g., pip install git+https://github.com/openai/CLIP.git)")
    clip = None # Set to None to avoid errors if not installed, though load_clip_to_cpu will fail

def load_clip_to_cpu(cfg):
    if clip is None:
        raise ImportError("CLIP library is required but not installed.")
    backbone_name = cfg.MODEL.BACKBONE.NAME
    try:
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, root=cfg.MODEL.WEIGHT_ROOT if hasattr(cfg.MODEL, 'WEIGHT_ROOT') else 'all_weights') # Use config path if available
    except KeyError:
        raise ValueError(f"Unsupported CLIP backbone name: {backbone_name}")
    except Exception as e:
        raise RuntimeError(f"Error downloading CLIP model: {e}")

    try:
        # Try loading JIT compiled model
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # Fallback to loading state dict
        state_dict = torch.load(model_path, map_location="cpu")

    # Build model from state dict (either loaded directly or from JIT model)
    model = clip.build_model(state_dict or model.state_dict())
    return model


# --- Define the function to replace the lambda ---
def convert_to_rgb(image):
    """Converts a PIL image to RGB."""
    # Ensure it's a PIL Image first (optional safeguard)
    if not isinstance(image, Image.Image):
         # If input might be a Tensor, convert back to PIL first
         # This depends heavily on the pipeline order - assuming it's PIL here
         pass # Or raise TypeError("Expected PIL Image for RGB conversion")
    return image.convert("RGB")
# -------------------------------------------------


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, random_transform):
        self.base_transform = base_transform  # from clip
        self.random_tranform = random_transform  # random transforms (currently from simsiam)

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.random_tranform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        # Ensure input is a PIL image for GaussianBlur
        if not isinstance(x, Image.Image):
             # Example: Convert tensor back to PIL if needed (adjust range if necessary)
             # x = transforms.ToPILImage()(x)
             pass # Assume input is PIL for now
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# Define interpolation mode using enum for modern torchvision
# This variable is used below to replace the deprecated integer codes
_INTERPOLATION_MODE = InterpolationMode.BICUBIC

def get_random_transform(ndim):
    # Define the normalization transform separately for clarity
    # Using standard CLIP normalization parameters
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))

    # Define Gaussian Blur instance
    blur = GaussianBlur(sigma=[0.1, 2.0]) # Standard SimCLR sigma range

    return transforms.Compose([
        # ***** THIS IS THE MODIFIED LINE *****
        transforms.RandomResizedCrop(ndim, scale=(0.2, 1.), interpolation=_INTERPOLATION_MODE), # Use InterpolationMode enum
        # *************************************
        transforms.RandomHorizontalFlip(p=0.5), # Specify probability explicitly
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) # Explicit parameters
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([blur], p=0.5), # Apply blur randomly like in SimCLR/MoCo v2
        transforms.ToTensor(), # Convert PIL Image to tensor
        normalize # Apply normalization
    ])

# Test Transform (using the defined function instead of lambda and correct interpolation)
te_transform = transforms.Compose([
    transforms.Resize(224, interpolation=_INTERPOLATION_MODE), # Already correct
    transforms.CenterCrop(224),
    convert_to_rgb, # Use the defined function here
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# Default Weakly Augmented Transform (using the defined function instead of lambda and correct interpolation)
transform_default_clip_weakly_aug = transforms.Compose([
    transforms.Resize(224, interpolation=_INTERPOLATION_MODE), # Already correct
    transforms.CenterCrop(224),
    convert_to_rgb, # Use the defined function here
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# Training Transform combining base and random augmentations
tr_transforms = TwoCropsTransform(transform_default_clip_weakly_aug, # Use the weakly augmented one as base for one crop
                                  get_random_transform(224))          # Use strong augmentation for the other (now correctly uses InterpolationMode)

# === Label Generation Functions (Unchanged from your provided code) ===

def gen_labels_with_templates(classes, descriptions):
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        if '_' in classname:
            classname = classname.replace('_', ' ')

        for descp in descriptions:
            descp = descp.format(classname)
            desc_.append(descp)
            labels.append(i)
    return desc_, labels


def gen_labels_with_captions(classes, folder_path, args):
    # Assuming 'args' has a 'dataset' attribute
    dataset_name = getattr(args, 'dataset', 'unknown') # Default if not present

    if dataset_name == 'imagenet':
        desc_ = []
        labels = []
        cls_name_dict = {}
        json_path = os.path.join(folder_path, 'imagenet_class_index.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"ImageNet class index file not found at {json_path}")
        with open(json_path) as f:
            cls_idx = json.load(f)
        # Map index string to class ID (nid) and readable name
        for k, v in cls_idx.items():
            # k is index (0-999), v is [class_id (e.g., n01440764), readable_name (e.g., tench)]
            cls_name_dict[v[0]] = v[1] # Store nid -> readable name mapping (if needed later)

        # Iterate through actual class IDs (nids) required by the 'classes' list
        for i, class_nid in enumerate(classes): # Assuming 'classes' contains nids for ImageNet
             caption_file_path = os.path.join(folder_path, f'{class_nid}.txt')
             if not os.path.exists(caption_file_path):
                 print(f"Warning: Caption file not found for class {class_nid}, skipping.")
                 continue
             with open(caption_file_path, 'r') as f:
                for line in f:
                    try:
                        # Assuming format "image_id caption text..."
                        caption = line.split(" ", 1)[1].replace("\n", "").strip().lower()
                        if caption: # Avoid empty captions
                            desc_.append(caption)
                            labels.append(i) # Use the index corresponding to the input 'classes' list
                    except IndexError:
                        print(f"Warning: Malformed line in {caption_file_path}: {line.strip()}")
        return desc_, labels

    # Fallback for non-ImageNet datasets (using your original logic)
    desc_ = []
    labels = []
    # These seem specific to CIFAR100 or similar?
    classes_to_care = ['aquarium_fish', 'lawn_mower', 'maple_tree', 'oak_tree', 'pickup_truck', 'pine_tree',
                       'sweet_pepper', 'willow_tree'] # Use underscore consistently if file names use it
    for i, classname in enumerate(classes):
        # Determine split based on classname presence in the special list
        split_ = 2 if classname in classes_to_care else 1
        caption_file_path = os.path.join(folder_path, f'{classname}.txt')
        if not os.path.exists(caption_file_path):
            print(f"Warning: Caption file not found for class {classname}, skipping.")
            continue
        try:
            with open(caption_file_path, 'r') as f:
                for line in f:
                    try:
                         # Format: "id part1 part2 ... caption" -> split accordingly
                        parts = line.strip().split(" ", split_)
                        if len(parts) > split_:
                            caption = parts[split_].replace("\n", "").strip().lower()
                            if caption:
                                desc_.append(caption)
                                labels.append(i)
                        else:
                            print(f"Warning: Malformed line (not enough parts) in {caption_file_path}: {line.strip()}")
                    except Exception as e_line: # Catch errors reading/splitting lines
                         print(f"Error processing line in {caption_file_path}: {line.strip()} -> {e_line}")
        except Exception as e_file: # Catch errors opening file
             print(f"Error opening or reading file {caption_file_path}: {e_file}")
    return desc_, labels


def gen_labels_with_captions_blip_2(classes, folder_path, args):
    # Assuming folder_path is the direct path to the single BLIP caption file
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"BLIP caption file not found at: {folder_path}")

    with open(folder_path) as f:
        lines = f.readlines()
    desc_ = []
    labels = []

    # Assuming class names in the file might have spaces or underscores
    # Normalize class names from input list and the file for comparison
    normalized_classes = {c.replace('_', ' '): i for i, c in enumerate(classes)}
    # This list seems dataset specific, adapt if needed
    classes_to_care_normalized = {'aquarium fish', 'lawn mower', 'maple tree', 'oak tree', 'pickup truck', 'pine tree',
                                  'sweet pepper', 'willow tree'}

    for line in lines:
        parts = line.strip().split(' ')
        if not parts: continue

        # Extract the class name from the image path (assuming format like 'path/to/class_name/image.jpg')
        try:
            file_class_name_from_path = parts[0].split('/')[-2].replace('_', ' ') # Normalize from path
        except IndexError:
            print(f"Warning: Could not extract class name from path in line: {line.strip()}")
            continue

        if file_class_name_from_path in normalized_classes:
            class_index = normalized_classes[file_class_name_from_path]
            # Determine split logic based on normalized class name
            split_ = 2 if file_class_name_from_path in classes_to_care_normalized else 1

            if len(parts) > split_:
                 caption = " ".join(parts[split_:]).replace("\n", "").strip().lower() # Join remaining parts for caption
                 if caption:
                    labels.append(class_index)
                    desc_.append(caption)
                    # print(caption) # Keep print if useful for debugging
            else:
                print(f"Warning: Malformed line (not enough parts for caption) in BLIP file: {line.strip()}")

    return desc_, labels


def gen_labels_with_classes(classes, descriptions=None): # descriptions often not used here
    # for direct class name -> class name as label description
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        # Clean up classname if needed (e.g., remove underscores)
        readable_classname = classname.replace('_', ' ')
        desc_.append(readable_classname)
        labels.append(i)
    return desc_, labels


def gen_labels_with_classes_and_simple_template(classes, descriptions=None):
    # for direct class name -> simple template like "a photo of a [class]"
    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        readable_classname = classname.replace('_', ' ')
        descp = f'a photo of a {readable_classname}' # Standard simple template
        desc_.append(descp)
        labels.append(i)
    return desc_, labels


def gen_labels_with_synonyms(classes, folder_path, args):
    # Assuming folder_path points to directory containing the JSON file
    # and args has 'dataset' attribute
    dataset_name = getattr(args, 'dataset', 'unknown')
    json_path = os.path.join(folder_path, f'{dataset_name}_cleaned.json')
    if not os.path.exists(json_path):
         raise FileNotFoundError(f"Synonym file not found: {json_path}")

    with open(json_path) as f:
        cls_synonyms_map = json.load(f) # Map from original class name -> comma-separated synonyms

    desc_ = []
    labels = []
    # Iterate through the original classes provided to maintain order and index
    for i, original_classname in enumerate(classes):
        if original_classname not in cls_synonyms_map:
            print(f"Warning: Class '{original_classname}' not found in synonym map. Using original name.")
            synonyms_str = original_classname # Use original name as fallback
        else:
            synonyms_str = cls_synonyms_map[original_classname]

        # Include the original class name itself along with synonyms
        all_names = [original_classname.replace('_', ' ')] + \
                    [syn.strip() for syn in synonyms_str.split(',') if syn.strip()]

        # Create descriptions using the simple template for each name
        for name in set(all_names): # Use set to avoid duplicate descriptions if name == synonym
            desc_.append(f'a photo of a {name}.') # Add period for consistency
            labels.append(i) # Assign the original class index

    return desc_, labels


def gen_labels_with_descrptions(classes, descriptions):
    # descriptions here is expected to be a dict: {classname: [desc1, desc2, ...]}
    if not isinstance(descriptions, dict):
        raise TypeError("Expected 'descriptions' to be a dictionary mapping class names to lists of descriptions.")

    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        if classname not in descriptions:
            print(f"Warning: Class '{classname}' not found in descriptions dictionary. Skipping.")
            continue
        class_descs = descriptions[classname]
        if not isinstance(class_descs, list):
             print(f"Warning: Descriptions for class '{classname}' is not a list. Skipping.")
             continue

        for desc in class_descs:
            if isinstance(desc, str) and desc.strip(): # Ensure description is a non-empty string
                desc_.append(desc.strip())
                labels.append(i)
            else:
                 print(f"Warning: Invalid description format for class '{classname}': {desc}")

    return desc_, labels


def gen_labels_with_expanded_labels_imagenet(folder, args):
    # folder is the directory containing 'imagenet_expanded_labels.txt'
    dataset_name = getattr(args, 'dataset', 'unknown')
    if dataset_name != 'imagenet':
        raise ValueError('This function is specifically for ImageNet expanded labels.')

    file_path = os.path.join(folder, 'imagenet_expanded_labels.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expanded labels file not found: {file_path}")

    with open(file_path) as f:
        data = f.readlines()

    exp_cls_map = {} # Map index to list of expanded class names
    for line_num, line in enumerate(data):
        try:
            # Assuming format: "index class_id 'comma,separated,labels'"
            parts = line.strip().split(' ', 2) # Split into 3 parts max
            if len(parts) < 3:
                 print(f"Warning: Malformed line {line_num+1} in expanded labels file: {line.strip()}")
                 continue
            # Clean the comma-separated string: remove surrounding quotes, split, strip whitespace
            raw_labels = parts[2].strip('\'"')
            expanded_names = [name.strip() for name in raw_labels.split(',') if name.strip()]
            exp_cls_map[line_num] = expanded_names # Store against line index (0-999)
        except Exception as e:
            print(f"Error processing line {line_num+1} in expanded labels file: {line.strip()} -> {e}")

    desc_ = []
    labels = []
    # Assuming the number of classes corresponds to the lines in the file (1000 for ImageNet)
    num_classes = len(exp_cls_map)
    for i in range(num_classes):
        if i not in exp_cls_map:
            print(f"Warning: Index {i} not found in expanded labels map. Skipping.")
            continue
        cls_names = exp_cls_map[i]
        for name in cls_names:
            desc_.append(f'a photo of a {name}') # Apply simple template
            labels.append(i) # Assign the original class index

    return desc_, labels


def gen_labels_with_descrptions_and_clsname(classes, descriptions):
     # descriptions is expected to be a dict: {classname: [desc1, desc2, ...]}
    if not isinstance(descriptions, dict):
        raise TypeError("Expected 'descriptions' to be a dictionary mapping class names to lists of descriptions.")

    desc_ = []
    labels = []
    for i, classname in enumerate(classes):
        readable_classname = classname.replace('_', ' ') # For display
        if classname not in descriptions:
            print(f"Warning: Class '{classname}' not found in descriptions dictionary. Skipping.")
            continue
        class_descs = descriptions[classname]
        if not isinstance(class_descs, list):
             print(f"Warning: Descriptions for class '{classname}' is not a list. Skipping.")
             continue

        for desc in class_descs:
             if isinstance(desc, str) and desc.strip():
                # Prepend class name to the description
                desc_.append(f'{readable_classname}: {desc.strip()}')
                labels.append(i)
             else:
                  print(f"Warning: Invalid description format for class '{classname}': {desc}")

    return desc_, labels

# --- Optional Main block for testing ---
if __name__ == '__main__':
    print("Testing CLIP model_utils functions...")

    # --- Test Transforms ---
    print("\nTesting Transforms:")
    try:
        dummy_pil_image = Image.new('L', (300, 300)) # Create a dummy grayscale PIL image
        print(f"Input dummy image mode: {dummy_pil_image.mode}, size: {dummy_pil_image.size}")

        # Test te_transform
        output_te = te_transform(dummy_pil_image)
        print(f"te_transform output shape: {output_te.shape}, dtype: {output_te.dtype}") # Should be [3, 224, 224]

         # Test transform_default_clip_weakly_aug
        output_weak = transform_default_clip_weakly_aug(dummy_pil_image)
        print(f"transform_default_clip_weakly_aug output shape: {output_weak.shape}, dtype: {output_weak.dtype}")

        # Test tr_transforms (TwoCropsTransform)
        output_tr = tr_transforms(dummy_pil_image)
        print(f"tr_transforms output type: {type(output_tr)}, length: {len(output_tr)}")
        print(f"  Crop 1 shape: {output_tr[0].shape}") # Should be [3, 224, 224] (weakly aug)
        print(f"  Crop 2 shape: {output_tr[1].shape}") # Should be [3, 224, 224] (strongly aug)

    except Exception as e:
        print(f"Error during transform testing: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Label Generation ---
    print("\nTesting Label Generation:")
    dummy_classes = ['cat', 'dog_animal', 'bird']
    dummy_templates = ["a photo of a {}", "an image depicting a {}"]
    dummy_descriptions_dict = {
        'cat': ['a feline animal', 'sleeps all day'],
        'dog_animal': ['a canine friend', 'barks loudly'],
        'bird': ['flies high', 'sings songs']
    }
    # Mock args for functions that need it
    class MockArgs: dataset = 'dummy_dataset'
    mock_args = MockArgs()

    try:
        # Test gen_labels_with_templates
        desc_t, labels_t = gen_labels_with_templates(dummy_classes, dummy_templates)
        print(f"gen_labels_with_templates: {len(desc_t)} descriptions generated.")
        # print(list(zip(desc_t, labels_t))) # Uncomment to view

        # Test gen_labels_with_classes
        desc_c, labels_c = gen_labels_with_classes(dummy_classes)
        print(f"gen_labels_with_classes: {len(desc_c)} descriptions generated.")
        # print(list(zip(desc_c, labels_c)))

        # Test gen_labels_with_classes_and_simple_template
        desc_st, labels_st = gen_labels_with_classes_and_simple_template(dummy_classes)
        print(f"gen_labels_with_classes_and_simple_template: {len(desc_st)} descriptions generated.")
        # print(list(zip(desc_st, labels_st)))

        # Test gen_labels_with_descrptions
        desc_d, labels_d = gen_labels_with_descrptions(dummy_classes, dummy_descriptions_dict)
        print(f"gen_labels_with_descrptions: {len(desc_d)} descriptions generated.")
        # print(list(zip(desc_d, labels_d)))

        # Test gen_labels_with_descrptions_and_clsname
        desc_dc, labels_dc = gen_labels_with_descrptions_and_clsname(dummy_classes, dummy_descriptions_dict)
        print(f"gen_labels_with_descrptions_and_clsname: {len(desc_dc)} descriptions generated.")
        # print(list(zip(desc_dc, labels_dc)))

        # Add tests for caption/synonym/expanded label functions if you have dummy data files
        # Example (requires setting up dummy files/folders):
        # print("Testing gen_labels_with_captions (requires dummy folder './dummy_captions')")
        # os.makedirs('./dummy_captions', exist_ok=True)
        # with open('./dummy_captions/cat.txt', 'w') as f: f.write("img1 meow\nimg2 purr")
        # ... create other dummy files ...
        # desc_cap, labels_cap = gen_labels_with_captions(dummy_classes, './dummy_captions', mock_args)
        # print(f"gen_labels_with_captions: {len(desc_cap)} descriptions.")

    except Exception as e:
        print(f"Error during label generation testing: {e}")
        import traceback
        traceback.print_exc()

    print("\nmodel_utils.py testing finished.")

