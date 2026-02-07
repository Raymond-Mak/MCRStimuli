from torch.nn import Conv2d, Dropout
import math
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
_tokenizer = _Tokenizer()
from utils.model_utils import *
from utils.utils import *
_tokenizer = _Tokenizer()
from functools import reduce
from operator import mul
from utils.data_utils import ds_specific_templates

# === NOTE ===
# The legacy LaFTer trainer in this file historically required templates/descriptions
# for text prompt engineering. For the new multimodal training stage based on
# pre-built image-text pairs, those requirements are removed. To make migration
# minimally invasive, we expose an MMTrainerDassl that reuses the modern
# `trainers/mm_trainer.py` loop (image+text pairs in, logits out) while keeping
# this module import path stable.
#
# Existing classes/functions remain for backward compatibility, but are no longer
# required by the new multimodal pipeline.

# Lightweight alias to the new trainer (keeps this module as a single import point)
try:
    from trainers.mm_trainer import MMTrainer as MMTrainerDassl  # noqa: F401
except Exception as _e:
    MMTrainerDassl = None
    print(f"Warning: MMTrainer import failed in LaFTer_trainers.py: {_e}")

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class LaFTerUFT(nn.Module):

    def __init__(self, model, classes, templates, ds_templates=None, device='cuda', log=None, dataset_name=None, txt_cls=None, cfg=None):
        super(LaFTerUFT, self).__init__()
        self.adapter_pl = None
        self.device = device
        self.cfg = cfg
        self.dataset_templates = ds_templates
        self.classes = classes
        self.dataset_name = dataset_name
        self.txt_cls = txt_cls
        self.log = log
        patch_size = (16, 16)
        self.model = model.to(device)
        self.templates = templates
        self.backbone_out_size = 512
        self.hidden_size = 768 # todo for ViT-L use 1024 - for ViT-B use 768
        self.num_tokens = 50 # todo all experiments are run with num_tokens = 50
        self.prompt_proj = nn.Identity()
        self.hidden_size_text = 512
        self.num_tokens_text = 77
        prompt_dim = self.hidden_size
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        self.prompt_dropout = Dropout(0.0)
        self.adapter = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(device)
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_tokens, self.hidden_size), requires_grad=True)
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        self.text_features = self.txt_features()

    def train_txt_clas(self, criteria):
        noise_std = 0.1
        noise = torch.randn(self.txt_features_for_text_cls.shape) * noise_std
        txt_feas = self.txt_features_for_text_cls
        txt_label = self.labels_for_text_cls
        feas = (self.adapter(txt_feas.to(torch.float32) + noise.cuda()))
        loss = criteria(feas, txt_label)
        return loss

    def txt_features_for_text_cls(self):
        """
        生成用于文本分类的特征和标签，确保数量完全匹配
        修复版本：解决模板数量增加后特征和标签不匹配的问题
        """
        
        if self.txt_cls == 'cls_only':
            gpt3_prompts = None
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'templates_only':
            gpt3_prompts = self.templates
            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'lafter':
            if self.dataset_name not in lafter_datasets:
                raise ValueError('Invalid dataset name for LaFTer')

            # 加载GPT生成的描述
            path_to_file = f'./descriptions/generic/{self.dataset_name}.json'
            with open(path_to_file) as f:
                gpt3_prompts = json.load(f)

            # # 生成基于描述的文本和标签
            desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)
            
            # 生成基于模板的文本和标签
            #templates, labels_for_templates = gen_labels_with_templates(self.classes, descriptions=self.dataset_templates)

            # 合并所有文本和标签
            #desc += templates
            #labels_for_descriptions += labels_for_templates
            
            # 关键调试信息：打印合并后的情况
            print(f"=== LaFTer模式数据统计 ===")
            #print(f"GPT描述文本数量: {len(desc) - len(templates)}")
            #print(f"模板文本数量: {len(templates)}")
            print(f"总描述文本数量: {len(desc)}")
            print(f"总标签数量: {len(labels_for_descriptions)}")
            print(f"类别数量: {len(self.classes)}")
            print(f"平均每类文本数量: {len(desc) / len(self.classes):.1f}")

        elif self.txt_cls == 'zero_shot':
            return None, None
        else:
            raise ValueError('Invalid txt_cls argument')

        if self.txt_cls in ['cls_only', 'templates_only', 'lafter']:
            Path(f'embeddings').mkdir(parents=True, exist_ok=True)
            
            # 构建缓存文件名，包含模板数量信息以避免缓存冲突
            # 关键修复：为了确保缓存的准确性，我们需要包含更多标识信息
            template_count = len(self.dataset_templates) if self.dataset_templates else 0
            total_text_count = len(desc)
            cache_filename = f'embeddings/{self.txt_cls}_{self.dataset_name}_templates{template_count}_total{total_text_count}_embeddings.pt'
            
            if os.path.isfile(cache_filename):
                print(f'******** 加载已保存的嵌入: {cache_filename} *********')
                try:
                    cached_data = torch.load(cache_filename)
                    # 检查缓存数据的完整性
                    if isinstance(cached_data, dict) and 'features' in cached_data and 'labels' in cached_data:
                        zeroshot_weights = cached_data['features']
                        labels_for_descriptions = cached_data['labels']
                        print(f"从缓存加载 - 特征形状: {zeroshot_weights.shape}, 标签形状: {labels_for_descriptions.shape}")
                    else:
                        # 旧格式缓存，只包含特征
                        zeroshot_weights = cached_data
                        labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()
                        print("检测到旧格式缓存，将重新生成以确保一致性...")
                        raise FileNotFoundError("需要重新生成缓存")
                        
                    # 验证缓存数据的一致性
                    if zeroshot_weights.shape[0] != len(labels_for_descriptions):
                        print(f"缓存数据不一致：特征数量 {zeroshot_weights.shape[0]} != 标签数量 {len(labels_for_descriptions)}")
                        raise FileNotFoundError("缓存数据不一致，需要重新生成")
                        
                except Exception as e:
                    print(f"加载缓存失败: {e}")
                    print("将重新生成嵌入...")
                    # 删除损坏的缓存文件
                    if os.path.exists(cache_filename):
                        os.remove(cache_filename)
                    # 继续到重新生成的逻辑
                    zeroshot_weights, labels_for_descriptions = self._generate_new_embeddings(desc, labels_for_descriptions, cache_filename)
            else:
                print('******** 未找到嵌入文件 --- 生成新嵌入 *********')
                zeroshot_weights, labels_for_descriptions = self._generate_new_embeddings(desc, labels_for_descriptions, cache_filename)

            # 最终的一致性检查和修复
            print(f"=== 最终数据一致性检查 ===")
            print(f"嵌入特征形状: {zeroshot_weights.shape}")
            print(f"标签形状: {labels_for_descriptions.shape}")
            
            # 确保特征和标签数量完全匹配
            if zeroshot_weights.shape[0] != labels_for_descriptions.shape[0]:
                print(f"警告：检测到数量不匹配！")
                print(f"特征数量: {zeroshot_weights.shape[0]}")
                print(f"标签数量: {labels_for_descriptions.shape[0]}")
                
                # 取较小的数量作为最终大小
                min_size = min(zeroshot_weights.shape[0], labels_for_descriptions.shape[0])
                zeroshot_weights = zeroshot_weights[:min_size]
                labels_for_descriptions = labels_for_descriptions[:min_size]
                
                print(f"已调整为一致大小: {min_size}")
                
                # 重新保存修正后的数据
                corrected_data = {
                    'features': zeroshot_weights,
                    'labels': labels_for_descriptions
                }
                torch.save(corrected_data, cache_filename)
                print(f"已保存修正后的数据到: {cache_filename}")
            
            print(f"=== 返回数据统计 ===")
            print(f"最终特征形状: {zeroshot_weights.shape}")
            print(f"最终标签形状: {labels_for_descriptions.shape}")
            
            return zeroshot_weights.squeeze(), labels_for_descriptions
        else:
            return None, None

    def _generate_new_embeddings(self, desc, labels_for_descriptions, cache_filename):
        """
        生成新的文本嵌入，确保特征和标签数量一致
        """
        print(f"正在处理 {len(desc)} 个文本描述...")
        
        # 确保标签是张量格式
        labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()
        print(f"初始标签数量: {len(labels_for_descriptions)}")
        
        zeroshot_weights = []
        processed_labels = []
        
        with torch.no_grad():
            for i, (text_desc, label) in enumerate(zip(tqdm(desc, desc="处理文本"), labels_for_descriptions)):
                try:
                    # 确保每个文本都被正确处理
                    text_tokens = tokenize([text_desc]).cuda()  # 注意：这里用列表包装确保批次维度
                    
                    # 获取文本嵌入
                    class_embeddings = self.model.encode_text(text_tokens)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    
                    # 由于我们只处理一个文本，所以嵌入应该是 [1, embedding_dim]
                    # 我们需要去掉批次维度
                    single_embedding = class_embeddings.squeeze(0)
                    
                    zeroshot_weights.append(single_embedding)
                    processed_labels.append(label)
                    
                except Exception as e:
                    print(f"处理第 {i} 个文本时出错: {text_desc[:50]}... - 错误: {e}")
                    continue
            
            # 将所有嵌入堆叠
            if len(zeroshot_weights) > 0:
                zeroshot_weights = torch.stack(zeroshot_weights).cuda()
                processed_labels = torch.stack(processed_labels).cuda()
            else:
                raise RuntimeError("没有成功处理任何文本嵌入")
            
            # 验证数量一致性
            print(f"生成的特征数量: {zeroshot_weights.shape[0]}")
            print(f"对应的标签数量: {processed_labels.shape[0]}")
            
            if zeroshot_weights.shape[0] != processed_labels.shape[0]:
                raise RuntimeError(f"特征和标签数量不匹配: {zeroshot_weights.shape[0]} vs {processed_labels.shape[0]}")
            
            # 保存嵌入到缓存文件 - 使用新的字典格式
            cache_data = {
                'features': zeroshot_weights,
                'labels': processed_labels,
                'metadata': {
                    'dataset_name': self.dataset_name,
                    'txt_cls_mode': self.txt_cls,
                    'total_texts': len(desc),
                    'num_classes': len(self.classes),
                    'template_count': len(self.dataset_templates) if self.dataset_templates else 0
                }
            }
            torch.save(cache_data, cache_filename)
            print(f"嵌入已保存到: {cache_filename}")
            
            return zeroshot_weights, processed_labels

    def txt_features(self):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classes):
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenize(texts).cuda()  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def image_features(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def eval_text_clas(self, batch):
        """
        Evaluate text classification performance
        """
        with torch.no_grad():
            txt_feas = self.txt_features_for_text_cls
            txt_label = self.labels_for_text_cls
            logits = self.adapter(txt_feas.to(torch.float32))
        return logits, txt_label

    def eval_clip(self, x):
        with torch.no_grad():
            img_features_2 = self.incorporate_prompt(x)
            img_features_2 = self.embeddings_after_prompts(img_features_2)
            img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def forward(self, x):
        # only used for 0-shot-eval
        with torch.no_grad():
            img_features = self.image_features(x)
            pseudo_label = img_features @ self.text_features
        return pseudo_label

    def forward_aug_with_prompts(self, x2):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        img_features_2 = self.incorporate_prompt(x2)
        img_features_2 = self.embeddings_after_prompts(img_features_2)
        img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def txt_cls_init(self):
        import copy
        self.adapter_pl = copy.deepcopy(self.adapter)

    def forward_normal_for_pl(self, x1):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        with torch.no_grad():
            img_features_1 = self.image_features(x1)
            pseudo_label = self.adapter_pl(img_features_1.float()).detach()
        return pseudo_label

    def incorporate_prompt(self, x, teacher=False):
        B = x.shape[0]
        x = self.patch_embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x

    def patch_embeddings(self, x: torch.tensor):
        return self.model.visual.embeddings_patch(x)

    def embeddings_after_prompts(self, x: torch.tensor):
        return self.model.visual.forward_after_patch_embeddings(x)

    def positional_embeddings_for_text(self, x: torch.tensor):
        return self.model.positional_embeddings(x)

    def embeddings_after_prompts_for_text(self, x: torch.tensor):
        return self.model.embeddings_after_prompting(x)


@TRAINER_REGISTRY.register()
class LaFTer(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")
        self.model = LaFTerUFT(model=clip_model, classes=classnames,
                                          templates=['a photo of a {}'], ds_templates = ds_specific_templates[cfg.DATASET.NAME], dataset_name= cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg)
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=tr_transforms)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def parse_batch_train(self, batch):

        if isinstance(batch, list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
            input = input.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
