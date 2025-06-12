# KAIST Multispectral Pedestrian Detection Dataset 특화 증강
"""
KAIST RGB-Thermal 데이터셋에 최적화된 데이터 증강 전략
- RGB/Thermal 독립적 증강 + 결합 증강
- 보행자 검출에 특화된 작은 객체 보존
- 다양한 조명/날씨 조건 시뮬레이션
"""

import random
import numpy as np
import cv2
from utils.general import LOGGER, colorstr


class KAISTAugmentations:
    """KAIST 멀티스펙트럴 보행자 검출에 최적화된 증강 클래스"""
    
    def __init__(self, size=640, prob=0.5):
        """
        Args:
            size: 입력 이미지 크기
            prob: 기본 증강 확률
        """
        self.size = size
        self.prob = prob
        self.transform = None
        
        prefix = colorstr("KAIST augmentations: ")
        try:
            import albumentations as A
            from albumentations import BboxParams
            
            # KAIST 데이터셋 특화 증강 설정
            self.rgb_transforms = A.Compose([
                # RGB 채널 특화 증강 (조명 조건 시뮬레이션)
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, 
                    contrast_limit=0.3, 
                    p=0.6
                ),
                A.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.15, 
                    hue=0.1, 
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.ToGray(p=0.1),  # 흑백 변환으로 조명 변화 시뮬레이션
            ])
            
            self.thermal_transforms = A.Compose([
                # Thermal 채널 특화 증강
                A.GaussNoise(var_limit=(10, 50), p=0.4),  # 열화상 노이즈
                A.RandomGamma(gamma_limit=(85, 115), p=0.5),  # 온도 변화
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.4, 
                    p=0.5
                ),
                A.Blur(blur_limit=3, p=0.2),  # 열화상 블러
            ])
            
            # 공통 기하학적 증강 (보행자 특화)
            self.geometric_transforms = A.Compose([
                # 보행자 검출에 중요한 기하학적 증강
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.2, 
                    rotate_limit=5,  # 작은 회전 (보행자 포즈 보존)
                    p=0.6
                ),
                A.RandomResizedCrop(
                    height=size, 
                    width=size, 
                    scale=(0.8, 1.0),  # 보행자가 잘리지 않도록 보수적 크롭
                    ratio=(0.9, 1.1), 
                    p=0.4
                ),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                
                # 날씨/환경 조건 시뮬레이션
                A.RandomRain(
                    slant_lower=-10, 
                    slant_upper=10, 
                    drop_length=20, 
                    drop_width=1, 
                    drop_color=(200, 200, 200), 
                    blur_value=1, 
                    brightness_coefficient=0.7, 
                    rain_type=None, 
                    p=0.1
                ),
                A.RandomFog(
                    fog_coef_lower=0.1, 
                    fog_coef_upper=0.3, 
                    alpha_coef=0.08, 
                    p=0.1
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1), 
                    num_shadows_lower=1, 
                    num_shadows_upper=2, 
                    shadow_dimension=5, 
                    p=0.2
                ),
            ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))
            
            # 고급 증강 (작은 객체 보존)
            self.advanced_transforms = A.Compose([
                # 작은 객체(보행자) 보존을 위한 증강
                A.SmallestMaxSize(max_size=int(size * 1.2), p=0.3),
                A.LongestMaxSize(max_size=size, p=0.3),
                
                # 이미지 품질 관련
                A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.MotionBlur(blur_limit=7, p=0.1),
            ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))
            
            LOGGER.info(f"{prefix}KAIST-specific augmentations initialized")
            
        except ImportError:
            LOGGER.warning(f"{prefix}⚠️ albumentations not found, install with `pip install albumentations`")
            self.rgb_transforms = None
            self.thermal_transforms = None
            self.geometric_transforms = None
            self.advanced_transforms = None
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")
    
    def __call__(self, rgb_img, thermal_img, labels, p=1.0):
        """
        KAIST RGB-Thermal 이미지에 증강 적용
        
        Args:
            rgb_img: RGB 이미지 (H, W, 3)
            thermal_img: Thermal 이미지 (H, W, 3) 또는 (H, W, 1)
            labels: YOLO 형식 라벨 (N, 5) [class, x_center, y_center, width, height]
            p: 증강 적용 확률
            
        Returns:
            augmented_rgb: 증강된 RGB 이미지
            augmented_thermal: 증강된 Thermal 이미지  
            augmented_labels: 증강된 라벨
        """
        if not self.geometric_transforms or random.random() > p:
            return rgb_img, thermal_img, labels
        
        try:
            # 라벨을 적절한 형식으로 변환
            if len(labels) > 0:
                bboxes = labels[:, 1:].tolist()  # [x_center, y_center, width, height]
                class_labels = labels[:, 0].astype(int).tolist()
            else:
                bboxes = []
                class_labels = []
            
            # 1. 공통 기하학적 증강 (RGB와 Thermal 동일하게 적용)
            if self.geometric_transforms and random.random() < 0.7:
                # RGB에 기하학적 증강 적용
                geo_result = self.geometric_transforms(
                    image=rgb_img, 
                    bboxes=bboxes, 
                    class_labels=class_labels
                )
                rgb_augmented = geo_result['image']
                updated_bboxes = geo_result['bboxes']
                updated_classes = geo_result['class_labels']
                
                # Thermal에 동일한 변환 적용 (bboxes는 이미 업데이트됨)
                thermal_augmented = self.geometric_transforms.replay(
                    saved_augmentations=geo_result['replay'],
                    image=thermal_img
                )['image']
            else:
                rgb_augmented = rgb_img.copy()
                thermal_augmented = thermal_img.copy()
                updated_bboxes = bboxes
                updated_classes = class_labels
            
            # 2. RGB 채널 특화 증강
            if self.rgb_transforms and random.random() < 0.6:
                rgb_augmented = self.rgb_transforms(image=rgb_augmented)['image']
            
            # 3. Thermal 채널 특화 증강
            if self.thermal_transforms and random.random() < 0.6:
                thermal_augmented = self.thermal_transforms(image=thermal_augmented)['image']
            
            # 4. 고급 증강 (확률적 적용)
            if self.advanced_transforms and random.random() < 0.3:
                adv_result = self.advanced_transforms(
                    image=rgb_augmented,
                    bboxes=updated_bboxes,
                    class_labels=updated_classes
                )
                rgb_augmented = adv_result['image']
                updated_bboxes = adv_result['bboxes']
                updated_classes = adv_result['class_labels']
            
            # 라벨을 원래 형식으로 복원
            if updated_bboxes:
                augmented_labels = np.array([
                    [cls, *bbox] for cls, bbox in zip(updated_classes, updated_bboxes)
                ])
            else:
                augmented_labels = np.array([]).reshape(0, 5)
            
            return rgb_augmented, thermal_augmented, augmented_labels
            
        except Exception as e:
            LOGGER.warning(f"KAIST augmentation failed: {e}")
            return rgb_img, thermal_img, labels


class KAISTPedestrianAugmentations:
    """보행자 검출에 특화된 단순화된 증강 클래스 (fallback용)"""
    
    def __init__(self, size=640):
        self.size = size
    
    def apply_lighting_augmentation(self, img, mode='rgb'):
        """조명 조건 증강"""
        if mode == 'rgb':
            # RGB용 조명 변화
            img = img.astype(np.float32)
            
            # 밝기 조정
            brightness = random.uniform(0.7, 1.3)
            img = img * brightness
            
            # 대비 조정  
            contrast = random.uniform(0.8, 1.2)
            img = (img - 127.5) * contrast + 127.5
            
            # 감마 보정
            gamma = random.uniform(0.8, 1.2)
            img = np.power(img / 255.0, gamma) * 255.0
            
            return np.clip(img, 0, 255).astype(np.uint8)
            
        elif mode == 'thermal':
            # Thermal용 온도 변화 시뮬레이션
            img = img.astype(np.float32)
            
            # 열화상 노이즈 추가
            noise = np.random.normal(0, random.uniform(5, 20), img.shape)
            img = img + noise
            
            # 온도 범위 조정
            temp_shift = random.uniform(-30, 30)
            img = img + temp_shift
            
            return np.clip(img, 0, 255).astype(np.uint8)
    
    def apply_geometric_augmentation(self, img, labels):
        """기하학적 증강 (보행자 보존)"""
        h, w = img.shape[:2]
        
        # 수평 플립
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]  # x_center 조정
        
        # 작은 스케일링 (보행자가 너무 작아지지 않도록)
        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            
            # 원래 크기로 패딩 또는 크롭
            if scale > 1.0:
                # 크롭
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img[start_h:start_h+h, start_w:start_w+w]
            else:
                # 패딩
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                img = cv2.copyMakeBorder(img, pad_h, h-new_h-pad_h, 
                                       pad_w, w-new_w-pad_w, 
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img, labels


# 기존 Albumentations 클래스 개선
def create_kaist_albumentations(size=640):
    """KAIST 데이터셋용 개선된 Albumentations 설정"""
    try:
        import albumentations as A
        
        transforms = [
            # 보행자 검출에 적합한 크롭 (보수적)
            A.RandomResizedCrop(
                height=size, width=size, 
                scale=(0.8, 1.0), ratio=(0.9, 1.11), 
                p=0.3
            ),
            
            # 조명 조건 다양화
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.6
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # 노이즈 및 블러 (현실적인 조건)
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.2),
            
            # 날씨 조건
            A.RandomRain(p=0.1),
            A.RandomFog(p=0.1),
            A.RandomShadow(p=0.2),
            
            # 색상 변화 (RGB 채널용)
            A.ColorJitter(
                brightness=0.2, contrast=0.2, 
                saturation=0.15, hue=0.1, p=0.4
            ),
            
            # 이미지 품질
            A.ImageCompression(quality_lower=75, quality_upper=95, p=0.2),
        ]
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"]
        ))
        
    except ImportError:
        LOGGER.warning("Albumentations not found for KAIST augmentations")
        return None
