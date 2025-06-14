#!/usr/bin/env python3
"""
⚡ Preference Optimizer: Direct Preference Optimization Engine

Implements DPO (Direct Preference Optimization) for GPRO model training
based on human feedback data.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from .gpro_model import GPROModel, GPROConfig
from .human_feedback import FeedbackData

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """최적화 설정"""
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    beta: float = 0.1  # DPO temperature
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 500
    evaluation_steps: int = 200


class PreferenceDataset(Dataset):
    """선호도 데이터셋"""
    
    def __init__(self, feedback_data: List[FeedbackData], tokenizer, max_length: int = 512):
        self.data = feedback_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feedback = self.data[idx]
        
        # 입력 텍스트 구성
        prompt = f"직책: {feedback.position}\n후보자: {feedback.candidate_info}"
        
        # 선호/거부 응답 토크나이징
        preferred_inputs = self.tokenizer(
            prompt + "\n" + feedback.preferred_response,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        rejected_inputs = self.tokenizer(
            prompt + "\n" + feedback.rejected_response,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'preferred_input_ids': preferred_inputs['input_ids'].squeeze(),
            'preferred_attention_mask': preferred_inputs['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_inputs['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_inputs['attention_mask'].squeeze(),
            'preference_strength': torch.tensor(feedback.preference_strength, dtype=torch.float),
            'position': feedback.position
        }


class PreferenceOptimizer:
    """
    Direct Preference Optimization 엔진
    
    인간 피드백 데이터를 사용하여 GPRO 모델을 DPO 방식으로 최적화합니다.
    """
    
    def __init__(self, model: GPROModel, config: OptimizationConfig):
        self.model = model
        self.config = config
        
        # 옵티마이저 설정
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        
        # 학습률 스케줄러
        self.scheduler = None
        
        # 학습 상태 추적
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        logger.info("Preference Optimizer 초기화 완료")
    
    def prepare_dataset(self, feedback_data: List[FeedbackData]) -> PreferenceDataset:
        """피드백 데이터를 학습용 데이터셋으로 변환"""
        return PreferenceDataset(
            feedback_data, 
            self.model.encoder.tokenizer,
            self.model.config.max_length
        )
    
    def compute_dpo_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """DPO 손실 계산"""
        # 선호되는 응답에 대한 모델 출력
        preferred_outputs = self.model(
            batch['preferred_input_ids'],
            batch['preferred_attention_mask'],
            batch.get('position')
        )
        
        # 거부되는 응답에 대한 모델 출력
        rejected_outputs = self.model(
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch.get('position')
        )
        
        # 로그 확률 계산
        preferred_logprobs = F.log_softmax(preferred_outputs['logits'], dim=-1)
        rejected_logprobs = F.log_softmax(rejected_outputs['logits'], dim=-1)
        
        # 최대 확률 클래스의 로그 확률
        preferred_max_logprob = preferred_logprobs.max(dim=-1)[0]
        rejected_max_logprob = rejected_logprobs.max(dim=-1)[0]
        
        # 선호도 차이 계산
        preference_diff = preferred_max_logprob - rejected_max_logprob
        
        # Beta로 온도 조절
        scaled_diff = self.config.beta * preference_diff
        
        # DPO 손실: -log(σ(β * (log π(y_w|x) - log π(y_l|x))))
        dpo_loss = -F.logsigmoid(scaled_diff)
        
        # 선호도 강도로 가중치 적용
        preference_weights = batch['preference_strength']
        weighted_loss = (dpo_loss * preference_weights).mean()
        
        # Constitutional AI 손실 추가
        constitutional_loss = self._compute_constitutional_loss(
            preferred_outputs, rejected_outputs
        )
        
        # 총 손실
        total_loss = weighted_loss + 0.1 * constitutional_loss
        
        # 메트릭 계산
        metrics = {
            'dpo_loss': weighted_loss.item(),
            'constitutional_loss': constitutional_loss.item(),
            'total_loss': total_loss.item(),
            'preference_accuracy': (preference_diff > 0).float().mean().item(),
            'avg_preference_strength': preference_weights.mean().item(),
            'preferred_confidence': preferred_outputs['confidence'].mean().item(),
            'rejected_confidence': rejected_outputs['confidence'].mean().item()
        }
        
        return total_loss, metrics
    
    def _compute_constitutional_loss(self, preferred_outputs: Dict, rejected_outputs: Dict) -> torch.Tensor:
        """Constitutional AI 손실 계산"""
        # 선호되는 응답의 constitutional score가 더 높아야 함
        preferred_score = preferred_outputs['constitutional_score']
        rejected_score = rejected_outputs['constitutional_score']
        
        # Constitutional score 차이를 최대화
        constitutional_loss = F.relu(rejected_score - preferred_score + 0.1).mean()
        
        return constitutional_loss
    
    def train_step(self, batch: Dict) -> Dict:
        """단일 학습 스텝"""
        self.model.train()
        
        # GPU로 이동
        batch = {k: v.to(next(self.model.parameters()).device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        loss, metrics = self.compute_dpo_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return metrics
    
    def evaluate(self, eval_dataset: PreferenceDataset) -> Dict:
        """모델 평가"""
        self.model.eval()
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(next(self.model.parameters()).device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                _, metrics = self.compute_dpo_loss(batch)
                
                # 메트릭 누적
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0
                    total_metrics[key] += value
                
                num_batches += 1
        
        # 평균 계산
        avg_metrics = {f"eval_{k}": v / num_batches for k, v in total_metrics.items()}
        
        return avg_metrics
    
    def train(
        self, 
        train_dataset: PreferenceDataset,
        eval_dataset: Optional[PreferenceDataset] = None,
        output_dir: str = "gpro_checkpoints"
    ) -> Dict:
        """전체 학습 과정"""
        logger.info(f"DPO 학습 시작 - 데이터: {len(train_dataset)}건, 에포크: {self.config.num_epochs}")
        
        # 데이터로더 생성
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 학습 히스토리
        training_history = {
            'train_loss': [],
            'eval_metrics': [],
            'best_checkpoints': []
        }
        
        # 학습 루프
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_metrics = []
            
            logger.info(f"에포크 {epoch + 1}/{self.config.num_epochs} 시작")
            
            for batch_idx, batch in enumerate(train_dataloader):
                # 학습 스텝
                step_metrics = self.train_step(batch)
                epoch_metrics.append(step_metrics)
                
                # 로깅
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = np.mean([m['total_loss'] for m in epoch_metrics[-self.config.logging_steps:]])
                    logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}")
                
                # 평가
                if eval_dataset and self.global_step % self.config.evaluation_steps == 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    training_history['eval_metrics'].append({
                        'step': self.global_step,
                        'metrics': eval_metrics
                    })
                    
                    logger.info(f"평가 결과: {eval_metrics}")
                
                # 체크포인트 저장
                if self.global_step % self.config.save_steps == 0:
                    checkpoint_path = output_path / f"checkpoint_step_{self.global_step}"
                    self.save_checkpoint(checkpoint_path)
            
            # 에포크 완료
            epoch_avg_loss = np.mean([m['total_loss'] for m in epoch_metrics])
            training_history['train_loss'].append(epoch_avg_loss)
            
            logger.info(f"에포크 {epoch + 1} 완료 - 평균 손실: {epoch_avg_loss:.4f}")
            
            # 최고 성능 모델 저장
            if epoch_avg_loss < self.best_loss:
                self.best_loss = epoch_avg_loss
                best_model_path = output_path / "best_model"
                self.model.save_model(str(best_model_path))
                training_history['best_checkpoints'].append({
                    'epoch': epoch + 1,
                    'loss': epoch_avg_loss,
                    'path': str(best_model_path)
                })
                logger.info(f"최고 성능 모델 저장: {best_model_path}")
        
        # 최종 모델 저장
        final_model_path = output_path / "final_model"
        self.model.save_model(str(final_model_path))
        
        logger.info("DPO 학습 완료!")
        
        return training_history
    
    def save_checkpoint(self, checkpoint_path: Path):
        """체크포인트 저장"""
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        self.model.save_model(str(checkpoint_path))
        
        # 옵티마이저 상태 저장
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'config': self.config
        }, checkpoint_path / "optimizer.pt")
        
        logger.info(f"체크포인트 저장: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint_path = Path(checkpoint_path)
        
        # 모델 로드
        self.model = GPROModel.load_model(str(checkpoint_path))
        
        # 옵티마이저 상태 로드
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            checkpoint = torch.load(optimizer_path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_step = checkpoint['global_step']
            self.current_epoch = checkpoint['current_epoch']
            self.best_loss = checkpoint['best_loss']
            
            logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
        else:
            logger.warning("옵티마이저 상태를 찾을 수 없습니다.")


# 편의 함수들
def train_gpro_with_feedback(
    model: GPROModel,
    feedback_data: List[FeedbackData],
    config: OptimizationConfig = None,
    validation_split: float = 0.2
) -> Tuple[GPROModel, Dict]:
    """피드백 데이터로 GPRO 모델 학습"""
    
    if config is None:
        config = OptimizationConfig()
    
    # 데이터 분할
    split_idx = int(len(feedback_data) * (1 - validation_split))
    train_data = feedback_data[:split_idx]
    val_data = feedback_data[split_idx:] if validation_split > 0 else None
    
    # 옵티마이저 생성
    optimizer = PreferenceOptimizer(model, config)
    
    # 데이터셋 준비
    train_dataset = optimizer.prepare_dataset(train_data)
    val_dataset = optimizer.prepare_dataset(val_data) if val_data else None
    
    # 학습 실행
    training_history = optimizer.train(train_dataset, val_dataset)
    
    return model, training_history


if __name__ == "__main__":
    # 테스트 코드
    print("⚡ Preference Optimizer 테스트")
    
    from .gpro_model import initialize_gpro_model
    from .human_feedback import HumanFeedbackSimulator, create_sample_candidates
    
    # GPRO 모델 생성
    model = initialize_gpro_model()
    
    # 피드백 데이터 생성
    simulator = HumanFeedbackSimulator()
    candidates = create_sample_candidates()
    
    # 샘플 피드백 생성
    sample_feedback = []
    for i in range(10):
        candidate = candidates[i % len(candidates)]
        
        # 가상 AI 추천
        ai_rec = {
            'prediction': '추천',
            'confidence': 0.8
        }
        
        feedback = simulator.generate_expert_feedback(
            candidate['info'], candidate['position'], ai_rec
        )
        sample_feedback.append(feedback)
    
    # 최적화 설정
    config = OptimizationConfig(
        learning_rate=1e-5,
        batch_size=2,
        num_epochs=1,
        logging_steps=5
    )
    
    # 옵티마이저 생성 및 학습
    optimizer = PreferenceOptimizer(model, config)
    train_dataset = optimizer.prepare_dataset(sample_feedback)
    
    print(f"학습 데이터: {len(train_dataset)}건")
    print("✅ Preference Optimizer 테스트 완료!") 