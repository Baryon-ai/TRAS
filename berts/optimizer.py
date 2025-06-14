"""
⚡ BERT 최적화 모듈
고속 추론과 메모리 효율성을 위한 최적화 기법들

- 모델 양자화 (Quantization)
- 동적 배치 크기 조정
- 캐싱 최적화
- 병렬 처리
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """최적화 결과 데이터 클래스"""
    original_time: float
    optimized_time: float
    speedup_ratio: float
    memory_saved: float
    accuracy_maintained: float
    optimization_methods: List[str]

class ModelQuantizer:
    """
    📊 모델 양자화
    float32 → int8 변환으로 속도와 메모리 효율성 향상
    """
    
    @staticmethod
    def quantize_model(model: nn.Module, calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        모델 양자화 수행
        
        Args:
            model: 양자화할 모델
            calibration_data: 캘리브레이션 데이터
            
        Returns:
            양자화된 모델
        """
        logger.info("🔧 모델 양자화 시작...")
        
        # 1. 모델을 평가 모드로 설정
        model.eval()
        
        # 2. 양자화 설정
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # 3. 양자화 준비
        quantization.prepare(model, inplace=True)
        
        # 4. 캘리브레이션 (선택적)
        if calibration_data is not None:
            with torch.no_grad():
                _ = model(calibration_data)
        
        # 5. 양자화 적용
        quantized_model = quantization.convert(model, inplace=True)
        
        logger.info("✅ 모델 양자화 완료")
        return quantized_model
    
    @staticmethod
    def measure_quantization_impact(
        original_model: nn.Module, 
        quantized_model: nn.Module,
        test_data: torch.Tensor
    ) -> Dict[str, float]:
        """양자화 영향 측정"""
        
        # 메모리 사용량 측정
        def get_model_size(model):
            param_size = 0
            buffer_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return (param_size + buffer_size) / 1024 / 1024  # MB
        
        original_size = get_model_size(original_model)
        quantized_size = get_model_size(quantized_model)
        
        # 추론 속도 측정
        def measure_inference_time(model, data, num_runs=10):
            model.eval()
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.time()
                    _ = model(data)
                    times.append(time.time() - start)
            return np.mean(times)
        
        original_time = measure_inference_time(original_model, test_data)
        quantized_time = measure_inference_time(quantized_model, test_data)
        
        return {
            'size_reduction': (original_size - quantized_size) / original_size,
            'speed_improvement': (original_time - quantized_time) / original_time,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'original_time_ms': original_time * 1000,
            'quantized_time_ms': quantized_time * 1000
        }

class DynamicBatchManager:
    """
    📈 동적 배치 크기 관리
    시스템 리소스에 따라 배치 크기를 자동 조정
    """
    
    def __init__(self, initial_batch_size: int = 8, max_batch_size: int = 64):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 1
        self.performance_history = []
        self.memory_threshold = 0.85  # 85% 메모리 사용 시 배치 크기 감소
        
    def get_optimal_batch_size(self) -> int:
        """현재 시스템 상태에 최적화된 배치 크기 반환"""
        
        # 메모리 사용률 확인
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        if memory_percent > self.memory_threshold:
            # 메모리 부족 시 배치 크기 감소
            self.current_batch_size = max(
                self.min_batch_size, 
                self.current_batch_size // 2
            )
            logger.info(f"🔻 메모리 부족으로 배치 크기 감소: {self.current_batch_size}")
        
        elif memory_percent < 0.6 and len(self.performance_history) > 0:
            # 메모리 여유 시 배치 크기 증가 시도
            recent_performance = np.mean(self.performance_history[-5:])
            if recent_performance > 0:  # 성능이 좋다면
                self.current_batch_size = min(
                    self.max_batch_size,
                    self.current_batch_size * 2
                )
                logger.info(f"🔺 메모리 여유로 배치 크기 증가: {self.current_batch_size}")
        
        return self.current_batch_size
    
    def record_performance(self, processing_time: float, batch_size: int):
        """성능 기록 및 학습"""
        throughput = batch_size / processing_time  # samples/second
        self.performance_history.append(throughput)
        
        # 최근 10개 기록만 유지
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)

class CacheManager:
    """
    💾 지능형 캐시 관리
    자주 사용되는 결과를 메모리에 캐시하여 반복 계산 방지
    """
    
    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_cache_size
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key in self.cache:
            self.hit_count += 1
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def put(self, key: str, value: Any):
        """캐시에 값 저장"""
        if len(self.cache) >= self.max_size:
            # LFU (Least Frequently Used) 정책으로 삭제
            least_used_key = min(self.access_count.keys(), 
                               key=lambda k: self.access_count[k])
            del self.cache[least_used_key]
            del self.access_count[least_used_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
    
    def get_stats(self) -> Dict[str, float]:
        """캐시 통계 반환"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }

class ParallelProcessor:
    """
    🔄 병렬 처리 관리
    멀티 스레딩을 통한 배치 처리 최적화
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, psutil.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def process_batch_parallel(
        self, 
        model: nn.Module, 
        data_batches: List[torch.Tensor],
        callback_fn: Optional[callable] = None
    ) -> List[Any]:
        """
        배치들을 병렬로 처리
        
        Args:
            model: 처리할 모델
            data_batches: 데이터 배치 리스트
            callback_fn: 각 배치 처리 후 호출할 콜백 함수
            
        Returns:
            처리 결과 리스트
        """
        def process_single_batch(batch):
            with torch.no_grad():
                result = model(batch)
                if callback_fn:
                    callback_fn(result)
                return result
        
        # 병렬 처리 실행
        futures = [
            self.executor.submit(process_single_batch, batch)
            for batch in data_batches
        ]
        
        # 결과 수집
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30초 타임아웃
                results.append(result)
            except Exception as e:
                logger.error(f"병렬 처리 오류: {e}")
                results.append(None)
        
        return results
    
    def shutdown(self):
        """스레드 풀 종료"""
        self.executor.shutdown(wait=True)

class BERTOptimizer:
    """
    🚀 통합 BERT 최적화 관리자
    모든 최적화 기법을 통합하여 관리
    """
    
    def __init__(self, model: nn.Module):
        self.original_model = model
        self.optimized_model = None
        
        # 최적화 컴포넌트들
        self.quantizer = ModelQuantizer()
        self.batch_manager = DynamicBatchManager()
        self.cache_manager = CacheManager()
        self.parallel_processor = ParallelProcessor()
        
        # 최적화 통계
        self.optimization_stats = {
            'total_optimizations': 0,
            'average_speedup': 0.0,
            'memory_savings': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("🚀 BERTOptimizer 초기화 완료")
    
    def optimize_model(
        self, 
        optimization_methods: List[str] = ['quantization', 'caching', 'dynamic_batching'],
        calibration_data: Optional[torch.Tensor] = None
    ) -> OptimizationResult:
        """
        모델 최적화 실행
        
        Args:
            optimization_methods: 적용할 최적화 방법들
            calibration_data: 양자화용 캘리브레이션 데이터
            
        Returns:
            최적화 결과
        """
        start_time = time.time()
        logger.info(f"🔧 모델 최적화 시작: {optimization_methods}")
        
        # 원본 성능 측정
        original_performance = self._measure_performance(self.original_model)
        
        # 최적화 적용
        optimized_model = self.original_model
        applied_methods = []
        
        if 'quantization' in optimization_methods:
            try:
                optimized_model = self.quantizer.quantize_model(
                    optimized_model, calibration_data
                )
                applied_methods.append('quantization')
                logger.info("✅ 양자화 적용 완료")
            except Exception as e:
                logger.warning(f"양자화 실패: {e}")
        
        # 최적화된 모델 저장
        self.optimized_model = optimized_model
        
        # 최적화 후 성능 측정
        optimized_performance = self._measure_performance(optimized_model)
        
        # 결과 계산
        speedup_ratio = original_performance['avg_time'] / optimized_performance['avg_time']
        memory_saved = (original_performance['memory_mb'] - optimized_performance['memory_mb']) / original_performance['memory_mb']
        
        # 통계 업데이트
        self.optimization_stats['total_optimizations'] += 1
        self.optimization_stats['average_speedup'] = (
            (self.optimization_stats['average_speedup'] * (self.optimization_stats['total_optimizations'] - 1) + speedup_ratio) / 
            self.optimization_stats['total_optimizations']
        )
        self.optimization_stats['memory_savings'] = memory_saved
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            original_time=original_performance['avg_time'],
            optimized_time=optimized_performance['avg_time'],
            speedup_ratio=speedup_ratio,
            memory_saved=memory_saved,
            accuracy_maintained=0.95,  # 간단히 고정값 사용
            optimization_methods=applied_methods
        )
        
        logger.info(f"🎉 최적화 완료 - 속도 향상: {speedup_ratio:.2f}x, 메모리 절약: {memory_saved:.1%}")
        return result
    
    def _measure_performance(self, model: nn.Module, num_runs: int = 5) -> Dict[str, float]:
        """모델 성능 측정"""
        model.eval()
        
        # 테스트 데이터 생성
        test_input = torch.randint(0, 1000, (8, 512))  # 배치 크기 8, 시퀀스 길이 512
        
        # 추론 시간 측정
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(test_input)
                times.append(time.time() - start)
        
        # 메모리 사용량 측정
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'memory_mb': model_size
        }
    
    def get_optimized_model(self) -> Optional[nn.Module]:
        """최적화된 모델 반환"""
        return self.optimized_model
    
    def process_with_optimization(
        self, 
        input_data: List[torch.Tensor],
        use_caching: bool = True,
        use_parallel: bool = True
    ) -> List[Any]:
        """
        최적화된 처리 파이프라인 실행
        
        Args:
            input_data: 입력 데이터 리스트
            use_caching: 캐싱 사용 여부
            use_parallel: 병렬 처리 사용 여부
            
        Returns:
            처리 결과 리스트
        """
        if not self.optimized_model:
            logger.warning("최적화된 모델이 없습니다. 원본 모델 사용")
            model = self.original_model
        else:
            model = self.optimized_model
        
        results = []
        
        for data in input_data:
            # 캐싱 확인
            cache_key = f"data_{hash(data.detach().numpy().tobytes())}" if use_caching else None
            
            if cache_key and use_caching:
                cached_result = self.cache_manager.get(cache_key)
                if cached_result is not None:
                    results.append(cached_result)
                    continue
            
            # 최적 배치 크기 결정
            optimal_batch_size = self.batch_manager.get_optimal_batch_size()
            
            # 모델 실행
            start_time = time.time()
            with torch.no_grad():
                result = model(data)
            processing_time = time.time() - start_time
            
            # 성능 기록
            self.batch_manager.record_performance(processing_time, data.size(0))
            
            # 캐싱 저장
            if cache_key and use_caching:
                self.cache_manager.put(cache_key, result)
            
            results.append(result)
        
        return results
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """최적화 리포트 생성"""
        cache_stats = self.cache_manager.get_stats()
        
        return {
            'optimization_stats': self.optimization_stats,
            'cache_stats': cache_stats,
            'current_batch_size': self.batch_manager.current_batch_size,
            'parallel_workers': self.parallel_processor.max_workers,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """성능 개선 권장사항 생성"""
        recommendations = []
        
        cache_stats = self.cache_manager.get_stats()
        if cache_stats['hit_rate'] < 0.5:
            recommendations.append("캐시 적중률이 낮습니다. 캐시 크기를 늘리거나 데이터 패턴을 확인하세요.")
        
        if self.optimization_stats['average_speedup'] < 1.5:
            recommendations.append("추가 최적화가 필요합니다. 양자화나 모델 압축을 고려하세요.")
        
        if self.batch_manager.current_batch_size < 4:
            recommendations.append("배치 크기가 작습니다. 시스템 메모리를 확인하세요.")
        
        return recommendations
    
    def cleanup(self):
        """리소스 정리"""
        self.parallel_processor.shutdown()
        self.cache_manager.cache.clear()
        logger.info("🧹 BERTOptimizer 리소스 정리 완료")

# 사용 예시
if __name__ == "__main__":
    print("⚡ BERT 최적화 모듈 테스트")
    print("=" * 50)
    
    # 간단한 테스트 모델 생성
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 128)
            self.relu = nn.ReLU()
            self.output = nn.Linear(128, 10)
        
        def forward(self, x):
            x = x.float()  # int를 float로 변환
            x = self.linear(x)
            x = self.relu(x)
            return self.output(x)
    
    # 테스트 모델과 데이터
    test_model = SimpleModel()
    test_data = [torch.randint(0, 100, (4, 512)) for _ in range(5)]
    
    # 최적화 실행
    optimizer = BERTOptimizer(test_model)
    
    print("🔧 모델 최적화 실행...")
    optimization_result = optimizer.optimize_model(['caching'])
    
    print(f"✅ 최적화 완료!")
    print(f"   속도 향상: {optimization_result.speedup_ratio:.2f}x")
    print(f"   메모리 절약: {optimization_result.memory_saved:.1%}")
    print(f"   적용된 방법: {optimization_result.optimization_methods}")
    
    # 최적화된 처리 테스트
    print(f"\n📊 최적화된 처리 테스트...")
    results = optimizer.process_with_optimization(test_data)
    print(f"✅ {len(results)}개 배치 처리 완료")
    
    # 최적화 리포트
    report = optimizer.get_optimization_report()
    print(f"\n📈 최적화 리포트:")
    print(f"   캐시 적중률: {report['cache_stats']['hit_rate']:.1%}")
    print(f"   현재 배치 크기: {report['current_batch_size']}")
    print(f"   병렬 워커 수: {report['parallel_workers']}")
    
    # 정리
    optimizer.cleanup()
    print(f"\n🎉 최적화 테스트 완료!") 