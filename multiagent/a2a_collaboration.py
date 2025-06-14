"""
🔗 A2A: Agent-to-Agent Collaboration
===================================

"Google의 똑똑한 협업 기술" - Section 6 강의 내용 기반 구현

여러 AI 에이전트가 효과적으로 협업하여 더 나은 결과를 
도출할 수 있도록 하는 협업 프레임워크입니다.

핵심 메커니즘:
- 병렬 처리 (Parallel Processing)
- 상호 검토 (Peer Review)
- 합의 구축 (Consensus Building)
- 계층적 처리 (Hierarchical Processing)
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import statistics
from datetime import datetime


class CollaborationPattern(Enum):
    """협업 패턴 정의"""
    PARALLEL = "parallel"              # 병렬 처리
    SEQUENTIAL = "sequential"          # 순차 처리  
    HIERARCHICAL = "hierarchical"      # 계층적 처리
    PEER_REVIEW = "peer_review"        # 상호 검토
    CONSENSUS = "consensus"            # 합의 구축
    HYBRID = "hybrid"                  # 혼합 방식


@dataclass
class CollaborationTask:
    """협업 작업 정의"""
    task_id: str
    input_data: Dict[str, Any]
    target_agents: List[str]
    pattern: CollaborationPattern
    priority: int = 1
    timeout: int = 300  # 5분 타임아웃
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class AgentResult:
    """에이전트 결과"""
    agent_id: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: str
    error: Optional[str] = None


@dataclass
class CollaborationResult:
    """협업 결과"""
    task_id: str
    pattern: CollaborationPattern
    agent_results: List[AgentResult]
    final_result: Dict[str, Any]
    consensus_level: float
    total_time: float
    success: bool
    metadata: Dict[str, Any]


class A2ACollaboration:
    """
    Google의 Agent-to-Agent 협업 프레임워크
    
    Section 6 강의에서 소개된 핵심 협업 메커니즘들을 구현
    """
    
    def __init__(self):
        """A2A 협업 시스템 초기화"""
        self.agents: Dict[str, Any] = {}
        self.collaboration_history: List[CollaborationResult] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 협업 패턴별 처리 함수 매핑
        self.collaboration_patterns = {
            CollaborationPattern.PARALLEL: self.parallel_processing,
            CollaborationPattern.SEQUENTIAL: self.sequential_processing,
            CollaborationPattern.HIERARCHICAL: self.hierarchical_processing,
            CollaborationPattern.PEER_REVIEW: self.peer_review_processing,
            CollaborationPattern.CONSENSUS: self.consensus_building,
            CollaborationPattern.HYBRID: self.hybrid_processing
        }
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "average_consensus_level": 0.0,
            "average_processing_time": 0.0,
            "pattern_usage": {pattern: 0 for pattern in CollaborationPattern}
        }
    
    def register_agent(self, agent_id: str, agent_instance: Any):
        """에이전트 등록"""
        self.agents[agent_id] = {
            "instance": agent_instance,
            "registered_at": datetime.now().isoformat(),
            "collaborations_count": 0,
            "average_confidence": 0.0,
            "specialties": getattr(agent_instance, 'specialties', [])
        }
    
    def collaborate(self, task: CollaborationTask) -> CollaborationResult:
        """
        협업 작업 실행
        
        Args:
            task: 협업 작업 정의
            
        Returns:
            협업 결과
        """
        start_time = time.time()
        
        try:
            # 협업 패턴에 따른 처리
            pattern_func = self.collaboration_patterns.get(task.pattern)
            if not pattern_func:
                raise ValueError(f"지원하지 않는 협업 패턴: {task.pattern}")
            
            # 협업 실행
            agent_results, final_result, consensus_level = pattern_func(
                task.input_data, task.target_agents
            )
            
            # 결과 생성
            total_time = time.time() - start_time
            result = CollaborationResult(
                task_id=task.task_id,
                pattern=task.pattern,
                agent_results=agent_results,
                final_result=final_result,
                consensus_level=consensus_level,
                total_time=total_time,
                success=True,
                metadata={
                    "participating_agents": len(task.target_agents),
                    "task_priority": task.priority,
                    "completion_time": datetime.now().isoformat()
                }
            )
            
            # 협업 기록 저장
            self.collaboration_history.append(result)
            self._update_performance_metrics(result)
            
            return result
            
        except Exception as e:
            # 실패 시 결과
            total_time = time.time() - start_time
            result = CollaborationResult(
                task_id=task.task_id,
                pattern=task.pattern,
                agent_results=[],
                final_result={"error": str(e)},
                consensus_level=0.0,
                total_time=total_time,
                success=False,
                metadata={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            self.collaboration_history.append(result)
            return result
    
    def parallel_processing(
        self, 
        task_data: Dict[str, Any], 
        agent_ids: List[str]
    ) -> Tuple[List[AgentResult], Dict[str, Any], float]:
        """
        병렬 처리: 여러 에이전트가 동시에 작업
        
        Args:
            task_data: 작업 데이터
            agent_ids: 참여 에이전트 ID 목록
            
        Returns:
            (에이전트 결과들, 통합 결과, 합의 수준)
        """
        print("🔄 병렬 처리 시작...")
        
        # 병렬 실행을 위한 Future 생성
        futures = {}
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent:
                future = self.executor.submit(
                    self._execute_agent_task, 
                    agent_id, 
                    agent["instance"], 
                    task_data
                )
                futures[future] = agent_id
        
        # 결과 수집
        agent_results = []
        for future in as_completed(futures.keys()):
            agent_id = futures[future]
            try:
                result = future.result(timeout=300)  # 5분 타임아웃
                agent_results.append(result)
                print(f"✅ {agent_id} 완료 (신뢰도: {result.confidence:.2f})")
            except Exception as e:
                error_result = AgentResult(
                    agent_id=agent_id,
                    result={"error": str(e)},
                    confidence=0.0,
                    processing_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                agent_results.append(error_result)
                print(f"❌ {agent_id} 실패: {e}")
        
        # 결과 통합
        combined_result = self.merge_parallel_results(agent_results)
        consensus_level = self._calculate_consensus_level(agent_results)
        
        return agent_results, combined_result, consensus_level
    
    def sequential_processing(
        self, 
        task_data: Dict[str, Any], 
        agent_ids: List[str]
    ) -> Tuple[List[AgentResult], Dict[str, Any], float]:
        """
        순차 처리: 에이전트가 순서대로 작업
        """
        print("🔄 순차 처리 시작...")
        
        agent_results = []
        accumulated_data = task_data.copy()
        
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent:
                try:
                    result = self._execute_agent_task(
                        agent_id, 
                        agent["instance"], 
                        accumulated_data
                    )
                    agent_results.append(result)
                    
                    # 다음 에이전트를 위해 결과 축적
                    accumulated_data["previous_results"] = accumulated_data.get("previous_results", [])
                    accumulated_data["previous_results"].append(result.result)
                    
                    print(f"✅ {agent_id} 완료 → 다음 에이전트로 전달")
                    
                except Exception as e:
                    error_result = AgentResult(
                        agent_id=agent_id,
                        result={"error": str(e)},
                        confidence=0.0,
                        processing_time=0.0,
                        timestamp=datetime.now().isoformat(),
                        error=str(e)
                    )
                    agent_results.append(error_result)
                    print(f"❌ {agent_id} 실패: {e}")
                    break  # 순차 처리에서는 하나 실패하면 중단
        
        # 최종 결과는 마지막 에이전트의 결과
        final_result = agent_results[-1].result if agent_results else {"error": "No results"}
        consensus_level = 1.0 if len(agent_results) == len(agent_ids) else 0.5
        
        return agent_results, final_result, consensus_level
    
    def hierarchical_processing(
        self, 
        task_data: Dict[str, Any], 
        agent_ids: List[str]
    ) -> Tuple[List[AgentResult], Dict[str, Any], float]:
        """
        계층적 처리: 전문가 → 검토자 → 결정자 순서
        """
        print("🏗️ 계층적 처리 시작...")
        
        # 에이전트를 역할별로 분류 (이름 기반)
        specialists = [aid for aid in agent_ids if "expert" in aid or "specialist" in aid]
        reviewers = [aid for aid in agent_ids if "reviewer" in aid or "checker" in aid]
        coordinators = [aid for aid in agent_ids if "coordinator" in aid or "master" in aid]
        
        all_results = []
        
        # 1단계: 전문가 분석 (병렬)
        if specialists:
            print("🎯 1단계: 전문가 분석")
            specialist_results, _, _ = self.parallel_processing(task_data, specialists)
            all_results.extend(specialist_results)
        
        # 2단계: 검토자 검토
        if reviewers:
            print("🔍 2단계: 검토자 검토")
            review_data = task_data.copy()
            review_data["specialist_results"] = [r.result for r in specialist_results]
            
            review_results, _, _ = self.parallel_processing(review_data, reviewers)
            all_results.extend(review_results)
        
        # 3단계: 코디네이터 최종 결정
        if coordinators:
            print("🎯 3단계: 최종 결정")
            final_data = task_data.copy()
            final_data["all_analyses"] = [r.result for r in all_results]
            
            coord_results, final_result, _ = self.sequential_processing(final_data, coordinators)
            all_results.extend(coord_results)
        else:
            # 코디네이터가 없으면 검토 결과를 통합
            final_result = self.merge_parallel_results(all_results)
        
        consensus_level = self._calculate_consensus_level(all_results)
        
        return all_results, final_result, consensus_level
    
    def peer_review_processing(
        self, 
        task_data: Dict[str, Any], 
        agent_ids: List[str]
    ) -> Tuple[List[AgentResult], Dict[str, Any], float]:
        """
        상호 검토: 에이전트들이 서로의 결과를 검증
        """
        print("🤝 상호 검토 처리 시작...")
        
        # 1단계: 모든 에이전트가 독립적으로 분석
        print("🔍 1단계: 독립 분석")
        initial_results, _, _ = self.parallel_processing(task_data, agent_ids)
        
        # 2단계: 상호 검토
        print("👥 2단계: 상호 검토")
        peer_reviews = []
        
        for reviewer_id in agent_ids:
            reviewer = self.agents.get(reviewer_id)
            if not reviewer:
                continue
                
            # 자신의 결과를 제외한 다른 에이전트들의 결과 검토
            reviews_for_others = {}
            for result in initial_results:
                if result.agent_id != reviewer_id:
                    try:
                        review_data = {
                            "original_task": task_data,
                            "analysis_to_review": result.result,
                            "reviewer_perspective": reviewer_id
                        }
                        
                        review_result = self._execute_agent_task(
                            f"{reviewer_id}_reviewing_{result.agent_id}",
                            reviewer["instance"],
                            review_data,
                            task_type="peer_review"
                        )
                        
                        reviews_for_others[result.agent_id] = review_result.result
                        
                    except Exception as e:
                        print(f"❌ {reviewer_id}의 {result.agent_id} 검토 실패: {e}")
            
            if reviews_for_others:
                peer_review_result = AgentResult(
                    agent_id=f"{reviewer_id}_peer_reviews",
                    result={"reviews": reviews_for_others},
                    confidence=0.8,  # 검토 결과의 기본 신뢰도
                    processing_time=time.time(),
                    timestamp=datetime.now().isoformat()
                )
                peer_reviews.append(peer_review_result)
        
        # 3단계: 검토 결과를 반영한 최종 결정
        print("🎯 3단계: 검토 반영 최종 결정")
        all_results = initial_results + peer_reviews
        
        # 상호 검토를 통한 가중치 조정
        final_result = self._synthesize_peer_reviewed_results(initial_results, peer_reviews)
        consensus_level = self._calculate_peer_review_consensus(initial_results, peer_reviews)
        
        return all_results, final_result, consensus_level
    
    def consensus_building(
        self, 
        task_data: Dict[str, Any], 
        agent_ids: List[str]
    ) -> Tuple[List[AgentResult], Dict[str, Any], float]:
        """
        합의 구축: 모든 의견을 종합해서 최종 결정
        """
        print("🎯 합의 구축 처리 시작...")
        
        # 1단계: 병렬 분석
        agent_results, _, _ = self.parallel_processing(task_data, agent_ids)
        
        # 2단계: 가중치 계산 (신뢰도와 전문성 기반)
        weights = self._calculate_agent_weights(agent_results)
        
        # 3단계: 가중 평균으로 최종 점수 계산
        final_scores = {}
        evaluation_criteria = ["technical", "policy", "leadership", "collaboration"]
        
        for criterion in evaluation_criteria:
            weighted_sum = 0
            total_weight = 0
            
            for result in agent_results:
                if criterion in result.result:
                    weight = weights.get(result.agent_id, 0.5)
                    weighted_sum += result.result[criterion] * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_scores[criterion] = weighted_sum / total_weight
            else:
                final_scores[criterion] = 0
        
        # 4단계: 최종 추천 등급 결정
        overall_score = sum(final_scores.values()) / len(final_scores) if final_scores else 0
        recommendation = self._score_to_recommendation(overall_score)
        
        # 5단계: 합의 수준 계산
        consensus_level = self._calculate_consensus_level(agent_results)
        
        final_result = {
            "final_recommendation": recommendation,
            "detailed_scores": final_scores,
            "overall_score": overall_score,
            "consensus_level": consensus_level,
            "reasoning": self._generate_consensus_reasoning(agent_results, weights),
            "participating_agents": [r.agent_id for r in agent_results],
            "decision_timestamp": datetime.now().isoformat()
        }
        
        return agent_results, final_result, consensus_level
    
    def hybrid_processing(
        self, 
        task_data: Dict[str, Any], 
        agent_ids: List[str]
    ) -> Tuple[List[AgentResult], Dict[str, Any], float]:
        """
        혼합 방식: 여러 협업 패턴을 조합
        """
        print("🔀 혼합 방식 처리 시작...")
        
        # 1단계: 병렬 분석
        initial_results, _, _ = self.parallel_processing(task_data, agent_ids)
        
        # 2단계: 상호 검토
        peer_review_data = task_data.copy()
        peer_review_data["initial_analyses"] = [r.result for r in initial_results]
        peer_reviews, _, _ = self.peer_review_processing(peer_review_data, agent_ids)
        
        # 3단계: 합의 구축
        consensus_data = task_data.copy()
        consensus_data["peer_reviewed_analyses"] = [r.result for r in peer_reviews]
        all_results, final_result, consensus_level = self.consensus_building(consensus_data, agent_ids)
        
        # 모든 단계의 결과 통합
        combined_results = initial_results + peer_reviews + all_results
        
        return combined_results, final_result, consensus_level
    
    def _execute_agent_task(
        self, 
        agent_id: str, 
        agent_instance: Any, 
        task_data: Dict[str, Any],
        task_type: str = "analysis"
    ) -> AgentResult:
        """에이전트 작업 실행"""
        start_time = time.time()
        
        try:
            # 에이전트 타입에 따른 메소드 호출
            if task_type == "peer_review":
                if hasattr(agent_instance, 'peer_review'):
                    result = agent_instance.peer_review(task_data)
                else:
                    result = agent_instance.analyze(task_data)
            else:
                if hasattr(agent_instance, 'analyze'):
                    result = agent_instance.analyze(task_data)
                elif hasattr(agent_instance, 'process'):
                    result = agent_instance.process(task_data)
                else:
                    result = {"error": "Agent has no analyze or process method"}
            
            processing_time = time.time() - start_time
            confidence = result.get("confidence", 0.5) if isinstance(result, dict) else 0.5
            
            return AgentResult(
                agent_id=agent_id,
                result=result,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return AgentResult(
                agent_id=agent_id,
                result={"error": str(e)},
                confidence=0.0,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
    
    def merge_parallel_results(self, agent_results: List[AgentResult]) -> Dict[str, Any]:
        """병렬 처리 결과 통합"""
        if not agent_results:
            return {"error": "No results to merge"}
        
        # 성공한 결과만 필터링
        successful_results = [r for r in agent_results if r.error is None]
        
        if not successful_results:
            return {"error": "All agents failed"}
        
        # 점수들의 평균 계산
        score_fields = ["technical_score", "policy_score", "leadership_score", "collaboration_score"]
        merged_scores = {}
        
        for field in score_fields:
            scores = [r.result.get(field, 0) for r in successful_results if field in r.result]
            if scores:
                merged_scores[field] = sum(scores) / len(scores)
        
        # 신뢰도 가중 평균
        total_confidence = sum(r.confidence for r in successful_results)
        avg_confidence = total_confidence / len(successful_results)
        
        # 공통 추천사항 추출
        recommendations = [r.result.get("recommendation", "") for r in successful_results]
        
        return {
            "merged_scores": merged_scores,
            "average_confidence": avg_confidence,
            "participating_agents": [r.agent_id for r in successful_results],
            "individual_recommendations": recommendations,
            "merge_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_consensus_level(self, agent_results: List[AgentResult]) -> float:
        """합의 수준 계산"""
        if len(agent_results) < 2:
            return 1.0
        
        # 신뢰도 기반 합의 수준
        confidences = [r.confidence for r in agent_results if r.confidence > 0]
        if not confidences:
            return 0.0
        
        # 표준편차가 낮을수록 합의 수준이 높음
        mean_confidence = statistics.mean(confidences)
        if len(confidences) > 1:
            std_confidence = statistics.stdev(confidences)
            consensus = max(0, 1 - (std_confidence / mean_confidence))
        else:
            consensus = mean_confidence
        
        return min(1.0, consensus)
    
    def _calculate_agent_weights(self, agent_results: List[AgentResult]) -> Dict[str, float]:
        """에이전트별 가중치 계산"""
        weights = {}
        total_confidence = sum(r.confidence for r in agent_results)
        
        if total_confidence == 0:
            # 모든 에이전트가 동일한 가중치
            equal_weight = 1.0 / len(agent_results)
            for result in agent_results:
                weights[result.agent_id] = equal_weight
        else:
            # 신뢰도 기반 가중치
            for result in agent_results:
                weights[result.agent_id] = result.confidence / total_confidence
        
        return weights
    
    def _score_to_recommendation(self, score: float) -> str:
        """점수를 추천 등급으로 변환"""
        if score >= 8.5:
            return "강력 추천"
        elif score >= 7.0:
            return "추천"
        elif score >= 5.5:
            return "조건부 추천"
        elif score >= 4.0:
            return "보류"
        else:
            return "비추천"
    
    def _generate_consensus_reasoning(self, agent_results: List[AgentResult], weights: Dict[str, float]) -> str:
        """합의 도출 근거 생성"""
        reasoning_parts = []
        
        # 참여 에이전트 정보
        reasoning_parts.append(f"총 {len(agent_results)}개 에이전트가 참여한 협업 분석")
        
        # 주요 의견들
        for result in agent_results:
            weight = weights.get(result.agent_id, 0)
            reasoning_parts.append(
                f"- {result.agent_id}: 신뢰도 {result.confidence:.2f}, 가중치 {weight:.2f}"
            )
        
        return " | ".join(reasoning_parts)
    
    def _synthesize_peer_reviewed_results(
        self, 
        initial_results: List[AgentResult], 
        peer_reviews: List[AgentResult]
    ) -> Dict[str, Any]:
        """상호 검토 결과 종합"""
        # 초기 분석과 검토 결과를 결합하여 최종 결과 생성
        synthesis = {
            "initial_analyses": len(initial_results),
            "peer_reviews": len(peer_reviews),
            "synthesis_method": "peer_review_weighted",
            "final_decision": "종합 검토 결과",
            "synthesis_timestamp": datetime.now().isoformat()
        }
        
        return synthesis
    
    def _calculate_peer_review_consensus(
        self, 
        initial_results: List[AgentResult], 
        peer_reviews: List[AgentResult]
    ) -> float:
        """상호 검토 기반 합의 수준 계산"""
        # 초기 분석 합의 수준
        initial_consensus = self._calculate_consensus_level(initial_results)
        
        # 검토 결과 존재 여부에 따른 가중치
        review_weight = 0.3 if peer_reviews else 0
        
        # 최종 합의 수준
        final_consensus = initial_consensus * (1 - review_weight) + review_weight * 0.8
        
        return final_consensus
    
    def _update_performance_metrics(self, result: CollaborationResult):
        """성능 메트릭 업데이트"""
        self.performance_metrics["total_collaborations"] += 1
        
        if result.success:
            self.performance_metrics["successful_collaborations"] += 1
        
        # 합의 수준 평균 업데이트
        total = self.performance_metrics["total_collaborations"]
        current_avg = self.performance_metrics["average_consensus_level"]
        self.performance_metrics["average_consensus_level"] = (
            (current_avg * (total - 1) + result.consensus_level) / total
        )
        
        # 처리 시간 평균 업데이트
        current_time_avg = self.performance_metrics["average_processing_time"]
        self.performance_metrics["average_processing_time"] = (
            (current_time_avg * (total - 1) + result.total_time) / total
        )
        
        # 패턴 사용 통계
        self.performance_metrics["pattern_usage"][result.pattern] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        return {
            "metrics": self.performance_metrics,
            "registered_agents": len(self.agents),
            "collaboration_history_size": len(self.collaboration_history),
            "success_rate": (
                self.performance_metrics["successful_collaborations"] / 
                max(1, self.performance_metrics["total_collaborations"])
            ),
            "report_timestamp": datetime.now().isoformat()
        }


# 사용 예시
def example_a2a_usage():
    """A2A 협업 사용 예시"""
    print("🔗 A2A (Agent-to-Agent) 협업 사용 예시")
    print("=" * 50)
    
    # A2A 시스템 초기화
    a2a = A2ACollaboration()
    
    # 가상의 에이전트들 등록 (실제로는 진짜 에이전트 인스턴스)
    class MockAgent:
        def __init__(self, name, specialty):
            self.name = name
            self.specialty = specialty
        
        def analyze(self, data):
            return {
                "agent_name": self.name,
                "specialty": self.specialty,
                "technical_score": 8.5,
                "policy_score": 7.2,
                "confidence": 0.85,
                "recommendation": f"{self.specialty} 관점에서 적합"
            }
    
    # 에이전트 등록
    a2a.register_agent("tech_expert", MockAgent("기술전문가", "AI/ML"))
    a2a.register_agent("policy_expert", MockAgent("정책전문가", "정부정책"))
    a2a.register_agent("hr_expert", MockAgent("인사전문가", "인재관리"))
    
    # 협업 작업 정의
    task = CollaborationTask(
        task_id="candidate_evaluation_001",
        input_data={
            "candidate_name": "김철수",
            "background": "AI 박사, 연구소 3년",
            "target_position": "AI정책관"
        },
        target_agents=["tech_expert", "policy_expert", "hr_expert"],
        pattern=CollaborationPattern.CONSENSUS
    )
    
    # 협업 실행
    result = a2a.collaborate(task)
    
    print(f"✅ 협업 완료: {result.success}")
    print(f"🎯 합의 수준: {result.consensus_level:.2f}")
    print(f"⏱️ 처리 시간: {result.total_time:.2f}초")
    print(f"📊 참여 에이전트: {len(result.agent_results)}개")
    
    # 성능 리포트
    report = a2a.get_performance_report()
    print(f"📈 성공률: {report['success_rate']:.2%}")


if __name__ == "__main__":
    example_a2a_usage() 