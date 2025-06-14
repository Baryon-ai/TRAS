"""
🏗️ Multi-Agent System Base
===========================

기본 멀티에이전트 시스템 아키텍처

Section 6 강의의 "오케스트라 vs 원맨밴드" 개념을 바탕으로
구현된 기본 멀티에이전트 시스템입니다.

핵심 구성 요소:
- BaseAgent: 모든 에이전트의 기본 클래스
- MultiAgentSystem: 에이전트들을 관리하는 시스템
- AgentRegistry: 에이전트 등록 및 관리
- TaskDistributor: 작업 분배 시스템
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class AgentStatus(Enum):
    """에이전트 상태"""
    IDLE = "idle"              # 대기 중
    WORKING = "working"        # 작업 중
    WAITING = "waiting"        # 대기 중 (다른 에이전트의 결과 기다림)
    ERROR = "error"            # 오류 발생
    OFFLINE = "offline"        # 오프라인


class TaskPriority(Enum):
    """작업 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class AgentCapability:
    """에이전트 능력 정의"""
    name: str                              # 능력 이름
    description: str                       # 능력 설명
    proficiency_level: float              # 숙련도 (0.0-1.0)
    processing_time_estimate: float       # 예상 처리 시간 (초)
    resource_requirements: Dict[str, Any] # 필요 리소스


@dataclass
class TaskRequest:
    """작업 요청"""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    priority: TaskPriority
    required_capabilities: List[str]
    deadline: Optional[str] = None
    requester: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class TaskResult:
    """작업 결과"""
    task_id: str
    agent_id: str
    result_data: Dict[str, Any]
    confidence: float
    processing_time: float
    status: str
    error_message: Optional[str] = None
    completed_at: str = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now().isoformat()


class BaseAgent(ABC):
    """
    모든 에이전트의 기본 클래스
    
    Section 6 강의의 "전문 악기 연주자" 개념을 구현
    각 에이전트는 특정 분야의 전문가 역할을 수행
    """
    
    def __init__(
        self, 
        agent_id: str, 
        name: str, 
        description: str,
        capabilities: List[AgentCapability] = None
    ):
        """
        베이스 에이전트 초기화
        
        Args:
            agent_id: 에이전트 고유 ID
            name: 에이전트 이름
            description: 에이전트 설명
            capabilities: 에이전트 능력 목록
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        
        # 상태 관리
        self.status = AgentStatus.IDLE
        self.current_task: Optional[TaskRequest] = None
        self.task_history: List[TaskResult] = []
        
        # 성능 메트릭
        self.performance_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # 등록 정보
        self.registered_at = datetime.now().isoformat()
        self.last_active = datetime.now().isoformat()
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        데이터 분석 (모든 에이전트가 구현해야 함)
        
        Args:
            data: 분석할 데이터
            
        Returns:
            분석 결과
        """
        pass
    
    def can_handle_task(self, task: TaskRequest) -> bool:
        """
        작업 처리 가능 여부 확인
        
        Args:
            task: 작업 요청
            
        Returns:
            처리 가능 여부
        """
        # 현재 상태 확인
        if self.status not in [AgentStatus.IDLE, AgentStatus.WAITING]:
            return False
        
        # 필요 능력 확인
        agent_capability_names = [cap.name for cap in self.capabilities]
        return all(
            required_cap in agent_capability_names 
            for required_cap in task.required_capabilities
        )
    
    def process_task(self, task: TaskRequest) -> TaskResult:
        """
        작업 처리
        
        Args:
            task: 처리할 작업
            
        Returns:
            작업 결과
        """
        start_time = datetime.now()
        self.status = AgentStatus.WORKING
        self.current_task = task
        
        try:
            # 실제 분석 수행
            result_data = self.analyze(task.input_data)
            
            # 신뢰도 계산 (서브클래스에서 오버라이드 가능)
            confidence = self._calculate_confidence(result_data)
            
            # 처리 시간 계산
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 결과 객체 생성
            result = TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                result_data=result_data,
                confidence=confidence,
                processing_time=processing_time,
                status="completed"
            )
            
            # 성능 메트릭 업데이트
            self._update_performance_metrics(result)
            
            # 작업 기록 저장
            self.task_history.append(result)
            
            # 상태 복원
            self.status = AgentStatus.IDLE
            self.current_task = None
            self.last_active = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            # 오류 처리
            processing_time = (datetime.now() - start_time).total_seconds()
            
            error_result = TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                result_data={"error": str(e)},
                confidence=0.0,
                processing_time=processing_time,
                status="failed",
                error_message=str(e)
            )
            
            # 오류 메트릭 업데이트
            self.performance_metrics["failed_tasks"] += 1
            self.performance_metrics["total_tasks"] += 1
            
            # 상태 복원
            self.status = AgentStatus.ERROR
            self.current_task = None
            
            return error_result
    
    def _calculate_confidence(self, result_data: Dict[str, Any]) -> float:
        """
        결과에 대한 신뢰도 계산
        
        서브클래스에서 오버라이드하여 특화된 신뢰도 계산 구현 가능
        """
        # 기본 신뢰도 계산 로직
        if "error" in result_data:
            return 0.0
        
        # 결과 데이터의 완성도 기반 신뢰도
        expected_fields = ["analysis", "recommendation", "score"]
        present_fields = sum(1 for field in expected_fields if field in result_data)
        completeness = present_fields / len(expected_fields)
        
        # 기본 신뢰도 0.5 + 완성도 보너스 0.5
        return 0.5 + (completeness * 0.5)
    
    def _update_performance_metrics(self, result: TaskResult):
        """성능 메트릭 업데이트"""
        self.performance_metrics["total_tasks"] += 1
        
        if result.status == "completed":
            self.performance_metrics["successful_tasks"] += 1
            
            # 평균 신뢰도 업데이트
            total = self.performance_metrics["total_tasks"]
            current_avg = self.performance_metrics["average_confidence"]
            self.performance_metrics["average_confidence"] = (
                (current_avg * (total - 1) + result.confidence) / total
            )
        
        # 평균 처리 시간 업데이트
        total = self.performance_metrics["total_tasks"]
        current_time_avg = self.performance_metrics["average_processing_time"]
        self.performance_metrics["average_processing_time"] = (
            (current_time_avg * (total - 1) + result.processing_time) / total
        )
        
        self.performance_metrics["total_processing_time"] += result.processing_time
    
    def get_status_info(self) -> Dict[str, Any]:
        """에이전트 상태 정보 반환"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "current_task_id": self.current_task.task_id if self.current_task else None,
            "capabilities": [cap.name for cap in self.capabilities],
            "performance": self.performance_metrics,
            "last_active": self.last_active
        }
    
    def reset_status(self):
        """에이전트 상태 초기화"""
        self.status = AgentStatus.IDLE
        self.current_task = None


class AgentRegistry:
    """
    에이전트 등록 및 관리 시스템
    
    "오케스트라의 악기 편성표" 역할
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.capability_index: Dict[str, List[str]] = {}  # 능력별 에이전트 인덱스
    
    def register_agent(self, agent: BaseAgent) -> bool:
        """
        에이전트 등록
        
        Args:
            agent: 등록할 에이전트
            
        Returns:
            등록 성공 여부
        """
        try:
            # 중복 ID 확인
            if agent.agent_id in self.agents:
                print(f"⚠️ 이미 등록된 에이전트 ID: {agent.agent_id}")
                return False
            
            # 에이전트 등록
            self.agents[agent.agent_id] = agent
            
            # 능력별 인덱스 업데이트
            for capability in agent.capabilities:
                if capability.name not in self.capability_index:
                    self.capability_index[capability.name] = []
                self.capability_index[capability.name].append(agent.agent_id)
            
            print(f"✅ 에이전트 등록 완료: {agent.name} ({agent.agent_id})")
            return True
            
        except Exception as e:
            print(f"❌ 에이전트 등록 실패: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """에이전트 등록 해제"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # 능력별 인덱스에서 제거
        for capability in agent.capabilities:
            if capability.name in self.capability_index:
                if agent_id in self.capability_index[capability.name]:
                    self.capability_index[capability.name].remove(agent_id)
        
        # 에이전트 제거
        del self.agents[agent_id]
        print(f"✅ 에이전트 등록 해제: {agent.name}")
        return True
    
    def find_capable_agents(self, required_capabilities: List[str]) -> List[str]:
        """
        필요 능력을 가진 에이전트 찾기
        
        Args:
            required_capabilities: 필요한 능력 목록
            
        Returns:
            조건을 만족하는 에이전트 ID 목록
        """
        candidate_agents = set()
        
        for capability in required_capabilities:
            if capability in self.capability_index:
                if not candidate_agents:
                    candidate_agents = set(self.capability_index[capability])
                else:
                    candidate_agents &= set(self.capability_index[capability])
        
        # 현재 사용 가능한 에이전트만 필터링
        available_agents = []
        for agent_id in candidate_agents:
            agent = self.agents.get(agent_id)
            if agent and agent.status in [AgentStatus.IDLE, AgentStatus.WAITING]:
                available_agents.append(agent_id)
        
        return available_agents
    
    def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """ID로 에이전트 조회"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """모든 에이전트 반환"""
        return self.agents.copy()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """등록소 통계"""
        status_counts = {}
        for agent in self.agents.values():
            status = agent.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_agents": len(self.agents),
            "status_distribution": status_counts,
            "available_capabilities": list(self.capability_index.keys()),
            "registry_created": datetime.now().isoformat()
        }


class TaskDistributor:
    """
    작업 분배 시스템
    
    "오케스트라 지휘자" 역할 - 적절한 에이전트에게 작업 할당
    """
    
    def __init__(self, agent_registry: AgentRegistry):
        self.agent_registry = agent_registry
        self.task_queue: List[TaskRequest] = []
        self.completed_tasks: List[TaskResult] = []
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
    
    def submit_task(self, task: TaskRequest) -> bool:
        """
        작업 제출
        
        Args:
            task: 제출할 작업
            
        Returns:
            제출 성공 여부
        """
        try:
            # 작업 처리 가능한 에이전트 확인
            capable_agents = self.agent_registry.find_capable_agents(
                task.required_capabilities
            )
            
            if not capable_agents:
                print(f"❌ 작업 처리 가능한 에이전트 없음: {task.task_type}")
                return False
            
            # 작업 큐에 추가
            self.task_queue.append(task)
            print(f"✅ 작업 제출 완료: {task.task_id}")
            return True
            
        except Exception as e:
            print(f"❌ 작업 제출 실패: {e}")
            return False
    
    def assign_task(self, task: TaskRequest) -> Optional[str]:
        """
        작업을 적절한 에이전트에게 할당
        
        Args:
            task: 할당할 작업
            
        Returns:
            할당된 에이전트 ID (실패 시 None)
        """
        # 능력을 가진 에이전트 찾기
        capable_agents = self.agent_registry.find_capable_agents(
            task.required_capabilities
        )
        
        if not capable_agents:
            return None
        
        # 에이전트 선택 전략 (성능 기반)
        best_agent_id = self._select_best_agent(capable_agents, task)
        
        if best_agent_id:
            self.task_assignments[task.task_id] = best_agent_id
            return best_agent_id
        
        return None
    
    def _select_best_agent(self, candidate_agents: List[str], task: TaskRequest) -> Optional[str]:
        """
        최적의 에이전트 선택
        
        성능 지표와 능력 수준을 고려한 선택
        """
        best_agent_id = None
        best_score = -1
        
        for agent_id in candidate_agents:
            agent = self.agent_registry.get_agent_by_id(agent_id)
            if not agent:
                continue
            
            # 선택 점수 계산
            score = self._calculate_agent_score(agent, task)
            
            if score > best_score:
                best_score = score
                best_agent_id = agent_id
        
        return best_agent_id
    
    def _calculate_agent_score(self, agent: BaseAgent, task: TaskRequest) -> float:
        """에이전트 선택 점수 계산"""
        score = 0.0
        
        # 성공률 (40%)
        total_tasks = agent.performance_metrics["total_tasks"]
        if total_tasks > 0:
            success_rate = agent.performance_metrics["successful_tasks"] / total_tasks
            score += success_rate * 0.4
        else:
            score += 0.2  # 신규 에이전트 기본 점수
        
        # 평균 신뢰도 (30%)
        score += agent.performance_metrics["average_confidence"] * 0.3
        
        # 능력 매칭도 (20%)
        matching_capabilities = 0
        for required_cap in task.required_capabilities:
            for agent_cap in agent.capabilities:
                if agent_cap.name == required_cap:
                    matching_capabilities += agent_cap.proficiency_level
                    break
        
        if task.required_capabilities:
            capability_score = matching_capabilities / len(task.required_capabilities)
            score += capability_score * 0.2
        
        # 가용성 (10%)
        if agent.status == AgentStatus.IDLE:
            score += 0.1
        
        return score
    
    def process_task_queue(self) -> List[TaskResult]:
        """작업 큐 처리"""
        results = []
        
        # 우선순위별로 정렬
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        # 작업 처리
        remaining_tasks = []
        for task in self.task_queue:
            agent_id = self.assign_task(task)
            if agent_id:
                agent = self.agent_registry.get_agent_by_id(agent_id)
                if agent:
                    result = agent.process_task(task)
                    results.append(result)
                    self.completed_tasks.append(result)
                    print(f"✅ 작업 완료: {task.task_id} by {agent.name}")
                else:
                    remaining_tasks.append(task)
            else:
                remaining_tasks.append(task)
                print(f"⏳ 작업 대기: {task.task_id} (처리 가능한 에이전트 없음)")
        
        self.task_queue = remaining_tasks
        return results
    
    def get_task_stats(self) -> Dict[str, Any]:
        """작업 통계"""
        return {
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "active_assignments": len(self.task_assignments),
            "stats_timestamp": datetime.now().isoformat()
        }


class MultiAgentSystem:
    """
    멀티에이전트 시스템 메인 클래스
    
    Section 6 강의의 "오케스트라" 전체를 관리하는 시스템
    """
    
    def __init__(self, system_name: str = "TRAS Multi-Agent System"):
        """
        멀티에이전트 시스템 초기화
        
        Args:
            system_name: 시스템 이름
        """
        self.system_name = system_name
        self.agent_registry = AgentRegistry()
        self.task_distributor = TaskDistributor(self.agent_registry)
        
        # 시스템 정보
        self.created_at = datetime.now().isoformat()
        self.system_id = str(uuid.uuid4())
        
        # 시스템 통계
        self.system_stats = {
            "total_agents_registered": 0,
            "total_tasks_processed": 0,
            "system_uptime": 0,
            "performance_score": 0.0
        }
    
    def add_agent(self, agent: BaseAgent) -> bool:
        """에이전트 추가"""
        success = self.agent_registry.register_agent(agent)
        if success:
            self.system_stats["total_agents_registered"] += 1
        return success
    
    def remove_agent(self, agent_id: str) -> bool:
        """에이전트 제거"""
        return self.agent_registry.unregister_agent(agent_id)
    
    def submit_task(self, task: TaskRequest) -> bool:
        """작업 제출"""
        return self.task_distributor.submit_task(task)
    
    def process_pending_tasks(self) -> List[TaskResult]:
        """대기 중인 작업들 처리"""
        results = self.task_distributor.process_task_queue()
        self.system_stats["total_tasks_processed"] += len(results)
        return results
    
    def get_system_overview(self) -> Dict[str, Any]:
        """시스템 전체 현황"""
        registry_stats = self.agent_registry.get_registry_stats()
        task_stats = self.task_distributor.get_task_stats()
        
        return {
            "system_info": {
                "name": self.system_name,
                "id": self.system_id,
                "created_at": self.created_at
            },
            "agents": registry_stats,
            "tasks": task_stats,
            "system_stats": self.system_stats,
            "overview_timestamp": datetime.now().isoformat()
        }
    
    def collaborative_analysis(
        self, 
        data: Dict[str, Any], 
        required_capabilities: List[str]
    ) -> Dict[str, Any]:
        """
        협업적 분석 수행
        
        여러 에이전트가 동시에 분석하여 결과를 통합
        """
        # 작업 요청 생성
        task = TaskRequest(
            task_id=f"collaborative_analysis_{int(datetime.now().timestamp())}",
            task_type="collaborative_analysis",
            input_data=data,
            priority=TaskPriority.NORMAL,
            required_capabilities=required_capabilities
        )
        
        # 작업 제출 및 처리
        if self.submit_task(task):
            results = self.process_pending_tasks()
            
            # 결과 통합
            if results:
                return self._integrate_collaborative_results(results)
            else:
                return {"error": "No results from collaborative analysis"}
        else:
            return {"error": "Failed to submit collaborative task"}
    
    def _integrate_collaborative_results(self, results: List[TaskResult]) -> Dict[str, Any]:
        """협업 결과 통합"""
        if not results:
            return {"error": "No results to integrate"}
        
        # 성공한 결과만 필터링
        successful_results = [r for r in results if r.status == "completed"]
        
        if not successful_results:
            return {"error": "All collaborative tasks failed"}
        
        # 결과 통합
        integrated = {
            "collaboration_summary": {
                "participating_agents": [r.agent_id for r in successful_results],
                "total_processing_time": sum(r.processing_time for r in successful_results),
                "average_confidence": sum(r.confidence for r in successful_results) / len(successful_results),
                "integration_timestamp": datetime.now().isoformat()
            },
            "individual_results": [
                {
                    "agent_id": r.agent_id,
                    "result": r.result_data,
                    "confidence": r.confidence
                }
                for r in successful_results
            ],
            "consensus_analysis": self._calculate_consensus(successful_results)
        }
        
        return integrated
    
    def _calculate_consensus(self, results: List[TaskResult]) -> Dict[str, Any]:
        """결과들 간의 합의 수준 계산"""
        if len(results) < 2:
            return {"consensus_level": 1.0, "note": "Single result, no consensus needed"}
        
        # 신뢰도 기반 합의 계산
        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        # 표준편차를 이용한 합의 수준 (낮을수록 높은 합의)
        import statistics
        if len(confidences) > 1:
            std_dev = statistics.stdev(confidences)
            consensus_level = max(0, 1 - std_dev)
        else:
            consensus_level = avg_confidence
        
        return {
            "consensus_level": consensus_level,
            "average_confidence": avg_confidence,
            "confidence_range": (min(confidences), max(confidences)),
            "agreement_strength": "높음" if consensus_level > 0.8 else "보통" if consensus_level > 0.5 else "낮음"
        }


# 사용 예시
def example_multi_agent_system():
    """멀티에이전트 시스템 사용 예시"""
    print("🏗️ Multi-Agent System 사용 예시")
    print("=" * 50)
    
    # 시스템 초기화
    mas = MultiAgentSystem("TRAS-Demo")
    
    # 예시 에이전트 구현
    class ExampleAgent(BaseAgent):
        def __init__(self, agent_id, name, specialty):
            capabilities = [
                AgentCapability(
                    name=specialty,
                    description=f"{specialty} 전문 분석",
                    proficiency_level=0.8,
                    processing_time_estimate=2.0,
                    resource_requirements={}
                )
            ]
            super().__init__(agent_id, name, f"{specialty} 전문가", capabilities)
            self.specialty = specialty
        
        def analyze(self, data):
            return {
                "specialty": self.specialty,
                "analysis": f"{self.specialty} 관점에서 분석 완료",
                "score": 8.5,
                "confidence": 0.85
            }
    
    # 에이전트 추가
    tech_agent = ExampleAgent("tech_001", "기술전문가", "technical_analysis")
    policy_agent = ExampleAgent("policy_001", "정책전문가", "policy_analysis")
    
    mas.add_agent(tech_agent)
    mas.add_agent(policy_agent)
    
    # 협업 분석 실행
    test_data = {
        "candidate": "김철수",
        "background": "AI 연구 5년",
        "position": "AI정책관"
    }
    
    result = mas.collaborative_analysis(
        data=test_data,
        required_capabilities=["technical_analysis", "policy_analysis"]
    )
    
    print("✅ 협업 분석 완료:")
    print(f"📊 참여 에이전트: {len(result.get('individual_results', []))}개")
    print(f"🎯 평균 신뢰도: {result.get('collaboration_summary', {}).get('average_confidence', 0):.2f}")
    
    # 시스템 현황
    overview = mas.get_system_overview()
    print(f"🏗️ 등록된 에이전트: {overview['agents']['total_agents']}개")
    print(f"📈 처리된 작업: {overview['system_stats']['total_tasks_processed']}개")


if __name__ == "__main__":
    example_multi_agent_system() 