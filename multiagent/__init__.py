"""
🤝 TRAS Multi-Agent Cooperation System
=====================================

Section 6 강의 내용을 바탕으로 구현된 멀티에이전트 협업 시스템
"여러 AI가 함께 더 똑똑하게"

주요 구성 요소:
- MCP (Model Context Protocol): AI 간 효율적 소통 프로토콜
- A2A (Agent-to-Agent): Google의 에이전트 협업 기술
- TRAS Multi-Agent System: 정부 인재 추천 전문 에이전트들
- Intelligent Message Classifier: 지능형 메시지 분류 시스템

Copyright (c) 2025 BarionLabs
License: MIT
"""

from .mcp import ModelContextProtocol, MCPMessage
from .a2a_collaboration import A2ACollaboration, CollaborationPattern
from .multi_agent_system import MultiAgentSystem, BaseAgent
from .tras_agents import (
    AITechnicalAgent,
    PolicyExpertAgent,
    LeadershipAgent,
    BiasDetectionAgent,
    MasterCoordinatorAgent
)
# from .message_classifier import IntelligentMessageClassifier, MessageCategory
from .integration import TRAS_6_0_MultiAgent

__version__ = "6.0.0"
__author__ = "BarionLabs"
__email__ = "admin@barion.ai"

# 패키지 메타데이터
__all__ = [
    # Core Protocols
    "ModelContextProtocol",
    "MCPMessage",
    
    # Collaboration Framework
    "A2ACollaboration", 
    "CollaborationPattern",
    
    # Multi-Agent Systems
    "MultiAgentSystem",
    "BaseAgent",
    
    # TRAS Specialist Agents
    "AITechnicalAgent",
    "PolicyExpertAgent", 
    "LeadershipAgent",
    "BiasDetectionAgent",
    "MasterCoordinatorAgent",
    
    # Message Classification
    # "IntelligentMessageClassifier",
    # "MessageCategory",
    
    # Integration
    "TRAS_6_0_MultiAgent"
]

# 시스템 정보
SYSTEM_INFO = {
    "name": "TRAS Multi-Agent System",
    "version": __version__,
    "description": "여러 AI가 함께 더 똑똑한 정부 인재 추천 시스템",
    "features": [
        "MCP 기반 AI 간 소통",
        "A2A 에이전트 협업",
        "전문가 에이전트 팀",
        "지능형 메시지 라우팅",
        "기존 TRAS 완전 통합"
    ],
    "compatibility": "TRAS 3.3.3+",
    "lecture_basis": "Section 6: Multi-Agent Cooperation"
}

def get_system_info():
    """시스템 정보 반환"""
    return SYSTEM_INFO

def quick_start():
    """빠른 시작 가이드"""
    return """
    🚀 TRAS Multi-Agent 빠른 시작:
    
    1. 기본 멀티에이전트 시스템:
       from multiagent import TRASMultiAgentSystem
       mas = TRASMultiAgentSystem()
       result = mas.comprehensive_evaluation(candidate_info, "AI정책관")
    
    2. 지능형 메시지 분류:
       from multiagent import IntelligentMessageClassifier
       classifier = IntelligentMessageClassifier()
       routes = classifier.intelligent_classify_and_route(messages)
    
    3. 기존 TRAS와 통합:
       from multiagent import TRAS_6_0_MultiAgent
       tras_6 = TRAS_6_0_MultiAgent()
       enhanced_result = tras_6.process_candidate_application(data)
    """

# 버전 호환성 확인
def check_compatibility():
    """TRAS 버전 호환성 확인"""
    try:
        import main
        # main.py의 버전 정보가 있다면 확인
        return True
    except ImportError:
        print("⚠️  기존 TRAS 시스템을 찾을 수 없습니다.")
        print("   multiagent 모듈은 TRAS 3.3.3+ 버전과 함께 사용하세요.")
        return False

# 패키지 로딩 시 자동 실행
if __name__ == "__main__":
    print(f"🤝 TRAS Multi-Agent System v{__version__} 로드됨!")
    print("   여러 AI가 함께 더 똑똑하게!")
    print(quick_start()) 