"""
📡 MCP: Model Context Protocol
=============================

"AI들의 공통 언어" - Section 6 강의 내용 기반 구현

AI 모델 간 효율적 소통을 위한 표준화된 프로토콜입니다.
여러 AI 에이전트가 일관되고 신뢰할 수 있는 방식으로 
정보를 교환할 수 있도록 합니다.

주요 기능:
- 표준화된 메시지 형식
- 브로드캐스트 통신
- 신뢰도 기반 메시지 검증
- 컨텍스트 공유 메커니즘
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class MessageType(Enum):
    """메시지 타입 정의"""
    ANALYSIS = "analysis"      # 분석 결과
    QUESTION = "question"      # 질문/요청
    RESPONSE = "response"      # 응답
    DECISION = "decision"      # 결정/판단
    BROADCAST = "broadcast"    # 브로드캐스트
    STATUS = "status"          # 상태 업데이트


@dataclass
class MCPMessage:
    """MCP 표준 메시지 형식"""
    # 필수 필드
    sender: str                    # 발신자 에이전트 ID
    receiver: str                  # 수신자 에이전트 ID
    message_type: MessageType      # 메시지 타입
    content: Dict[str, Any]        # 실제 메시지 내용
    
    # 자동 생성 필드
    message_id: str = None
    timestamp: str = None
    
    # 선택적 필드
    context: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    priority: int = 1
    reply_to: Optional[str] = None
    
    def __post_init__(self):
        """메시지 생성 후 자동 필드 설정"""
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = asdict(self)
        result['message_type'] = self.message_type.value
        return result
    
    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ModelContextProtocol:
    """
    AI 모델 간 효율적 소통을 위한 프로토콜
    
    Section 6 강의에서 소개된 "AI들의 공통 언어" 구현
    """
    
    def __init__(self, agent_id: str):
        """MCP 인스턴스 초기화"""
        self.agent_id = agent_id
        self.message_queue: List[MCPMessage] = []
        self.sent_messages: List[MCPMessage] = []
        self.shared_context: Dict[str, Any] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        
        # 메시지 통계
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts_sent": 0,
            "errors": 0
        }
    
    def create_message(
        self,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.ANALYSIS,
        confidence: Optional[float] = None,
        priority: int = 1
    ) -> MCPMessage:
        """표준화된 메시지 생성"""
        message = MCPMessage(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            context=self.get_shared_context(),
            confidence=confidence,
            priority=priority
        )
        
        return message
    
    def send_message(self, message: MCPMessage) -> bool:
        """메시지 전송"""
        try:
            # 메시지 검증
            if not self._validate_message(message):
                self.stats["errors"] += 1
                return False
            
            # 메시지 큐에 추가
            self.message_queue.append(message)
            self.sent_messages.append(message)
            
            # 통계 업데이트
            self.stats["messages_sent"] += 1
            if message.receiver == "all_agents":
                self.stats["broadcasts_sent"] += 1
            
            return True
            
        except Exception as e:
            print(f"❌ 메시지 전송 실패: {e}")
            self.stats["errors"] += 1
            return False
    
    def broadcast_analysis(
        self,
        sender: str,
        analysis_result: Dict[str, Any],
        confidence: Optional[float] = None
    ) -> bool:
        """분석 결과를 모든 에이전트에게 브로드캐스트"""
        message = self.create_message(
            sender=sender,
            receiver="all_agents",
            content={
                "type": "analysis_broadcast",
                "analysis": analysis_result,
                "broadcast_time": datetime.now().isoformat()
            },
            message_type=MessageType.BROADCAST,
            confidence=confidence,
            priority=3
        )
        
        return self.send_message(message)
    
    def receive_messages(self) -> List[MCPMessage]:
        """수신된 메시지 조회"""
        received = []
        remaining = []
        
        for message in self.message_queue:
            if (message.receiver == self.agent_id or 
                message.receiver == "all_agents"):
                received.append(message)
                self.stats["messages_received"] += 1
            else:
                remaining.append(message)
        
        self.message_queue = remaining
        return received
    
    def get_shared_context(self) -> Dict[str, Any]:
        """현재 공유 컨텍스트 반환"""
        return self.shared_context.copy()
    
    def _validate_message(self, message: MCPMessage) -> bool:
        """메시지 유효성 검증"""
        try:
            # 필수 필드 확인
            if not all([message.sender, message.receiver, message.content]):
                return False
            
            # 신뢰도 범위 확인
            if message.confidence is not None:
                if not (0.0 <= message.confidence <= 1.0):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """MCP 사용 통계 반환"""
        return {
            **self.stats,
            "agent_id": self.agent_id,
            "registered_agents": len(self.agent_registry),
            "queue_size": len(self.message_queue)
        }


# 사용 예시
if __name__ == "__main__":
    print("📡 MCP (Model Context Protocol) 테스트")
    
    # MCP 인스턴스 생성
    mcp = ModelContextProtocol("test_agent")
    
    # 테스트 메시지 생성
    test_message = mcp.create_message(
        sender="test_agent",
        receiver="target_agent",
        content={"test": "Hello MCP!"},
        confidence=0.9
    )
    
    # 메시지 전송
    success = mcp.send_message(test_message)
    print(f"✅ 메시지 전송: {success}")
    
    # 통계 확인
    stats = mcp.get_statistics()
    print(f"📊 통계: {stats}") 