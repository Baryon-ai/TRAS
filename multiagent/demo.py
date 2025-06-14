#!/usr/bin/env python3
"""
🎭 TRAS 6.0 Multi-Agent System Demo
===================================

Section 6 강의 내용을 바탕으로 구현된 멀티에이전트 시스템의
실제 동작을 보여주는 종합 데모 스크립트입니다.

사용법:
    python demo.py
    
또는:
    uv run python multiagent/demo.py
"""

import sys
import os
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from multiagent.integration import TRAS_6_0_MultiAgent
    from multiagent.mcp import ModelContextProtocol, MessageType
    from multiagent.tras_agents import (
        AITechnicalAgent, PolicyExpertAgent, LeadershipAgent,
        BiasDetectionAgent, MasterCoordinatorAgent
    )
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("multiagent 패키지가 올바르게 설치되었는지 확인하세요.")
    sys.exit(1)


def print_banner():
    """데모 시작 배너"""
    print("🎭" + "=" * 60 + "🎭")
    print("   TRAS 6.0 Multi-Agent System 종합 데모")
    print("   Section 6: Multi-Agent Cooperation 구현")
    print("🎭" + "=" * 60 + "🎭")
    print()


def demo_individual_agents():
    """개별 에이전트 데모"""
    print("🔍 1단계: 개별 전문가 에이전트 테스트")
    print("-" * 50)
    
    # 테스트 후보자 데이터
    candidate_data = {
        "name": "김AI",
        "background": "AI 박사, 정부 연구소 3년, 프로젝트 리더 경험, 정책 자문위원",
        "education": "컴퓨터공학 박사 (KAIST)",
        "experience": "AI정책연구소 선임연구원, 머신러닝 프로젝트 8개 관리",
        "target_position": "AI정책관"
    }
    
    print(f"📋 분석 대상: {candidate_data['name']}")
    print(f"🎯 목표 직책: {candidate_data['target_position']}")
    print()
    
    # 각 전문가 에이전트 테스트
    agents = [
        ("🤖", AITechnicalAgent()),
        ("🏛️", PolicyExpertAgent()),
        ("👑", LeadershipAgent()),
        ("⚖️", BiasDetectionAgent())
    ]
    
    agent_results = {}
    
    for emoji, agent in agents:
        try:
            print(f"{emoji} {agent.name} 분석 중...")
            result = agent.analyze(candidate_data)
            agent_results[agent.agent_id] = result
            
            # 결과 요약 출력
            if "recommendation" in result:
                print(f"   ✅ 추천: {result['recommendation']}")
            if "confidence" in result:
                print(f"   📊 신뢰도: {result['confidence']:.2f}")
            
            # 특별한 메트릭 출력
            if hasattr(agent, 'specialty') and "기술" in agent.specialty:
                tech_score = result.get("technical_score", 0)
                print(f"   🔧 기술 점수: {tech_score}")
            elif hasattr(agent, 'specialty') and "정책" in agent.specialty:
                policy_score = result.get("policy_score", 0)
                print(f"   🏛️ 정책 점수: {policy_score}")
            elif hasattr(agent, 'specialty') and "리더십" in agent.specialty:
                leadership_score = result.get("leadership_score", 0)
                print(f"   👑 리더십 점수: {leadership_score}")
            elif hasattr(agent, 'specialty') and "편향" in agent.specialty:
                fairness_score = result.get("fairness_score", 0)
                print(f"   ⚖️ 공정성 점수: {fairness_score}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ 분석 실패: {e}")
            print()
    
    return agent_results


def demo_mcp_protocol():
    """MCP 프로토콜 데모"""
    print("📡 2단계: MCP (Model Context Protocol) 테스트")
    print("-" * 50)
    
    # MCP 인스턴스 생성
    mcp_tech = ModelContextProtocol("tech_agent")
    mcp_policy = ModelContextProtocol("policy_agent")
    
    print("🔧 MCP 인스턴스 초기화 완료")
    
    # 테스트 메시지 생성
    test_analysis = {
        "candidate": "김AI",
        "technical_score": 9.2,
        "assessment": "AI 전문성 우수",
        "recommendation": "강력 추천"
    }
    
    # 메시지 생성 및 전송
    message = mcp_tech.create_message(
        sender="tech_agent",
        receiver="policy_agent",
        content=test_analysis,
        message_type=MessageType.ANALYSIS,
        confidence=0.92
    )
    
    print(f"📨 메시지 생성: {message.message_id[:8]}...")
    
    # 메시지 전송
    success = mcp_tech.send_message(message)
    print(f"📤 메시지 전송: {'성공' if success else '실패'}")
    
    # 브로드캐스트 테스트
    broadcast_success = mcp_tech.broadcast_analysis(
        sender="tech_agent",
        analysis_result=test_analysis,
        confidence=0.92
    )
    print(f"📡 브로드캐스트: {'성공' if broadcast_success else '실패'}")
    
    # 통계 확인
    stats = mcp_tech.get_statistics()
    print(f"📊 MCP 통계:")
    print(f"   전송된 메시지: {stats['messages_sent']}개")
    print(f"   브로드캐스트: {stats['broadcasts_sent']}개")
    print(f"   에러: {stats['errors']}개")
    print()


def demo_master_coordinator(agent_results):
    """마스터 코디네이터 데모"""
    print("🎯 3단계: 마스터 코디네이터 최종 결정")
    print("-" * 50)
    
    # 마스터 코디네이터 초기화
    coordinator = MasterCoordinatorAgent()
    print(f"🎯 {coordinator.name} 초기화 완료")
    
    # 편향 검사 결과 (간단한 가정)
    bias_check = {
        "audit_passed": True,
        "fairness_score": 8.5,
        "bias_risks": {
            "gender_bias": 0.1,
            "education_bias": 0.2,
            "regional_bias": 0.05
        }
    }
    
    try:
        # 최종 결정 수행
        final_decision = coordinator.synthesize_decision(
            specialist_analyses=agent_results,
            peer_reviews={},
            bias_check=bias_check,
            position_requirements={}
        )
        
        print("🎯 최종 결정 완료!")
        
        # 결과 상세 출력
        decision = final_decision["final_decision"]
        print(f"📋 추천 등급: {decision['recommendation_level']}")
        print(f"📊 종합 점수: {decision['overall_score']}")
        print(f"🤝 합의 수준: {decision['consensus_level']:.2f}")
        print(f"⚖️ 편향 검사: {'통과' if decision['bias_audit_passed'] else '재검토'}")
        print(f"🎯 의사결정 신뢰도: {final_decision['confidence']:.2f}")
        print()
        
        # 결정 근거 출력
        rationale = final_decision.get("decision_rationale", "근거 정보 없음")
        print(f"💭 결정 근거: {rationale}")
        print()
        
        return final_decision
        
    except Exception as e:
        print(f"❌ 마스터 코디네이터 실행 실패: {e}")
        return None


def demo_full_system():
    """전체 시스템 통합 데모"""
    print("🚀 4단계: TRAS 6.0 전체 시스템 통합 테스트")
    print("-" * 50)
    
    try:
        # TRAS 6.0 시스템 초기화
        tras_6 = TRAS_6_0_MultiAgent()
        
        # 종합 후보자 데이터
        comprehensive_candidate = {
            "name": "이정부",
            "background": "AI 정책학 박사, 정부 부처 5년 근무, 팀장 경험, 국제 협력 프로젝트 관리",
            "education": "정책학 박사 (서울대), 컴퓨터공학 석사 (KAIST)",
            "experience": "과학기술정보통신부 과장, AI 정책 기획팀장, 디지털 뉴딜 정책 수립",
            "target_position": "AI정책관",
            "contact_info": {"email": "lee.gov@example.com"},
            "additional_info": "정부 AI 윤리 가이드라인 작성, 국제 AI 거버넌스 포럼 참석"
        }
        
        print(f"👤 종합 분석 대상: {comprehensive_candidate['name']}")
        print(f"🎯 목표 직책: {comprehensive_candidate['target_position']}")
        print()
        
        # 전체 시스템 분석 실행
        full_result = tras_6.enhanced_analysis(comprehensive_candidate)
        
        # 결과 출력
        print("🎉 TRAS 6.0 분석 완료!")
        
        final_decision = full_result["final_decision"]["final_decision"]
        print(f"🏆 최종 결정: {final_decision['recommendation_level']}")
        print(f"📊 종합 점수: {final_decision['overall_score']}")
        print(f"🤝 전문가 합의: {final_decision['consensus_level']:.2f}")
        
        # 성능 개선 효과 출력
        print(f"\n🚀 TRAS 6.0 성능 개선 효과:")
        improvements = full_result["performance_improvements"]
        for metric, data in improvements.items():
            print(f"   {metric}: {data['from']}% → {data['to']}% (+{data['improvement']}%)")
        
        # 데이터베이스 저장 시도
        print(f"\n💾 데이터베이스 저장 시도...")
        save_success = tras_6.save_to_database(full_result)
        if save_success:
            print("✅ 데이터베이스 저장 성공!")
        else:
            print("⚠️ 데이터베이스 저장 실패 (정상적 - 테스트 환경)")
        
        return full_result
        
    except Exception as e:
        print(f"❌ 전체 시스템 테스트 실패: {e}")
        return None


def demo_performance_comparison():
    """성능 비교 데모"""
    print("📈 5단계: 성능 비교 및 시스템 통계")
    print("-" * 50)
    
    # TRAS 버전별 성능 비교
    performance_history = {
        "TRAS 1.0": {"accuracy": 65, "bias": 40, "explanation": 30},
        "TRAS 4.0 (GRPO)": {"accuracy": 89, "bias": 15, "explanation": 90},
        "TRAS 6.0 (Multi-Agent)": {"accuracy": 94, "bias": 8, "explanation": 97}
    }
    
    print("📊 TRAS 시스템 진화 비교:")
    print()
    
    for version, metrics in performance_history.items():
        print(f"🔸 {version}:")
        print(f"   정확도: {metrics['accuracy']}%")
        print(f"   편향률: {metrics['bias']}%")
        print(f"   설명품질: {metrics['explanation']}%")
        print()
    
    # 개선 효과 계산
    v1_to_v6_accuracy = performance_history["TRAS 6.0 (Multi-Agent)"]["accuracy"] - performance_history["TRAS 1.0"]["accuracy"]
    v4_to_v6_accuracy = performance_history["TRAS 6.0 (Multi-Agent)"]["accuracy"] - performance_history["TRAS 4.0 (GRPO)"]["accuracy"]
    
    print(f"🚀 전체 개선 효과 (v1.0 → v6.0):")
    print(f"   정확도 개선: +{v1_to_v6_accuracy}%p")
    print(f"   편향 감소: -{performance_history['TRAS 1.0']['bias'] - performance_history['TRAS 6.0 (Multi-Agent)']['bias']}%p")
    print(f"   설명품질 향상: +{performance_history['TRAS 6.0 (Multi-Agent)']['explanation'] - performance_history['TRAS 1.0']['explanation']}%p")
    print()
    
    print(f"🔥 최근 개선 효과 (v4.0 → v6.0):")
    print(f"   정확도: +{v4_to_v6_accuracy}%p (멀티에이전트 협업 효과)")
    print(f"   편향 감소: -{performance_history['TRAS 4.0 (GRPO)']['bias'] - performance_history['TRAS 6.0 (Multi-Agent)']['bias']}%p (전담 편향 검사 에이전트)")
    print(f"   설명품질: +{performance_history['TRAS 6.0 (Multi-Agent)']['explanation'] - performance_history['TRAS 4.0 (GRPO)']['explanation']}%p (전문가별 상세 근거)")


def main():
    """메인 데모 실행"""
    print_banner()
    
    try:
        # 1단계: 개별 에이전트 테스트
        agent_results = demo_individual_agents()
        
        # 2단계: MCP 프로토콜 테스트
        demo_mcp_protocol()
        
        # 3단계: 마스터 코디네이터 테스트
        coordinator_result = demo_master_coordinator(agent_results)
        
        # 4단계: 전체 시스템 통합 테스트
        full_system_result = demo_full_system()
        
        # 5단계: 성능 비교
        demo_performance_comparison()
        
        # 최종 요약
        print("🎉" + "=" * 60 + "🎉")
        print("   TRAS 6.0 Multi-Agent System 데모 완료!")
        print("   여러 AI가 함께 더 똑똑한 정부 인재 추천!")
        print("🎉" + "=" * 60 + "🎉")
        print()
        
        if full_system_result:
            final_rec = full_system_result["final_decision"]["final_decision"]["recommendation_level"]
            print(f"✨ 데모 후보자 최종 결과: {final_rec}")
        
        print(f"⏰ 데모 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 사용법 안내
        print("📚 추가 정보:")
        print("   - Section 6 강의자료: slides/section6_multiagent_cooperation.md")
        print("   - 모듈 사용법: multiagent/README.md")
        print("   - 전체 시스템: main.py")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 데모 실행 중 오류 발생: {e}")
        print("🔧 문제 해결 방법:")
        print("   1. 필요한 모듈이 모두 설치되었는지 확인")
        print("   2. Python 경로 설정 확인")
        print("   3. multiagent 패키지 구조 확인")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 