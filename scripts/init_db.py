"""
Database Schema Initialization Script
Creates tables for Task Executions, Agent Audit Logs, and Persistent Context.
"""
import asyncio
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.core.database import Base, engine

class TaskExecutionModel(Base):
    __tablename__ = "task_executions"

    id = Column(String(64), primary_key=True, index=True)
    goal = Column(Text, nullable=False)
    status = Column(String(32), default="PENDING", index=True)
    session_id = Column(String(64), index=True)
    payload = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class AgentAuditLogModel(Base):
    __tablename__ = "agent_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), index=True)
    agent_name = Column(String(64), index=True)
    action = Column(String(64))
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

async def init_tables():
    print("🚀 Initializing PostgreSQL Database Tables for Multi-Agent Orchestration...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ All tables created successfully!")

if __name__ == "__main__":
    asyncio.run(init_tables())
