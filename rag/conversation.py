"""
Conversation history management for multi-turn chat support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Message:
    """Single message in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float


class ConversationManager:
    """
    Manages conversation history for multi-turn chat.
    
    Stores conversation history per session, with automatic cleanup
    of old conversations to prevent memory leaks.
    """
    
    def __init__(self, max_conversations: int = 1000, max_history_per_conversation: int = 50):
        """
        Initialize conversation manager.
        
        Args:
            max_conversations: Maximum number of conversations to keep in memory
            max_history_per_conversation: Maximum messages per conversation
        """
        self._conversations: Dict[str, List[Message]] = {}
        self.max_conversations = max_conversations
        self.max_history_per_conversation = max_history_per_conversation
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to conversation history.
        
        Args:
            session_id: Unique session identifier
            role: "user" or "assistant"
            content: Message content
        """
        if session_id not in self._conversations:
            # Cleanup old conversations if we're at limit
            if len(self._conversations) >= self.max_conversations:
                # Remove oldest conversation (simple FIFO)
                oldest_key = next(iter(self._conversations))
                del self._conversations[oldest_key]
            
            self._conversations[session_id] = []
        
        # Add message
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
        )
        self._conversations[session_id].append(message)
        
        # Trim history if too long
        if len(self._conversations[session_id]) > self.max_history_per_conversation:
            # Keep only the most recent messages
            self._conversations[session_id] = self._conversations[session_id][-self.max_history_per_conversation:]
    
    def get_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            max_messages: Optional limit on number of messages to return
            
        Returns:
            List of messages in format [{"role": "user", "content": "..."}, ...]
        """
        if session_id not in self._conversations:
            return []
        
        messages = self._conversations[session_id]
        
        if max_messages:
            messages = messages[-max_messages:]
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session"""
        if session_id in self._conversations:
            del self._conversations[session_id]
    
    def format_history_for_llm(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> str:
        """
        Format conversation history as a string for LLM context.
        
        Args:
            session_id: Unique session identifier
            max_messages: Optional limit on number of messages
            
        Returns:
            Formatted string with conversation history
        """
        history = self.get_history(session_id, max_messages)
        
        if not history:
            return ""
        
        formatted = []
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg['content']}")
        
        return "\n\n".join(formatted)
    
    def get_conversation_count(self) -> int:
        """Get total number of active conversations"""
        return len(self._conversations)

